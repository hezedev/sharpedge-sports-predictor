#!/usr/bin/env python3
"""
Full-game historical replay backtest.

Scores every historical game out-of-sample using expanding-window retraining.
Unlike the existing bet-focused backtester, this script evaluates model quality
across all games, not just simulated wagers.

Usage:
    .venv/bin/python backtest_all_games.py --sports soccer basketball
    .venv/bin/python backtest_all_games.py --sports soccer --window-days 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from retrain_and_calibrate import SPORT_CONFIGS
from src.models.trainer import ModelTrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("all_games_backtest")


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def _log_loss_score(y_true: np.ndarray, y_proba: np.ndarray, labels: list[int]) -> float:
    return float(log_loss(y_true, y_proba, labels=labels))


def _brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_onehot = np.zeros_like(y_proba)
    for i, label in enumerate(y_true):
        y_onehot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def _ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def _prepare_xy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    to_drop = [c for c in cfg["drop_cols"] if c in df.columns]
    X = df.drop(columns=to_drop + [cfg["target_col"]], errors="ignore")
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        X = X.drop(columns=obj_cols)
    X = X.fillna(0)
    y = df[cfg["target_col"]].astype(int)
    return X, y


def _load_features(sport: str, cfg: dict) -> pd.DataFrame:
    path = Path(cfg["raw_cache"])
    if not path.exists():
        raise FileNotFoundError(f"No feature cache found for {sport}: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "target" not in df.columns:
        raise ValueError(f"{sport} feature cache has no target column")
    return df


def replay_sport(
    sport: str,
    cfg: dict,
    window_days: int,
    min_train_games: int,
    initial_train_days: int,
) -> dict[str, Any]:
    df = _load_features(sport, cfg)
    X_all, y_all = _prepare_xy(df, cfg)

    unique_dates = pd.Series(df["date"].sort_values().unique())
    if unique_dates.empty:
        raise ValueError(f"{sport}: no dates available")

    start_date = df["date"].min() + pd.Timedelta(days=initial_train_days)
    train_end = max(start_date, df.iloc[min(min_train_games, len(df) - 1)]["date"])
    max_date = df["date"].max()

    all_true: list[int] = []
    all_pred: list[int] = []
    all_proba: list[list[float]] = []
    period_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    period = 0
    while train_end < max_date:
        test_end = train_end + pd.Timedelta(days=window_days)

        train_mask = df["date"] < train_end
        test_mask = (df["date"] >= train_end) & (df["date"] < test_end)

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        if len(train_idx) < min_train_games or len(test_idx) == 0:
            train_end = test_end
            continue

        X_train = X_all.loc[train_idx]
        y_train = y_all.loc[train_idx]
        X_test = X_all.loc[test_idx]
        y_test = y_all.loc[test_idx]

        # Use the last 20% of the current training slice as an internal
        # validation split for early stopping.
        split_idx = max(1, int(len(X_train) * 0.8))
        if split_idx >= len(X_train):
            train_end = test_end
            continue

        trainer = ModelTrainer(sport=sport)
        trainer.train(
            X_train.iloc[:split_idx],
            y_train.iloc[:split_idx],
            X_train.iloc[split_idx:],
            y_train.iloc[split_idx:],
        )
        trainer.build_ensemble(X_train, y_train)

        if trainer.ensemble_model is None:
            raise RuntimeError(f"{sport}: ensemble model was not built")

        proba = trainer.ensemble_model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)

        accuracy = float(np.mean(pred == y_test.values))
        log_loss = _log_loss_score(y_test.values, proba, labels=sorted(y_train.unique().tolist()))

        period_rows.append(
            {
                "period": period,
                "train_end": str(pd.Timestamp(train_end)),
                "test_end": str(pd.Timestamp(test_end)),
                "train_games": int(len(train_idx)),
                "test_games": int(len(test_idx)),
                "accuracy": accuracy,
                "log_loss": log_loss,
            }
        )

        for local_i, row_idx in enumerate(test_idx):
            row = df.loc[row_idx]
            prediction_rows.append(
                {
                    "sport": sport,
                    "date": str(pd.Timestamp(row["date"])),
                    "match_id": str(row.get("match_id", row_idx)),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "actual_target": int(y_test.iloc[local_i]),
                    "predicted_target": int(pred[local_i]),
                    "confidence": float(np.max(proba[local_i])),
                    "probabilities": [float(x) for x in proba[local_i]],
                }
            )

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())
        all_proba.extend(proba.tolist())

        logger.info(
            "[%s] period %d train=%d test=%d accuracy=%.4f log_loss=%.4f",
            sport, period, len(train_idx), len(test_idx), accuracy, log_loss,
        )

        train_end = test_end
        period += 1

    if not all_true:
        raise RuntimeError(f"{sport}: no out-of-sample games were scored")

    proba_arr = np.array(all_proba)
    results = {
        "sport": sport,
        "games_scored": len(all_true),
        "periods": len(period_rows),
        "overall_accuracy": _accuracy(np.array(all_true), np.array(all_pred)),
        "overall_log_loss": _log_loss_score(np.array(all_true), proba_arr, labels=sorted(set(all_true))),
        "overall_brier": _brier_score(np.array(all_true), proba_arr),
        "overall_ece": _ece(np.array(all_true), proba_arr),
        "period_details": period_rows,
    }

    results_dir = Path("reports") / "backtests"
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"{sport}_all_games_predictions.json"
    pred_path.write_text(json.dumps(prediction_rows, indent=2))
    results["predictions_path"] = str(pred_path)

    summary_path = results_dir / f"{sport}_all_games_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    results["summary_path"] = str(summary_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest all historical games out-of-sample")
    parser.add_argument(
        "--sports",
        nargs="+",
        default=["soccer", "basketball"],
        choices=sorted(SPORT_CONFIGS.keys()),
    )
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--min-train-games", type=int, default=300)
    parser.add_argument("--initial-train-days", type=int, default=180)
    args = parser.parse_args()

    summary: dict[str, Any] = {}
    for sport in args.sports:
        logger.info("=" * 72)
        logger.info("Replay backtest: %s", sport)
        result = replay_sport(
            sport=sport,
            cfg=SPORT_CONFIGS[sport],
            window_days=args.window_days,
            min_train_games=args.min_train_games,
            initial_train_days=args.initial_train_days,
        )
        summary[sport] = {
            "games_scored": result["games_scored"],
            "periods": result["periods"],
            "overall_accuracy": result["overall_accuracy"],
            "overall_log_loss": result["overall_log_loss"],
            "overall_brier": result["overall_brier"],
            "overall_ece": result["overall_ece"],
            "summary_path": result["summary_path"],
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
