"""
Market-specific walk-forward backtesting.

Extends the existing all-games replay idea to multiple betting markets such as
totals, BTTS, spreads, draw-no-bet, double chance, team totals, and a few
tennis derivatives. The goal is to evaluate predictive quality market by
market, even when historical odds-line snapshots are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from retrain_and_calibrate import SPORT_CONFIGS
from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


TargetBuilder = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class MarketSpec:
    key: str
    sport: str
    market_type: str
    target_builder: TargetBuilder
    required_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    line: Optional[float] = None
    notes: str = ""


def _binary_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    pos_prob = y_proba[:, 1]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pos_prob > lo) & (pos_prob <= hi)
        if mask.sum() == 0:
            continue
        empirical = y_true[mask].mean()
        predicted = pos_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(empirical - predicted)
    return float(ece)


def _multiclass_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
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


def _compute_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if y_proba.shape[1] == 2:
        return float(np.mean((y_proba[:, 1] - y_true) ** 2))
    y_onehot = np.zeros_like(y_proba)
    for i, label in enumerate(y_true):
        y_onehot[i, int(label)] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def _compute_ece(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if y_proba.shape[1] == 2:
        return _binary_ece(y_true, y_proba)
    return _multiclass_ece(y_true, y_proba)


def _parse_tennis_games(score: Any) -> Optional[int]:
    if not isinstance(score, str) or not score.strip():
        return None
    pairs = re.findall(r"(\d+)-(\d+)", score)
    if not pairs:
        return None
    total_games = 0
    for a, b in pairs:
        total_games += int(a) + int(b)
    return total_games


def _parse_tennis_set_wins(score: Any) -> Optional[tuple[int, int]]:
    if not isinstance(score, str) or not score.strip():
        return None
    pairs = re.findall(r"(\d+)-(\d+)", score)
    if not pairs:
        return None
    p1_sets = 0
    p2_sets = 0
    for a, b in pairs:
        ga = int(a)
        gb = int(b)
        if ga == gb:
            continue
        if ga > gb:
            p1_sets += 1
        else:
            p2_sets += 1
    if p1_sets == 0 and p2_sets == 0:
        return None
    return p1_sets, p2_sets


def _build_moneyline_target(df: pd.DataFrame, sport: str) -> pd.Series:
    cfg = SPORT_CONFIGS[sport]
    return df[cfg["target_col"]].astype(float)


def _over_total_target(home_col: str, away_col: str, line: float) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        total = df[home_col].astype(float) + df[away_col].astype(float)
        return (total > line).astype(float)
    return _builder


def _under_total_target(home_col: str, away_col: str, line: float) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        total = df[home_col].astype(float) + df[away_col].astype(float)
        return (total < line).astype(float)
    return _builder


def _team_total_target(score_col: str, line: float) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        return (df[score_col].astype(float) > line).astype(float)
    return _builder


def _spread_target(home_col: str, away_col: str, line: float) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        margin = df[home_col].astype(float) - df[away_col].astype(float)
        return (margin > line).astype(float)
    return _builder


def _handicap_target(home_col: str, away_col: str, handicap: float) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        margin = df[home_col].astype(float) - df[away_col].astype(float)
        return ((margin + handicap) > 0).astype(float)
    return _builder


def _soccer_btts_target(df: pd.DataFrame) -> pd.Series:
    return ((df["home_goals"].astype(float) > 0) & (df["away_goals"].astype(float) > 0)).astype(float)


def _result_binary_target(positive: set[str], exclude: Optional[set[str]] = None) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        result = df["result"].astype(str)
        out = pd.Series(np.where(result.isin(positive), 1.0, 0.0), index=df.index, dtype=float)
        if exclude:
            out.loc[result.isin(exclude)] = np.nan
        return out
    return _builder


def _tennis_total_games_target(line: float) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        total_games = df["score"].apply(_parse_tennis_games)
        return pd.Series(np.where(total_games > line, 1.0, 0.0), index=df.index, dtype=float)
    return _builder


def _tennis_straight_sets_home(df: pd.DataFrame) -> pd.Series:
    vals = []
    for score in df["score"]:
        parsed = _parse_tennis_set_wins(score)
        if parsed is None:
            vals.append(np.nan)
            continue
        p1_sets, p2_sets = parsed
        vals.append(1.0 if p1_sets > 0 and p2_sets == 0 else 0.0)
    return pd.Series(vals, index=df.index, dtype=float)


def _corners_over_target(line: float) -> TargetBuilder:
    def _builder(df: pd.DataFrame) -> pd.Series:
        total = df["home_corners"].astype(float) + df["away_corners"].astype(float)
        return (total > line).astype(float)
    return _builder


MARKET_SPECS: dict[str, list[MarketSpec]] = {
    "soccer": [
        MarketSpec("moneyline", "soccer", "moneyline", lambda df: _build_moneyline_target(df, "soccer"), notes="1X2 result replay."),
        MarketSpec("totals_over_0_5", "soccer", "totals", _over_total_target("home_goals", "away_goals", 0.5), ("home_goals", "away_goals"), line=0.5),
        MarketSpec("totals_over_1_5", "soccer", "totals", _over_total_target("home_goals", "away_goals", 1.5), ("home_goals", "away_goals"), line=1.5),
        MarketSpec("totals_over_2_5", "soccer", "totals", _over_total_target("home_goals", "away_goals", 2.5), ("home_goals", "away_goals"), line=2.5),
        MarketSpec("totals_over_3_5", "soccer", "totals", _over_total_target("home_goals", "away_goals", 3.5), ("home_goals", "away_goals"), line=3.5),
        MarketSpec("totals_under_2_5", "soccer", "totals", _under_total_target("home_goals", "away_goals", 2.5), ("home_goals", "away_goals"), line=2.5),
        MarketSpec("btts_yes", "soccer", "btts", _soccer_btts_target, ("home_goals", "away_goals")),
        MarketSpec("home_draw_no_bet", "soccer", "draw_no_bet", _result_binary_target({"home_win"}, exclude={"draw"}), ("result",)),
        MarketSpec("away_draw_no_bet", "soccer", "draw_no_bet", _result_binary_target({"away_win"}, exclude={"draw"}), ("result",)),
        MarketSpec("double_chance_home_or_draw", "soccer", "double_chance", _result_binary_target({"home_win", "draw"}), ("result",)),
        MarketSpec("double_chance_away_or_draw", "soccer", "double_chance", _result_binary_target({"away_win", "draw"}), ("result",)),
        MarketSpec("double_chance_home_or_away", "soccer", "double_chance", _result_binary_target({"home_win", "away_win"}), ("result",)),
        MarketSpec("home_asian_minus_1_5", "soccer", "spreads", _handicap_target("home_goals", "away_goals", -1.5), ("home_goals", "away_goals"), line=-1.5, notes="Asian-handicap style replay using half-goal handicap."),
        MarketSpec("away_asian_minus_1_5", "soccer", "spreads", _handicap_target("away_goals", "home_goals", -1.5), ("home_goals", "away_goals"), line=-1.5, notes="Asian-handicap style replay using half-goal handicap."),
        MarketSpec("home_asian_plus_1_5", "soccer", "spreads", _handicap_target("home_goals", "away_goals", 1.5), ("home_goals", "away_goals"), line=1.5, notes="Asian-handicap style replay using half-goal handicap."),
        MarketSpec("away_asian_plus_1_5", "soccer", "spreads", _handicap_target("away_goals", "home_goals", 1.5), ("home_goals", "away_goals"), line=1.5, notes="Asian-handicap style replay using half-goal handicap."),
        MarketSpec("home_team_total_over_1_5", "soccer", "team_total", _team_total_target("home_goals", 1.5), ("home_goals",), line=1.5),
        MarketSpec("away_team_total_over_1_5", "soccer", "team_total", _team_total_target("away_goals", 1.5), ("away_goals",), line=1.5),
        MarketSpec("corners_over_9_5", "soccer", "corners", _corners_over_target(9.5), ("home_corners", "away_corners"), line=9.5, notes="Runs only if corners columns exist."),
    ],
    "basketball": [
        MarketSpec("moneyline", "basketball", "moneyline", lambda df: _build_moneyline_target(df, "basketball"), notes="Home/away winner replay."),
        MarketSpec("totals_over_220_5", "basketball", "totals", _over_total_target("home_score", "away_score", 220.5), ("home_score", "away_score"), line=220.5),
        MarketSpec("home_cover_minus_4_5", "basketball", "spreads", _spread_target("home_score", "away_score", 4.5), ("home_score", "away_score"), line=4.5),
        MarketSpec("home_team_total_over_110_5", "basketball", "team_total", _team_total_target("home_score", 110.5), ("home_score",), line=110.5),
        MarketSpec("away_team_total_over_110_5", "basketball", "team_total", _team_total_target("away_score", 110.5), ("away_score",), line=110.5),
    ],
    "mlb": [
        MarketSpec("moneyline", "mlb", "moneyline", lambda df: _build_moneyline_target(df, "mlb")),
        MarketSpec("totals_over_8_5", "mlb", "totals", _over_total_target("home_score", "away_score", 8.5), ("home_score", "away_score"), line=8.5),
        MarketSpec("home_cover_minus_1_5", "mlb", "spreads", _spread_target("home_score", "away_score", 1.5), ("home_score", "away_score"), line=1.5),
        MarketSpec("home_team_total_over_4_5", "mlb", "team_total", _team_total_target("home_score", 4.5), ("home_score",), line=4.5),
        MarketSpec("away_team_total_over_4_5", "mlb", "team_total", _team_total_target("away_score", 4.5), ("away_score",), line=4.5),
    ],
    "nhl": [
        MarketSpec("moneyline", "nhl", "moneyline", lambda df: _build_moneyline_target(df, "nhl")),
        MarketSpec("totals_over_5_5", "nhl", "totals", _over_total_target("home_score", "away_score", 5.5), ("home_score", "away_score"), line=5.5),
        MarketSpec("home_cover_minus_1_5", "nhl", "spreads", _spread_target("home_score", "away_score", 1.5), ("home_score", "away_score"), line=1.5),
        MarketSpec("home_team_total_over_2_5", "nhl", "team_total", _team_total_target("home_score", 2.5), ("home_score",), line=2.5),
        MarketSpec("away_team_total_over_2_5", "nhl", "team_total", _team_total_target("away_score", 2.5), ("away_score",), line=2.5),
    ],
    "tennis": [
        MarketSpec("moneyline", "tennis", "moneyline", lambda df: _build_moneyline_target(df, "tennis")),
        MarketSpec("total_games_over_22_5", "tennis", "totals", _tennis_total_games_target(22.5), ("score",), line=22.5),
        MarketSpec("home_wins_in_straight_sets", "tennis", "set_betting", _tennis_straight_sets_home, ("score",)),
    ],
}


def _load_feature_cache(sport: str, cfg: dict) -> pd.DataFrame:
    path = Path(cfg["raw_cache"])
    if not path.exists():
        raise FileNotFoundError(f"No feature cache found for {sport}: {path}")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _augment_with_raw_outcomes(sport: str, df: pd.DataFrame, required_columns: Iterable[str], cfg: dict) -> pd.DataFrame:
    missing = [c for c in required_columns if c not in df.columns]
    if not missing:
        return df

    try:
        fetcher_mod = importlib.import_module(cfg["fetcher_module"])
        fetcher_cls = getattr(fetcher_mod, cfg["fetcher_cls"])
        fetcher = fetcher_cls()
        raw = fetcher.fetch_all_seasons()
    except Exception as exc:
        logger.warning("%s: could not augment missing labels %s from raw fetcher: %s", sport, missing, exc)
        return df

    if raw.empty:
        logger.warning("%s: raw fetcher returned no rows while augmenting %s", sport, missing)
        return df

    raw = raw.copy()
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"])

    join_keys_map = {
        "soccer": ["date", "home_team", "away_team"],
        "basketball": ["date", "home_team", "away_team"],
        "mlb": ["date", "home_team", "away_team"],
        "nhl": ["date", "home_team", "away_team"],
        "tennis": ["date", "player1_name", "player2_name"],
    }
    join_keys = [c for c in join_keys_map.get(sport, ["date"]) if c in df.columns and c in raw.columns]
    if not join_keys:
        logger.warning("%s: no join keys available to augment missing labels %s", sport, missing)
        return df

    raw_subset_cols = join_keys + [c for c in missing if c in raw.columns]
    if len(raw_subset_cols) == len(join_keys):
        logger.warning("%s: raw fetcher did not provide requested label columns %s", sport, missing)
        return df

    raw_subset = raw[raw_subset_cols].drop_duplicates(subset=join_keys, keep="last")
    merged = df.merge(raw_subset, on=join_keys, how="left", suffixes=("", "__raw"))
    for col in missing:
        raw_col = f"{col}__raw"
        if col not in merged.columns and raw_col in merged.columns:
            merged[col] = merged[raw_col]
        elif raw_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[raw_col])
        if raw_col in merged.columns:
            merged = merged.drop(columns=[raw_col])
    return merged


def _prepare_xy(df: pd.DataFrame, cfg: dict, spec: MarketSpec, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    keep_mask = y.notna()
    df = df.loc[keep_mask].copy()
    y = y.loc[keep_mask].copy()

    extra_drop = set(spec.required_columns) | set(spec.drop_columns)
    to_drop = [c for c in cfg["drop_cols"] if c in df.columns]
    to_drop.extend(c for c in extra_drop if c in df.columns)
    to_drop.extend(c for c in ["target", "result"] if c in df.columns and c not in spec.required_columns)
    X = df.drop(columns=list(dict.fromkeys(to_drop)), errors="ignore")
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        X = X.drop(columns=obj_cols)
    X = X.fillna(0)
    return X, y


def _evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    pred = np.argmax(y_proba, axis=1)
    labels = list(range(y_proba.shape[1]))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
        "brier": _compute_brier(y_true, y_proba),
        "ece": _compute_ece(y_true, y_proba),
    }


def _replay_event_identity_columns(sport: str, df: pd.DataFrame) -> list[str]:
    by_sport = {
        "soccer": ["date", "home_team", "away_team"],
        "basketball": ["date", "home_team", "away_team"],
        "mlb": ["date", "home_team", "away_team"],
        "nhl": ["date", "home_team", "away_team"],
        "tennis": ["date", "player1_name", "player2_name"],
    }
    cols = [c for c in by_sport.get(sport, ["date"]) if c in df.columns]
    return cols or [c for c in ["date"] if c in df.columns]


def _build_replay_event_rows(
    *,
    sport: str,
    spec: MarketSpec,
    df: pd.DataFrame,
    test_idx: pd.Index,
    y_test: pd.Series,
    proba: np.ndarray,
    pred: np.ndarray,
    period: int,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    event_df = df.loc[test_idx].copy()
    identity_cols = _replay_event_identity_columns(sport, event_df)
    event_df = event_df[identity_cols].copy()

    if "home_team" in event_df.columns and "away_team" in event_df.columns:
        event_df["match_id"] = (
            event_df["home_team"].astype(str).str.strip() + " vs " + event_df["away_team"].astype(str).str.strip()
        )
    elif "player1_name" in event_df.columns and "player2_name" in event_df.columns:
        event_df["match_id"] = (
            event_df["player1_name"].astype(str).str.strip() + " vs " + event_df["player2_name"].astype(str).str.strip()
        )
    else:
        event_df["match_id"] = ""

    confidence = np.max(proba, axis=1)
    event_df["sport"] = sport
    event_df["market"] = spec.key
    event_df["market_type"] = spec.market_type
    event_df["line"] = spec.line
    event_df["period"] = int(period)
    event_df["train_end"] = pd.Timestamp(train_end)
    event_df["test_end"] = pd.Timestamp(test_end)
    event_df["y_true"] = y_test.astype(int).to_numpy()
    event_df["y_pred"] = pred.astype(int)
    event_df["pred_confidence"] = confidence.astype(float)
    event_df["correct"] = (event_df["y_true"].to_numpy() == event_df["y_pred"].to_numpy()).astype(int)
    event_df["event_log_loss"] = [
        float(-np.log(max(float(proba[i, int(y_true)]), 1e-12)))
        for i, y_true in enumerate(event_df["y_true"].to_numpy())
    ]

    for cls in range(proba.shape[1]):
        event_df[f"pred_prob_{cls}"] = proba[:, cls].astype(float)

    return event_df.reset_index(drop=True)


def replay_market(
    sport: str,
    spec: MarketSpec,
    window_days: int = 30,
    min_train_games: int = 300,
    initial_train_days: int = 180,
) -> dict[str, Any]:
    cfg = SPORT_CONFIGS[sport]
    df = _load_feature_cache(sport, cfg)
    df = _augment_with_raw_outcomes(sport, df, spec.required_columns, cfg)

    missing = [c for c in spec.required_columns if c not in df.columns]
    if missing:
        return {
            "sport": sport,
            "market": spec.key,
            "status": "unsupported",
            "reason": f"Missing required columns: {', '.join(missing)}",
        }

    y = spec.target_builder(df)
    X, y = _prepare_xy(df, cfg, spec, y)
    df = df.loc[y.index].copy()

    if len(X) < min_train_games:
        return {
            "sport": sport,
            "market": spec.key,
            "status": "unsupported",
            "reason": f"Not enough valid rows for replay ({len(X)} < {min_train_games})",
        }

    unique_dates = pd.Series(df["date"].sort_values().unique())
    if unique_dates.empty:
        return {
            "sport": sport,
            "market": spec.key,
            "status": "unsupported",
            "reason": "No dates available in source dataframe",
        }

    start_date = df["date"].min() + pd.Timedelta(days=initial_train_days)
    train_end = max(start_date, df.iloc[min(min_train_games, len(df) - 1)]["date"])
    max_date = df["date"].max()

    period = 0
    all_true: list[float] = []
    all_pred: list[int] = []
    all_proba: list[list[float]] = []
    period_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []

    while train_end < max_date:
        test_end = train_end + pd.Timedelta(days=window_days)
        train_mask = df["date"] < train_end
        test_mask = (df["date"] >= train_end) & (df["date"] < test_end)
        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        if len(train_idx) < min_train_games or len(test_idx) == 0:
            train_end = test_end
            continue

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx].astype(int)
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx].astype(int)

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
            raise RuntimeError(f"{sport}/{spec.key}: ensemble model not built")

        proba = trainer.ensemble_model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)
        metrics = _evaluate_predictions(y_test.values, proba)
        period_rows.append(
            {
                "period": period,
                "train_end": str(pd.Timestamp(train_end)),
                "test_end": str(pd.Timestamp(test_end)),
                "train_games": int(len(train_idx)),
                "test_games": int(len(test_idx)),
                **metrics,
            }
        )

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())
        all_proba.extend(proba.tolist())
        event_frames.append(
            _build_replay_event_rows(
                sport=sport,
                spec=spec,
                df=df,
                test_idx=test_idx,
                y_test=y_test,
                proba=proba,
                pred=pred,
                period=period,
                train_end=pd.Timestamp(train_end),
                test_end=pd.Timestamp(test_end),
            )
        )
        logger.info(
            "[%s/%s] period %d train=%d test=%d accuracy=%.4f log_loss=%.4f",
            sport, spec.key, period, len(train_idx), len(test_idx), metrics["accuracy"], metrics["log_loss"],
        )

        train_end = test_end
        period += 1

    if not all_true:
        return {
            "sport": sport,
            "market": spec.key,
            "status": "unsupported",
            "reason": "No out-of-sample periods were scored",
        }

    y_true = np.array(all_true, dtype=int)
    y_proba = np.array(all_proba, dtype=float)
    labels = list(range(y_proba.shape[1]))
    summary = {
        "sport": sport,
        "market": spec.key,
        "market_type": spec.market_type,
        "status": "ok",
        "line": spec.line,
        "notes": spec.notes,
        "games_scored": len(all_true),
        "periods": len(period_rows),
        "target_mean": float(np.mean(y_true)) if len(set(y_true.tolist())) <= 2 else None,
        "overall_accuracy": float(accuracy_score(y_true, np.array(all_pred))),
        "overall_log_loss": float(log_loss(y_true, y_proba, labels=labels)),
        "overall_brier": _compute_brier(y_true, y_proba),
        "overall_ece": _compute_ece(y_true, y_proba),
        "period_details": period_rows,
    }

    results_dir = Path("reports") / "backtests" / "markets"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{sport}_{spec.key}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    if event_frames:
        events_df = pd.concat(event_frames, ignore_index=True)
        events_path = results_dir / f"{sport}_{spec.key}_events.parquet"
        events_df.to_parquet(events_path, index=False)
        summary["event_rows"] = int(len(events_df))
        summary["events_path"] = str(events_path)
    return summary


def replay_sport_markets(
    sport: str,
    market_keys: Optional[Iterable[str]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    specs = MARKET_SPECS[sport]
    selected = [spec for spec in specs if market_keys is None or spec.key in set(market_keys)]
    results = {}
    for spec in selected:
        results[spec.key] = replay_market(sport=sport, spec=spec, **kwargs)
    return results
