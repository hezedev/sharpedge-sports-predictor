"""
fit_calibrators.py
==================
Standalone script that:
  1. Re-loads feature-engineered data for each sport
  2. Splits off the last 20% as calibration set + last 10% as test set
  3. Rebuilds the ensemble for any sport missing one
  4. Fits the adaptive EnsembleCalibrator on the cal set
  5. Evaluates on the held-out test set
  6. Saves calibrator to data/models/{sport}/calibrator_{tag}.joblib

Run:
    python fit_calibrators.py [--sports soccer basketball tennis]
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_calibrators")

sys.path.insert(0, ".")
from config import settings
from src.models.artifacts import calibrator_path_for_tag, get_current_model_tag
from src.models.calibration import EnsembleCalibrator
from src.models.trainer import ModelTrainer, _SoftVotingWrapper


# ──────────────────────────────────────────────────────────────────────────────
# Per-sport configurations
# ──────────────────────────────────────────────────────────────────────────────

SPORT_CONFIGS = {
    "soccer": {
        "tag": "pl_2024_25",
        "target_col": "target",
        "drop_cols": [
            "result", "match_id", "date",
            "home_goals", "away_goals", "home_score", "away_score",
            "home_team", "away_team", "season", "competition",
        ],
        "fetcher_cls": "SoccerFetcher",
        "feature_cls": "SoccerFeatureEngineer",
        "fetcher_module": "src.data.soccer_fetcher",
        "feature_module": "src.features.soccer_features",
    },
    "basketball": {
        "tag": "nba_2025_26",
        "target_col": "target",
        "drop_cols": [
            "result", "match_id", "date", "league_id", "league_name",
            "season", "home_team", "away_team", "home_team_id", "away_team_id",
            "home_score", "away_score", "status",
            "home_first_half", "home_second_half", "away_first_half", "away_second_half",
            "home_q1", "home_q2", "home_q3", "home_q4",
            "away_q1", "away_q2", "away_q3", "away_q4", "home_ot", "away_ot",
            "point_diff", "total_points", "neg_point_diff",
        ],
        "fetcher_cls": "BallDontLieFetcher",
        "feature_cls": "BasketballFeatureEngineer",
        "fetcher_module": "src.data.balldontlie_fetcher",
        "feature_module": "src.features.basketball_features",
    },
    "tennis": {
        "tag": "atp_2022_25",
        "target_col": "target",
        "drop_cols": [
            "result", "match_id", "date", "player1_name", "player2_name",
            "tourney_name", "tourney_id", "tourney_date", "tourney_level",
            "tourney_level_name", "surface",
            "player1_id", "player2_id", "player1_entry", "player2_entry",
            "player1_ioc", "player2_ioc",
            "score", "minutes", "winner_ioc",
            "p1_svpt", "p1_1stIn", "p1_ace", "p1_df", "p1_bpSaved", "p1_bpFaced",
            "p1_1stWon", "p1_2ndWon", "p1_1st_pct", "p1_ace_rate", "p1_bp_save",
            "p2_svpt", "p2_1stIn", "p2_ace", "p2_df", "p2_bpSaved", "p2_bpFaced",
            "p2_1stWon", "p2_2ndWon", "p2_1st_pct", "p2_ace_rate", "p2_bp_save",
            "player1_rank", "player2_rank", "player1_rank_pts", "player2_rank_pts",
            "player1_seed", "player2_seed", "player1_age", "player2_age",
            "player1_ht", "player2_ht", "round_num", "best_of",
        ],
        "fetcher_cls": "TennisFetcher",
        "feature_cls": "TennisFeatureEngineer",
        "fetcher_module": "src.data.tennis_fetcher",
        "feature_module": "src.features.tennis_features",
    },
    "tennis_wta": {
        "tag": "wta_2022_24",
        "model_sport": "tennis",
        "cache_path": "data/cache/tennis_wta_features.parquet",
        "target_col": "target",
        "drop_cols": [
            "result", "match_id", "date", "player1_name", "player2_name",
            "tourney_name", "tourney_id", "tourney_date", "tourney_level",
            "tourney_level_name", "surface",
            "player1_id", "player2_id", "player1_entry", "player2_entry",
            "player1_ioc", "player2_ioc",
            "score", "minutes", "winner_ioc",
            "p1_svpt", "p1_1stIn", "p1_ace", "p1_df", "p1_bpSaved", "p1_bpFaced",
            "p1_1stWon", "p1_2ndWon", "p1_1st_pct", "p1_ace_rate", "p1_bp_save",
            "p2_svpt", "p2_1stIn", "p2_ace", "p2_df", "p2_bpSaved", "p2_bpFaced",
            "p2_1stWon", "p2_2ndWon", "p2_1st_pct", "p2_ace_rate", "p2_bp_save",
            "player1_rank", "player2_rank", "player1_rank_pts", "player2_rank_pts",
            "player1_seed", "player2_seed", "player1_age", "player2_age",
            "player1_ht", "player2_ht", "round_num", "best_of",
        ],
        "fetcher_cls": "TennisFetcher",
        "feature_cls": "TennisFeatureEngineer",
        "fetcher_module": "src.data.tennis_fetcher",
        "feature_module": "src.features.tennis_features",
    },
    "mlb": {
        "tag": "mlb_2024_25",
        "target_col": "target",
        "drop_cols": [
            "result", "game_pk", "date", "season",
            "home_team", "away_team",
            "home_score", "away_score", "home_hits", "away_hits",
            "home_errors", "away_errors", "home_innings", "away_innings",
        ],
        "fetcher_cls": "MLBFetcher",
        "feature_cls": "MLBFeatureEngineer",
        "fetcher_module": "src.data.mlb_fetcher",
        "feature_module": "src.features.mlb_features",
    },
    "nhl": {
        "tag": "nhl_2024_25",
        "target_col": "target",
        "drop_cols": [
            "result", "game_id", "date", "season",
            "home_team", "away_team",
            "home_score", "away_score",
            "home_shots", "away_shots",
            "home_pp_goals", "away_pp_goals",
            "went_to_ot",
            "home_corsi", "away_corsi",
            "home_fenwick", "away_fenwick",
            "home_xg", "away_xg",
            "home_pp_opp", "away_pp_opp",
        ],
        "fetcher_cls": "NHLFetcher",
        "feature_cls": "NHLFeatureEngineer",
        "fetcher_module": "src.data.nhl_fetcher",
        "feature_module": "src.features.nhl_features",
    },
}


def _get_feature_matrix(sport: str, cfg: dict):
    """Load or (re-)compute the feature matrix for a sport."""
    import importlib

    cache_path = Path(cfg.get("cache_path", f"data/cache/{sport}_features.parquet"))
    if cache_path.exists():
        logger.info("Loading cached features from %s", cache_path)
        df = pd.read_parquet(cache_path)
        return df

    logger.info("Building features from scratch for %s", sport)

    fetcher_mod = importlib.import_module(cfg["fetcher_module"])
    feature_mod = importlib.import_module(cfg["feature_module"])

    FetcherCls = getattr(fetcher_mod, cfg["fetcher_cls"])
    FeatureCls = getattr(feature_mod, cfg["feature_cls"])

    fetcher = FetcherCls()
    df_raw = fetcher.fetch_all_seasons()

    engineer = FeatureCls()
    df_feat = engineer.engineer_features(df_raw)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(cache_path)
    logger.info("Cached features → %s", cache_path)
    return df_feat


def _prepare_X_y(df: pd.DataFrame, cfg: dict):
    """Extract X, y; drop non-feature columns."""
    target_col = cfg["target_col"]
    drop_cols = cfg["drop_cols"] + [target_col]

    # Only drop columns that actually exist
    to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=to_drop)

    # Drop any remaining object columns
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        logger.warning("Dropping remaining object columns: %s", obj_cols)
        X = X.drop(columns=obj_cols)

    X = X.fillna(0)
    y = df[target_col].astype(int)
    return X, y


def _load_or_rebuild_ensemble(sport: str, tag: str, X_train, y_train, model_sport: str = None):
    """Load ensemble if it exists, otherwise rebuild from individual models."""
    model_dir = Path(f"data/models/{model_sport or sport}")
    ensemble_path = model_dir / f"ensemble_{tag}.joblib"

    if ensemble_path.exists():
        logger.info("Ensemble found at %s", ensemble_path)
        return joblib.load(ensemble_path)

    logger.info("No ensemble found — rebuilding from individual models")
    trainer = ModelTrainer(sport)
    trainer.load_models(tag)

    if not trainer.trained_models:
        raise RuntimeError(f"No individual models found for {model_sport or sport}/{tag}")

    ensemble = trainer.build_ensemble(X_train, y_train)
    joblib.dump(ensemble, ensemble_path)
    logger.info("Saved rebuilt ensemble → %s", ensemble_path)
    return ensemble


def fit_sport(sport: str, cfg: dict, cal_frac: float = 0.20, test_frac: float = 0.10):
    model_sport = cfg.get("model_sport", sport)
    # For derived sports (e.g. tennis_wta saves into the tennis dir), use the
    # config tag directly so we target the correct ensemble artifact.
    if model_sport != sport:
        model_tag = cfg["tag"]
    else:
        model_tag = get_current_model_tag(model_sport, fallback=cfg["tag"])
    logger.info("=" * 65)
    logger.info("Sport: %s  model_sport: %s  tag: %s", sport, model_sport, model_tag)

    # ── 1. Feature matrix ────────────────────────────────────────────────
    df = _get_feature_matrix(sport, cfg)
    X, y = _prepare_X_y(df, cfg)
    logger.info("Total samples: %d  |  classes: %s", len(y), sorted(y.unique()))

    # ── 2. Temporal splits (no shuffle — time-ordered) ───────────────────
    n = len(X)
    test_start  = int(n * (1 - test_frac))
    cal_start   = int(n * (1 - cal_frac - test_frac))

    X_train = X.iloc[:cal_start]
    y_train = y.iloc[:cal_start]
    X_cal   = X.iloc[cal_start:test_start]
    y_cal   = y.iloc[cal_start:test_start]
    X_test  = X.iloc[test_start:]
    y_test  = y.iloc[test_start:]

    logger.info(
        "Split → train=%d  cal=%d  test=%d",
        len(y_train), len(y_cal), len(y_test),
    )

    # ── 3. Ensemble ───────────────────────────────────────────────────────
    ensemble = _load_or_rebuild_ensemble(sport, model_tag, X_train, y_train, model_sport)

    # ── 4. Fit calibrator ─────────────────────────────────────────────────
    cal = EnsembleCalibrator()
    cal.fit(ensemble, X_cal, y_cal)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    metrics = cal.evaluate(ensemble, X_test, y_test)
    logger.info("Test-set evaluation for %s:", sport)
    logger.info(
        "  log_loss  raw=%.4f  cal=%.4f  improvement=%+.4f",
        metrics["raw_log_loss"], metrics["cal_log_loss"],
        metrics["log_loss_improvement"],
    )
    logger.info(
        "  brier     raw=%.4f  cal=%.4f  improvement=%+.4f",
        metrics["raw_brier"], metrics["cal_brier"],
        metrics["brier_improvement"],
    )

    # ── 6. Save ───────────────────────────────────────────────────────────
    save_path = calibrator_path_for_tag(model_sport, model_tag)
    ll_imp = metrics["log_loss_improvement"]
    bs_imp = metrics["brier_improvement"]
    if ll_imp > 0 and bs_imp > 0:
        cal.save(save_path)
        logger.info("Calibrator saved → %s", save_path)
    elif save_path.exists():
        save_path.unlink()
        logger.info("Removed stale calibrator → %s", save_path)
    else:
        logger.info("Calibration not saved for %s (held-out metrics did not improve)", sport)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fit probability calibrators for all sports")
    parser.add_argument(
        "--sports",
        nargs="+",
        default=list(SPORT_CONFIGS.keys()),
        choices=list(SPORT_CONFIGS.keys()),
    )
    parser.add_argument(
        "--cal-frac",
        type=float,
        default=0.20,
        help="Fraction of data for calibration set (default 0.20)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.10,
        help="Fraction of data for test set (default 0.10)",
    )
    args = parser.parse_args()

    summary = {}
    for sport in args.sports:
        cfg = SPORT_CONFIGS[sport]
        try:
            metrics = fit_sport(sport, cfg, args.cal_frac, args.test_frac)
            summary[sport] = metrics
        except Exception as exc:
            logger.error("Failed for %s: %s", sport, exc, exc_info=True)
            summary[sport] = {"error": str(exc)}

    print("\n" + "=" * 65)
    print("CALIBRATION SUMMARY")
    print("=" * 65)
    for sport, m in summary.items():
        if "error" in m:
            print(f"{sport:12s}  ERROR: {m['error']}")
        else:
            ll_imp = m.get("log_loss_improvement", 0)
            bs_imp = m.get("brier_improvement", 0)
            print(
                f"{sport:12s}  log_loss Δ{ll_imp:+.4f}  brier Δ{bs_imp:+.4f}"
                f"  ({'✓ improved' if ll_imp > 0 else '↔ passthrough'})"
            )


if __name__ == "__main__":
    main()
