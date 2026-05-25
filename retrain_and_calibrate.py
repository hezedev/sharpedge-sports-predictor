"""
retrain_and_calibrate.py
========================
Retrains soccer, tennis, and basketball models on the current feature set,
then fits the adaptive EnsembleCalibrator for each sport.

Basketball now uses the BallDontLie free-tier API (NBA data 2022-2024).

Usage:
    python retrain_and_calibrate.py [--sports soccer tennis basketball]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from config import settings
from src.models.artifacts import calibrator_path_for_tag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain")
sys.path.insert(0, ".")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

SPORT_CONFIGS = {
    "basketball": {
        "tag": "nba_2025_26",
        "target_col": "target",
        "raw_cache": "data/cache/basketball_features.parquet",
        "split_ratios": {"train": 0.70, "val": 0.10, "cal": 0.15, "test": 0.05},
        "calibration_methods": ["temperature", "sigmoid"],
        "active_recalibration_methods": ["temperature"],
        "fetcher_cls": "BallDontLieFetcher",
        "feature_cls": "BasketballFeatureEngineer",
        "fetcher_module": "src.data.balldontlie_fetcher",
        "feature_module": "src.features.basketball_features",
        "drop_cols": [
            "result", "match_id", "date", "league_id", "league_name",
            "season", "home_team", "away_team", "home_team_id", "away_team_id",
            "home_score", "away_score", "status",
            # Drop raw quarter SCORES (result leakage) but KEEP derived features:
            # home_half_ratio, away_half_ratio, went_to_ot, home_q4_avg, away_q4_avg
            # are now filled by FeatureStore at inference time — so they must be in the model.
            "home_first_half", "home_second_half", "away_first_half", "away_second_half",
            "home_q1", "home_q2", "home_q3", "home_q4",
            "away_q1", "away_q2", "away_q3", "away_q4", "home_ot", "away_ot",
            "point_diff", "total_points", "neg_point_diff",
        ],
    },
    "soccer": {
        "tag": "pl_2024_25",
        "target_col": "target",
        "raw_cache": "data/cache/soccer_features.parquet",
        "active_recalibration_methods": ["temperature"],
        "active_recalibration_split_ratios": {"train": 0.70, "val": 0.10, "cal": 0.15, "test": 0.05},
        "fetcher_cls": "SoccerFetcher",
        "feature_cls": "SoccerFeatureEngineer",
        "fetcher_module": "src.data.soccer_fetcher",
        "feature_module": "src.features.soccer_features",
        "drop_cols": [
            "result", "match_id", "date",
            "home_goals", "away_goals", "home_score", "away_score",
            "home_team", "away_team", "season", "competition",
        ],
    },
    "tennis": {
        "tag": "atp_2022_25",
        "target_col": "target",
        "raw_cache": "data/cache/tennis_features.parquet",
        "split_ratios": {"train": 0.70, "val": 0.10, "cal": 0.15, "test": 0.05},
        "fetcher_cls": "TennisFetcher",
        "feature_cls": "TennisFeatureEngineer",
        "fetcher_module": "src.data.tennis_fetcher",
        "feature_module": "src.features.tennis_features",
        "fetcher_kwargs": {"seasons": [2022, 2023, 2024, 2025, 2026], "tours": ["atp"]},
        "drop_cols": [
            "result", "match_id", "date", "player1_name", "player2_name",
            "tourney_name", "tourney_id", "tourney_date", "tourney_level",
            "tourney_level_name", "surface", "round",
            "player1_id", "player2_id", "player1_entry", "player2_entry",
            "player1_ioc", "player2_ioc", "player1_hand", "player2_hand",
            "score", "minutes", "winner_ioc",
            # Raw per-match serve stats (post-match leakage — keep only rolling equivalents)
            "p1_svpt", "p1_1stIn", "p1_ace", "p1_df", "p1_bpSaved", "p1_bpFaced",
            "p1_1stWon", "p1_2ndWon", "p1_1st_pct", "p1_ace_rate", "p1_bp_save",
            "p2_svpt", "p2_1stIn", "p2_ace", "p2_df", "p2_bpSaved", "p2_bpFaced",
            "p2_1stWon", "p2_2ndWon", "p2_1st_pct", "p2_ace_rate", "p2_bp_save",
            "player1_rank", "player2_rank", "player1_rank_pts", "player2_rank_pts",
            "player1_seed", "player2_seed", "player1_age", "player2_age",
            "player1_ht", "player2_ht", "round_num", "best_of",
        ],
    },
    "mlb": {
        "tag": "mlb_2024_25",
        "target_col": "target",
        "raw_cache": "data/cache/mlb_features.parquet",
        "fetcher_cls": "MLBFetcher",
        "feature_cls": "MLBFeatureEngineer",
        "fetcher_module": "src.data.mlb_fetcher",
        "feature_module": "src.features.mlb_features",
        "drop_cols": [
            "result", "game_pk", "date", "season",
            "home_team", "away_team",
            "home_score", "away_score", "home_hits", "away_hits",
            "home_errors", "away_errors", "home_innings", "away_innings",
        ],
    },
    "nhl": {
        "tag": "nhl_2024_25",
        "target_col": "target",
        "raw_cache": "data/cache/nhl_features.parquet",
        "fetcher_cls": "NHLFetcher",
        "feature_cls": "NHLFeatureEngineer",
        "fetcher_module": "src.data.nhl_fetcher",
        "feature_module": "src.features.nhl_features",
        "drop_cols": [
            "result", "game_id", "date", "season",
            "home_team", "away_team",
            "home_score", "away_score",
            "home_shots", "away_shots",
            "home_pp_goals", "away_pp_goals",
            "went_to_ot",
            # Per-game shot/xG stats (post-match leakage — keep only rolling window versions)
            "home_corsi", "away_corsi",
            "home_fenwick", "away_fenwick",
            "home_xg", "away_xg",
            "home_pp_opp", "away_pp_opp",
        ],
    },
    "tennis_wta": {
        "tag": "wta_2022_24",
        "model_sport": "tennis",   # save artifacts into data/models/tennis/
        "target_col": "target",
        "raw_cache": "data/cache/tennis_wta_features.parquet",
        "split_ratios": {"train": 0.70, "val": 0.10, "cal": 0.15, "test": 0.05},
        "fetcher_cls": "TennisFetcher",
        "feature_cls": "TennisFeatureEngineer",
        "fetcher_module": "src.data.tennis_fetcher",
        "feature_module": "src.features.tennis_features",
        "fetcher_kwargs": {"tours": ["wta"], "seasons": [2022, 2023, 2024, 2025, 2026]},
        "drop_cols": [
            "result", "match_id", "date", "player1_name", "player2_name",
            "tourney_name", "tourney_id", "tourney_date", "tourney_level",
            "tourney_level_name", "surface", "round",
            "player1_id", "player2_id", "player1_entry", "player2_entry",
            "player1_ioc", "player2_ioc", "player1_hand", "player2_hand",
            "score", "minutes", "winner_ioc",
            # Raw per-match serve stats (post-match leakage — keep only rolling equivalents)
            "p1_svpt", "p1_1stIn", "p1_ace", "p1_df", "p1_bpSaved", "p1_bpFaced",
            "p1_1stWon", "p1_2ndWon", "p1_1st_pct", "p1_ace_rate", "p1_bp_save",
            "p2_svpt", "p2_1stIn", "p2_ace", "p2_df", "p2_bpSaved", "p2_bpFaced",
            "p2_1stWon", "p2_2ndWon", "p2_1st_pct", "p2_ace_rate", "p2_bp_save",
            "player1_rank", "player2_rank", "player1_rank_pts", "player2_rank_pts",
            "player1_seed", "player2_seed", "player1_age", "player2_age",
            "player1_ht", "player2_ht", "round_num", "best_of",
        ],
    },
}

_FEATURE_CACHE_VERSIONS = {
    "soccer": 3,
    "basketball": 2,
    "tennis": 4,
    "tennis_wta": 4,
    "mlb": 2,
    "nhl": 2,
}


def _feature_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _feature_cache_scope_signature(sport: str) -> dict:
    sport_cfg = settings.get("sports", {}).get(sport, {})
    apis_cfg = settings.get("apis", {})
    signature = {
        "sport": sport,
        "seasons_to_fetch": sport_cfg.get("seasons_to_fetch"),
        "feature_version": _FEATURE_CACHE_VERSIONS.get(sport, 1),
    }
    if sport == "soccer":
        signature["competitions"] = list(apis_cfg.get("football_data", {}).get("competitions", []))
        signature["api_sports_competitions"] = [
            {
                "key": item.get("key"),
                "name": item.get("name"),
                "country": item.get("country"),
                "season_mode": item.get("season_mode"),
                "league_id": item.get("league_id"),
                "season_years": item.get("season_years"),
            }
            for item in sport_cfg.get("api_sports_competitions", [])
        ]
    if sport == "tennis":
        signature["tours"] = list(sport_cfg.get("tours", []))
    return signature


def _feature_cache_matches_scope(sport: str, cache_path: Path) -> bool:
    meta_path = _feature_cache_meta_path(cache_path)
    if not meta_path.exists():
        return False
    try:
        saved = json.loads(meta_path.read_text())
    except Exception:
        return False
    return saved == _feature_cache_scope_signature(sport)


def _write_feature_cache_scope(sport: str, cache_path: Path) -> None:
    _feature_cache_meta_path(cache_path).write_text(
        json.dumps(_feature_cache_scope_signature(sport), indent=2)
    )


def _build_features(sport: str, cfg: dict) -> pd.DataFrame:
    """Fetch raw data and engineer features; cache result to parquet."""
    import importlib
    cache_path = Path(cfg["raw_cache"])

    fetcher_mod  = importlib.import_module(cfg["fetcher_module"])
    feature_mod  = importlib.import_module(cfg["feature_module"])
    FetcherCls   = getattr(fetcher_mod, cfg["fetcher_cls"])
    FeatureCls   = getattr(feature_mod, cfg["feature_cls"])

    fetcher_kwargs = cfg.get("fetcher_kwargs", {})
    fetcher = FetcherCls(**fetcher_kwargs) if fetcher_kwargs else FetcherCls()
    df_raw  = fetcher.fetch_all_seasons()
    if df_raw.empty:
        raise RuntimeError(f"No raw data returned for {sport}")

    engineer = FeatureCls()
    df_feat  = engineer.engineer_features(df_raw)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Coerce mixed-type object columns to string so parquet serialization doesn't fail
    # (e.g. seed columns in WTA data mix bytes like b"WC" with float NaN)
    for col in df_feat.select_dtypes(include="object").columns:
        df_feat[col] = df_feat[col].where(df_feat[col].isna(), df_feat[col].astype(str))
    df_feat.to_parquet(cache_path)
    _write_feature_cache_scope(sport, cache_path)
    logger.info("Feature cache written → %s  (%d rows)", cache_path, len(df_feat))
    return df_feat


def prepare_data(sport: str, cfg: dict):
    """Load (or build) features, return X, y with current feature set."""
    cache_path = Path(cfg["raw_cache"])
    if cache_path.exists() and _feature_cache_matches_scope(sport, cache_path):
        df = pd.read_parquet(cache_path)
        logger.info("Loaded %d rows from cache %s", len(df), cache_path)
    else:
        if cache_path.exists():
            logger.info("Feature cache scope changed — rebuilding features from scratch for %s", sport)
        else:
            logger.info("Cache not found — building features from scratch for %s", sport)
        df = _build_features(sport, cfg)
    logger.info("Loaded %d rows from %s", len(df), cache_path)

    to_drop = [c for c in cfg["drop_cols"] if c in df.columns]
    X = df.drop(columns=to_drop + [cfg["target_col"]])

    non_numeric_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if non_numeric_cols:
        logger.warning("Dropping additional non-numeric columns: %s", non_numeric_cols)
        X = X.drop(columns=non_numeric_cols)

    X = X.fillna(0)
    y = df[cfg["target_col"]].astype(int)

    logger.info(
        "%s: %d samples × %d features  |  classes: %s",
        sport, len(y), X.shape[1], sorted(y.unique()),
    )
    return X, y


def retrain_and_save(sport: str, cfg: dict, X_train, y_train, X_val, y_val):
    """Retrain models on X_train, evaluate on X_val, save to disk."""
    from src.models.trainer import ModelTrainer
    try:
        import lightgbm as lgb
        _lgb = lgb
    except ImportError:
        _lgb = None

    model_sport = cfg.get("model_sport", sport)
    trainer = ModelTrainer(model_sport)
    logger.info("Training %s models on %d samples …", sport, len(y_train))
    trainer.train(X_train, y_train, X_val=X_val, y_val=y_val)

    ensemble = trainer.build_ensemble(X_train, y_train)
    eval_results = trainer.evaluate(X_val, y_val)

    for name, m in eval_results.items():
        logger.info("  %-12s  acc=%.4f  log_loss=%.4f", name, m["accuracy"], m["log_loss"])

    paths = trainer.save_models(cfg["tag"])
    logger.info("Saved models: %s", list(paths.keys()))

    return trainer, ensemble


def fit_calibrator(sport: str, cfg: dict, ensemble, X_cal, y_cal, X_test, y_test):
    """Fit adaptive calibrator, evaluate on test set, save only if it helps."""
    from src.models.calibration import EnsembleCalibrator

    cal = EnsembleCalibrator()
    cal.fit(ensemble, X_cal, y_cal, allowed_methods=cfg.get("calibration_methods"))

    metrics = cal.evaluate(ensemble, X_test, y_test)
    ll_imp = metrics["log_loss_improvement"]
    bs_imp = metrics["brier_improvement"]

    logger.info(
        "Calibration %s:  log_loss Δ%+.4f  brier Δ%+.4f  (%s)",
        sport, ll_imp, bs_imp,
        "✓ improved" if ll_imp > 0 and bs_imp > 0 else "✗ not saving",
    )

    model_sport = cfg.get("model_sport", sport)
    save_path = Path(f"data/models/{model_sport}/calibrator_{cfg['tag']}.joblib")
    saved = False
    if ll_imp > 0 and bs_imp > 0:
        cal.save(save_path)
        saved = True
    elif save_path.exists():
        save_path.unlink()
        logger.info("Removed stale calibrator → %s", save_path)
    metrics["saved"] = saved
    return metrics


def _align_to_model_features(X: pd.DataFrame, trainer) -> pd.DataFrame:
    sample_model = next(iter(trainer.trained_models.values()))
    feature_cols = list(sample_model.feature_names_in_)
    aligned = X.copy()
    missing = [c for c in feature_cols if c not in aligned.columns]
    if missing:
        logger.warning("Adding zero-filled missing %s features: %s", trainer.sport, missing)
        for col in missing:
            aligned[col] = 0.0
    return aligned.reindex(columns=feature_cols, fill_value=0.0).fillna(0)


def _compute_temporal_split_indices(n: int, cfg: dict) -> tuple[int, int, int]:
    """Return train/val/cal cut points for a sport's temporal split config."""
    split_cfg = cfg.get("split_ratios", {})
    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("val", 0.15))
    cal_ratio = float(split_cfg.get("cal", 0.05))
    test_ratio = float(split_cfg.get("test", 0.10))
    ratio_total = train_ratio + val_ratio + cal_ratio + test_ratio
    if abs(ratio_total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_total:.4f}")

    i_val = int(n * train_ratio)
    i_cal = int(n * (train_ratio + val_ratio))
    i_test = int(n * (train_ratio + val_ratio + cal_ratio))
    return i_val, i_cal, i_test


def process_sport(sport: str):
    cfg = SPORT_CONFIGS[sport]
    logger.info("=" * 65)
    logger.info("Processing: %s", sport)
    from src.models.artifacts import get_current_model_tag, set_current_model_tag
    previous_tag = get_current_model_tag(cfg.get("model_sport", sport), fallback=cfg["tag"])

    X, y = prepare_data(sport, cfg)

    # Temporal splits default to 70/15/5/10, but sports can override.
    n = len(X)
    i_val, i_cal, i_test = _compute_temporal_split_indices(n, cfg)

    X_train, y_train = X.iloc[:i_val],  y.iloc[:i_val]
    X_val,   y_val   = X.iloc[i_val:i_cal], y.iloc[i_val:i_cal]
    X_cal,   y_cal   = X.iloc[i_cal:i_test], y.iloc[i_cal:i_test]
    X_test,  y_test  = X.iloc[i_test:], y.iloc[i_test:]

    logger.info(
        "Splits → train=%d  val=%d  cal=%d  test=%d",
        len(y_train), len(y_val), len(y_cal), len(y_test),
    )

    trainer, ensemble = retrain_and_save(sport, cfg, X_train, y_train, X_val, y_val)
    metrics = fit_calibrator(sport, cfg, ensemble, X_cal, y_cal, X_test, y_test)
    if not metrics.get("saved", False) and previous_tag:
        set_current_model_tag(cfg.get("model_sport", sport), previous_tag)
        logger.info(
            "Calibration did not improve for %s — restored current_tag.txt → %s",
            sport,
            previous_tag,
        )

    # After WTA training, restore the ATP tag as primary so daily_scan.py
    # continues using the ATP model for ATP events.
    if sport == "tennis_wta":
        from src.models.artifacts import set_current_model_tag, get_current_model_tag
        atp_tag = SPORT_CONFIGS.get("tennis", {}).get("tag", "atp_2022_25")
        set_current_model_tag("tennis", atp_tag)
        logger.info("Restored tennis current_tag.txt → %s", atp_tag)

    return metrics


def recalibrate_active_sport(sport: str):
    cfg = SPORT_CONFIGS[sport]
    logger.info("=" * 65)
    logger.info("Recalibrating active tag: %s", sport)
    from src.models.artifacts import get_current_model_tag
    from src.models.calibration import EnsembleCalibrator
    from src.models.trainer import ModelTrainer, _SoftVotingWrapper

    model_sport = cfg.get("model_sport", sport)
    active_tag = get_current_model_tag(model_sport, fallback=cfg["tag"])
    if not active_tag:
        raise RuntimeError(f"No active tag found for {sport}")

    X, y = prepare_data(sport, cfg)
    trainer = ModelTrainer(sport=model_sport)
    trainer.load_models(tag=active_tag)
    if not trainer.trained_models:
        raise RuntimeError(f"No trained models found for {sport} tag {active_tag}")

    X = _align_to_model_features(X, trainer)
    n = len(X)
    split_cfg = dict(cfg)
    active_split = cfg.get("active_recalibration_split_ratios")
    if active_split:
        split_cfg["split_ratios"] = active_split
    i_val, i_cal, i_test = _compute_temporal_split_indices(n, split_cfg)
    X_cal, y_cal = X.iloc[i_cal:i_test], y.iloc[i_cal:i_test]
    X_test, y_test = X.iloc[i_test:], y.iloc[i_test:]

    trainer.ensemble_model = _SoftVotingWrapper(
        estimators=list(trainer.trained_models.items()),
        weights=None,
        classes=np.array(sorted(y.unique())),
    )
    cal = EnsembleCalibrator()
    cal.fit(
        trainer.ensemble_model,
        X_cal,
        y_cal,
        allowed_methods=cfg.get("active_recalibration_methods", cfg.get("calibration_methods")),
    )
    metrics = cal.evaluate(trainer.ensemble_model, X_test, y_test)
    ll_imp = metrics["log_loss_improvement"]
    bs_imp = metrics["brier_improvement"]
    logger.info(
        "Active recalibration %s/%s: log_loss Δ%+.4f  brier Δ%+.4f  (%s)",
        sport,
        active_tag,
        ll_imp,
        bs_imp,
        "✓ improved" if ll_imp > 0 and bs_imp > 0 else "✗ not saving",
    )
    save_path = calibrator_path_for_tag(model_sport, active_tag)
    saved = False
    if ll_imp > 0 and bs_imp > 0:
        cal.save(save_path)
        saved = True
    metrics["saved"] = saved
    metrics["tag"] = active_tag
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sports", nargs="+", default=list(SPORT_CONFIGS.keys()))
    parser.add_argument("--recalibrate-active", action="store_true")
    args = parser.parse_args()

    summary = {}
    for sport in args.sports:
        if sport not in SPORT_CONFIGS:
            logger.error("Unknown sport: %s. Choices: %s", sport, list(SPORT_CONFIGS))
            continue
        try:
            if args.recalibrate_active:
                summary[sport] = recalibrate_active_sport(sport)
            else:
                summary[sport] = process_sport(sport)
        except Exception as exc:
            logger.error("FAILED %s: %s", sport, exc, exc_info=True)
            summary[sport] = {"error": str(exc)}

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for sport, m in summary.items():
        if "error" in m:
            print(f"{sport:12s}  ERROR: {m['error']}")
        else:
            ll_imp = m.get("log_loss_improvement", 0)
            bs_imp = m.get("brier_improvement", 0)
            print(f"{sport:12s}  log_loss Δ{ll_imp:+.4f}  brier Δ{bs_imp:+.4f}")


if __name__ == "__main__":
    main()
