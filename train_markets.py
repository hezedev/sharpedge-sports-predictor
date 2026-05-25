"""
train_markets.py
================
Trains Over/Under (totals) and spread-cover models for all sports.
Reuses the same cached feature parquets as retrain_and_calibrate.py —
no new data fetching needed.

Usage:
    python train_markets.py                          # all sports, both markets
    python train_markets.py --sports soccer nhl      # specific sports
    python train_markets.py --markets totals         # totals only
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_markets")
sys.path.insert(0, ".")


# ── Sport configs — mirrors retrain_and_calibrate.py ──────────────────────────

SPORT_CONFIGS = {
    "soccer": {
        "cache": "data/cache/soccer_features.parquet",
        "drop_cols": [
            "result", "match_id", "date",
            "home_goals", "away_goals", "home_score", "away_score",
            "home_team", "away_team", "season", "competition", "target",
        ],
        "totals_line": 2.5,
        "spreads_line": 0.5,   # Asian Handicap 0.5 (home wins = covers)
    },
    "basketball": {
        "cache": "data/cache/basketball_features.parquet",
        "drop_cols": [
            "result", "match_id", "date", "league_id", "league_name",
            "season", "home_team", "away_team", "home_team_id", "away_team_id",
            "home_score", "away_score", "status",
            "home_first_half", "home_second_half", "away_first_half", "away_second_half",
            "home_q1", "home_q2", "home_q3", "home_q4",
            "away_q1", "away_q2", "away_q3", "away_q4", "home_ot", "away_ot",
            "point_diff", "total_points", "neg_point_diff", "target",
        ],
        "totals_line": 220.0,  # reference; actual line from market at inference
        "spreads_line": 0.0,   # home covers = home wins
    },
    "nhl": {
        "cache": "data/cache/nhl_features.parquet",
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
            "target",
        ],
        "totals_line": 5.5,
        "spreads_line": 1.5,   # puck line
    },
    "mlb": {
        "cache": "data/cache/mlb_features.parquet",
        "drop_cols": [
            "result", "game_pk", "date", "season",
            "home_team", "away_team",
            "home_score", "away_score", "home_hits", "away_hits",
            "home_errors", "away_errors", "home_innings", "away_innings",
            "target",
        ],
        "totals_line": 8.5,
        "spreads_line": 1.5,   # run line
    },
}


def load_features(sport: str, cfg: dict) -> pd.DataFrame:
    path = Path(cfg["cache"])
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}. Run retrain_and_calibrate.py first.")
    df = pd.read_parquet(path)
    logger.info("%s: loaded %d rows from %s", sport, len(df), path)
    return df


def prepare_X(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """Drop result/target columns and non-numeric; fill NaN."""
    to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=to_drop)
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        logger.warning("Dropping object columns: %s", obj_cols)
        X = X.drop(columns=obj_cols)
    return X.fillna(0)


def train_totals(sport: str, cfg: dict, df: pd.DataFrame) -> bool:
    """Train Over/Under model. Returns True if successful."""
    from src.models.totals_trainer import TotalsTrainer

    line = cfg["totals_line"]
    y = TotalsTrainer.make_totals_target(df, sport, line)
    if y is None:
        logger.warning("%s totals: no target available — skipping", sport)
        return False

    over_pct = y.mean()
    logger.info("%s totals (line=%.1f): %d samples | over=%.1f%%",
                sport, line, len(y), over_pct * 100)

    if over_pct < 0.15 or over_pct > 0.85:
        logger.warning("%s totals: extreme class imbalance (%.0f%% over) — model may be unreliable",
                       sport, over_pct * 100)

    X = prepare_X(df, cfg["drop_cols"])

    # Temporal splits: 70/15/10/5
    n = len(X)
    i_val  = int(n * 0.70)
    i_cal  = int(n * 0.85)
    i_test = int(n * 0.90)

    X_train, y_train = X.iloc[:i_val],      y.iloc[:i_val]
    X_val,   y_val   = X.iloc[i_val:i_cal], y.iloc[i_val:i_cal]
    X_cal,   y_cal   = X.iloc[i_cal:i_test],y.iloc[i_cal:i_test]
    X_test,  y_test  = X.iloc[i_test:],     y.iloc[i_test:]

    logger.info("  Splits → train=%d  val=%d  cal=%d  test=%d",
                len(y_train), len(y_val), len(y_cal), len(y_test))

    trainer = TotalsTrainer(sport, market="totals", line=line)
    trainer.fit(X_train, y_train, X_val, y_val)
    trainer.fit_calibrator(X_cal, y_cal, X_test, y_test)
    trainer.save()
    logger.info("%s totals model saved ✓", sport)
    return True


def train_spreads(sport: str, cfg: dict, df: pd.DataFrame) -> bool:
    """Train spread-cover model. Returns True if successful."""
    from src.models.totals_trainer import TotalsTrainer

    line = cfg["spreads_line"]
    y = TotalsTrainer.make_spreads_target(df, sport, line)
    if y is None:
        logger.warning("%s spreads: no target available — skipping", sport)
        return False

    covers_pct = y.mean()
    logger.info("%s spreads (line=%.1f): %d samples | home_covers=%.1f%%",
                sport, line, len(y), covers_pct * 100)

    X = prepare_X(df, cfg["drop_cols"])

    n = len(X)
    i_val  = int(n * 0.70)
    i_cal  = int(n * 0.85)
    i_test = int(n * 0.90)

    X_train, y_train = X.iloc[:i_val],      y.iloc[:i_val]
    X_val,   y_val   = X.iloc[i_val:i_cal], y.iloc[i_val:i_cal]
    X_cal,   y_cal   = X.iloc[i_cal:i_test],y.iloc[i_cal:i_test]
    X_test,  y_test  = X.iloc[i_test:],     y.iloc[i_test:]

    logger.info("  Splits → train=%d  val=%d  cal=%d  test=%d",
                len(y_train), len(y_val), len(y_cal), len(y_test))

    trainer = TotalsTrainer(sport, market="spreads", line=line)
    trainer.fit(X_train, y_train, X_val, y_val)
    trainer.fit_calibrator(X_cal, y_cal, X_test, y_test)
    trainer.save()
    logger.info("%s spreads model saved ✓", sport)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sports", nargs="+", default=list(SPORT_CONFIGS.keys()))
    parser.add_argument("--markets", nargs="+", default=["totals", "spreads"])
    args = parser.parse_args()

    results = {}
    for sport in args.sports:
        if sport not in SPORT_CONFIGS:
            logger.error("Unknown sport: %s", sport)
            continue

        cfg = SPORT_CONFIGS[sport]
        logger.info("=" * 60)
        logger.info("Sport: %s", sport)

        try:
            df = load_features(sport, cfg)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        sport_results = {}
        if "totals" in args.markets:
            sport_results["totals"] = train_totals(sport, cfg, df)
        if "spreads" in args.markets:
            sport_results["spreads"] = train_spreads(sport, cfg, df)
        results[sport] = sport_results

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sport, r in results.items():
        for market, ok in r.items():
            status = "✓ trained" if ok else "✗ skipped (no target data)"
            print(f"  {sport:12s} {market:8s}  {status}")


if __name__ == "__main__":
    main()
