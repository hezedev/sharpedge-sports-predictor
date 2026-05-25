from __future__ import annotations

import argparse
from pathlib import Path

from daily_scan import _load_features_cached
from src.data.soccer_fetcher import SoccerFetcher
from src.features.soccer_features import SoccerFeatureEngineer


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh local historical feature caches outside the live scan path.")
    parser.add_argument("--sport", choices=["soccer"], default="soccer")
    args = parser.parse_args()

    if args.sport == "soccer":
        features_df, _engineer = _load_features_cached(
            "soccer",
            str(Path("data/cache/soccer_features.parquet")),
            SoccerFetcher,
            SoccerFeatureEngineer,
            allow_live_refresh=True,
        )
        print(f"Refreshed soccer feature cache: {len(features_df)} rows")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
