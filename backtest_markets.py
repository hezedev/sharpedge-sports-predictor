#!/usr/bin/env python3
"""
Market-specific historical replay backtests.

Examples:
    .venv/bin/python backtest_markets.py --sports soccer mlb nhl
    .venv/bin/python backtest_markets.py --sports soccer --markets totals_over_2_5 btts_yes
    .venv/bin/python backtest_markets.py --sports soccer --markets totals_over_0_5 totals_over_1_5 totals_over_2_5 totals_over_3_5 totals_under_2_5 home_asian_minus_1_5 away_asian_plus_1_5
"""

from __future__ import annotations

import argparse
import json
import logging

from src.evaluation.market_backtest import MARKET_SPECS, replay_sport_markets


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("market_backtests")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay historical backtests across multiple betting markets")
    parser.add_argument(
        "--sports",
        nargs="+",
        default=["soccer", "basketball", "mlb", "nhl", "tennis"],
        choices=sorted(MARKET_SPECS.keys()),
    )
    parser.add_argument(
        "--markets",
        nargs="*",
        default=None,
        help="Optional list of market keys to restrict the replay",
    )
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--min-train-games", type=int, default=300)
    parser.add_argument("--initial-train-days", type=int, default=180)
    args = parser.parse_args()

    summary = {}
    for sport in args.sports:
        logger.info("=" * 72)
        logger.info("Market replay backtest: %s", sport)
        summary[sport] = replay_sport_markets(
            sport=sport,
            market_keys=args.markets,
            window_days=args.window_days,
            min_train_games=args.min_train_games,
            initial_train_days=args.initial_train_days,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
