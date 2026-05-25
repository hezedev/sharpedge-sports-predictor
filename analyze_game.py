"""CLI for deep one-game analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.analysis import ManualGameAnalyst


def _write_output(path_str: Optional[str], content: str) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep manual analysis for a single game and bet.")
    parser.add_argument("--sport", required=True, choices=["soccer", "basketball", "mlb", "nhl"])
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--bet", required=True, help="Free-text bet description, e.g. 'Arsenal moneyline'")
    parser.add_argument("--market", default="h2h", help="Odds market key, default: h2h")
    parser.add_argument("--selection", default=None, help="Optional normalized selection: home, away, draw, over, under")
    parser.add_argument("--price", type=float, default=None, help="Optional manual price to evaluate instead of best live price")
    parser.add_argument("--json-out", default=None, help="Optional path for JSON report")
    parser.add_argument("--md-out", default=None, help="Optional path for markdown report")
    args = parser.parse_args()

    analyst = ManualGameAnalyst()
    report = analyst.analyze_game(
        sport=args.sport,
        home_team=args.home_team,
        away_team=args.away_team,
        bet=args.bet,
        market=args.market,
        selection=args.selection,
        price=args.price,
    )

    markdown = report.to_markdown()
    json_payload = json.dumps(report.to_dict(), indent=2, default=str)
    print(markdown)
    _write_output(args.md_out, markdown)
    _write_output(args.json_out, json_payload)


if __name__ == "__main__":
    main()
