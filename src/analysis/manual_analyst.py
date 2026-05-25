"""Deep one-game analyst that blends online data into a manual report."""

from __future__ import annotations

import difflib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from src.analysis.schemas import AnalysisReport, AnalysisSignal, SourceNote
from src.data.api_football_enricher import APIFootballEnricher
from src.data.basketball_fetcher import BasketballFetcher
from src.data.mlb_fetcher import MLBFetcher
from src.data.nhl_fetcher import NHLFetcher
from src.data.odds_fetcher import OddsFetcher, SPORT_KEYS
from src.data.soccer_fetcher import SoccerFetcher
from src.features.feature_store import build_entity_alias_map, resolve_canonical_name
from src.utils.logger import setup_logger
from src.utils.odds_quota import get_odds_budget_status
from src.utils.sport_registry import SOCCER_ODDS_TO_COMPETITION

logger = setup_logger(__name__)

@dataclass
class TeamSnapshot:
    """Recent-team summary used for matchup analysis."""

    team: str
    games: int
    wins: int
    draws: int
    losses: int
    scored_avg: float
    allowed_avg: float
    margin_avg: float
    unbeaten_streak: int
    losing_streak: int
    scoring_streak: int
    conceding_streak: int
    last_game_date: Optional[pd.Timestamp]


def _normalize_team_name(name: str) -> str:
    clean = name.lower()
    clean = re.sub(r"[^a-z0-9\s]", " ", clean)
    clean = re.sub(r"\b(fc|cf|sc|afc|bc)\b", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _current_soccer_season_start(today: datetime) -> str:
    return str(today.year if today.month >= 7 else today.year - 1)


def _current_basketball_season(today: datetime) -> str:
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{start_year + 1}"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class ManualGameAnalyst:
    """Analyze a single game and betting angle using live/contextual data."""

    def __init__(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.weather_api_key = os.environ.get("OPENWEATHER_API_KEY", "")

    def analyze_game(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        bet: str,
        market: str = "h2h",
        selection: Optional[str] = None,
        price: Optional[float] = None,
        fair_prob: Optional[float] = None,
    ) -> AnalysisReport:
        """Build a deep report for one game."""
        normalized_sport = sport.lower()
        resolved_selection = self._resolve_selection(
            selection=selection,
            bet=bet,
            home_team=home_team,
            away_team=away_team,
        )
        report = AnalysisReport(
            sport=normalized_sport,
            home_team=home_team,
            away_team=away_team,
            market=market,
            bet=bet,
            selection=resolved_selection,
            verdict="pass",
            confidence=0.45,
        )

        market_context = self._fetch_market_context(
            sport=normalized_sport,
            home_team=home_team,
            away_team=away_team,
            market=market,
        )
        self._merge_market_context(report, market_context, explicit_price=price)
        if report.fair_prob is None and fair_prob is not None:
            report.fair_prob = fair_prob
            if report.price_used:
                implied = 1.0 / report.price_used
                report.edge_pct = report.fair_prob - implied

        if normalized_sport == "soccer":
            self._analyze_soccer(report)
        elif normalized_sport == "basketball":
            self._analyze_basketball(report)
        elif normalized_sport == "mlb":
            self._analyze_mlb(report)
        elif normalized_sport == "nhl":
            self._analyze_nhl(report)
        elif normalized_sport == "tennis":
            self._analyze_tennis(report)
        else:
            report.warnings.append(
                f"Sport '{sport}' is not supported yet. Supported sports: soccer, basketball, mlb, nhl, tennis."
            )

        self._add_market_signal(report)
        self._finalize_report(report)
        return report

    def _merge_market_context(
        self,
        report: AnalysisReport,
        market_context: Dict[str, Any],
        explicit_price: Optional[float],
    ) -> None:
        report.data_points["market"] = market_context
        report.sources.extend(market_context.get("sources", []))
        report.warnings.extend(market_context.get("warnings", []))
        report.unknowns.extend(market_context.get("unknowns", []))

        selection_label = self._selection_market_label(
            market=report.market,
            selection=report.selection,
            home_team=report.home_team,
            away_team=report.away_team,
        )
        report.data_points["selection_market_label"] = selection_label

        fair_probs = market_context.get("fair_probabilities", {})
        best_prices = market_context.get("best_prices", {})
        report.fair_prob = _safe_float(fair_probs.get(selection_label))
        report.price_used = explicit_price or _safe_float(best_prices.get(selection_label))
        if report.fair_prob is not None and report.price_used:
            implied = 1.0 / report.price_used
            report.edge_pct = report.fair_prob - implied

    def _fetch_market_context(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        market: str,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "sources": [],
            "warnings": [],
            "unknowns": [],
            "best_prices": {},
            "fair_probabilities": {},
        }
        cached_context = self._fetch_cached_market_context(
            sport=sport,
            home_team=home_team,
            away_team=away_team,
            market=market,
        )
        if cached_context is not None:
            return cached_context

        if sport == "soccer" and len(SPORT_KEYS.get("soccer", [])) > 12:
            context["unknowns"].append(
                "Skipped a live soccer odds sweep because the expanded league universe is cache-first to protect quota."
            )
            context["warnings"].append(
                "Soccer manual analysis is using cached odds only unless the matchup is already present in a saved odds board."
            )
            return context

        budget = get_odds_budget_status(os.environ.get("ODDS_API_KEY", ""))
        if (
            budget.get("daily_allowance") is not None
            and budget.get("used_today", 0) >= budget.get("daily_allowance", 0)
        ):
            context["unknowns"].append(
                "Skipped fresh Odds API lookup because today's odds budget is already spent."
            )
            context["warnings"].append(
                f"Odds budget guard: used {budget.get('used_today', 0)} of "
                f"{budget.get('daily_allowance', 0)} planned calls today."
            )
            return context

        try:
            fetcher = OddsFetcher(sport=sport)
            odds_df = fetcher.fetch_odds(markets=[market])
            if odds_df.empty:
                context["unknowns"].append("No live odds were returned by The Odds API.")
                return context

            event_rows = self._match_event_rows(odds_df, home_team, away_team)
            if event_rows.empty:
                context["unknowns"].append(
                    "Could not match this game to the live odds board, so market context is partial."
                )
                return context

            event_id = event_rows["event_id"].iloc[0]
            event_df = odds_df[odds_df["event_id"] == event_id].copy()
            best = fetcher.get_best_odds(event_df)
            consensus = fetcher.get_consensus_odds(event_df)

            context["event"] = {
                "event_id": event_id,
                "sport_key": event_rows["sport_key"].iloc[0],
                "commence_time": str(event_rows["commence_time"].iloc[0]),
                "bookmakers": int(event_df["bookmaker"].nunique()),
            }
            context["best_prices"] = {
                row["outcome"]: row["price"] for _, row in best.iterrows()
            }
            context["fair_probabilities"] = {
                row["outcome"]: row["fair_prob"] for _, row in consensus.iterrows()
            }
            context["sources"].append(
                SourceNote(
                    name="The Odds API",
                    detail=(
                        f"Live {market} market snapshot across "
                        f"{event_df['bookmaker'].nunique()} bookmakers."
                    ),
                    url="https://the-odds-api.com/",
                )
            )
            return context
        except Exception as exc:
            logger.error("Market context failed: %s", exc)
            context["warnings"].append(f"Market lookup failed: {exc}")
            return context

    def _fetch_cached_market_context(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        market: str,
    ) -> Optional[Dict[str, Any]]:
        """Use daily_scan disk cache before spending live odds quota."""
        sport_keys = SPORT_KEYS.get(sport, [])
        fetcher = OddsFetcher(sport=sport)

        for sport_key in sport_keys:
            cache_path = self.project_root / "data" / "cache" / "odds" / f"{sport_key}.json"
            if not cache_path.exists():
                continue
            try:
                games = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(games, list) or not games:
                    continue
                odds_df = fetcher._parse_odds(games, sport_key=sport_key)
                event_rows = self._match_event_rows(odds_df, home_team, away_team)
                if event_rows.empty:
                    continue
                event_id = event_rows["event_id"].iloc[0]
                event_df = odds_df[odds_df["event_id"] == event_id].copy()
                best = fetcher.get_best_odds(event_df)
                consensus = fetcher.get_consensus_odds(event_df)
                return {
                    "sources": [
                        SourceNote(
                            name="Odds disk cache",
                            detail=(
                                f"Reused cached {market} market snapshot from {cache_path.name} "
                                "instead of spending a live Odds API call."
                            ),
                        )
                    ],
                    "warnings": [],
                    "unknowns": [],
                    "best_prices": {
                        row["outcome"]: row["price"] for _, row in best.iterrows()
                    },
                    "fair_probabilities": {
                        row["outcome"]: row["fair_prob"] for _, row in consensus.iterrows()
                    },
                    "event": {
                        "event_id": event_id,
                        "sport_key": sport_key,
                        "commence_time": str(event_rows["commence_time"].iloc[0]),
                        "bookmakers": int(event_df["bookmaker"].nunique()),
                        "cached": True,
                    },
                }
            except Exception as exc:
                logger.debug("Cached odds context failed for %s: %s", cache_path.name, exc)
        return None

    def _match_event_rows(
        self,
        odds_df: pd.DataFrame,
        home_team: str,
        away_team: str,
    ) -> pd.DataFrame:
        home_norm = _normalize_team_name(home_team)
        away_norm = _normalize_team_name(away_team)
        work = odds_df.copy()
        work["_home_norm"] = work["home_team"].map(_normalize_team_name)
        work["_away_norm"] = work["away_team"].map(_normalize_team_name)

        exact = work[(work["_home_norm"] == home_norm) & (work["_away_norm"] == away_norm)]
        if not exact.empty:
            return exact

        candidates = work[["event_id", "home_team", "away_team", "_home_norm", "_away_norm"]].drop_duplicates()
        scored_rows: List[Tuple[float, str]] = []
        for _, row in candidates.iterrows():
            score = difflib.SequenceMatcher(None, row["_home_norm"], home_norm).ratio()
            score += difflib.SequenceMatcher(None, row["_away_norm"], away_norm).ratio()
            scored_rows.append((score, row["event_id"]))
        if not scored_rows:
            return pd.DataFrame()

        best_score, best_event_id = max(scored_rows, key=lambda item: item[0])
        if best_score < 1.55:
            return pd.DataFrame()
        return work[work["event_id"] == best_event_id]

    def _resolve_selection(
        self,
        selection: Optional[str],
        bet: str,
        home_team: str,
        away_team: str,
    ) -> str:
        if selection:
            return selection.strip()

        bet_norm = bet.lower()
        if "draw" in bet_norm:
            if "or draw" in bet_norm and _normalize_team_name(away_team) in _normalize_team_name(bet):
                return "away_or_draw"
            if "or draw" in bet_norm:
                return "home_or_draw"
            return "draw"
        if "dnb" in bet_norm or "draw no bet" in bet_norm:
            if _normalize_team_name(away_team) in _normalize_team_name(bet):
                return "away"
            return "home"
        if _normalize_team_name(home_team) in _normalize_team_name(bet):
            return "home"
        if _normalize_team_name(away_team) in _normalize_team_name(bet):
            return "away"
        if "over" in bet_norm:
            return "over"
        if "under" in bet_norm:
            return "under"
        return "home"

    def _selection_market_label(
        self,
        market: str,
        selection: str,
        home_team: str,
        away_team: str,
    ) -> str:
        selection_norm = selection.lower()
        if market == "h2h":
            if selection_norm == "home":
                return home_team
            if selection_norm == "away":
                return away_team
            if selection_norm == "draw":
                return "Draw"
        if market == "double_chance":
            if selection_norm == "home_or_draw":
                return f"{home_team} or Draw"
            if selection_norm == "away_or_draw":
                return f"{away_team} or Draw"
            if selection_norm == "home_or_away":
                return f"{home_team} or {away_team}"
        if market == "draw_no_bet":
            if selection_norm == "home":
                return f"{home_team} DNB"
            if selection_norm == "away":
                return f"{away_team} DNB"
        return selection

    def _team_games(self, df: pd.DataFrame, team: str) -> pd.DataFrame:
        return df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")

    def _build_snapshot(
        self,
        games: pd.DataFrame,
        team: str,
        result_map: Dict[str, Tuple[str, str, str]],
        scored_cols: Tuple[str, str],
        limit: int,
    ) -> TeamSnapshot:
        rows = self._team_games(games, team).tail(limit)
        wins = draws = losses = 0
        scored: List[float] = []
        allowed: List[float] = []
        results_seq: List[str] = []
        scored_flags: List[bool] = []
        conceded_flags: List[bool] = []
        for _, row in rows.iterrows():
            is_home = row["home_team"] == team
            scored_col = scored_cols[0] if is_home else scored_cols[1]
            allowed_col = scored_cols[1] if is_home else scored_cols[0]
            team_scored = float(row[scored_col])
            team_allowed = float(row[allowed_col])
            scored.append(team_scored)
            allowed.append(team_allowed)
            scored_flags.append(team_scored > 0)
            conceded_flags.append(team_allowed > 0)

            home_result, draw_result, away_result = result_map["result"]
            result = row["result"]
            if result == draw_result:
                draws += 1
                results_seq.append("draw")
            elif (is_home and result == home_result) or (not is_home and result == away_result):
                wins += 1
                results_seq.append("win")
            else:
                losses += 1
                results_seq.append("loss")

        last_game = rows["date"].max() if not rows.empty else None
        games_count = len(rows)
        scored_avg = float(pd.Series(scored).mean()) if scored else 0.0
        allowed_avg = float(pd.Series(allowed).mean()) if allowed else 0.0

        def _trailing_count(values: List[Any], predicate) -> int:
            total = 0
            for value in reversed(values):
                if predicate(value):
                    total += 1
                else:
                    break
            return total

        return TeamSnapshot(
            team=team,
            games=games_count,
            wins=wins,
            draws=draws,
            losses=losses,
            scored_avg=scored_avg,
            allowed_avg=allowed_avg,
            margin_avg=scored_avg - allowed_avg,
            unbeaten_streak=_trailing_count(results_seq, lambda item: item != "loss"),
            losing_streak=_trailing_count(results_seq, lambda item: item == "loss"),
            scoring_streak=_trailing_count(scored_flags, lambda item: bool(item)),
            conceding_streak=_trailing_count(conceded_flags, lambda item: bool(item)),
            last_game_date=last_game,
        )

    def _rest_days(self, snapshot: TeamSnapshot) -> Optional[int]:
        if snapshot.last_game_date is None:
            return None
        return int((pd.Timestamp(self.now.date()) - pd.Timestamp(snapshot.last_game_date).normalize()).days)

    def _perspective_multiplier(self, selection: str) -> int:
        if selection.lower() in {"away", "away_or_draw", "away_dnb"}:
            return -1
        return 1

    def _analyze_soccer(self, report: AnalysisReport) -> None:
        fetcher = SoccerFetcher()
        season = _current_soccer_season_start(self.now)
        matches = fetcher.fetch_matches(season=season)
        if matches.empty:
            report.unknowns.append("Soccer match history was unavailable from football-data.org.")
            return

        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        home_snapshot = self._build_snapshot(
            games=matches,
            team=report.home_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_goals", "away_goals"),
            limit=5,
        )
        away_snapshot = self._build_snapshot(
            games=matches,
            team=report.away_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_goals", "away_goals"),
            limit=5,
        )
        report.data_points["home_recent"] = home_snapshot.__dict__
        report.data_points["away_recent"] = away_snapshot.__dict__
        report.sources.append(
            SourceNote(
                name="football-data.org",
                detail=f"Current-season results and standings for season {season}.",
                url="https://www.football-data.org/documentation/quickstart",
            )
        )

        multiplier = self._perspective_multiplier(report.selection)
        self._append_form_signal(report, home_snapshot, away_snapshot, multiplier, "recent form")
        self._append_scoring_signal(report, home_snapshot, away_snapshot, multiplier, "goal balance")
        self._append_streak_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_rest_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_h2h_signal(report, matches, "home_goals", "away_goals")
        self._append_standings_signal(report, fetcher, season)
        self._append_weather_signal(report)
        self._append_soccer_enrichment(report)

    def _analyze_basketball(self, report: AnalysisReport) -> None:
        fetcher = BasketballFetcher()
        season = _current_basketball_season(self.now)
        matches = fetcher.fetch_matches(season=season)
        if matches.empty:
            report.unknowns.append("Basketball game history was unavailable from API-Sports.")
            return

        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        home_snapshot = self._build_snapshot(
            games=matches,
            team=report.home_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_score", "away_score"),
            limit=10,
        )
        away_snapshot = self._build_snapshot(
            games=matches,
            team=report.away_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_score", "away_score"),
            limit=10,
        )
        report.data_points["home_recent"] = home_snapshot.__dict__
        report.data_points["away_recent"] = away_snapshot.__dict__
        report.sources.append(
            SourceNote(
                name="API-Sports Basketball",
                detail=f"NBA results and standings for season {season}.",
                url="https://api-sports.io/sports/basketball",
            )
        )

        multiplier = self._perspective_multiplier(report.selection)
        self._append_form_signal(report, home_snapshot, away_snapshot, multiplier, "recent form")
        self._append_scoring_signal(report, home_snapshot, away_snapshot, multiplier, "point differential")
        self._append_streak_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_rest_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_h2h_signal(report, matches, "home_score", "away_score")
        self._append_basketball_standings_signal(report, fetcher, season)
        self._append_basketball_team_stats(report, fetcher, season, multiplier)

    def _analyze_mlb(self, report: AnalysisReport) -> None:
        fetcher = MLBFetcher()
        season = str(self.now.year)
        matches = fetcher.fetch_matches(season=season)
        if matches.empty:
            report.unknowns.append("MLB recent-game data was unavailable.")
            return

        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        home_snapshot = self._build_snapshot(
            games=matches,
            team=report.home_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_score", "away_score"),
            limit=10,
        )
        away_snapshot = self._build_snapshot(
            games=matches,
            team=report.away_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_score", "away_score"),
            limit=10,
        )
        report.sources.append(
            SourceNote(
                name="MLB Stats API",
                detail=f"Regular-season results for season {season}.",
                url="https://statsapi.mlb.com/",
            )
        )
        multiplier = self._perspective_multiplier(report.selection)
        self._append_form_signal(report, home_snapshot, away_snapshot, multiplier, "recent form")
        self._append_scoring_signal(report, home_snapshot, away_snapshot, multiplier, "run differential")
        self._append_streak_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_rest_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_mlb_probable_starters(report)

    def _analyze_nhl(self, report: AnalysisReport) -> None:
        fetcher = NHLFetcher()
        matches = fetcher.fetch_matches(season=str(self.now.year))
        if matches.empty:
            report.unknowns.append("NHL recent-game data was unavailable.")
            return

        matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
        home_snapshot = self._build_snapshot(
            games=matches,
            team=report.home_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_score", "away_score"),
            limit=10,
        )
        away_snapshot = self._build_snapshot(
            games=matches,
            team=report.away_team,
            result_map={"result": ("home_win", "draw", "away_win")},
            scored_cols=("home_score", "away_score"),
            limit=10,
        )
        report.sources.append(
            SourceNote(
                name="NHL API",
                detail="Recent NHL game and shot-profile data.",
                url="https://api-web.nhle.com/",
            )
        )
        multiplier = self._perspective_multiplier(report.selection)
        self._append_form_signal(report, home_snapshot, away_snapshot, multiplier, "recent form")
        self._append_scoring_signal(report, home_snapshot, away_snapshot, multiplier, "goal differential")
        self._append_streak_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_rest_signal(report, home_snapshot, away_snapshot, multiplier)
        self._append_h2h_signal(report, matches, "home_score", "away_score")

    def _append_form_signal(
        self,
        report: AnalysisReport,
        home_snapshot: TeamSnapshot,
        away_snapshot: TeamSnapshot,
        multiplier: int,
        label: str,
    ) -> None:
        if home_snapshot.games == 0 or away_snapshot.games == 0:
            report.unknowns.append(f"Not enough recent games to score {label}.")
            return
        home_rate = (home_snapshot.wins + 0.5 * home_snapshot.draws) / max(home_snapshot.games, 1)
        away_rate = (away_snapshot.wins + 0.5 * away_snapshot.draws) / max(away_snapshot.games, 1)
        delta = (home_rate - away_rate) * multiplier
        report.signals.append(
            AnalysisSignal(
                name=label,
                score=max(-1.0, min(1.0, delta * 2.0)),
                confidence=0.68,
                summary=(
                    f"{report.home_team} recent points rate {home_rate:.2f} vs "
                    f"{report.away_team} {away_rate:.2f}."
                ),
                data={"home_rate": home_rate, "away_rate": away_rate},
            )
        )

    def _append_scoring_signal(
        self,
        report: AnalysisReport,
        home_snapshot: TeamSnapshot,
        away_snapshot: TeamSnapshot,
        multiplier: int,
        label: str,
    ) -> None:
        delta = (home_snapshot.margin_avg - away_snapshot.margin_avg) * multiplier
        report.signals.append(
            AnalysisSignal(
                name=label,
                score=max(-1.0, min(1.0, delta / 2.5)),
                confidence=0.63,
                summary=(
                    f"{report.home_team} average margin {home_snapshot.margin_avg:+.2f} vs "
                    f"{report.away_team} {away_snapshot.margin_avg:+.2f}."
                ),
                data={
                    "home_margin_avg": home_snapshot.margin_avg,
                    "away_margin_avg": away_snapshot.margin_avg,
                },
            )
        )

    def _append_rest_signal(
        self,
        report: AnalysisReport,
        home_snapshot: TeamSnapshot,
        away_snapshot: TeamSnapshot,
        multiplier: int,
    ) -> None:
        home_rest = self._rest_days(home_snapshot)
        away_rest = self._rest_days(away_snapshot)
        report.data_points["rest_days"] = {"home": home_rest, "away": away_rest}
        if home_rest is None or away_rest is None:
            report.unknowns.append("Rest-day comparison was unavailable.")
            return

        delta = (home_rest - away_rest) * multiplier
        report.signals.append(
            AnalysisSignal(
                name="rest differential",
                score=max(-0.7, min(0.7, delta / 4.0)),
                confidence=0.55,
                summary=f"Rest days: {report.home_team} {home_rest}, {report.away_team} {away_rest}.",
                data={"home_rest_days": home_rest, "away_rest_days": away_rest},
            )
        )

    def _append_streak_signal(
        self,
        report: AnalysisReport,
        home_snapshot: TeamSnapshot,
        away_snapshot: TeamSnapshot,
        multiplier: int,
    ) -> None:
        home_strength = (
            home_snapshot.unbeaten_streak
            + 0.4 * home_snapshot.scoring_streak
            - 0.9 * home_snapshot.losing_streak
            - 0.3 * home_snapshot.conceding_streak
        )
        away_strength = (
            away_snapshot.unbeaten_streak
            + 0.4 * away_snapshot.scoring_streak
            - 0.9 * away_snapshot.losing_streak
            - 0.3 * away_snapshot.conceding_streak
        )
        delta = (home_strength - away_strength) * multiplier
        report.signals.append(
            AnalysisSignal(
                name="recent streaks",
                score=max(-0.75, min(0.75, delta / 4.0)),
                confidence=0.61,
                summary=(
                    f"{report.home_team} unbeaten {home_snapshot.unbeaten_streak}, losing {home_snapshot.losing_streak}, "
                    f"scored in {home_snapshot.scoring_streak} straight; "
                    f"{report.away_team} unbeaten {away_snapshot.unbeaten_streak}, losing {away_snapshot.losing_streak}, "
                    f"conceded in {away_snapshot.conceding_streak} straight."
                ),
                data={
                    "home_unbeaten_streak": home_snapshot.unbeaten_streak,
                    "away_unbeaten_streak": away_snapshot.unbeaten_streak,
                    "home_losing_streak": home_snapshot.losing_streak,
                    "away_losing_streak": away_snapshot.losing_streak,
                    "home_scoring_streak": home_snapshot.scoring_streak,
                    "away_scoring_streak": away_snapshot.scoring_streak,
                    "home_conceding_streak": home_snapshot.conceding_streak,
                    "away_conceding_streak": away_snapshot.conceding_streak,
                },
            )
        )

    def _append_h2h_signal(
        self,
        report: AnalysisReport,
        matches: pd.DataFrame,
        home_score_col: str,
        away_score_col: str,
    ) -> None:
        home_n = _normalize_team_name(report.home_team)
        away_n = _normalize_team_name(report.away_team)
        hn = matches["home_team"].map(_normalize_team_name)
        an = matches["away_team"].map(_normalize_team_name)
        mask = ((hn == home_n) & (an == away_n)) | ((hn == away_n) & (an == home_n))
        h2h = matches.loc[mask].sort_values("date").tail(5)
        if h2h.empty:
            report.unknowns.append("No recent head-to-head sample was found.")
            return

        home_points = 0.0
        for _, row in h2h.iterrows():
            if row["result"] == "draw":
                home_points += 0.5
            elif ((row["home_team"] == report.home_team) and (row["result"] == "home_win")) or (
                (row["away_team"] == report.home_team) and (row["result"] == "away_win")
            ):
                home_points += 1.0
        home_rate = home_points / len(h2h)
        multiplier = self._perspective_multiplier(report.selection)
        report.signals.append(
            AnalysisSignal(
                name="head-to-head",
                score=max(-0.5, min(0.5, (home_rate - 0.5) * 2.0 * multiplier)),
                confidence=0.35,
                summary=(
                    f"{report.home_team} took {home_points:.1f} points-equivalent "
                    f"from the last {len(h2h)} meetings."
                ),
                data={"meetings": len(h2h), "home_points_equivalent": home_points},
            )
        )

    def _append_standings_signal(
        self,
        report: AnalysisReport,
        fetcher: SoccerFetcher,
        season: str,
    ) -> None:
        market_event = ((report.data_points or {}).get("market") or {}).get("event") or {}
        market_sport_key = str(market_event.get("sport_key") or "").strip()
        competitions = []
        mapped_comp = SOCCER_ODDS_TO_COMPETITION.get(market_sport_key)
        if mapped_comp:
            competitions.append(mapped_comp)
        competitions.extend([comp for comp in fetcher._competitions if comp not in competitions])
        best_table = pd.DataFrame()
        for comp in competitions:
            try:
                standings = fetcher.fetch_standings(season=season, competition=comp)
            except Exception:
                continue
            if standings.empty:
                continue
            teams = standings["team_name"].map(_normalize_team_name)
            if _normalize_team_name(report.home_team) in teams.values and _normalize_team_name(report.away_team) in teams.values:
                best_table = standings
                report.data_points["competition"] = comp
                break

        if best_table.empty:
            report.unknowns.append("Standings context was unavailable for this soccer matchup.")
            return

        home_row = self._find_team_row(best_table, "team_name", report.home_team)
        away_row = self._find_team_row(best_table, "team_name", report.away_team)
        if home_row is None or away_row is None:
            report.unknowns.append("Could not map both teams into the standings table.")
            return

        home_ppg = home_row["points"] / max(home_row["played"], 1)
        away_ppg = away_row["points"] / max(away_row["played"], 1)
        multiplier = self._perspective_multiplier(report.selection)
        report.signals.append(
            AnalysisSignal(
                name="table strength",
                score=max(-0.8, min(0.8, (home_ppg - away_ppg) * 0.9 * multiplier)),
                confidence=0.62,
                summary=(
                    f"Points per game: {report.home_team} {home_ppg:.2f}, "
                    f"{report.away_team} {away_ppg:.2f}."
                ),
                data={
                    "home_position": int(home_row["position"]),
                    "away_position": int(away_row["position"]),
                    "home_ppg": home_ppg,
                    "away_ppg": away_ppg,
                },
            )
        )

    def _append_basketball_standings_signal(
        self,
        report: AnalysisReport,
        fetcher: BasketballFetcher,
        season: str,
    ) -> None:
        try:
            standings = fetcher.fetch_standings(season=season)
        except Exception as exc:
            report.warnings.append(f"Basketball standings lookup failed: {exc}")
            return
        if standings.empty:
            report.unknowns.append("Basketball standings were unavailable.")
            return

        home_row = self._find_team_row(standings, "team_name", report.home_team)
        away_row = self._find_team_row(standings, "team_name", report.away_team)
        if home_row is None or away_row is None:
            report.unknowns.append("Could not map both NBA teams to the standings table.")
            return

        home_wpct = _safe_float(home_row["win_pct"]) or 0.0
        away_wpct = _safe_float(away_row["win_pct"]) or 0.0
        multiplier = self._perspective_multiplier(report.selection)
        report.signals.append(
            AnalysisSignal(
                name="season strength",
                score=max(-0.8, min(0.8, (home_wpct - away_wpct) * 2.0 * multiplier)),
                confidence=0.64,
                summary=(
                    f"Season win rate: {report.home_team} {home_wpct:.3f}, "
                    f"{report.away_team} {away_wpct:.3f}."
                ),
                data={
                    "home_group": home_row.get("group"),
                    "away_group": away_row.get("group"),
                    "home_win_pct": home_wpct,
                    "away_win_pct": away_wpct,
                },
            )
        )

    def _append_basketball_team_stats(
        self,
        report: AnalysisReport,
        fetcher: BasketballFetcher,
        season: str,
        multiplier: int,
    ) -> None:
        try:
            standings = fetcher.fetch_standings(season=season)
        except Exception:
            return
        if standings.empty:
            return
        home_row = self._find_team_row(standings, "team_name", report.home_team)
        away_row = self._find_team_row(standings, "team_name", report.away_team)
        if home_row is None or away_row is None:
            return

        try:
            home_stats = fetcher.fetch_team_statistics(int(home_row["team_id"]), season)
            away_stats = fetcher.fetch_team_statistics(int(away_row["team_id"]), season)
        except Exception as exc:
            report.warnings.append(f"Basketball team-stat lookup failed: {exc}")
            return

        home_points = _safe_float((home_stats or {}).get("points", {}).get("for", {}).get("average", {}).get("all"))
        away_points = _safe_float((away_stats or {}).get("points", {}).get("for", {}).get("average", {}).get("all"))
        home_allowed = _safe_float((home_stats or {}).get("points", {}).get("against", {}).get("average", {}).get("all"))
        away_allowed = _safe_float((away_stats or {}).get("points", {}).get("against", {}).get("average", {}).get("all"))
        if None in (home_points, away_points, home_allowed, away_allowed):
            report.unknowns.append("Full NBA offense/defense averages were unavailable.")
            return

        home_net = home_points - home_allowed
        away_net = away_points - away_allowed
        report.signals.append(
            AnalysisSignal(
                name="team efficiency proxy",
                score=max(-1.0, min(1.0, (home_net - away_net) / 12.0 * multiplier)),
                confidence=0.58,
                summary=(
                    f"Net scoring proxy: {report.home_team} {home_net:+.1f}, "
                    f"{report.away_team} {away_net:+.1f}."
                ),
                data={"home_net_proxy": home_net, "away_net_proxy": away_net},
            )
        )

    def _append_soccer_enrichment(self, report: AnalysisReport) -> None:
        if not os.environ.get("API_SPORTS_KEY"):
            report.unknowns.append(
                "API_SPORTS_KEY is not configured, so xG/corners/injury-style soccer enrichment is unavailable."
            )
            return
        try:
            enricher = APIFootballEnricher()
            enrichment = enricher._fetch_match_enrichment(
                report.home_team,
                report.away_team,
                fields=["form", "xg", "corners", "h2h"],
            )
        except Exception as exc:
            report.warnings.append(f"Soccer enrichment lookup failed: {exc}")
            return

        if not enrichment:
            report.unknowns.append("API-Football enrichment returned no matchup-specific data.")
            return

        report.data_points["soccer_enrichment"] = enrichment
        report.sources.append(
            SourceNote(
                name="API-Football",
                detail="Match enrichment for form, xG, corners, and head-to-head.",
                url="https://www.api-football.com/documentation-v3",
            )
        )

        home_xg = _safe_float(enrichment.get("home_xg"))
        away_xg = _safe_float(enrichment.get("away_xg"))
        if home_xg is not None and away_xg is not None:
            multiplier = self._perspective_multiplier(report.selection)
            report.signals.append(
                AnalysisSignal(
                    name="xg edge",
                    score=max(-0.7, min(0.7, (home_xg - away_xg) * 0.7 * multiplier)),
                    confidence=0.54,
                    summary=(
                        f"API-Football xG snapshot: {report.home_team} {home_xg:.2f}, "
                        f"{report.away_team} {away_xg:.2f}."
                    ),
                    data={"home_xg": home_xg, "away_xg": away_xg},
                )
            )

    def _append_weather_signal(self, report: AnalysisReport) -> None:
        if report.sport not in {"soccer", "mlb"}:
            return
        if not self.weather_api_key:
            report.unknowns.append(
                "OPENWEATHER_API_KEY is not configured, so weather risk is not included."
            )
            return

        query = f"{report.home_team}"
        try:
            geo_resp = requests.get(
                "http://api.openweathermap.org/geo/1.0/direct",
                params={"q": query, "limit": 1, "appid": self.weather_api_key},
                timeout=10,
            )
            geo_resp.raise_for_status()
            geos = geo_resp.json()
            if not geos:
                report.unknowns.append("Weather lookup could not determine a venue/city.")
                return
            geo = geos[0]
            weather_resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": geo["lat"], "lon": geo["lon"], "appid": self.weather_api_key, "units": "metric"},
                timeout=10,
            )
            weather_resp.raise_for_status()
            weather = weather_resp.json()
        except Exception as exc:
            report.warnings.append(f"Weather lookup failed: {exc}")
            return

        wind = _safe_float((weather.get("wind") or {}).get("speed")) or 0.0
        rain = _safe_float((weather.get("rain") or {}).get("1h")) or 0.0
        desc = ((weather.get("weather") or [{}])[0]).get("description", "unknown conditions")
        report.sources.append(
            SourceNote(
                name="OpenWeather",
                detail=f"Current weather near '{query}' for outdoor-risk context.",
                url="https://openweathermap.org/api",
            )
        )
        if wind >= 8 or rain > 0:
            report.signals.append(
                AnalysisSignal(
                    name="weather volatility",
                    score=-0.25,
                    confidence=0.42,
                    summary=f"Outdoor conditions look noisy: {desc}, wind {wind:.1f} m/s, rain {rain:.1f} mm.",
                    data={"description": desc, "wind_mps": wind, "rain_mm": rain},
                )
            )

    def _find_team_row(
        self,
        df: pd.DataFrame,
        column: str,
        team_name: str,
    ) -> Optional[pd.Series]:
        if df.empty:
            return None
        target = _normalize_team_name(team_name)
        norm = df[column].fillna("").map(_normalize_team_name)
        exact = df.loc[norm == target]
        if not exact.empty:
            return exact.iloc[0]

        if norm.empty:
            return None
        scores = norm.map(lambda candidate: difflib.SequenceMatcher(None, candidate, target).ratio())
        if scores.max() < 0.72:
            return None
        return df.loc[scores.idxmax()]

    # ------------------------------------------------------------------
    # MLB probable starters
    # ------------------------------------------------------------------

    def _append_mlb_probable_starters(self, report: AnalysisReport) -> None:
        """Fetch today's probable starters and add a pitching matchup signal."""
        try:
            fetcher = MLBFetcher()
            pitchers = fetcher.fetch_todays_probable_pitchers(report.home_team, report.away_team)
        except Exception as exc:
            report.unknowns.append(f"Starting pitcher lookup failed: {exc}")
            return

        if pitchers is None:
            report.unknowns.append(
                "Probable starting pitchers have not been announced yet or game not found."
            )
            return

        home_era  = pitchers["home_sp_era"]
        away_era  = pitchers["away_sp_era"]
        home_whip = pitchers["home_sp_whip"]
        away_whip = pitchers["away_sp_whip"]
        home_gs   = pitchers.get("home_sp_gs", 0)
        away_gs   = pitchers.get("away_sp_gs", 0)
        home_name = pitchers.get("home_pitcher_name", "TBD")
        away_name = pitchers.get("away_pitcher_name", "TBD")

        # ERA + WHIP combined edge: positive = home pitcher advantage
        era_edge  = away_era  - home_era
        whip_edge = away_whip - home_whip
        combined_edge = 0.6 * era_edge + 0.4 * whip_edge

        multiplier  = self._perspective_multiplier(report.selection)
        score       = max(-1.0, min(1.0, combined_edge / 1.2 * multiplier))
        confidence  = 0.66 if min(home_gs, away_gs) >= 5 else 0.44

        report.data_points["probable_pitchers"] = {
            "home": {"name": home_name, "era": home_era, "whip": home_whip, "gs": home_gs},
            "away": {"name": away_name, "era": away_era, "whip": away_whip, "gs": away_gs},
        }
        report.sources.append(SourceNote(
            name="MLB Stats API – probable pitchers",
            detail=f"{home_name} (ERA {home_era:.2f}) vs {away_name} (ERA {away_era:.2f}).",
            url="https://statsapi.mlb.com/",
        ))
        report.signals.append(AnalysisSignal(
            name="pitching matchup",
            score=score,
            confidence=confidence,
            summary=(
                f"{home_name} ERA {home_era:.2f} WHIP {home_whip:.2f} vs "
                f"{away_name} ERA {away_era:.2f} WHIP {away_whip:.2f}."
            ),
            data=report.data_points["probable_pitchers"],
        ))
        if min(home_gs, away_gs) < 5:
            report.unknowns.append(
                f"One or both starters has < 5 starts (home {home_gs}, away {away_gs}) "
                "— ERA/WHIP sample size is small."
            )

    # ------------------------------------------------------------------
    # Tennis analyst
    # ------------------------------------------------------------------

    @staticmethod
    def _tennis_name_variants(query: str, all_players: set) -> List[str]:
        """
        Return every canonical name in the cache that shares a surname with query.

        Handles the Sackmann ("Aryna Sabalenka") vs tennis-data.co.uk
        ("Sabalenka A.") naming split by matching on the longest token (surname).
        """
        query_tokens = re.sub(r"[^a-z ]", "", query.lower()).split()
        # Pick tokens longer than 2 chars as surname candidates (excludes initials)
        surnames = {t for t in query_tokens if len(t) > 2}
        if not surnames:
            # Fall back to full fuzzy resolve
            return [resolve_canonical_name(query, all_players,
                                           alias_map=build_entity_alias_map(all_players))]
        variants = []
        for name in all_players:
            name_tokens = re.sub(r"[^a-z ]", "", name.lower()).split()
            name_surnames = {t for t in name_tokens if len(t) > 2}
            if surnames & name_surnames:
                variants.append(name)
        return variants

    def _analyze_tennis(self, report: AnalysisReport) -> None:
        """
        Analyze a tennis match using our own ATP/WTA feature cache.

        No external API needed — all signals come from historical match data
        already used for model training (form, surface win rate, H2H, serve quality).
        """
        # Determine surface from the market sport_key
        market_event = ((report.data_points or {}).get("market") or {}).get("event") or {}
        sport_key = str(market_event.get("sport_key") or "").lower()
        if any(t in sport_key for t in ("clay", "roland_garros", "madrid",
                                         "barcelona", "rome", "monte_carlo",
                                         "hamburg", "gstaad", "bastad")):
            surface = "Clay"
        elif any(t in sport_key for t in ("wimbledon", "queens", "halle",
                                           "grass", "eastbourne")):
            surface = "Grass"
        else:
            surface = "Hard"

        project_root = Path(__file__).resolve().parent.parent.parent
        cache_candidates = [
            (project_root / "data" / "cache" / "tennis_features.parquet",     "ATP"),
            (project_root / "data" / "cache" / "tennis_wta_features.parquet", "WTA"),
        ]

        fdf: Optional[pd.DataFrame] = None
        p1_names: List[str] = []
        p2_names: List[str] = []
        tour_label = "Tennis"

        for cache_path, label in cache_candidates:
            if not cache_path.exists():
                continue
            try:
                _df = pd.read_parquet(cache_path)
                _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
                _all = (
                    set(_df["player1_name"].dropna().astype(str).unique())
                    | set(_df["player2_name"].dropna().astype(str).unique())
                )
                _p1v = self._tennis_name_variants(report.home_team, _all)
                _p2v = self._tennis_name_variants(report.away_team, _all)
                # Verify at least one variant has real history (≥5 appearances)
                def _has_history(df, variants, min_rows=5):
                    return any(
                        ((df["player1_name"] == n) | (df["player2_name"] == n)).sum() >= min_rows
                        for n in variants
                    )
                p1_ok = _has_history(_df, _p1v)
                p2_ok = _has_history(_df, _p2v)
                if p1_ok and p2_ok:
                    fdf, p1_names, p2_names, tour_label = _df, _p1v, _p2v, label
                    break
                if fdf is None and (p1_ok or p2_ok):
                    fdf, p1_names, p2_names, tour_label = _df, _p1v, _p2v, label
            except Exception as exc:
                logger.debug("Tennis cache %s failed: %s", label, exc)

        if fdf is None:
            report.unknowns.append("Tennis feature cache was unavailable.")
            return
        if not p1_names:
            report.unknowns.append(f"Could not find {report.home_team} in {tour_label} match history.")
        if not p2_names:
            report.unknowns.append(f"Could not find {report.away_team} in {tour_label} match history.")
        if not p1_names or not p2_names:
            return

        report.sources.append(SourceNote(
            name=f"{tour_label} match history (2022–2026)",
            detail="Form, surface win rate, serve stats, and H2H from internal feature cache.",
        ))

        multiplier = self._perspective_multiplier(report.selection)

        # Filter rows for each player across all their name variants
        # p1_all / p2_all used for serve stats (need deeper history for Sackmann data)
        # p1_rows / p2_rows limited to last 15 for form signals
        p1_mask = fdf["player1_name"].isin(p1_names) | fdf["player2_name"].isin(p1_names)
        p2_mask = fdf["player1_name"].isin(p2_names) | fdf["player2_name"].isin(p2_names)
        p1_all  = fdf.loc[p1_mask].sort_values("date")
        p2_all  = fdf.loc[p2_mask].sort_values("date")
        p1_rows = p1_all.tail(15)
        p2_rows = p2_all.tail(15)

        # ---- Recent form (last 15 matches) --------------------------------
        def _win_rate(rows: pd.DataFrame, player_names: List[str]) -> Optional[float]:
            if len(rows) < 3:
                return None
            wins = sum(
                1 for _, r in rows.iterrows()
                if (r["player1_name"] in player_names and r.get("result") == "player1_win")
                or (r["player2_name"] in player_names and r.get("result") == "player2_win")
            )
            return wins / len(rows)

        p1_form = _win_rate(p1_rows, p1_names)
        p2_form = _win_rate(p2_rows, p2_names)
        if p1_form is not None and p2_form is not None:
            delta = (p1_form - p2_form) * multiplier
            report.signals.append(AnalysisSignal(
                name="recent form",
                score=max(-1.0, min(1.0, delta * 2.0)),
                confidence=0.65,
                summary=(
                    f"Last 15 matches: {report.home_team} {p1_form:.0%} vs "
                    f"{report.away_team} {p2_form:.0%}."
                ),
                data={"p1_form": p1_form, "p2_form": p2_form},
            ))
        else:
            report.unknowns.append("Insufficient match history for recent form comparison.")

        # ---- Surface form -------------------------------------------------
        if "surface" in fdf.columns:
            p1_surf_rows = p1_rows[p1_rows["surface"] == surface]
            p2_surf_rows = p2_rows[p2_rows["surface"] == surface]
            p1_surf = _win_rate(p1_surf_rows, p1_names)
            p2_surf = _win_rate(p2_surf_rows, p2_names)
            if p1_surf is not None and p2_surf is not None:
                delta = (p1_surf - p2_surf) * multiplier
                report.signals.append(AnalysisSignal(
                    name=f"{surface.lower()} court form",
                    score=max(-1.0, min(1.0, delta * 2.0)),
                    confidence=0.62,
                    summary=(
                        f"{surface} win rate: {report.home_team} {p1_surf:.0%}, "
                        f"{report.away_team} {p2_surf:.0%}."
                    ),
                    data={"surface": surface, "p1_surf_form": p1_surf, "p2_surf_form": p2_surf},
                ))
            else:
                report.unknowns.append(
                    f"Insufficient {surface} court sample to score surface form."
                )

        # ---- Head-to-head -------------------------------------------------
        h2h_mask = (
            (fdf["player1_name"].isin(p1_names) & fdf["player2_name"].isin(p2_names))
            | (fdf["player1_name"].isin(p2_names) & fdf["player2_name"].isin(p1_names))
        )
        h2h_rows = fdf.loc[h2h_mask].sort_values("date").tail(10)
        if len(h2h_rows) >= 2:
            p1_wins = sum(
                1 for _, r in h2h_rows.iterrows()
                if (r["player1_name"] in p1_names and r.get("result") == "player1_win")
                or (r["player2_name"] in p1_names and r.get("result") == "player2_win")
            )
            total = len(h2h_rows)
            p1_h2h = p1_wins / total
            delta  = (p1_h2h - 0.5) * multiplier
            report.signals.append(AnalysisSignal(
                name="head-to-head",
                score=max(-0.5, min(0.5, delta)),
                confidence=0.55,
                summary=(
                    f"{report.home_team} won {p1_wins}/{total} prior meetings "
                    f"({p1_h2h:.0%})."
                ),
                data={"p1_wins": p1_wins, "total": total, "p1_h2h_rate": p1_h2h},
            ))
        else:
            report.unknowns.append("No head-to-head history found between these players.")

        # ---- Serve quality (rolling bp save + ace rate) -------------------
        def _rolling_stat(rows: pd.DataFrame, player_names: List[str], stat: str) -> Optional[float]:
            p1_col = f"roll_p1_{stat}"
            p2_col = f"roll_p2_{stat}"
            vals: List[float] = []
            if p1_col in rows.columns:
                mask_p1 = rows["player1_name"].isin(player_names)
                vals += rows.loc[mask_p1, p1_col].replace(0, float("nan")).dropna().tolist()
            if p2_col in rows.columns:
                mask_p2 = rows["player2_name"].isin(player_names)
                vals += rows.loc[mask_p2, p2_col].replace(0, float("nan")).dropna().tolist()
            return float(pd.Series(vals).mean()) if len(vals) >= 3 else None

        p1_bp = _rolling_stat(p1_all, p1_names, "bp_save")
        p2_bp = _rolling_stat(p2_all, p2_names, "bp_save")
        if p1_bp is not None and p2_bp is not None:
            delta = (p1_bp - p2_bp) * multiplier
            report.signals.append(AnalysisSignal(
                name="break point defence",
                score=max(-0.7, min(0.7, delta * 3.0)),
                confidence=0.55,
                summary=(
                    f"Rolling BP save rate: {report.home_team} {p1_bp:.0%}, "
                    f"{report.away_team} {p2_bp:.0%}."
                ),
                data={"p1_bp_save": p1_bp, "p2_bp_save": p2_bp},
            ))

        p1_ace = _rolling_stat(p1_all, p1_names, "ace_rate")
        p2_ace = _rolling_stat(p2_all, p2_names, "ace_rate")
        if p1_ace is not None and p2_ace is not None:
            delta = (p1_ace - p2_ace) * multiplier
            report.signals.append(AnalysisSignal(
                name="serve aggressiveness",
                score=max(-0.5, min(0.5, delta * 20.0)),
                confidence=0.45,
                summary=(
                    f"Rolling ace rate: {report.home_team} {p1_ace:.1%}, "
                    f"{report.away_team} {p2_ace:.1%}."
                ),
                data={"p1_ace_rate": p1_ace, "p2_ace_rate": p2_ace},
            ))

    # ------------------------------------------------------------------

    def _add_market_signal(self, report: AnalysisReport) -> None:
        if report.fair_prob is None or report.price_used is None:
            report.unknowns.append(
                "No clean market edge estimate was available for the requested selection."
            )
            return
        implied = 1.0 / report.price_used
        edge = report.fair_prob - implied
        report.signals.append(
            AnalysisSignal(
                name="market edge",
                score=max(-1.0, min(1.0, edge * 8.0)),
                confidence=0.76,
                summary=(
                    f"Consensus fair probability {report.fair_prob:.1%} vs "
                    f"price-implied {implied:.1%}."
                ),
                data={"fair_prob": report.fair_prob, "implied_prob": implied, "edge": edge},
            )
        )

    def _finalize_report(self, report: AnalysisReport) -> None:
        unique_warnings = list(dict.fromkeys(report.warnings))
        unique_unknowns = list(dict.fromkeys(report.unknowns))
        report.warnings = unique_warnings
        report.unknowns = unique_unknowns

        weighted_scores = [signal.score * signal.confidence for signal in report.signals]
        report.score = float(sum(weighted_scores))
        avg_conf = sum(signal.confidence for signal in report.signals) / max(len(report.signals), 1)
        report.confidence = max(0.25, min(0.9, 0.35 + 0.25 * min(abs(report.score), 2.0) + 0.2 * avg_conf))

        if report.selection.lower() in {"over", "under"}:
            report.warnings.append(
                "Totals markets are only partially supported in this first version; verdict leans on general matchup context."
            )
        if report.selection.lower() == "draw":
            report.warnings.append(
                "Draw analysis has weaker sport-specific signals than home/away moneyline analysis."
            )

        if report.edge_pct is not None and report.edge_pct <= -0.015:
            report.verdict = "avoid"
            return
        if report.score >= 1.1:
            report.verdict = "support"
        elif report.score >= 0.35:
            report.verdict = "lean"
        elif report.score <= -0.55:
            report.verdict = "avoid"
        else:
            report.verdict = "pass"
