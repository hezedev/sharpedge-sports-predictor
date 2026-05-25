"""
Parlay Builder
==============
Constructs optimal parlays from a pool of candidate bets to hit target
combined-odds brackets (5x, 10x, 20x) while maximising expected value.

Design principles:
- Parlay EV  = Π(ml_prob_i) × Π(decimal_odds_i)
- Parlay edge = EV − 1
- Kelly stake  = (edge / (combined_odds − 1)) × fraction
- Legs from the same match are mutually exclusive (never combined).
- Legs are validated for minimum individual confidence before combining.

Two selection pools are supported:
    VALUE       — tighter, higher-discipline combinations from the strongest
                  positive-edge legs
    LONGSHOT    — more aggressive upside builds that still require some model
                  support and at least one genuinely higher-odds leg

Target brackets (default):
    5x  : combined odds in [4.00, 6.50)
    10x : combined odds in [6.50, 14.00)
    20x : combined odds in [20.00, 40.00)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from config import settings

logger = logging.getLogger(__name__)


_PARLAY_CFG = (((settings or {}).get("betting") or {}).get("parlay") or {})


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParlayLeg:
    """A single selection inside a parlay."""
    sport: str           # 'soccer' | 'basketball'
    match_id: str        # unique match key (e.g. 'home_team vs away_team')
    team: str            # selected team / outcome label
    odds: float          # decimal odds
    ml_prob: float       # model-estimated win probability
    fair_prob: float     # vig-removed bookmaker implied probability
    edge: float          # ml_prob − fair_prob
    commence: str        # ISO commence time string
    window: str = "today"  # 'today' (before midnight Vienna) or 'overnight' (00:00–07:00)
    market: str = ""
    home_team: str = ""
    away_team: str = ""


@dataclass
class Parlay:
    """A multi-leg parlay with computed EV and Kelly stake."""
    legs: List[ParlayLeg]
    combined_odds: float = field(init=False)
    combined_prob: float = field(init=False)
    ev: float            = field(init=False)
    edge: float          = field(init=False)
    kelly_stake_pct: float = field(init=False)
    tier: str            = ""   # 'value' | 'speculative'
    target_bracket: str  = ""   # '5x' | '10x' | '20x'
    risk_tier: str       = ""
    build_verdict: str   = ""
    weakest_leg: Optional[dict[str, Any]] = None
    duplicate_games: List[str] = field(default_factory=list)
    conflicting_picks: List[str] = field(default_factory=list)
    correlated_pick_groups: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.combined_odds = float(np.prod([leg.odds for leg in self.legs]))
        self.combined_prob = float(np.prod([leg.ml_prob for leg in self.legs]))
        self.ev = self.combined_prob * self.combined_odds
        self.edge = self.ev - 1.0
        weakest = min(self.legs, key=lambda leg: (leg.ml_prob, leg.edge, -leg.odds), default=None)
        if weakest is not None:
            self.weakest_leg = {
                "team": weakest.team,
                "match_id": weakest.match_id,
                "sport": weakest.sport,
                "market": weakest.market,
                "odds": weakest.odds,
                "ml_prob": weakest.ml_prob,
                "edge": weakest.edge,
            }
        # Fractional Kelly for parlay
        self.kelly_stake_pct = self._kelly()

    def _kelly(self, fraction: float = 0.25, max_pct: float = 3.0) -> float:
        """
        Kelly Criterion adapted for a parlay.

        full_kelly = edge / (combined_odds − 1)
        We cap at max_pct to control risk on long-shot parlays.
        """
        denom = self.combined_odds - 1.0
        if denom <= 0 or self.combined_odds < 1.01:
            return 0.0
        full_kelly = self.edge / denom
        return round(min(max(full_kelly * fraction * 100, 0.0), max_pct), 2)

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    def summary_line(self) -> str:
        teams = " + ".join(f"{leg.team}({leg.odds})" for leg in self.legs)
        return (
            f"[{self.target_bracket}] {teams} → "
            f"odds={self.combined_odds:.2f}  EV={self.ev:.3f}  "
            f"Kelly={self.kelly_stake_pct:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Target brackets
# ─────────────────────────────────────────────────────────────────────────────

BRACKETS: Dict[str, Tuple[float, float]] = {
    "5x":  (4.00,  6.50),
    "10x": (6.50, 14.00),
    "20x": (20.00, 40.00),
}

# Minimum legs required per bracket (relaxed for shorter-odds brackets)
BRACKET_MIN_LEGS: Dict[str, int] = {
    "5x":  2,  # 2 legs at ~2.2 odds = 4.84 ✓
    "10x": 2,  # 2 legs at ~3.1 or 3 at ~2.0
    "20x": 3,  # 3 legs at ~2.7 or higher
}


def classify_bracket(odds: float) -> Optional[str]:
    for name, (lo, hi) in BRACKETS.items():
        if lo <= odds < hi:
            return name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

class ParlayBuilder:
    """
    Build optimal parlays from a candidate bet pool.

    Parameters
    ----------
    min_edge : float
        Minimum ML edge for a leg to qualify as 'value'.
    min_prob : float
        Minimum ML probability for a leg to qualify as a longshot-only add.
    min_legs : int
        Minimum number of legs per parlay (default 3).
    max_legs : int
        Maximum number of legs per parlay (default 6).
    top_n : int
        Number of top parlays to return per (bracket × tier) combination.
    fraction : float
        Fractional Kelly multiplier (default 0.25).
    max_kelly_pct : float
        Hard cap on recommended stake as % of bankroll (default 3.0%).
    """

    def __init__(
        self,
        min_edge: float = 0.03,
        min_prob: float = 0.50,
        min_legs: int = 3,
        max_legs: int = 6,
        top_n: int = 3,
        fraction: float = 0.25,
        max_kelly_pct: float = 3.0,
        value_min_prob: float = 0.54,
        value_max_leg_odds: float = 3.25,
        longshot_min_prob: float = 0.42,
        longshot_min_leg_odds: float = 1.8,
        longshot_min_combined_odds: float = 20.0,
        longshot_min_aggressive_leg_odds: float = 3.0,
    ) -> None:
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.min_legs = min_legs
        self.max_legs = max_legs
        self.top_n = top_n
        self.fraction = fraction
        self.max_kelly_pct = max_kelly_pct
        self.value_min_prob = value_min_prob
        self.value_max_leg_odds = value_max_leg_odds
        self.longshot_min_prob = longshot_min_prob
        self.longshot_min_leg_odds = longshot_min_leg_odds
        self.longshot_min_combined_odds = longshot_min_combined_odds
        self.longshot_min_aggressive_leg_odds = longshot_min_aggressive_leg_odds
        self.conservative_max_legs = int(_PARLAY_CFG.get("conservative_max_legs", 5) or 5)
        self.conservative_min_legs = int(_PARLAY_CFG.get("conservative_min_legs", 3) or 3)
        self.conservative_min_leg_prob = float(_PARLAY_CFG.get("conservative_min_leg_prob", 0.54) or 0.54)
        self.conservative_min_average_prob = float(_PARLAY_CFG.get("conservative_min_average_prob", 0.57) or 0.57)
        self.medium_risk_min_leg_prob = float(_PARLAY_CFG.get("medium_risk_min_leg_prob", 0.50) or 0.50)
        self.speculative_min_leg_prob = float(_PARLAY_CFG.get("speculative_min_leg_prob", 0.46) or 0.46)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        candidates: List[ParlayLeg],
    ) -> Dict[str, Dict[str, List[Parlay]]]:
        """
        Build parlays from the candidate pool.

        Two tiers are produced:

        VALUE
            All legs have ML edge ≥ min_edge. Strictly positive-expectation
            parlays — the model beats the bookmaker on every leg.

        SPECULATIVE
            Legs where ML prob ≥ min_prob (model predicts this team wins) but
            edge requirement is relaxed. Includes favourite picks at low odds
            (1.15–2.0) that allow smaller 5x/10x parlays to be assembled.

        Parameters
        ----------
        candidates : list[ParlayLeg]
            All available bet selections with ML probabilities.

        Returns
        -------
        dict
            {
                'value':       {'5x': [...], '10x': [...], '20x': [...]},
                'speculative': {'5x': [...], '10x': [...], '20x': [...]},
            }
        """
        value_pool = [l for l in candidates if l.edge >= self.min_edge]
        disciplined_value_pool = [
            l for l in value_pool
            if l.ml_prob >= self.value_min_prob and l.odds <= self.value_max_leg_odds
        ]
        if len(disciplined_value_pool) >= 2:
            value_pool = disciplined_value_pool

        # Longshot-only pool: some model support, but aimed at higher-upside
        # legs rather than merely mixing in short-odds favourites.
        speculative_pool = [
            l for l in candidates
            if l.ml_prob >= self.longshot_min_prob
            and l.odds >= self.longshot_min_leg_odds
            and l.edge >= -0.01
            and l.edge < self.min_edge
        ]

        # Combined longshot pool: stronger value legs plus at least one true
        # upside leg from the speculative pool.
        combined_spec_pool = value_pool + speculative_pool
        speculative_keys = {(l.match_id, l.team) for l in speculative_pool}

        logger.info(
            "Parlay pools — value: %d legs, longshot_only: %d legs, combined: %d legs",
            len(value_pool), len(speculative_pool), len(combined_spec_pool),
        )

        results: Dict[str, Dict[str, List[Parlay]]] = {
            "value": {b: [] for b in BRACKETS},
            "speculative": {b: [] for b in BRACKETS},
        }

        if value_pool:
            results["value"] = self._search(value_pool, tier="value")
        if combined_spec_pool and speculative_pool:
            # Only build longshot tier if there are non-value legs to add.
            results["speculative"] = self._search(
                combined_spec_pool,
                tier="speculative",
                required_leg_keys=speculative_keys,
            )

        return results

    # ------------------------------------------------------------------
    # Internal search
    # ------------------------------------------------------------------

    def _search(
        self,
        pool: List[ParlayLeg],
        tier: str,
        required_leg_keys: Optional[set[tuple[str, str]]] = None,
    ) -> Dict[str, List[Parlay]]:
        """
        Enumerate all valid leg combinations and bucket by bracket.

        Pruning:
        - Skip combinations containing two legs from the same match.
        - Skip combinations whose combined EV ≤ 1.0 (negative expectation).
        - Short-circuit early if combined odds already exceed 26 during build.
        """
        bracket_best: Dict[str, List[Parlay]] = {b: [] for b in BRACKETS}
        total_evaluated = 0

        # Pre-sort pool by odds ascending so lower-odds legs come first,
        # making it easier to enumerate small-odds 5x/10x combos first.
        pool_sorted = sorted(pool, key=lambda l: l.odds)

        for n_legs in range(self.min_legs, self.max_legs + 1):
            for combo in combinations(pool_sorted, n_legs):
                # --- Constraint: no two legs from the same match ---
                assessment = self.assess_legs(list(combo), tier=tier)
                if assessment["build_verdict"] == "DO NOT BUILD":
                    continue

                if tier == "value":
                    sport_counts: Dict[str, int] = {}
                    for leg in combo:
                        sport_counts[leg.sport] = sport_counts.get(leg.sport, 0) + 1
                    if any(count > 2 for count in sport_counts.values()):
                        continue
                    if sum(1 for leg in combo if leg.odds >= 3.0) > 1:
                        continue
                    if float(np.mean([leg.ml_prob for leg in combo])) < self.value_min_prob:
                        continue

                if tier == "speculative":
                    if required_leg_keys and not any((leg.match_id, leg.team) in required_leg_keys for leg in combo):
                        continue
                    if n_legs < 3:
                        continue
                    if max(leg.odds for leg in combo) < self.longshot_min_aggressive_leg_odds:
                        continue
                    if sum(1 for leg in combo if leg.odds >= 2.2) < 2:
                        continue

                total_evaluated += 1

                combined_odds = float(np.prod([l.odds for l in combo]))
                if tier == "speculative" and combined_odds < self.longshot_min_combined_odds:
                    continue
                bracket = classify_bracket(combined_odds)
                if bracket is None:
                    continue

                # Enforce per-bracket minimum legs
                bracket_min = BRACKET_MIN_LEGS.get(bracket, self.min_legs)
                if n_legs < bracket_min:
                    continue

                combined_prob = float(np.prod([l.ml_prob for l in combo]))
                ev = combined_prob * combined_odds
                if ev <= 1.0:
                    continue  # negative expectation

                parlay = self._create_parlay(list(combo), tier=tier, target_bracket=bracket, assessment=assessment)
                bracket_best[bracket].append(parlay)

        logger.info(
            "[%s] evaluated %d combinations → found %s",
            tier, total_evaluated,
            {b: len(v) for b, v in bracket_best.items()},
        )

        # Keep top N per bracket by tier-specific rank:
        # value -> safer / cleaner combinations
        # speculative -> more aggressive upside while staying +EV
        return {
            bracket: sorted(parlays, key=lambda p: self._rank_key(p, tier), reverse=True)[: self.top_n]
            for bracket, parlays in bracket_best.items()
        }

    def _rank_key(self, parlay: Parlay, tier: str):
        avg_edge = float(np.mean([leg.edge for leg in parlay.legs])) if parlay.legs else 0.0
        max_leg_odds = max((leg.odds for leg in parlay.legs), default=1.0)
        if tier == "value":
            return (
                1 if parlay.risk_tier == "conservative" else 0,
                parlay.combined_prob,
                avg_edge,
                -parlay.combined_odds,
            )
        min_leg_prob = min((leg.ml_prob for leg in parlay.legs), default=0.0)
        aggressive_legs = sum(1 for leg in parlay.legs if leg.odds >= self.longshot_min_aggressive_leg_odds)
        return (
            1 if parlay.risk_tier == "medium-risk" else 0,
            parlay.combined_odds,
            aggressive_legs,
            parlay.edge * parlay.combined_prob,
            min_leg_prob,
            avg_edge,
            max_leg_odds,
        )

    def _create_parlay(
        self,
        legs: List[ParlayLeg],
        tier: str,
        target_bracket: str,
        assessment: Optional[dict[str, Any]] = None,
    ) -> Parlay:
        assessment = assessment or self.assess_legs(legs, tier=tier)
        return Parlay(
            legs=legs,
            tier=tier,
            target_bracket=target_bracket,
            risk_tier=str(assessment.get("risk_tier", "")),
            build_verdict=str(assessment.get("build_verdict", "")),
            duplicate_games=list(assessment.get("duplicate_games", [])),
            conflicting_picks=list(assessment.get("conflicting_picks", [])),
            correlated_pick_groups=list(assessment.get("correlated_pick_groups", [])),
            validation_notes=list(assessment.get("notes", [])),
        )

    def assess_legs(self, legs: List[ParlayLeg], tier: str = "") -> dict[str, Any]:
        notes: List[str] = []
        duplicate_games: List[str] = []
        conflicting_picks: List[str] = []
        correlated_pick_groups: List[str] = []

        match_groups: Dict[str, List[ParlayLeg]] = {}
        for leg in legs:
            match_groups.setdefault(leg.match_id, []).append(leg)

        for match_id, grouped in match_groups.items():
            if len(grouped) > 1:
                duplicate_games.append(match_id)
                teams = {str(leg.team).strip().lower() for leg in grouped if str(leg.team).strip()}
                markets = {str(leg.market).strip().lower() for leg in grouped if str(leg.market).strip()}
                if len(teams) > 1 or len(markets) > 1:
                    conflicting_picks.append(match_id)

        sport_counts: Dict[str, int] = {}
        sport_market_counts: Dict[tuple[str, str], int] = {}
        for leg in legs:
            sport = str(leg.sport or "").lower()
            market = str(leg.market or "").lower()
            sport_counts[sport] = sport_counts.get(sport, 0) + 1
            sport_market_counts[(sport, market)] = sport_market_counts.get((sport, market), 0) + 1

        for sport, count in sport_counts.items():
            if count >= 3:
                correlated_pick_groups.append(f"{sport}: {count} legs")
        for (sport, market), count in sport_market_counts.items():
            if sport and market and count >= 2:
                correlated_pick_groups.append(f"{sport}/{market}: {count} legs")

        if duplicate_games:
            notes.append("Duplicate game detected in parlay candidate.")
        if conflicting_picks:
            notes.append("Conflicting or overlapping picks detected from the same game.")
        if correlated_pick_groups:
            notes.append("Correlation risk detected from clustered sport/market exposure.")

        n_legs = len(legs)
        weakest = min(legs, key=lambda leg: (leg.ml_prob, leg.edge, -leg.odds), default=None)
        min_prob = weakest.ml_prob if weakest is not None else 0.0
        avg_prob = float(np.mean([leg.ml_prob for leg in legs])) if legs else 0.0

        build_verdict = "BUILD"
        if n_legs < 2 or duplicate_games or conflicting_picks:
            build_verdict = "DO NOT BUILD"

        if build_verdict == "DO NOT BUILD":
            risk_tier = "do-not-build"
        elif n_legs > self.conservative_max_legs:
            risk_tier = "speculative" if n_legs >= 7 or min_prob < self.speculative_min_leg_prob else "high-risk"
        elif correlated_pick_groups:
            risk_tier = "high-risk" if len(correlated_pick_groups) >= 2 or min_prob < self.medium_risk_min_leg_prob else "medium-risk"
        elif (
            tier == "value"
            and self.conservative_min_legs <= n_legs <= self.conservative_max_legs
            and min_prob >= self.conservative_min_leg_prob
            and avg_prob >= self.conservative_min_average_prob
        ):
            risk_tier = "conservative"
        elif n_legs <= self.conservative_max_legs and min_prob >= self.medium_risk_min_leg_prob:
            risk_tier = "medium-risk"
        else:
            risk_tier = "high-risk"

        if n_legs > self.conservative_max_legs:
            notes.append("Parlays above 5 legs cannot be labelled conservative.")

        return {
            "risk_tier": risk_tier,
            "build_verdict": build_verdict,
            "duplicate_games": duplicate_games,
            "conflicting_picks": conflicting_picks,
            "correlated_pick_groups": correlated_pick_groups,
            "notes": notes,
            "weakest_leg": weakest,
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(results: Dict[str, Dict[str, List[Parlay]]], bankroll: float = 1000.0) -> str:
        """
        Render a human-readable parlay report.

        Parameters
        ----------
        results : dict
            Output of ParlayBuilder.build().
        bankroll : float
            Current bankroll in currency units (for stake calculation).

        Returns
        -------
        str
            Formatted markdown report section.
        """
        lines: List[str] = []

        tier_labels = {
            "value": "🎯 Value Parlays (all legs have edge ≥ 3%)",
            "speculative": "⚡ Longshot Parlays (higher-upside builds with at least one aggressive leg)",
        }

        for tier_key, tier_label in tier_labels.items():
            tier_data = results.get(tier_key, {})
            has_any = any(len(v) > 0 for v in tier_data.values())
            if not has_any:
                continue

            lines += [f"### {tier_label}", ""]

            for bracket in ("5x", "10x", "20x"):
                parlays = tier_data.get(bracket, [])
                if not parlays:
                    lines.append(f"**{bracket} target** — no valid parlays found  \n")
                    continue

                lines.append(f"**{bracket} target odds**")
                for i, parlay in enumerate(parlays, 1):
                    stake_amount = bankroll * parlay.kelly_stake_pct / 100
                    lines += [
                        f"",
                        f"**Parlay {i}** ({parlay.n_legs} legs | combined odds: {parlay.combined_odds:.2f})",
                        f"- Risk tier: {parlay.risk_tier}",
                        f"- Build verdict: {parlay.build_verdict}",
                        f"- Win probability: {parlay.combined_prob*100:.2f}%",
                        f"- Expected value: {parlay.ev:.3f}x (edge: {parlay.edge*100:+.1f}%)",
                        f"- Kelly stake: {parlay.kelly_stake_pct:.1f}% of bankroll"
                        + (f" = £{stake_amount:.2f}" if bankroll else ""),
                        f"- Weakest leg: {parlay.weakest_leg['team']} @ {parlay.weakest_leg['odds']}"
                        if parlay.weakest_leg else "- Weakest leg: n/a",
                        f"- Legs:",
                    ]
                    for note in parlay.validation_notes:
                        lines.append(f"- Note: {note}")
                    for leg in parlay.legs:
                        lines.append(
                            f"  - [{leg.sport.upper()}] **{leg.team}** "
                            f"vs {leg.match_id.replace(leg.team + ' vs ', '').replace(' vs ' + leg.team, '')} "
                            f"@ {leg.odds} "
                            f"(ML: {leg.ml_prob*100:.1f}%  edge: {leg.edge*100:+.1f}%)"
                        )
                lines.append("")

        return "\n".join(lines)
