"""
feature_store.py
================
Production-grade feature store for inference time.

Problem it solves
-----------------
During TRAINING we have the full game record — quarters, OT flag, etc.
During INFERENCE we only have the upcoming game and historical averages.
Instead of dropping features that are "missing" at inference (a leaky fix),
this store computes them from team history so training and inference use
the same feature set.

Features currently handled
--------------------------
Basketball:
  - home_half_ratio / away_half_ratio  (2nd-half pts / 1st-half pts, rolling avg)
  - went_to_ot                          (team's historical OT rate)
  - home_q4_avg / away_q4_avg           (avg Q4 pts from history)

Usage
-----
    from src.features.feature_store import FeatureStore
    store = FeatureStore(features_df, sport="basketball")
    extra = store.get_inference_features(home_team, away_team, as_home=True)
    # extra is a dict of feature_name → value ready to inject into the snapshot
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Path to the master team ID registry
_TEAM_IDS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "team_ids.json"

_CONNECTOR_WORDS = {
    "and", "club", "de", "del", "der", "di", "do", "fc", "cf", "sc", "ac", "bc",
    "sv", "kv", "rc", "afc", "clube", "calcio", "balompie", "ud", "cd",
}


def normalize_entity_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_entity_alias_map(
    canonical_names: Iterable[str],
    *,
    extra_aliases: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical in canonical_names:
        raw = str(canonical or "").strip()
        if not raw:
            continue
        normalized = normalize_entity_name(raw)
        if not normalized:
            continue
        alias_map[normalized] = raw

        tokens = [token for token in normalized.split() if token not in _CONNECTOR_WORDS]
        if tokens:
            alias_map[" ".join(tokens)] = raw
        if len(tokens) >= 2:
            alias_map[" ".join(tokens[:-1])] = raw
            alias_map[" ".join(tokens[1:])] = raw
        if len(tokens) == 2 and len(tokens[0]) > 2:
            alias_map[tokens[0]] = raw

    for alias, canonical in (extra_aliases or {}).items():
        normalized_alias = normalize_entity_name(alias)
        if normalized_alias:
            alias_map[normalized_alias] = canonical
    return alias_map


def resolve_canonical_name(
    name: str,
    canonical_names: Iterable[str],
    *,
    alias_map: Optional[dict[str, str]] = None,
) -> str:
    raw = str(name or "").strip()
    if not raw:
        return raw
    alias_map = alias_map or build_entity_alias_map(canonical_names)
    normalized = normalize_entity_name(raw)
    if normalized in alias_map:
        return alias_map[normalized]

    name_tokens = set(token for token in normalized.split() if token)
    candidates = []
    for canonical in canonical_names:
        canonical_norm = normalize_entity_name(canonical)
        canonical_tokens = set(token for token in canonical_norm.split() if token)
        overlap = len(name_tokens & canonical_tokens)
        if not overlap:
            continue
        if normalized in canonical_norm or canonical_norm in normalized:
            score = overlap / max(len(name_tokens), len(canonical_tokens), 1)
        else:
            score = overlap / max(len(name_tokens), len(canonical_tokens), 1)
        candidates.append((score, canonical))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_name = candidates[0]
        threshold = 0.5 if len(name_tokens) <= 2 else 0.6
        if best_score >= threshold:
            return best_name
    return raw


# ── Team entity resolver ──────────────────────────────────────────────────────

class TeamResolver:
    """
    Resolves any team name alias → canonical training name.

    Uses data/team_ids.json as the single source of truth.
    Adding a new alias never requires code changes — just edit the JSON.
    """

    def __init__(self, sport: str) -> None:
        self._sport = sport
        self._alias_map: Dict[str, str] = {}          # alias → team_id
        self._canonical_map: Dict[str, str] = {}      # team_id → canonical name
        self._normalized_alias_map: Dict[str, str] = {}
        self._normalized_canonical_map: Dict[str, str] = {}
        self._load()

    @staticmethod
    def _normalize(name: str) -> str:
        text = normalize_entity_name(name)
        text = re.sub(r"\b(fc|cf|sc|ac|bc|sv|kv|rc|afc|club|clube|calcio|balompie|de|del|der|di|do)\b", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _load(self) -> None:
        if not _TEAM_IDS_PATH.exists():
            logger.warning("team_ids.json not found at %s — entity resolution disabled", _TEAM_IDS_PATH)
            return
        try:
            data = json.loads(_TEAM_IDS_PATH.read_text())
            sport_data = data.get(self._sport, {})
            self._alias_map     = sport_data.get("_aliases", {})
            self._canonical_map = sport_data.get("_id_to_canonical", {})
            self._normalized_alias_map = {
                self._normalize(alias): team_id for alias, team_id in self._alias_map.items()
            }
            self._normalized_canonical_map = {
                self._normalize(canonical): canonical for canonical in self._canonical_map.values()
            }
            logger.debug("TeamResolver[%s]: loaded %d aliases, %d canonical names",
                         self._sport, len(self._alias_map), len(self._canonical_map))
        except Exception as exc:
            logger.warning("TeamResolver: failed to load team_ids.json — %s", exc)

    def resolve(self, name: str) -> str:
        """
        Return the canonical training-data name for any alias.
        If no mapping found, returns the original name unchanged.
        """
        team_id = self._alias_map.get(name)
        if team_id:
            canonical = self._canonical_map.get(team_id, name)
            if canonical != name:
                logger.debug("TeamResolver[%s]: '%s' → '%s' (via %s)",
                             self._sport, name, canonical, team_id)
            return canonical
        normalized = self._normalize(name)
        if normalized:
            team_id = self._normalized_alias_map.get(normalized)
            if team_id:
                return self._canonical_map.get(team_id, name)
            canonical = self._normalized_canonical_map.get(normalized)
            if canonical:
                return canonical
            candidates = []
            name_tokens = set(normalized.split())
            for norm_canonical, canonical_name in self._normalized_canonical_map.items():
                canonical_tokens = set(norm_canonical.split())
                overlap = len(name_tokens & canonical_tokens)
                if overlap and (normalized in norm_canonical or norm_canonical in normalized or overlap >= max(1, min(len(name_tokens), len(canonical_tokens)) - 1)):
                    score = overlap / max(len(name_tokens), len(canonical_tokens), 1)
                    candidates.append((score, canonical_name))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                best_score, best_name = candidates[0]
                if best_score >= 0.6:
                    return best_name
        return name

    def resolve_id(self, name: str) -> Optional[str]:
        """Return the unique team ID (e.g. 'NHL_CAR') for a given name."""
        return self._alias_map.get(name)


# ── Feature Store ─────────────────────────────────────────────────────────────

class FeatureStore:
    """
    Computes inference-time features from historical team data.

    Instead of dropping features that are unavailable at runtime
    (e.g. quarter scores), we estimate them from rolling team history.

    Parameters
    ----------
    features_df : pd.DataFrame
        The full training feature DataFrame (including raw columns).
    sport : str
        'basketball', 'nhl', 'mlb', 'soccer', 'tennis'
    window : int
        Rolling history window for averages (default 10 games).
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        sport: str,
        window: int = 10,
    ) -> None:
        self._df = features_df
        self._sport = sport
        self._window = window
        self._resolver = TeamResolver(sport if sport != "basketball" else "nba")
        self._cache: Dict[str, dict] = {}  # team → computed features

    def _team_history(self, team: str, as_home: bool) -> pd.DataFrame:
        """Return historical rows for a team (as home or away)."""
        col = "home_team" if as_home else "away_team"
        resolved = self._resolver.resolve(team)
        rows = self._df[self._df[col] == resolved]
        return rows.tail(self._window) if len(rows) > 0 else pd.DataFrame()

    # ── Basketball-specific features ──────────────────────────────────────────

    def _basketball_half_ratio(self, team: str, as_home: bool) -> float:
        """
        Estimate home_half_ratio / away_half_ratio from historical quarter data.
        half_ratio = avg(2nd_half_pts) / avg(1st_half_pts).
        Falls back to 1.0 (neutral) if quarter data unavailable.
        """
        prefix = "home" if as_home else "away"
        hist = self._team_history(team, as_home)
        if hist.empty:
            return 1.0

        q1_col = f"{prefix}_q1"
        q2_col = f"{prefix}_q2"
        q3_col = f"{prefix}_q3"
        q4_col = f"{prefix}_q4"

        if all(c in hist.columns for c in [q1_col, q2_col, q3_col, q4_col]):
            first_half  = (hist[q1_col].fillna(0) + hist[q2_col].fillna(0))
            second_half = (hist[q3_col].fillna(0) + hist[q4_col].fillna(0))
            # Only use rows where we actually have quarter data
            valid = first_half > 0
            if valid.sum() >= 3:
                ratio = (second_half[valid] / first_half[valid].replace(0, np.nan)).mean()
                return float(np.clip(ratio, 0.5, 2.0)) if not np.isnan(ratio) else 1.0

        # Fallback: use scoring margin as proxy for 2nd-half surge
        # Teams that tend to win (positive margin) are often strong in 4th quarter
        margin_col = f"{prefix}_scoring_margin"
        if margin_col in hist.columns:
            avg_margin = hist[margin_col].fillna(0).mean()
            # Rough heuristic: 1.0 + 0.005 per point of margin (bounded 0.85–1.15)
            return float(np.clip(1.0 + avg_margin * 0.005, 0.85, 1.15))

        return 1.0

    def _basketball_went_to_ot(self, team: str, as_home: bool) -> float:
        """
        Estimate probability this game goes to OT based on team's historical OT rate.
        Returns float (0–1) — historical fraction of games that went to OT.
        """
        prefix = "home" if as_home else "away"
        hist = self._team_history(team, as_home)
        if hist.empty:
            return 0.08  # league average ~8% of games go to OT

        ot_col = f"{prefix}_ot"
        if ot_col in hist.columns:
            went = (hist[ot_col].fillna(0) > 0).mean()
            return float(went) if not np.isnan(went) else 0.08

        # Fallback: use margin std — teams with tight average margins go to OT more
        margin_col = f"{prefix}_margin_std"
        if margin_col in hist.columns:
            std = hist[margin_col].fillna(10).mean()
            # tighter games (std < 8) → ~12% OT rate; blowout teams (std > 15) → ~5%
            return float(np.clip(0.08 + (8 - std) * 0.005, 0.04, 0.15))

        return 0.08

    def _basketball_q4_avg(self, team: str, as_home: bool) -> float:
        """Rolling average Q4 points from history."""
        prefix = "home" if as_home else "away"
        hist = self._team_history(team, as_home)
        q4_col = f"{prefix}_q4"
        if not hist.empty and q4_col in hist.columns:
            valid = hist[q4_col].dropna()
            if len(valid) >= 3:
                return float(valid.mean())
        # League average Q4 ≈ 27 pts
        return 27.0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_basketball_extras(self, home_team: str, away_team: str) -> dict:
        """
        Return inference-time estimates for basketball features that are derived
        from raw quarter data (unavailable at bet time).

        Returns
        -------
        dict with keys matching the training feature names:
            home_half_ratio, away_half_ratio, went_to_ot,
            home_q4_avg, away_q4_avg
        """
        return {
            "home_half_ratio": self._basketball_half_ratio(home_team, as_home=True),
            "away_half_ratio": self._basketball_half_ratio(away_team, as_home=False),
            "went_to_ot":      (self._basketball_went_to_ot(home_team, as_home=True)
                                + self._basketball_went_to_ot(away_team, as_home=False)) / 2,
            "home_q4_avg":     self._basketball_q4_avg(home_team, as_home=True),
            "away_q4_avg":     self._basketball_q4_avg(away_team, as_home=False),
        }

    def resolve_team(self, name: str) -> str:
        """Resolve any API name or alias to the canonical training name."""
        return self._resolver.resolve(name)

    def resolve_team_id(self, name: str) -> Optional[str]:
        """Return the unique team ID string (e.g. 'NHL_CAR')."""
        return self._resolver.resolve_id(name)
