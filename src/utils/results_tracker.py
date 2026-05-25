"""
results_tracker.py
==================
Records predictions made by daily_scan.py and, once results are known,
tracks P&L, ROI, and closing-line value (CLV).

Storage layout (all in data/tracker/):
    predictions.parquet   — one row per prediction placed
    settled.parquet       — one row per settled bet (joined with actual result)
    summary.parquet       — rolling daily P&L summary

Prediction schema
-----------------
pred_id         str     UUID for the prediction
recorded_at     datetime  when daily_scan.py ran
commence_time   datetime  match start time
sport           str     soccer | basketball | tennis
match_id        str     unique match identifier
team_or_player  str     side we're predicting for
market          str     market family at time of publication (moneyline, spreads, totals, etc.)
market_status   str     policy status at time of publication (preferred, experimental, disabled)
tier            str     user-facing market tier label at time of publication (Preferred, Limited, Experimental)
ml_prob         float   model probability (post-calibration)
fair_prob       float   fair probability after vig removal
bet_odds        float   decimal odds at time of prediction
bookmaker       str     name of bookmaker used
edge            float   (fair_prob × bet_odds − 1)
kelly_stake_pct float   Kelly fraction (fraction of bankroll)
stake_units     float   actual units staked (bankroll × kelly × fraction)
is_parlay_leg   bool    was this used inside a parlay?
parlay_id       str     parlay identifier if applicable
status          str     pending | won | lost | void | push

Settled schema (adds)
---------------------
settled_at      datetime
actual_result   str     e.g. "home_win", "player1_win", etc.
won             bool
profit_units    float   net profit/loss in stake units
closing_odds    float   odds at close (for CLV)
clv             float   closing line value = bet_odds/closing_odds − 1
"""

from __future__ import annotations

import logging
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from config import settings

logger = logging.getLogger(__name__)

_POST_RESULT_CFG = (((settings or {}).get("betting") or {}).get("post_result") or {})

_TRACKER_DIR  = Path("data/tracker")
_PRED_FILE    = _TRACKER_DIR / "predictions.parquet"
_SETTLED_FILE = _TRACKER_DIR / "settled.parquet"
_SUMMARY_FILE = _TRACKER_DIR / "summary.parquet"
_PARLAY_FILE  = _TRACKER_DIR / "parlays.parquet"

_PARLAY_SCHEMA: Dict[str, str] = {
    "parlay_id":       "string",
    "recorded_at":     "datetime64[ns, UTC]",
    "tier":            "string",
    "bracket":         "string",
    "n_legs":          "int64",
    "combined_odds":   "float64",
    "combined_prob":   "float64",
    "ev":              "float64",
    "edge":            "float64",
    "kelly_stake_pct": "float64",
    "stake_units":     "float64",
    "legs_json":       "string",
    "risk_tier":       "string",
    "build_verdict":   "string",
    "weakest_leg_json":"string",
    "version_snapshot":"string",
    "status":          "string",
    "settled_at":      "datetime64[ns, UTC]",
    "won":             "bool",
    "profit_units":    "float64",
    "mistake_classification": "string",
}

_PRED_SCHEMA: Dict[str, str] = {
    "pred_id":        "string",
    "recorded_at":    "datetime64[ns, UTC]",
    "commence_time":  "datetime64[ns, UTC]",
    "sport":          "string",
    "match_id":       "string",
    "team_or_player": "string",
    "market":         "string",
    "market_status":  "string",
    "tier":           "string",
    "ml_prob":        "float64",
    "fair_prob":      "float64",
    "bet_odds":       "float64",
    "bookmaker":      "string",
    "edge":           "float64",
    "kelly_stake_pct":"float64",
    "stake_units":    "float64",
    "is_parlay_leg":  "bool",
    "parlay_id":      "string",
    "decision_status":"string",
    "decision_reason":"string",
    "lower_bound_passed":"bool",
    "minimum_acceptable_odds":"float64",
    "freshness_check":"string",
    "odds_freshness":"string",
    "lineup_freshness":"string",
    "injury_news_freshness":"string",
    "standings_freshness":"string",
    "fixture_verified":"bool",
    "market_suitable":"bool",
    "recommended_market":"string",
    "context_factor_names":"string",
    "probability_debug_json":"string",
    "probability_context_applied":"bool",
    "probability_context_reasons":"string",
    "structural_available":"bool",
    "structural_weight":"float64",
    "version_snapshot":"string",
    "status":         "string",
}

_MISTAKE_CATEGORIES = [
    "normal variance",
    "overconfidence error",
    "motivation error",
    "rotation error",
    "lineup/injury error",
    "stale-data error",
    "market-selection error",
    "wrong-conversion error",
    "favourite-trap error",
    "underdog-resistance error",
    "parlay-construction error",
    "odds/value error",
]


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    _TRACKER_DIR.mkdir(parents=True, exist_ok=True)


def _load(path: Path, schema: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    if schema:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
    return pd.DataFrame()


def _save(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir()
    # Coerce any Timestamp columns that ended up with mixed object dtype to str
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].apply(
                    lambda v: v.isoformat() if hasattr(v, "isoformat") else v
                )
            except Exception:
                pass
    df.to_parquet(path, index=False)


def _json_dumps_safe(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _probability_diagnostics_from_prediction(p: Dict) -> Dict:
    debug_payload = p.get("probability_debug") or p.get("probability_debug_json") or {}
    if isinstance(debug_payload, str):
        try:
            debug_payload = json.loads(debug_payload) if debug_payload.strip() else {}
        except Exception:
            debug_payload = {}
    if not isinstance(debug_payload, dict):
        debug_payload = {}

    context_debug = debug_payload.get("context_probability_adjustment") or {}
    if not isinstance(context_debug, dict):
        context_debug = {}

    reasons = context_debug.get("reasons") or p.get("probability_context_reasons") or []
    if isinstance(reasons, str):
        try:
            parsed = json.loads(reasons)
            reasons = parsed if isinstance(parsed, list) else [reasons]
        except Exception:
            reasons = [reasons] if reasons.strip() else []
    reasons = [str(item) for item in (reasons or []) if str(item).strip()]

    structural_weight = p.get("structural_weight")
    if structural_weight is None:
        structural_weight = debug_payload.get("structural_weight")
    try:
        structural_weight_value = float(structural_weight) if structural_weight is not None else 0.0
    except Exception:
        structural_weight_value = 0.0

    context_applied = p.get("probability_context_applied")
    if context_applied is None:
        context_applied = context_debug.get("applied", False)

    structural_available = p.get("structural_available")
    if structural_available is None:
        structural_available = debug_payload.get("structural_available")

    return {
        "probability_debug_json": _json_dumps_safe(debug_payload),
        "probability_context_applied": bool(context_applied),
        "probability_context_reasons": _json_dumps_safe(reasons),
        "structural_available": bool(structural_available) if structural_available is not None else False,
        "structural_weight": round(structural_weight_value, 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def record_prediction(
    *,
    sport: str,
    match_id: str,
    team_or_player: str,
    commence_time: datetime,
    market: str = "moneyline",
    market_status: str = "experimental",
    tier: str = "Experimental",
    ml_prob: float,
    fair_prob: float,
    bet_odds: float,
    bookmaker: str = "unknown",
    edge: float = 0.0,
    kelly_stake_pct: float = 0.0,
    stake_units: float = 0.0,
    is_parlay_leg: bool = False,
    parlay_id: str = "",
    decision_status: str = "",
    decision_reason: str = "",
    lower_bound_passed: Optional[bool] = None,
    minimum_acceptable_odds: Optional[float] = None,
    freshness_check: str = "",
    odds_freshness: str = "",
    lineup_freshness: str = "",
    injury_news_freshness: str = "",
    standings_freshness: str = "",
    fixture_verified: Optional[bool] = None,
    market_suitable: Optional[bool] = None,
    recommended_market: str = "",
    context_factor_names: str = "",
    probability_debug: Optional[Dict] = None,
    probability_debug_json: str = "",
    probability_context_applied: Optional[bool] = None,
    probability_context_reasons: str = "",
    structural_available: Optional[bool] = None,
    structural_weight: Optional[float] = None,
    version_snapshot: str = "",
) -> str:
    """
    Persist a single prediction.  Returns the assigned pred_id.
    """
    pred_id = str(uuid.uuid4())
    now     = datetime.now(tz=timezone.utc)

    probability_diagnostics = _probability_diagnostics_from_prediction({
        "probability_debug": probability_debug,
        "probability_debug_json": probability_debug_json,
        "probability_context_applied": probability_context_applied,
        "probability_context_reasons": probability_context_reasons,
        "structural_available": structural_available,
        "structural_weight": structural_weight,
    })

    row = {
        "pred_id":        pred_id,
        "recorded_at":    now,
        "commence_time":  pd.Timestamp(commence_time).tz_localize("UTC")
                          if commence_time.tzinfo is None
                          else pd.Timestamp(commence_time).tz_convert("UTC"),
        "sport":          sport,
        "match_id":       match_id,
        "team_or_player": team_or_player,
        "market":         market,
        "market_status":  market_status,
        "tier":           tier,
        "ml_prob":        round(float(ml_prob), 6),
        "fair_prob":      round(float(fair_prob), 6),
        "bet_odds":       round(float(bet_odds), 4),
        "bookmaker":      bookmaker,
        "edge":           round(float(edge), 6),
        "kelly_stake_pct":round(float(kelly_stake_pct), 6),
        "stake_units":    round(float(stake_units), 4),
        "is_parlay_leg":  bool(is_parlay_leg),
        "parlay_id":      parlay_id or "",
        "decision_status": str(decision_status or ""),
        "decision_reason": str(decision_reason or ""),
        "lower_bound_passed": bool(lower_bound_passed) if lower_bound_passed is not None else False,
        "minimum_acceptable_odds": round(float(minimum_acceptable_odds), 4) if minimum_acceptable_odds else None,
        "freshness_check": str(freshness_check or ""),
        "odds_freshness": str(odds_freshness or ""),
        "lineup_freshness": str(lineup_freshness or ""),
        "injury_news_freshness": str(injury_news_freshness or ""),
        "standings_freshness": str(standings_freshness or ""),
        "fixture_verified": bool(fixture_verified) if fixture_verified is not None else True,
        "market_suitable": bool(market_suitable) if market_suitable is not None else True,
        "recommended_market": str(recommended_market or ""),
        "context_factor_names": str(context_factor_names or ""),
        **probability_diagnostics,
        "version_snapshot": str(version_snapshot or ""),
        "status":         "pending",
    }

    df = _load(_PRED_FILE, _PRED_SCHEMA)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df, _PRED_FILE)
    logger.info(
        "Recorded prediction %s: %s %s @ %.2f (edge %.1f%%)",
        pred_id[:8], sport, team_or_player, bet_odds, edge * 100,
    )
    return pred_id


def record_predictions_bulk(predictions: List[Dict]) -> List[str]:
    """Record multiple predictions in one write. Returns list of pred_ids.

    Deduplication: a prediction is skipped if an entry with the same
    (match_id, team_or_player, commence_time date) already exists in any
    status — prevents double-recording when the scan runs more than once
    per day on the same games.
    """
    now  = datetime.now(tz=timezone.utc)
    df   = _load(_PRED_FILE, _PRED_SCHEMA)

    # Build a set of already-recorded keys: (match_id, team_or_player, date)
    existing_keys: set = set()
    if not df.empty and "commence_time" in df.columns:
        ct_col = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
        for _, row in df.iterrows():
            key = (
                str(row["match_id"]).lower(),
                str(row["team_or_player"]).lower(),
                ct_col[row.name].date(),
            )
            existing_keys.add(key)

    rows = []
    ids  = []
    skipped = 0
    for p in predictions:
        ct  = p.get("commence_time", now)
        ct_ts = pd.Timestamp(ct).tz_localize("UTC") if pd.Timestamp(ct).tzinfo is None \
                else pd.Timestamp(ct).tz_convert("UTC")

        key = (
            str(p.get("match_id", "")).lower(),
            str(p.get("team_or_player", "")).lower(),
            ct_ts.date(),
        )
        if key in existing_keys:
            skipped += 1
            continue

        pid = str(uuid.uuid4())
        ids.append(pid)
        existing_keys.add(key)   # prevent dupes within this same batch
        probability_diagnostics = _probability_diagnostics_from_prediction(p)
        rows.append({
            "pred_id":        pid,
            "recorded_at":    now,
            "commence_time":  ct_ts,
            "sport":          p.get("sport", ""),
            "match_id":       p.get("match_id", ""),
            "team_or_player": p.get("team_or_player", ""),
            "market":         p.get("market", "moneyline"),
            "market_status":  p.get("market_status", "experimental"),
            "tier":           p.get("tier", "Experimental"),
            "ml_prob":        round(float(p.get("ml_prob", 0)), 6),
            "fair_prob":      round(float(p.get("fair_prob", 0)), 6),
            "bet_odds":       round(float(p.get("bet_odds", 0)), 4),
            "bookmaker":      p.get("bookmaker", "unknown"),
            "edge":           round(float(p.get("edge", 0)), 6),
            "kelly_stake_pct":round(float(p.get("kelly_stake_pct", 0)), 6),
            "stake_units":    round(float(p.get("stake_units", 0)), 4),
            "is_parlay_leg":  bool(p.get("is_parlay_leg", False)),
            "parlay_id":      p.get("parlay_id", "") or "",
            "decision_status": str(p.get("decision_status", "") or ""),
            "decision_reason": str(p.get("decision_reason", "") or ""),
            "lower_bound_passed": bool(p.get("lower_bound_passed", False)),
            "minimum_acceptable_odds": round(float(p.get("minimum_acceptable_odds", 0) or 0), 4) if p.get("minimum_acceptable_odds") is not None else None,
            "freshness_check": str(p.get("freshness_check", "") or ""),
            "odds_freshness": str(p.get("odds_freshness", "") or ""),
            "lineup_freshness": str(p.get("lineup_freshness", "") or ""),
            "injury_news_freshness": str(p.get("injury_news_freshness", "") or ""),
            "standings_freshness": str(p.get("standings_freshness", "") or ""),
            "fixture_verified": bool(p.get("fixture_verified", True)),
            "market_suitable": bool(p.get("market_suitable", True)),
            "recommended_market": str(p.get("recommended_market", "") or ""),
            "context_factor_names": str(p.get("context_factor_names", "") or ""),
            **probability_diagnostics,
            "version_snapshot": str(p.get("version_snapshot", "") or ""),
            "status":         "pending",
        })

    if skipped:
        logger.info("Dedup: skipped %d already-recorded prediction(s)", skipped)

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        _save(df, _PRED_FILE)
        logger.info("Recorded %d new prediction(s) to tracker", len(rows))
    else:
        logger.info("No new predictions to record (all duplicates)")

    return ids


def settle_prediction(
    pred_id: str,
    actual_result: str,
    *,
    closing_odds: Optional[float] = None,
    void: bool = False,
    won: Optional[bool] = None,
) -> Optional[Dict]:
    """
    Settle a prediction given the actual result.

    Parameters
    ----------
    pred_id : str
    actual_result : str
        The outcome that happened, e.g. "home_win", "player1_win".
        Should match the team_or_player field in the prediction.
    closing_odds : float, optional
        Market odds at match close (for CLV calculation).
    void : bool
        Mark as void (abandoned match, etc.) without P&L impact.

    Returns
    -------
    dict with settlement details, or None if pred_id not found.
    """
    df = _load(_PRED_FILE, _PRED_SCHEMA)
    mask = df["pred_id"] == pred_id
    if not mask.any():
        logger.warning("settle_prediction: pred_id %s not found", pred_id)
        return None

    row  = df[mask].iloc[0].to_dict()
    if void:
        df.loc[mask, "status"] = "void"
        _save(df, _PRED_FILE)
        settled_row = {
            "pred_id": pred_id,
            "settled_at": datetime.now(tz=timezone.utc),
            "sport": row["sport"],
            "match_id": row["match_id"],
            "team_or_player": row["team_or_player"],
            "commence_time": row["commence_time"],
            "recorded_at": row["recorded_at"],
            "market": row.get("market", "moneyline"),
            "market_status": row.get("market_status", "experimental"),
            "tier": row.get("tier", "Experimental"),
            "bet_odds": row["bet_odds"],
            "bookmaker": row.get("bookmaker", "unknown"),
            "edge": row["edge"],
            "ml_prob": row["ml_prob"],
            "fair_prob": row["fair_prob"],
            "stake_units": row["stake_units"],
            "kelly_stake_pct": row["kelly_stake_pct"],
            "is_parlay_leg": row["is_parlay_leg"],
            "version_snapshot": row.get("version_snapshot", ""),
            "actual_result": actual_result,
            "won": False,
            "profit_units": 0.0,
            "closing_odds": closing_odds,
            "closing_line_value": None,
            "clv": None,
            "decision_status": row.get("decision_status", ""),
            "decision_reason": row.get("decision_reason", ""),
            "lower_bound_passed": row.get("lower_bound_passed", False),
            "minimum_acceptable_odds": row.get("minimum_acceptable_odds"),
            "freshness_check": row.get("freshness_check", ""),
            "odds_freshness": row.get("odds_freshness", ""),
            "lineup_freshness": row.get("lineup_freshness", ""),
            "injury_news_freshness": row.get("injury_news_freshness", ""),
            "standings_freshness": row.get("standings_freshness", ""),
            "fixture_verified": row.get("fixture_verified", True),
            "market_suitable": row.get("market_suitable", True),
            "recommended_market": row.get("recommended_market", ""),
            "context_factor_names": row.get("context_factor_names", ""),
            "probability_debug_json": row.get("probability_debug_json", ""),
            "probability_context_applied": row.get("probability_context_applied", False),
            "probability_context_reasons": row.get("probability_context_reasons", ""),
            "structural_available": row.get("structural_available", False),
            "structural_weight": row.get("structural_weight", 0.0),
            "mistake_classification": "void",
            "status": "void",
        }
        settled_df = _load(_SETTLED_FILE)
        settled_df = pd.concat([settled_df, pd.DataFrame([settled_row])], ignore_index=True)
        _save(settled_df, _SETTLED_FILE)
        return {"pred_id": pred_id, "status": "void"}

    # Determine win/loss — caller may pass won=True/False directly to bypass
    # the fuzzy label matching (used when settle.py has match context).
    if won is None:
        won = _is_win(row["team_or_player"], actual_result)
    profit = (row["bet_odds"] - 1) * row["stake_units"] if won else -row["stake_units"]
    status = "won" if won else "lost"

    df.loc[mask, "status"] = status
    _save(df, _PRED_FILE)

    # CLV
    clv = None
    if closing_odds and row["bet_odds"]:
        clv = row["bet_odds"] / closing_odds - 1.0

    settled_row = {
        "pred_id":       pred_id,
        "settled_at":    datetime.now(tz=timezone.utc),
        "sport":         row["sport"],
        "match_id":      row["match_id"],
        "team_or_player":row["team_or_player"],
        "commence_time": row["commence_time"],
        "recorded_at":   row["recorded_at"],
        "market":        row.get("market", "moneyline"),
        "market_status": row.get("market_status", "experimental"),
        "tier":          row.get("tier", "Experimental"),
        "bet_odds":      row["bet_odds"],
        "bookmaker":     row.get("bookmaker", "unknown"),
        "edge":          row["edge"],
        "ml_prob":       row["ml_prob"],
        "fair_prob":     row["fair_prob"],
        "stake_units":   row["stake_units"],
        "kelly_stake_pct": row["kelly_stake_pct"],
        "is_parlay_leg": row["is_parlay_leg"],
        "version_snapshot": row.get("version_snapshot", ""),
        "actual_result": actual_result,
        "won":           won,
        "profit_units":  round(profit, 4),
        "closing_odds":  closing_odds,
        "closing_line_value": round(clv, 6) if clv is not None else None,
        "clv":           round(clv, 6) if clv is not None else None,
        "decision_status": row.get("decision_status", ""),
        "decision_reason": row.get("decision_reason", ""),
        "lower_bound_passed": row.get("lower_bound_passed", False),
        "minimum_acceptable_odds": row.get("minimum_acceptable_odds"),
        "freshness_check": row.get("freshness_check", ""),
        "odds_freshness": row.get("odds_freshness", ""),
        "lineup_freshness": row.get("lineup_freshness", ""),
        "injury_news_freshness": row.get("injury_news_freshness", ""),
        "standings_freshness": row.get("standings_freshness", ""),
        "fixture_verified": row.get("fixture_verified", True),
        "market_suitable": row.get("market_suitable", True),
        "recommended_market": row.get("recommended_market", ""),
        "context_factor_names": row.get("context_factor_names", ""),
        "probability_debug_json": row.get("probability_debug_json", ""),
        "probability_context_applied": row.get("probability_context_applied", False),
        "probability_context_reasons": row.get("probability_context_reasons", ""),
        "structural_available": row.get("structural_available", False),
        "structural_weight": row.get("structural_weight", 0.0),
        "status":        status,
    }
    settled_row["mistake_classification"] = _classify_mistake(settled_row)

    settled_df = _load(_SETTLED_FILE)
    settled_df = pd.concat([settled_df, pd.DataFrame([settled_row])], ignore_index=True)
    _save(settled_df, _SETTLED_FILE)

    logger.info(
        "Settled %s: %s %s — %s  P&L=%.3f units  CLV=%s",
        pred_id[:8], row["sport"], row["team_or_player"],
        status.upper(),
        profit,
        f"{clv*100:+.2f}%" if clv is not None else "n/a",
    )
    return settled_row


def _is_win(team_or_player: str, actual_result: str) -> bool:
    """Return True if the prediction side matches the actual result.

    Handles both result-label predictions (home_win / away_win / draw) and
    actual team-name predictions (e.g. 'Arsenal', 'LA Lakers'), as well as
    double-chance ('Arsenal or Draw') and DNB markets.
    """
    t = team_or_player.strip().lower()
    a = actual_result.strip().lower()

    if t == a:
        return True

    # Result-label aliases (backward compat)
    aliases = {
        "home_win":    ["home", "home_win", "1"],
        "away_win":    ["away", "away_win", "2"],
        "draw":        ["draw", "x", "tie"],
        "player1_win": ["player1", "player1_win", "p1"],
        "player2_win": ["player2", "player2_win", "p2"],
    }
    for canonical, alts in aliases.items():
        if t in alts and a in alts:
            return True
        if t == canonical and a in alts:
            return True
        if a == canonical and t in alts:
            return True

    # Draw
    if a in ("draw", "x", "tie"):
        if " or draw" in t:
            return True
        return False

    # Double-chance "Team or Draw": wins when team wins OR draw
    if " or draw" in t:
        base = t.replace(" or draw", "").strip()
        return len(base) >= 3 and (base in a or a in base)

    # Draw-no-bet "Team DNB": wins when team wins (draw handled as void externally)
    if " dnb" in t:
        base = t.replace(" dnb", "").strip()
        return len(base) >= 3 and (base in a or a in base)

    # Fuzzy team-name substring match (handles 'Arsenal' vs 'Arsenal FC' etc.)
    if len(t) >= 4 and len(a) >= 4:
        return t in a or a in t

    return False


def get_pending() -> pd.DataFrame:
    """Return all predictions with status == 'pending'."""
    df = _load(_PRED_FILE, _PRED_SCHEMA)
    return df[df["status"] == "pending"].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Parlay tracking
# ──────────────────────────────────────────────────────────────────────────────

def record_parlay(parlay, *, bankroll: float = 1000.0, version_snapshot: str = "") -> str:
    """
    Persist a parlay record.  Returns the assigned parlay_id.

    Parameters
    ----------
    parlay : Parlay
        Object from src.risk.parlay_builder.
    bankroll : float
        Current bankroll (for stake-units calculation).
    """
    import json as _json

    parlay_id = str(uuid.uuid4())
    now       = datetime.now(tz=timezone.utc)

    legs_data = [
        {
            "sport":     leg.sport,
            "match_id":  leg.match_id,
            "team":      leg.team,
            "odds":      leg.odds,
            "ml_prob":   leg.ml_prob,
            "fair_prob": leg.fair_prob,
            "edge":      leg.edge,
            "commence":  leg.commence,
            "market":    getattr(leg, "market", ""),
        }
        for leg in parlay.legs
    ]

    # Dedup: skip if identical leg combination already recorded today
    df = _load(_PARLAY_FILE, _PARLAY_SCHEMA)
    existing_key = frozenset((l["match_id"], l["team"]) for l in legs_data)
    if not df.empty:
        today = now.date()
        for _, r in df.iterrows():
            try:
                r_date = pd.Timestamp(r["recorded_at"]).date()
                if r_date != today:
                    continue
                r_legs = _json.loads(r["legs_json"])
                if frozenset((l["match_id"], l["team"]) for l in r_legs) == existing_key:
                    logger.debug("Parlay already recorded today — skipping duplicate")
                    return r["parlay_id"]
            except Exception:
                pass

    row = {
        "parlay_id":       parlay_id,
        "recorded_at":     now,
        "tier":            parlay.tier,
        "bracket":         parlay.target_bracket,
        "n_legs":          parlay.n_legs,
        "combined_odds":   round(parlay.combined_odds, 4),
        "combined_prob":   round(parlay.combined_prob, 6),
        "ev":              round(parlay.ev, 6),
        "edge":            round(parlay.edge, 6),
        "kelly_stake_pct": round(parlay.kelly_stake_pct, 4),
        "stake_units":     round(parlay.kelly_stake_pct / 100, 6),
        "legs_json":       _json.dumps(legs_data),
        "risk_tier":       str(getattr(parlay, "risk_tier", "") or ""),
        "build_verdict":   str(getattr(parlay, "build_verdict", "") or ""),
        "weakest_leg_json": _json.dumps(getattr(parlay, "weakest_leg", None)),
        "version_snapshot": str(version_snapshot or ""),
        "status":          "pending",
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df, _PARLAY_FILE)
    logger.info(
        "Recorded parlay %s: %s/%s legs=%d odds=%.2f EV=%.3f Kelly=%.1f%%",
        parlay_id[:8], parlay.tier, parlay.target_bracket,
        parlay.n_legs, parlay.combined_odds, parlay.ev, parlay.kelly_stake_pct,
    )
    return parlay_id


def get_pending_parlays() -> pd.DataFrame:
    """Return all parlays with status == 'pending'."""
    df = _load(_PARLAY_FILE, _PARLAY_SCHEMA)
    if df.empty:
        return df
    return df[df["status"] == "pending"].copy()


def settle_parlay(parlay_id: str, won: bool, *, void: bool = False) -> Optional[Dict]:
    """Settle a parlay given the combined outcome."""
    df = _load(_PARLAY_FILE, _PARLAY_SCHEMA)
    mask = df["parlay_id"] == parlay_id
    if not mask.any():
        logger.warning("settle_parlay: parlay_id %s not found", parlay_id)
        return None

    row = df[mask].iloc[0].to_dict()

    if void:
        df.loc[mask, "status"] = "void"
        df.loc[mask, "settled_at"] = datetime.now(tz=timezone.utc)
        df.loc[mask, "won"] = False
        df.loc[mask, "profit_units"] = 0.0
        df.loc[mask, "mistake_classification"] = "void"
        _save(df, _PARLAY_FILE)
        return {"parlay_id": parlay_id, "status": "void", "won": False, "profit_units": 0.0}

    status = "won" if won else "lost"
    profit = (row["combined_odds"] - 1) * row["stake_units"] if won else -row["stake_units"]
    df.loc[mask, "status"] = status
    df.loc[mask, "settled_at"] = datetime.now(tz=timezone.utc)
    df.loc[mask, "won"] = bool(won)
    df.loc[mask, "profit_units"] = round(profit, 4)
    df.loc[mask, "mistake_classification"] = "won" if won else "parlay-construction error"
    _save(df, _PARLAY_FILE)

    logger.info(
        "Settled parlay %s: %s  P&L=%.3f units",
        parlay_id[:8], status.upper(), profit,
    )
    return {"parlay_id": parlay_id, "status": status, "won": won, "profit_units": round(profit, 4)}


def parlay_breakdown() -> pd.DataFrame:
    """Return all settled parlays with P&L."""
    df = _load(_PARLAY_FILE, _PARLAY_SCHEMA)
    if df.empty:
        return pd.DataFrame()
    resolved = df[df["status"].isin(["won", "lost"])].copy()
    if resolved.empty:
        return pd.DataFrame()
    resolved["won"] = resolved["status"] == "won"
    resolved["profit_units"] = resolved.apply(
        lambda r: (r["combined_odds"] - 1) * r["stake_units"] if r["won"] else -r["stake_units"],
        axis=1,
    )
    return resolved


def get_settled() -> pd.DataFrame:
    """Return all settled bets."""
    df = _load(_SETTLED_FILE)
    if df.empty:
        return df
    return _augment_mistake_fields(df)


def _contains_any(text: str, keywords: list[str]) -> bool:
    lower = str(text or "").lower()
    return any(keyword in lower for keyword in keywords)


def _classify_mistake(row: Dict) -> str:
    status = str(row.get("status", "") or "").lower()
    if status == "won":
        return "won"
    if status == "void":
        return "void"
    if status != "lost":
        return ""

    context_names = str(row.get("context_factor_names", "") or "").lower()
    decision_reason = str(row.get("decision_reason", "") or "").lower()
    market = str(row.get("market", "") or "").lower()
    recommended_market = str(row.get("recommended_market", "") or "").lower()
    ml_prob = float(row.get("ml_prob", 0) or 0)
    edge = float(row.get("edge", 0) or 0)
    bet_odds = float(row.get("bet_odds", 0) or 0)
    clv = row.get("clv")
    if clv is not None:
        try:
            clv = float(clv)
        except Exception:
            clv = None

    freshness_values = " ".join(
        str(row.get(key, "") or "")
        for key in (
            "freshness_check",
            "odds_freshness",
            "lineup_freshness",
            "injury_news_freshness",
            "standings_freshness",
        )
    ).lower()
    if _contains_any(freshness_values, ["stale", "missing", "failed"]):
        if "lineup" in freshness_values or "injury" in freshness_values or "news" in freshness_values:
            return "lineup/injury error"
        return "stale-data error"

    if not bool(row.get("fixture_verified", True)):
        return "stale-data error"

    if not bool(row.get("market_suitable", True)) or (recommended_market and recommended_market not in {"", market}):
        if market in {"moneyline", "spreads", "draw_no_bet", "double_chance"} and recommended_market in {"double_chance", "draw_no_bet", "spreads", "no_bet"}:
            return "wrong-conversion error"
        return "market-selection error"

    if _contains_any(context_names + " " + decision_reason, ["rotation", "cup_rotation", "european_rotation"]):
        return "rotation error"

    if _contains_any(context_names + " " + decision_reason, ["motivation", "playoff", "relegation", "title_", "nothing_to_play_for", "final_day"]):
        return "motivation error"

    if _contains_any(context_names + " " + decision_reason, ["lineup", "injury", "goalie", "starter", "availability"]):
        return "lineup/injury error"

    negative_clv_error_threshold = float(_POST_RESULT_CFG.get("negative_clv_error_threshold", -0.03) or -0.03)
    overconfidence_prob_threshold = float(_POST_RESULT_CFG.get("overconfidence_prob_threshold", 0.65) or 0.65)
    overconfidence_edge_threshold = float(_POST_RESULT_CFG.get("overconfidence_edge_threshold", 0.12) or 0.12)
    favourite_trap_max_odds = float(_POST_RESULT_CFG.get("favourite_trap_max_odds", 1.75) or 1.75)
    underdog_resistance_min_odds = float(_POST_RESULT_CFG.get("underdog_resistance_min_odds", 2.35) or 2.35)

    if clv is not None and clv <= negative_clv_error_threshold:
        return "odds/value error"

    if ml_prob >= overconfidence_prob_threshold or edge >= overconfidence_edge_threshold or not bool(row.get("lower_bound_passed", True)):
        return "overconfidence error"

    if bet_odds and bet_odds <= favourite_trap_max_odds:
        return "favourite-trap error"

    if bet_odds and bet_odds >= underdog_resistance_min_odds:
        return "underdog-resistance error"

    return "normal variance"


def _augment_mistake_fields(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "mistake_classification" not in enriched.columns:
        enriched["mistake_classification"] = ""
    for col in [
        "decision_status",
        "decision_reason",
        "freshness_check",
        "odds_freshness",
        "lineup_freshness",
        "injury_news_freshness",
        "standings_freshness",
        "recommended_market",
        "context_factor_names",
    ]:
        if col not in enriched.columns:
            enriched[col] = ""
    for col, default in [
        ("fixture_verified", True),
        ("market_suitable", True),
        ("lower_bound_passed", True),
    ]:
        if col not in enriched.columns:
            enriched[col] = default
    if "closing_line_value" not in enriched.columns:
        enriched["closing_line_value"] = enriched["clv"] if "clv" in enriched.columns else None
    mask = enriched["mistake_classification"].astype(str).str.strip() == ""
    if mask.any():
        enriched.loc[mask, "mistake_classification"] = enriched.loc[mask].apply(
            lambda r: _classify_mistake(r.to_dict()), axis=1
        )
    return enriched


def _report_window(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
    start_norm = pd.Timestamp(start_date)
    if start_norm.tzinfo is None:
        start_norm = start_norm.tz_localize("UTC")
    else:
        start_norm = start_norm.tz_convert("UTC")
    start_norm = start_norm.normalize()

    end_norm = pd.Timestamp(end_date)
    if end_norm.tzinfo is None:
        end_norm = end_norm.tz_localize("UTC")
    else:
        end_norm = end_norm.tz_convert("UTC")
    end_norm = end_norm.normalize()

    if df.empty:
        return {
            "start_date": str(start_norm.date()),
            "end_date": str(end_norm.date()),
            "losses": 0,
            "categories": {k: 0 for k in _MISTAKE_CATEGORIES},
            "top_category": None,
        }
    work = df.copy()
    event_dates = pd.to_datetime(work.get("commence_time"), utc=True, errors="coerce").dt.normalize()
    mask = (event_dates >= start_norm) & (event_dates <= end_norm)
    work = work.loc[mask]
    losses = work[work["status"] == "lost"].copy()
    if losses.empty:
        return {
            "start_date": str(start_norm.date()),
            "end_date": str(end_norm.date()),
            "losses": 0,
            "categories": {k: 0 for k in _MISTAKE_CATEGORIES},
            "top_category": None,
        }
    counts = losses["mistake_classification"].value_counts().to_dict()
    categories = {k: int(counts.get(k, 0)) for k in _MISTAKE_CATEGORIES}
    top_category = max(categories.items(), key=lambda item: item[1])[0] if any(categories.values()) else None
    return {
        "start_date": str(start_norm.date()),
        "end_date": str(end_norm.date()),
        "losses": int(len(losses)),
        "categories": categories,
        "top_category": top_category,
    }


def mistake_report(target_date: Optional[str] = None) -> Dict:
    settled = get_settled()
    parlay_df = parlay_breakdown()
    if not parlay_df.empty:
        if "mistake_classification" not in parlay_df.columns:
            parlay_df["mistake_classification"] = parlay_df["status"].apply(
                lambda s: "parlay-construction error" if str(s).lower() == "lost" else ("won" if str(s).lower() == "won" else str(s).lower())
            )
        if "commence_time" not in parlay_df.columns:
            parlay_df["commence_time"] = parlay_df["recorded_at"]
        if "bet_odds" not in parlay_df.columns:
            parlay_df["bet_odds"] = parlay_df["combined_odds"]
        settled = pd.concat([settled, parlay_df], ignore_index=True, sort=False)

    if settled.empty:
        today = pd.Timestamp(target_date) if target_date else pd.Timestamp(datetime.now(tz=timezone.utc).date())
        return {
            "daily": _report_window(pd.DataFrame(), today, today),
            "weekly": _report_window(pd.DataFrame(), today - pd.Timedelta(days=6), today),
        }

    settled = _augment_mistake_fields(settled)
    target = pd.Timestamp(target_date) if target_date else pd.to_datetime(settled["commence_time"], utc=True, errors="coerce").max().normalize()
    daily = _report_window(settled, target, target)
    weekly = _report_window(settled, target - pd.Timedelta(days=6), target)
    return {"daily": daily, "weekly": weekly}


def compute_summary(settled: Optional[pd.DataFrame] = None) -> Dict:
    """
    Compute overall performance summary.

    Returns
    -------
    dict with keys: n_bets, n_won, win_rate, total_stake, total_profit,
                    roi, avg_odds, avg_edge, avg_clv, clv_positive_pct
    """
    if settled is None:
        settled = get_settled()

    if settled.empty:
        return {"n_bets": 0, "message": "No settled bets yet"}

    resolved = settled[settled["status"].isin(["won", "lost"])]
    if resolved.empty:
        return {"n_bets": 0, "message": "No resolved bets (only voids/pushes)"}

    n       = len(resolved)
    n_won   = resolved["won"].sum()
    total_stake  = resolved["stake_units"].sum()
    total_profit = resolved["profit_units"].sum()
    roi     = total_profit / total_stake if total_stake > 0 else 0.0

    clv_valid  = resolved["clv"].dropna()
    avg_clv    = float(clv_valid.mean()) if not clv_valid.empty else None
    clv_pos    = (clv_valid > 0).mean() if not clv_valid.empty else None

    return {
        "n_bets":          int(n),
        "n_won":           int(n_won),
        "win_rate":        round(float(n_won / n), 4),
        "total_stake":     round(float(total_stake), 4),
        "total_profit":    round(float(total_profit), 4),
        "roi":             round(float(roi), 4),
        "avg_odds":        round(float(resolved["bet_odds"].mean()), 4),
        "avg_edge":        round(float(resolved["edge"].mean()), 4),
        "avg_clv":         round(avg_clv, 4) if avg_clv is not None else None,
        "clv_positive_pct":round(float(clv_pos), 4) if clv_pos is not None else None,
    }


def daily_pnl() -> pd.DataFrame:
    """Return P&L grouped by date."""
    settled = get_settled()
    if settled.empty:
        return pd.DataFrame()
    resolved = settled[settled["status"].isin(["won", "lost"])].copy()
    resolved["date"] = pd.to_datetime(resolved["commence_time"]).dt.date
    return (
        resolved.groupby("date")
        .agg(
            n_bets=("pred_id", "count"),
            n_won=("won", "sum"),
            profit_units=("profit_units", "sum"),
            total_stake=("stake_units", "sum"),
        )
        .assign(
            roi=lambda d: d["profit_units"] / d["total_stake"].replace(0, float("nan")),
            cumulative_profit=lambda d: d["profit_units"].cumsum(),
        )
        .reset_index()
    )


def sport_breakdown() -> pd.DataFrame:
    """Return P&L broken down by sport."""
    settled = get_settled()
    if settled.empty:
        return pd.DataFrame()
    resolved = settled[settled["status"].isin(["won", "lost"])].copy()
    return (
        resolved.groupby("sport")
        .agg(
            n_bets=("pred_id", "count"),
            n_won=("won", "sum"),
            profit_units=("profit_units", "sum"),
            total_stake=("stake_units", "sum"),
            avg_clv=("clv", "mean"),
        )
        .assign(
            win_rate=lambda d: d["n_won"] / d["n_bets"],
            roi=lambda d: d["profit_units"] / d["total_stake"].replace(0, float("nan")),
        )
        .reset_index()
    )


def print_dashboard() -> None:
    """Print a quick dashboard to stdout."""
    summary = compute_summary()
    print("\n" + "=" * 55)
    print("  BETTING TRACKER DASHBOARD")
    print("=" * 55)
    if "message" in summary:
        print(f"  {summary['message']}")
        print("=" * 55)
        return

    print(f"  Bets:        {summary['n_bets']}  ({summary['n_won']} won, "
          f"{summary['win_rate']*100:.1f}% win rate)")
    print(f"  Total stake: {summary['total_stake']:.2f} units")
    print(f"  Profit:      {summary['total_profit']:+.2f} units  "
          f"(ROI {summary['roi']*100:+.1f}%)")
    print(f"  Avg odds:    {summary['avg_odds']:.3f}")
    print(f"  Avg edge:    {summary['avg_edge']*100:+.2f}%")
    if summary.get("avg_clv") is not None:
        print(f"  Avg CLV:     {summary['avg_clv']*100:+.2f}%  "
              f"(positive {summary['clv_positive_pct']*100:.0f}% of bets)")

    # By sport
    sport_df = sport_breakdown()
    if not sport_df.empty:
        print("\n  By sport:")
        for _, row in sport_df.iterrows():
            print(
                f"    {row['sport']:12s}  {row['n_bets']:3d} bets  "
                f"ROI {row['roi']*100:+.1f}%  "
                f"CLV {row['avg_clv']*100:+.2f}%" if pd.notna(row.get('avg_clv')) else
                f"    {row['sport']:12s}  {row['n_bets']:3d} bets  "
                f"ROI {row['roi']*100:+.1f}%"
            )

    # Daily P&L (last 7 days)
    pnl = daily_pnl()
    if not pnl.empty:
        print("\n  Recent P&L (last 7 days):")
        for _, row in pnl.tail(7).iterrows():
            bar = "█" * max(0, int(row["profit_units"] * 5)) if row["profit_units"] > 0 \
                  else "▓" * max(0, int(-row["profit_units"] * 5))
            print(
                f"    {row['date']}  {row['profit_units']:+.2f}u  {bar}"
            )

    # Parlay section
    p_df = parlay_breakdown()
    p_pending = get_pending_parlays()
    if not p_df.empty or not p_pending.empty:
        print("\n  Parlays:")
        if not p_df.empty:
            n_p = len(p_df)
            n_p_won = int(p_df["won"].sum())
            p_profit = float(p_df["profit_units"].sum())
            p_stake  = float(p_df["stake_units"].sum())
            p_roi    = p_profit / p_stake if p_stake > 0 else 0.0
            print(f"    Settled: {n_p} ({n_p_won} won)  "
                  f"P&L {p_profit:+.2f}u  ROI {p_roi*100:+.1f}%")
        if not p_pending.empty:
            print(f"    Pending: {len(p_pending)} parlay(s) awaiting leg results")

    print("=" * 55)
