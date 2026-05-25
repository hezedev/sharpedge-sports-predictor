"""
Tennis live edge model.

Three edge markets (API odds available):
  h2h     — live match winner
  totals  — total games over/under
  spreads — games handicap

Three probability signals (model-only, no market odds):
  set_winner   — P(each player wins current set)
  next_game    — P(server holds) / P(receiver breaks)
  tiebreak     — P(current set reaches tiebreak)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_EDGE   = 0.06
_MIN_CONF   = 60
_MAX_RATIO  = 2.2
_CACHE: Optional[pd.DataFrame] = None


# ── Feature cache ─────────────────────────────────────────────────────────────

def _load_cache() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    try:
        atp = pd.read_parquet("data/cache/tennis_features.parquet")
        wta = pd.read_parquet("data/cache/tennis_wta_features.parquet")
        _CACHE = pd.concat([atp, wta], ignore_index=True)
    except Exception as exc:
        logger.warning("Could not load tennis feature cache: %s", exc)
        _CACHE = pd.DataFrame()
    return _CACHE


def _player_baselines(player_name: str) -> dict:
    df = _load_cache()
    norm = lambda s: re.sub(r"[^a-z]", "", s.lower())
    target = norm(player_name)

    for col, pfx in [("player1_name", "p1"), ("player2_name", "p2")]:
        if col not in df.columns:
            continue
        mask = df[col].fillna("").apply(norm) == target
        rows = df[mask].tail(20)
        if len(rows) >= 3:
            return _extract(rows, pfx)

    # Surname fallback
    surname = target[-6:] if len(target) > 6 else target
    for col, pfx in [("player1_name", "p1"), ("player2_name", "p2")]:
        if col not in df.columns:
            continue
        mask = df[col].fillna("").apply(lambda s: surname in norm(s))
        rows = df[mask].tail(20)
        if len(rows) >= 3:
            return _extract(rows, pfx)

    return _tour_avg()


def _extract(rows: pd.DataFrame, pfx: str) -> dict:
    def mean(col):
        c = f"roll_{pfx}_{col}"
        if c not in rows.columns:
            return None
        vals = pd.to_numeric(rows[c], errors="coerce").replace(0, np.nan).dropna()
        return float(vals.mean()) if len(vals) else None

    first_in = mean("1st_pct") or 0.60
    bp_save  = mean("bp_save") or 0.65
    ace_rate = mean("ace_rate") or 0.06
    hold = first_in * 0.73 + (1 - first_in) * 0.50
    hold = hold * 0.7 + bp_save * 0.3
    return {
        "hold_rate": max(0.45, min(0.90, hold)),
        "bp_save":   bp_save,
        "ace_rate":  ace_rate,
        "first_in":  first_in,
    }


def _tour_avg() -> dict:
    return {"hold_rate": 0.635, "bp_save": 0.65, "ace_rate": 0.06, "first_in": 0.60}


# ── Markov core ───────────────────────────────────────────────────────────────

def _hold_from_live(live: dict, key: str, fallback: float) -> float:
    pct = live.get(key)
    if pct:
        h = pct * 0.73 + (1 - pct) * 0.50
        return max(0.45, min(0.90, h))
    return fallback


def _set_win_prob(p1_hold: float, p2_hold: float,
                  g1: int, g2: int, p1_serves: bool) -> float:
    memo: Dict[Tuple, float] = {}

    def w(a, b, p1s):
        if a >= 6 and a - b >= 2:
            return 1.0
        if b >= 6 and b - a >= 2:
            return 0.0
        if a == 6 and b == 6:
            return (p1_hold + (1 - p2_hold)) / 2
        k = (a, b, p1s)
        if k in memo:
            return memo[k]
        pg = p1_hold if p1s else (1 - p2_hold)
        res = pg * w(a + 1, b, not p1s) + (1 - pg) * w(a, b + 1, not p1s)
        memo[k] = res
        return res

    games_played = g1 + g2
    cur_p1_serves = p1_serves if (games_played % 2 == 0) else not p1_serves
    return w(g1, g2, cur_p1_serves)


def _series_win(p_set: float, need1: int, need2: int) -> float:
    memo: Dict[Tuple, float] = {}

    def f(n1, n2):
        if n1 == 0: return 1.0
        if n2 == 0: return 0.0
        k = (n1, n2)
        if k in memo: return memo[k]
        r = p_set * f(n1 - 1, n2) + (1 - p_set) * f(n1, n2 - 1)
        memo[k] = r
        return r

    return f(need1, need2)


def _match_win_prob(p1_hold, p2_hold, p1_sets, p2_sets,
                    p1_games, p2_games, server, best_of=3) -> float:
    sets_needed = (best_of + 1) // 2
    if p1_sets >= sets_needed: return 1.0
    if p2_sets >= sets_needed: return 0.0

    if server is None:
        return 0.5 * (_match_win_prob(p1_hold, p2_hold, p1_sets, p2_sets,
                                      p1_games, p2_games, 1, best_of) +
                      _match_win_prob(p1_hold, p2_hold, p1_sets, p2_sets,
                                      p1_games, p2_games, 2, best_of))

    p1_serves_first_in_set = (server == 1)
    p1_set = _set_win_prob(p1_hold, p2_hold, p1_games, p2_games, p1_serves_first_in_set)
    return _series_win(p1_set, sets_needed - p1_sets, sets_needed - p2_sets)


def _expected_games_per_set(p1_hold: float, p2_hold: float) -> float:
    """
    Compute E[games in a set] via Markov chain over all reachable set scores.
    """
    memo: Dict[Tuple, float] = {}

    def eg(a, b, p1s):
        if a >= 6 and a - b >= 2: return 0.0
        if b >= 6 and b - a >= 2: return 0.0
        if a == 6 and b == 6:
            # Tiebreak adds ~13 points ≈ 1 game in expectation
            return 1.0
        k = (a, b, p1s)
        if k in memo: return memo[k]
        pg = p1_hold if p1s else (1 - p2_hold)
        res = 1 + pg * eg(a + 1, b, not p1s) + (1 - pg) * eg(a, b + 1, not p1s)
        memo[k] = res
        return res

    return eg(0, 0, True)


def _expected_sets(p1_hold: float, p2_hold: float,
                   p1_sets: int, p2_sets: int, best_of: int = 3) -> float:
    """Expected total sets remaining from current state."""
    needed = (best_of + 1) // 2
    p1_set = _set_win_prob(p1_hold, p2_hold, 0, 0, True)

    memo: Dict[Tuple, float] = {}

    def es(n1, n2):
        if n1 == 0 or n2 == 0: return 0.0
        k = (n1, n2)
        if k in memo: return memo[k]
        r = 1 + p1_set * es(n1 - 1, n2) + (1 - p1_set) * es(n1, n2 - 1)
        memo[k] = r
        return r

    return es(needed - p1_sets, needed - p2_sets)


def _expected_total_games(p1_hold: float, p2_hold: float,
                           p1_sets: int, p2_sets: int,
                           p1_games: int, p2_games: int,
                           best_of: int = 3) -> float:
    games_played = (
        sum(range(p1_games + p2_games))   # already counted below
    )
    # Games already played (completed sets + current set games)
    # We don't track per-set history easily here, so use a proxy:
    # approximate from set score
    sets_completed = p1_sets + p2_sets
    avg_per_set = _expected_games_per_set(p1_hold, p2_hold)

    # Games already played = (sets_completed × avg_per_set) + current set games
    already = sets_completed * avg_per_set + p1_games + p2_games

    # Games remaining = expected remaining sets × avg games per set
    remaining_sets = _expected_sets(p1_hold, p2_hold, p1_sets, p2_sets, best_of)
    # Remaining in current set
    current_set_remaining = max(0, avg_per_set - p1_games - p2_games)

    return already + current_set_remaining + remaining_sets * avg_per_set


def _tiebreak_prob(p1_hold: float, p2_hold: float,
                   g1: int, g2: int, p1_serves: bool) -> float:
    """P(current set reaches 6-6 tiebreak from current game score)."""
    memo: Dict[Tuple, float] = {}

    def p(a, b, p1s):
        if a >= 6 and a - b >= 2: return 0.0
        if b >= 6 and b - a >= 2: return 0.0
        if a == 6 and b == 6:    return 1.0
        k = (a, b, p1s)
        if k in memo: return memo[k]
        pg = p1_hold if p1s else (1 - p2_hold)
        res = pg * p(a + 1, b, not p1s) + (1 - pg) * p(a, b + 1, not p1s)
        memo[k] = res
        return res

    games_played = g1 + g2
    cur_p1s = p1_serves if (games_played % 2 == 0) else not p1_serves
    return p(g1, g2, cur_p1s)


def _momentum_adj(recent_p1: int, recent_p2: int) -> float:
    total = recent_p1 + recent_p2
    if total == 0: return 0.0
    return max(-0.04, min(0.04, (recent_p1 / total - 0.5) * 0.08))


# ── Edge / signal dataclass ───────────────────────────────────────────────────

@dataclass
class LiveEdge:
    p1_name:     str
    p2_name:     str
    bet_on:      str
    model_prob:  float
    market_prob: float
    odds:        float
    edge:        float
    confidence:  int
    sport_key:   str
    bookmaker:   str
    market:      str
    triggers:    List[str] = field(default_factory=list)
    is_signal:   bool = False   # True = no direct market odds, informational only
    signal_label: str = ""

    def to_dict(self) -> dict:
        return {
            "p1_name":     self.p1_name,
            "p2_name":     self.p2_name,
            "bet_on":      self.bet_on,
            "model_prob":  round(self.model_prob, 4),
            "market_prob": round(self.market_prob, 4),
            "odds":        round(self.odds, 3),
            "edge":        round(self.edge, 4),
            "edge_pct":    f"{self.edge*100:+.1f}%",
            "confidence":  self.confidence,
            "sport_key":   self.sport_key,
            "bookmaker":   self.bookmaker,
            "market":      self.market,
            "triggers":    self.triggers,
            "is_signal":   self.is_signal,
            "signal_label": self.signal_label,
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_edges(match_state: dict, odds_entries: List[dict]) -> List[LiveEdge]:
    """
    Compute all edges and signals for one live match.

    odds_entries: all market entries for this match (h2h, totals, spreads).
    """
    p1_name = match_state["p1_name"]
    p2_name = match_state["p2_name"]
    live    = match_state.get("live_stats") or {}

    p1_base = _player_baselines(p1_name)
    p2_base = _player_baselines(p2_name)

    p1_hold = _hold_from_live(live, "p1_1st_pct", p1_base["hold_rate"])
    p2_hold = _hold_from_live(live, "p2_1st_pct", p2_base["hold_rate"])

    sport_key = next((e["sport_key"] for e in odds_entries if e.get("sport_key")), "")
    best_of = 5 if any(gs in sport_key for gs in (
        "french_open", "wimbledon", "us_open", "australian_open")) else 3

    # Core match-win probability
    model_p1 = _match_win_prob(
        p1_hold, p2_hold,
        match_state["p1_sets"], match_state["p2_sets"],
        match_state["p1_games_in_set"], match_state["p2_games_in_set"],
        match_state.get("server"), best_of,
    )
    adj = _momentum_adj(match_state.get("recent_p1", 0), match_state.get("recent_p2", 0))
    model_p1 = max(0.05, min(0.95, model_p1 + adj))
    model_p2 = 1.0 - model_p1

    edges: List[LiveEdge] = []

    for entry in odds_entries:
        mkt = entry.get("market")
        if mkt == "h2h":
            edges.extend(_h2h_edges(match_state, entry, model_p1, model_p2,
                                     p1_base, p2_base, live, adj, sport_key))
        elif mkt == "totals":
            edges.extend(_totals_edges(match_state, entry, p1_hold, p2_hold,
                                        best_of, sport_key))
        elif mkt == "spreads":
            edges.extend(_spreads_edges(match_state, entry, p1_hold, p2_hold,
                                         model_p1, best_of, sport_key))

    # Model-only signals (no market odds needed)
    edges.extend(_set_winner_signals(match_state, p1_hold, p2_hold, sport_key))
    edges.extend(_next_game_signals(match_state, p1_hold, p2_hold, sport_key))
    edges.extend(_tiebreak_signal(match_state, p1_hold, p2_hold, sport_key))

    return edges


# ── H2H (match winner) ────────────────────────────────────────────────────────

def _h2h_edges(state, entry, model_p1, model_p2,
                p1_base, p2_base, live, adj, sport_key) -> List[LiveEdge]:
    p1_odds = entry.get("p1_odds")
    p2_odds = entry.get("p2_odds")
    if not p1_odds or not p2_odds:
        return []

    total_imp = 1 / p1_odds + 1 / p2_odds
    mkt_p1 = (1 / p1_odds) / total_imp
    mkt_p2 = (1 / p2_odds) / total_imp

    out = []
    for player, model_p, mkt_p, odds in [
        (state["p1_name"], model_p1, mkt_p1, p1_odds),
        (state["p2_name"], model_p2, mkt_p2, p2_odds),
    ]:
        edge = model_p * odds - 1.0
        if edge < _MIN_EDGE: continue
        if model_p > _MAX_RATIO * mkt_p: continue
        conf = _h2h_confidence(state, live, model_p, mkt_p)
        if conf < _MIN_CONF: continue
        triggers = _h2h_triggers(player == state["p1_name"], state,
                                   p1_base, p2_base, model_p, mkt_p, adj, live)
        out.append(LiveEdge(
            p1_name=state["p1_name"], p2_name=state["p2_name"],
            bet_on=player, model_prob=model_p, market_prob=mkt_p,
            odds=odds, edge=edge, confidence=conf,
            sport_key=sport_key, bookmaker=entry.get("bookmaker", ""),
            market="Live Match Winner", triggers=triggers,
        ))
    return out


def _h2h_confidence(state, live, model_p, mkt_p) -> int:
    score = 50
    total_games = (sum(state.get("p1_games_history") or []) +
                   sum(state.get("p2_games_history") or []) +
                   state.get("p1_games_in_set", 0) + state.get("p2_games_in_set", 0))
    score += min(15, total_games)
    if live.get("p1_1st_pct") and live.get("p2_1st_pct"): score += 10
    if state.get("server"): score += 5
    if (model_p - mkt_p) > 0.08: score += 10
    elif (model_p - mkt_p) > 0.05: score += 5
    if total_games < 3: score -= 15
    return max(0, min(100, score))


def _h2h_triggers(is_p1, state, p1_base, p2_base, model_p, mkt_p, adj, live) -> List[str]:
    t = []
    pname = state["p1_name"] if is_p1 else state["p2_name"]
    oname = state["p2_name"] if is_p1 else state["p1_name"]
    base  = p1_base if is_p1 else p2_base
    obase = p2_base if is_p1 else p1_base
    gap = (model_p - mkt_p) * 100
    t.append(f"Model: {model_p*100:.0f}% vs market: {mkt_p*100:.0f}% (gap {gap:+.0f}pp)")
    if is_p1 and adj > 0.02:
        t.append(f"{pname} won {state['recent_p1']} of last 5 games — momentum")
    elif not is_p1 and adj < -0.02:
        t.append(f"{pname} won {state['recent_p2']} of last 5 games — momentum")
    if base.get("hold_rate", 0) > 0.70:
        t.append(f"{pname} holds at {base['hold_rate']*100:.0f}% historically — strong server")
    if obase.get("hold_rate", 1) < 0.60:
        t.append(f"{oname} holds at only {obase['hold_rate']*100:.0f}% — vulnerable on serve")
    lv = live.get("p1_1st_pct" if is_p1 else "p2_1st_pct")
    if lv and lv > 0.65:
        t.append(f"First serve at {lv*100:.0f}% today — serving well")
    elif lv and lv < 0.50:
        t.append(f"First serve only {lv*100:.0f}% — under pressure on serve")
    return t


# ── Totals (total games O/U) ──────────────────────────────────────────────────

def _totals_edges(state, entry, p1_hold, p2_hold,
                   best_of, sport_key) -> List[LiveEdge]:
    line      = entry.get("line")
    over_odds = entry.get("over_odds")
    under_odds = entry.get("under_odds")
    if not line or not over_odds or not under_odds:
        return []

    expected = _expected_total_games(
        p1_hold, p2_hold,
        state["p1_sets"], state["p2_sets"],
        state["p1_games_in_set"], state["p2_games_in_set"],
        best_of,
    )

    # Use a normal approximation around expected total to get P(over) / P(under)
    # Std dev ≈ 2.5 games (empirical estimate for best-of-3)
    import math
    sigma = 2.5 if best_of == 3 else 3.5
    z_over  = (line - expected) / sigma
    p_over  = 1 - _normal_cdf(z_over)
    p_under = 1 - p_over

    total_imp = 1 / over_odds + 1 / under_odds
    mkt_over  = (1 / over_odds) / total_imp
    mkt_under = (1 / under_odds) / total_imp

    out = []
    for side, model_p, mkt_p, odds, label in [
        ("Over",  p_over,  mkt_over,  over_odds,  f"Over {line}"),
        ("Under", p_under, mkt_under, under_odds, f"Under {line}"),
    ]:
        edge = model_p * odds - 1.0
        if edge < _MIN_EDGE: continue
        if model_p > _MAX_RATIO * mkt_p: continue
        conf = _totals_confidence(state, expected, line)
        if conf < _MIN_CONF: continue
        diff = expected - line
        triggers = [
            f"Expected total games: {expected:.1f} vs line {line} (diff {diff:+.1f})",
            f"Model {side} probability: {model_p*100:.0f}% vs market: {mkt_p*100:.0f}%",
            f"Hold rates — {state['p1_name']}: {p1_hold*100:.0f}%  {state['p2_name']}: {p2_hold*100:.0f}%",
        ]
        out.append(LiveEdge(
            p1_name=state["p1_name"], p2_name=state["p2_name"],
            bet_on=label, model_prob=model_p, market_prob=mkt_p,
            odds=odds, edge=edge, confidence=conf,
            sport_key=sport_key, bookmaker=entry.get("bookmaker", ""),
            market="Total Games", triggers=triggers,
        ))
    return out


def _totals_confidence(state, expected, line) -> int:
    score = 55
    diff = abs(expected - line)
    if diff > 3: score += 15
    elif diff > 1.5: score += 8
    total_games = (state.get("p1_games_in_set", 0) + state.get("p2_games_in_set", 0) +
                   sum(state.get("p1_games_history") or []) +
                   sum(state.get("p2_games_history") or []))
    score += min(10, total_games // 2)
    return max(0, min(100, score))


def _normal_cdf(z: float) -> float:
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ── Spreads (games handicap) ──────────────────────────────────────────────────

def _spreads_edges(state, entry, p1_hold, p2_hold,
                    model_p1, best_of, sport_key) -> List[LiveEdge]:
    p1_spread      = entry.get("p1_spread")
    p1_spread_odds = entry.get("p1_spread_odds")
    p2_spread      = entry.get("p2_spread")
    p2_spread_odds = entry.get("p2_spread_odds")
    if p1_spread is None or not p1_spread_odds:
        return []

    # Use win-probability as a proxy for covering spread:
    # If model_p1 >> market-implied, the spread odds are also mispriced.
    # More precisely: P(p1 covers spread X) ≈ f(model_p1, X)
    # Approximation: P(cover) ≈ model_p1 adjusted by spread direction
    # For a -3.5 favourite: covering is harder than winning → discount
    # For a +3.5 underdog:  covering is easier than winning → inflate

    total_imp_p1 = 1 / p1_spread_odds
    total_imp_p2 = 1 / p2_spread_odds
    vig = total_imp_p1 + total_imp_p2
    mkt_p1_spread = total_imp_p1 / vig
    mkt_p2_spread = total_imp_p2 / vig

    # Expected game margin scales with win prob
    expected_margin = (model_p1 - 0.5) * 8   # rough: 50% win → 0 margin, 80% → +2.4
    sigma = 3.0
    import math
    p1_covers = 1 - _normal_cdf((-p1_spread - expected_margin) / sigma) if p1_spread < 0 \
                else 1 - _normal_cdf((p1_spread - expected_margin) / sigma)

    out = []
    for player, model_p, mkt_p, odds, spread, label in [
        (state["p1_name"], p1_covers,       mkt_p1_spread, p1_spread_odds,
         p1_spread, f"{state['p1_name']} {p1_spread:+.1f} games"),
        (state["p2_name"], 1 - p1_covers,   mkt_p2_spread, p2_spread_odds,
         p2_spread, f"{state['p2_name']} {p2_spread:+.1f} games"),
    ]:
        edge = model_p * odds - 1.0
        if edge < _MIN_EDGE: continue
        if model_p > _MAX_RATIO * mkt_p: continue
        conf = max(0, min(100, 55 + int((model_p - mkt_p) * 100)))
        if conf < _MIN_CONF: continue
        triggers = [
            f"Model win prob {model_p1*100:.0f}% implies expected margin {expected_margin:+.1f} games",
            f"Spread: {label} — model covers prob {model_p*100:.0f}% vs market {mkt_p*100:.0f}%",
        ]
        out.append(LiveEdge(
            p1_name=state["p1_name"], p2_name=state["p2_name"],
            bet_on=label, model_prob=model_p, market_prob=mkt_p,
            odds=odds, edge=edge, confidence=conf,
            sport_key=sport_key, bookmaker=entry.get("bookmaker", ""),
            market="Games Handicap", triggers=triggers,
        ))
    return out


# ── Model-only signals ────────────────────────────────────────────────────────

def _set_winner_signals(state, p1_hold, p2_hold, sport_key) -> List[LiveEdge]:
    g1 = state["p1_games_in_set"]
    g2 = state["p2_games_in_set"]
    server = state.get("server")
    p1_serves = (server == 1) if server else True

    p1_set = _set_win_prob(p1_hold, p2_hold, g1, g2, p1_serves)
    p2_set = 1 - p1_set

    out = []
    for player, prob, opp_prob in [
        (state["p1_name"], p1_set, p2_set),
        (state["p2_name"], p2_set, p1_set),
    ]:
        if prob < 0.55 or prob > 0.95:
            continue  # Only surface meaningful skews
        out.append(LiveEdge(
            p1_name=state["p1_name"], p2_name=state["p2_name"],
            bet_on=player, model_prob=prob, market_prob=0.5,
            odds=round(1 / prob, 2), edge=0.0, confidence=65,
            sport_key=sport_key, bookmaker="model",
            market="Current Set Winner",
            is_signal=True,
            signal_label=f"Set {state['current_set']}: {player} at {prob*100:.0f}%",
            triggers=[
                f"Games: {g1}-{g2} in Set {state['current_set']}",
                f"Server: {state['p1_name'] if server==1 else state['p2_name'] if server==2 else 'unknown'}",
            ],
        ))
    return out[:1]  # Only show the player with the edge


def _next_game_signals(state, p1_hold, p2_hold, sport_key) -> List[LiveEdge]:
    server = state.get("server")
    if server is None:
        return []

    if server == 1:
        hold_prob = p1_hold
        server_name = state["p1_name"]
        receiver_name = state["p2_name"]
    else:
        hold_prob = p2_hold
        server_name = state["p2_name"]
        receiver_name = state["p1_name"]

    break_prob = 1 - hold_prob
    out = []

    if hold_prob > 0.72:
        out.append(LiveEdge(
            p1_name=state["p1_name"], p2_name=state["p2_name"],
            bet_on=server_name, model_prob=hold_prob, market_prob=0.635,
            odds=round(1 / hold_prob, 2), edge=0.0, confidence=70,
            sport_key=sport_key, bookmaker="model",
            market="Server to Hold",
            is_signal=True,
            signal_label=f"{server_name} holds at {hold_prob*100:.0f}%",
            triggers=[f"Historical hold rate {hold_prob*100:.0f}% — strong server advantage"],
        ))
    elif break_prob > 0.45:
        out.append(LiveEdge(
            p1_name=state["p1_name"], p2_name=state["p2_name"],
            bet_on=receiver_name, model_prob=break_prob, market_prob=0.365,
            odds=round(1 / break_prob, 2), edge=0.0, confidence=65,
            sport_key=sport_key, bookmaker="model",
            market="Receiver to Break",
            is_signal=True,
            signal_label=f"{receiver_name} break chance {break_prob*100:.0f}%",
            triggers=[f"Server hold rate only {hold_prob*100:.0f}% — returner has edge"],
        ))
    return out


def _tiebreak_signal(state, p1_hold, p2_hold, sport_key) -> List[LiveEdge]:
    g1 = state["p1_games_in_set"]
    g2 = state["p2_games_in_set"]

    # Only relevant once set is 4+ games in and close
    if g1 + g2 < 6 or abs(g1 - g2) > 2:
        return []

    server = state.get("server")
    p1_serves = (server == 1) if server else True
    tb_prob = _tiebreak_prob(p1_hold, p2_hold, g1, g2, p1_serves)

    if tb_prob < 0.20:
        return []

    return [LiveEdge(
        p1_name=state["p1_name"], p2_name=state["p2_name"],
        bet_on="Tiebreak", model_prob=tb_prob, market_prob=0.20,
        odds=round(1 / tb_prob, 2), edge=0.0, confidence=65,
        sport_key=sport_key, bookmaker="model",
        market="Tiebreak in Set",
        is_signal=True,
        signal_label=f"Tiebreak probability: {tb_prob*100:.0f}% (Score: {g1}-{g2})",
        triggers=[
            f"Set score {g1}-{g2} — both players on serve, converging toward tiebreak",
            f"Hold rates: {state['p1_name'].split()[-1]} {p1_hold*100:.0f}%  {state['p2_name'].split()[-1]} {p2_hold*100:.0f}%",
        ],
    )]
