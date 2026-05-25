"""
src/notifications/telegram.py
==============================
Sends formatted Telegram messages with today's top bets and parlays.

Setup (one-time):
  1. Message @BotFather on Telegram → /newbot → copy the token
  2. Message your new bot once, then visit:
     https://api.telegram.org/bot<TOKEN>/getUpdates
     to find your chat_id
  3. Add to .env:
       TELEGRAM_TOKEN=123456:ABC-your-token
       TELEGRAM_CHAT_ID=987654321

Usage:
    from src.notifications.telegram import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.send_daily_report(all_bets, parlay_results, bankroll=1000)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"

# Emoji map per sport
_SPORT_EMOJI = {
    "soccer":     "⚽",
    "basketball": "🏀",
    "tennis":     "🎾",
}

# EV quality tiers  (edge = true EV = (ml_prob × odds) − 1)
def _edge_tag(edge: float) -> str:
    if edge >= 0.30:  return "🔥🔥"   # +30% EV and above
    if edge >= 0.15:  return "🔥"     # +15–30% EV
    if edge >= 0.05:  return "✅"     # +5–15% EV (min threshold)
    return "📊"


class TelegramNotifier:
    """
    Sends betting alerts via the Telegram Bot API.

    Parameters
    ----------
    token : str, optional
        Bot token. Falls back to TELEGRAM_TOKEN env var.
    chat_id : str, optional
        Target chat / channel ID. Falls back to TELEGRAM_CHAT_ID env var.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.token   = token   or os.environ.get("TELEGRAM_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

        if not self.token:
            logger.warning(
                "TELEGRAM_TOKEN not set. Messages will not be sent. "
                "Add it to your .env file."
            )
        if not self.chat_id:
            logger.warning(
                "TELEGRAM_CHAT_ID not set. Messages will not be sent. "
                "Add it to your .env file."
            )

    @property
    def _configured(self) -> bool:
        return bool(self.token and self.chat_id)

    # ──────────────────────────────────────────────────────────────────────
    # Low-level send
    # ──────────────────────────────────────────────────────────────────────

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a raw message. Returns True on success.

        Uses HTML parse mode by default so <b>, <i>, <code> tags work.
        Max message length is 4096 chars; longer messages are auto-split.
        """
        if not self._configured:
            logger.info("[Telegram disabled] Would send:\n%s", text[:300])
            return False

        # Split long messages
        chunks = _split_message(text, 4096)
        success = True
        for chunk in chunks:
            url = _API_BASE.format(token=self.token, method="sendMessage")
            payload = {
                "chat_id":    self.chat_id,
                "text":       chunk,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            try:
                r = requests.post(url, json=payload, timeout=10)
                r.raise_for_status()
            except Exception as exc:
                logger.error("Telegram send failed: %s", exc)
                success = False
        return success

    # ──────────────────────────────────────────────────────────────────────
    # High-level formatters
    # ──────────────────────────────────────────────────────────────────────

    def send_daily_report(
        self,
        all_bets: List[dict],
        parlay_results: Optional[Dict] = None,
        bankroll: float = 1000.0,
        top_n_bets: int = 5,
    ) -> bool:
        """
        Format and send the full daily scan report.

        Parameters
        ----------
        all_bets : list of dicts
            Value bets from daily_scan.py (already edge-filtered).
        parlay_results : dict, optional
            Output of ParlayBuilder.build().
        bankroll : float
            Current bankroll for stake calculation.
        top_n_bets : int
            How many single bets to show (sorted by edge desc).
        """
        today = datetime.now().strftime("%a %d %b %Y")
        lines = [f"<b>📊 Daily Betting Report — {today}</b>", ""]

        # ── Single bets — top 2 per sport, then fill remaining slots ────
        if not all_bets:
            lines.append("No value bets found today.")
        else:
            # Build a diverse top-N: take the best 2 from each active sport
            # first, then fill up to top_n_bets with whatever remains by edge.
            by_sport: Dict[str, List[dict]] = {}
            for b in sorted(all_bets, key=lambda b: b["edge"], reverse=True):
                by_sport.setdefault(b.get("sport", "other"), []).append(b)

            # 2 best per sport (preserves variety)
            per_sport_slots = 2
            selected: List[dict] = []
            seen_ids: set = set()
            for sport_bets in by_sport.values():
                for b in sport_bets[:per_sport_slots]:
                    _id = id(b)
                    if _id not in seen_ids:
                        selected.append(b)
                        seen_ids.add(_id)

            # Fill remaining slots from the global sorted list
            for b in sorted(all_bets, key=lambda b: b["edge"], reverse=True):
                if len(selected) >= top_n_bets:
                    break
                if id(b) not in seen_ids:
                    selected.append(b)
                    seen_ids.add(id(b))

            # Final sort by edge for display
            selected = sorted(selected, key=lambda b: b["edge"], reverse=True)[:top_n_bets]

            lines.append(
                f"<b>Value Bets</b> ({len(all_bets)} found, showing top {len(selected)})"
            )
            for bet in selected:
                sport   = bet.get("sport", "")
                emoji   = _SPORT_EMOJI.get(sport, "🏟")
                tag     = _edge_tag(bet["edge"])
                home    = bet.get("home", "")
                away    = bet.get("away", "")
                bk      = bet.get("bookmaker", "")
                stake   = bankroll * bet["kelly_stake_pct"] / 100
                bk_str  = f" @ <i>{bk}</i>" if bk else ""
                flags = []
                if bet.get("stale_line"):
                    flags.append("⚠️ stale line — verify odds")
                elif bet.get("flagged"):
                    flags.append("⚠️ verify manually")
                flag_str = f"  <i>{' | '.join(flags)}</i>" if flags else ""
                kick_off = bet.get("kick_off", "")
                kick_str = f"  🕐 <i>{kick_off}</i>" if kick_off else ""
                lines += [
                    "",
                    f"{emoji} {tag} <b>{bet['team']}</b>{bk_str}{flag_str}",
                    f"  <code>{away} vs {home}</code>{kick_str}",
                    f"  Odds: <b>{bet['odds']}</b>  EV: <b>+{bet['edge']*100:.1f}%</b>",
                    f"  Model: {bet['ml_prob']*100:.0f}%  Mkt implied: {bet['fair_prob']*100:.0f}%",
                    f"  Kelly stake: <b>£{stake:.0f}</b>  ({bet['kelly_stake_pct']:.1f}% bankroll)",
                ]

        # ── Parlays — split into Today (before midnight) and Overnight (00:00–07:00) ──
        if parlay_results:
            has_any = any(
                v
                for tier_data in parlay_results.values()
                for v in tier_data.values()
            )
            if has_any:
                # Classify each parlay: "overnight" if ANY leg is tagged overnight,
                # otherwise "today". Deduplicate across both sections by leg fingerprint.
                today_parlays: list = []
                overnight_parlays: list = []
                seen_leg_combos: set = set()

                for bracket in ("5x", "10x", "20x"):
                    for tier in ("value", "speculative"):
                        parlays = parlay_results.get(tier, {}).get(bracket, [])
                        if not parlays:
                            continue
                        best = parlays[0]
                        leg_key = frozenset((l.team, l.odds) for l in best.legs)
                        if leg_key in seen_leg_combos:
                            continue
                        seen_leg_combos.add(leg_key)
                        # Check if any leg is from the overnight window
                        has_overnight = any(
                            getattr(l, "window", "today") == "overnight"
                            for l in best.legs
                        )
                        tier_label = "VALUE" if tier == "value" else "SPEC"
                        stake = bankroll * best.kelly_stake_pct / 100
                        # Include local kick-off time for overnight legs
                        def _leg_label(l) -> str:
                            label = f"{l.team} ({l.odds})"
                            if getattr(l, "window", "today") == "overnight":
                                try:
                                    import zoneinfo
                                    utc_dt = datetime.fromisoformat(
                                        l.commence.replace("Z", "+00:00")
                                    )
                                    local_dt = utc_dt.astimezone(
                                        zoneinfo.ZoneInfo("Europe/Vienna")
                                    )
                                    label += f" 🕐{local_dt.strftime('%H:%M')}"
                                except Exception:
                                    pass
                            return label

                        legs_str = "  +  ".join(_leg_label(l) for l in best.legs)
                        entry = (bracket, tier_label, best, stake, legs_str)
                        if has_overnight:
                            overnight_parlays.append(entry)
                        else:
                            today_parlays.append(entry)

                def _parlay_block(entries: list) -> list:
                    out = []
                    for bracket, tier_label, best, stake, legs_str in entries:
                        out += [
                            "",
                            f"<b>[{bracket} {tier_label}]</b> {best.combined_odds:.2f}x",
                            f"  {legs_str}",
                            f"  Win prob: {best.combined_prob*100:.1f}%  "
                            f"EV: {best.ev:.3f}x  Kelly: £{stake:.0f}",
                        ]
                    return out

                if today_parlays:
                    lines += ["", "─────────────────────────", "<b>🎯 Today's Parlays</b>"]
                    lines += _parlay_block(today_parlays)

                if overnight_parlays:
                    lines += ["", "─────────────────────────",
                              "<b>🌙 Overnight Parlays</b>  <i>(NBA · 01:00–06:00 Vienna)</i>"]
                    lines += _parlay_block(overnight_parlays)

                if not today_parlays and not overnight_parlays:
                    lines += ["", "─────────────────────────", "<b>🎯 Parlays</b>",
                              "", "<i>No qualifying parlays found today.</i>"]

        # ── Footer ────────────────────────────────────────────────────────
        lines += [
            "",
            "─────────────────────────",
            f"<i>Bankroll: £{bankroll:,.0f}  |  "
            f"Generated {datetime.now().strftime('%H:%M')}</i>",
        ]

        return self.send("\n".join(lines))

    def send_settlement_summary(
        self,
        n_settled: int,
        profit_units: float,
        roi: float,
        avg_clv: Optional[float] = None,
    ) -> bool:
        """Send a short P&L update after running settle.py."""
        direction = "📈" if profit_units >= 0 else "📉"
        clv_line  = f"\nAvg CLV: <b>{avg_clv*100:+.2f}%</b>" if avg_clv is not None else ""
        text = (
            f"{direction} <b>Settlement Update</b>\n"
            f"Settled: {n_settled} bet(s)\n"
            f"P&L: <b>{profit_units:+.2f} units</b>\n"
            f"ROI: <b>{roi*100:+.1f}%</b>"
            f"{clv_line}"
        )
        return self.send(text)

    def send_alert(self, message: str) -> bool:
        """Send a plain alert message."""
        return self.send(f"⚠️ {message}")

    def test(self) -> bool:
        """Send a test message to confirm the bot is working."""
        return self.send(
            "✅ <b>Sports Betting Bot connected!</b>\n"
            "Daily reports will be sent here each morning."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split_message(text: str, max_len: int = 4096) -> List[str]:
    """Split a message into chunks that fit within Telegram's limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Split at last newline before limit
        cut = text.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks
