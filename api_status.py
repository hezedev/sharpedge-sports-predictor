#!/usr/bin/env python3
"""
api_status.py — Check the health and quota of all APIs used by sports_predictor.

Usage:
    python api_status.py             # check all APIs
    python api_status.py --update    # update a key in .env interactively
    python api_status.py odds        # check only The Odds API
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv, set_key

ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE)

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(s):    return f"{GREEN}✓ {s}{RESET}"
def warn(s):  return f"{YELLOW}⚠ {s}{RESET}"
def err(s):   return f"{RED}✗ {s}{RESET}"
def info(s):  return f"{BLUE}ℹ {s}{RESET}"
def bold(s):  return f"{BOLD}{s}{RESET}"
def dim(s):   return f"{DIM}{s}{RESET}"

def bar(used, total, width=28):
    if not total:
        return dim("─" * width)
    pct = used / total
    filled = int(pct * width)
    color = GREEN if pct < 0.5 else YELLOW if pct < 0.8 else RED
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET} {int(pct*100)}%"

# ── Individual checks ─────────────────────────────────────────────────────────

def check_odds_api():
    key = os.getenv("ODDS_API_KEY", "")
    if not key:
        return {"name": "The Odds API", "status": "missing_key", "detail": "ODDS_API_KEY not set"}
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": key},
            timeout=10,
        )
        remaining = int(r.headers.get("x-requests-remaining", -1))
        used      = int(r.headers.get("x-requests-used", -1))
        total     = used + remaining if used >= 0 and remaining >= 0 else 500
        if r.status_code == 401:
            return {"name": "The Odds API", "status": "invalid_key", "detail": "401 Unauthorised — key rejected"}
        if r.status_code == 200:
            status = "ok" if remaining > 100 else ("warn" if remaining > 10 else "critical")
            return {
                "name": "The Odds API",
                "status": status,
                "detail": f"{remaining}/{total} requests remaining this month",
                "used": used, "remaining": remaining, "total": total,
            }
        return {"name": "The Odds API", "status": "error", "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"name": "The Odds API", "status": "error", "detail": str(e)}


def check_football_data():
    key = os.getenv("FOOTBALL_DATA_API_KEY", "")
    if not key:
        return {"name": "Football-Data.org", "status": "missing_key", "detail": "FOOTBALL_DATA_API_KEY not set"}
    try:
        r = requests.get(
            "https://api.football-data.org/v4/competitions",
            headers={"X-Auth-Token": key},
            timeout=10,
        )
        if r.status_code == 200:
            return {"name": "Football-Data.org", "status": "ok", "detail": "10 req/min, key valid"}
        if r.status_code == 403:
            return {"name": "Football-Data.org", "status": "invalid_key", "detail": "403 Forbidden — key rejected"}
        return {"name": "Football-Data.org", "status": "error", "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"name": "Football-Data.org", "status": "error", "detail": str(e)}


def check_balldontlie():
    key = os.getenv("BALLDONTLIE_API_KEY", "")
    if not key:
        return {"name": "BallDontLie", "status": "missing_key", "detail": "BALLDONTLIE_API_KEY not set"}
    try:
        r = requests.get(
            "https://api.balldontlie.io/v1/teams",
            headers={"Authorization": key},
            timeout=10,
        )
        if r.status_code == 200:
            return {"name": "BallDontLie (NBA)", "status": "ok", "detail": "Free tier, key valid"}
        if r.status_code == 401:
            return {"name": "BallDontLie (NBA)", "status": "invalid_key", "detail": "401 Unauthorised"}
        return {"name": "BallDontLie (NBA)", "status": "error", "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"name": "BallDontLie (NBA)", "status": "error", "detail": str(e)}


def check_mlb_api():
    """MLB Stats API is public — just ping it."""
    try:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/sports",
            timeout=10,
        )
        if r.status_code == 200:
            return {"name": "MLB Stats API", "status": "ok", "detail": "Public API, no key required"}
        return {"name": "MLB Stats API", "status": "error", "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"name": "MLB Stats API", "status": "error", "detail": str(e)}


def check_nhl_api():
    """NHL Web API is public — just ping it."""
    try:
        r = requests.get(
            "https://api-web.nhle.com/v1/standings/now",
            timeout=10,
        )
        if r.status_code == 200:
            return {"name": "NHL Web API", "status": "ok", "detail": "Public API, no key required"}
        return {"name": "NHL Web API", "status": "error", "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"name": "NHL Web API", "status": "error", "detail": str(e)}


def check_telegram():
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token:
        return {"name": "Telegram Bot", "status": "missing_key", "detail": "TELEGRAM_TOKEN not set"}
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=10,
        )
        if r.status_code == 200:
            bot_name = r.json().get("result", {}).get("username", "unknown")
            return {"name": "Telegram Bot", "status": "ok",
                    "detail": f"@{bot_name} | chat_id={chat_id or 'not set'}"}
        return {"name": "Telegram Bot", "status": "invalid_key", "detail": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"name": "Telegram Bot", "status": "error", "detail": str(e)}


CHECKS = {
    "odds":      check_odds_api,
    "football":  check_football_data,
    "balldontlie": check_balldontlie,
    "mlb":       check_mlb_api,
    "nhl":       check_nhl_api,
    "telegram":  check_telegram,
}

# ── Display ───────────────────────────────────────────────────────────────────

def status_icon(status):
    return {
        "ok": ok("OK"),
        "warn": warn("LOW"),
        "critical": err("CRITICAL"),
        "missing_key": warn("NO KEY"),
        "invalid_key": err("INVALID"),
        "error": err("ERROR"),
    }.get(status, warn(status.upper()))


def print_result(result):
    name   = bold(result["name"].ljust(22))
    status = status_icon(result["status"])
    detail = dim(result["detail"])
    print(f"  {name}  {status}  {detail}")

    if "used" in result:
        used, total = result["used"], result["total"]
        print(f"  {'':22}  {bar(used, total)}  {result['remaining']} left")


def run_checks(names=None):
    checks = {k: v for k, v in CHECKS.items() if not names or k in names}
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  Sports Predictor — API Status Check{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}\n")

    results = {}
    for name, fn in checks.items():
        sys.stdout.write(f"  Checking {name}… ")
        sys.stdout.flush()
        r = fn()
        sys.stdout.write("\r" + " " * 40 + "\r")
        print_result(r)
        results[name] = r

    print(f"\n{BOLD}{'─'*60}{RESET}")
    ok_count   = sum(1 for r in results.values() if r["status"] == "ok")
    warn_count = sum(1 for r in results.values() if r["status"] in ("warn", "missing_key"))
    err_count  = sum(1 for r in results.values() if r["status"] in ("invalid_key", "error", "critical"))
    total = len(results)
    print(f"  {ok(f'{ok_count}/{total} OK')}  {warn(f'{warn_count} warnings')}  {err(f'{err_count} errors')}\n")
    return results


def update_key():
    """Interactive key updater."""
    print(f"\n{BOLD}Update API Key{RESET}\n")
    env_vars = {
        "1": ("ODDS_API_KEY",          "The Odds API"),
        "2": ("FOOTBALL_DATA_API_KEY", "Football-Data.org"),
        "3": ("BALLDONTLIE_API_KEY",   "BallDontLie (NBA)"),
        "4": ("API_SPORTS_KEY",        "API-Sports"),
        "5": ("RAPIDAPI_KEY",          "RapidAPI"),
        "6": ("TELEGRAM_TOKEN",        "Telegram Bot token"),
        "7": ("TELEGRAM_CHAT_ID",      "Telegram Chat ID"),
    }
    for num, (var, label) in env_vars.items():
        current = os.getenv(var, "")
        masked  = (current[:6] + "••••" + current[-4:]) if len(current) > 10 else current or dim("not set")
        print(f"  {num}. {label.ljust(26)} {dim(masked)}")

    choice = input("\n  Select number (or q to quit): ").strip()
    if choice == "q" or choice not in env_vars:
        return

    var, label = env_vars[choice]
    new_val = input(f"  New value for {label}: ").strip()
    if not new_val:
        print(warn("  Empty value — no changes made."))
        return

    set_key(str(ENV_FILE), var, new_val)
    print(ok(f"  {var} updated in {ENV_FILE.name}"))
    print(info("  Restart daily_scan.py to pick up the new key."))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check API health for sports_predictor")
    parser.add_argument("apis", nargs="*",
                        default=["all"], help="Which APIs to check (odds football balldontlie mlb nhl telegram)")
    parser.add_argument("--update", action="store_true", help="Update a key in .env")
    args = parser.parse_args()

    if args.update:
        update_key()
    else:
        names = None if "all" in args.apis or not args.apis else args.apis
        run_checks(names)
