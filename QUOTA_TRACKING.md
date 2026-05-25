# API Quota Tracking & Monitoring

## Overview

The hybrid odds fetcher automatically logs all API requests to `data/quota_tracker.json`. This enables visibility into quota burn across Betfair (unlimited), The Odds API (500/month), and API-Football (100/day).

---

## Quota Limits Reference

| Source | Limit | Period | Free Tier | Cost |
|--------|-------|--------|-----------|------|
| **Betfair** | Unlimited | ∞ | ✅ Yes | $0 |
| **The Odds API** | 500 requests | 1 month | ✅ Yes (free tier) | $0 (with quota) |
| **API-Football** | 100 requests | 1 day | ✅ Yes (RapidAPI) | $0 (with quota) |

---

## Tracker File Format

Location: `data/quota_tracker.json`

```json
{
  "2026-04-20": {
    "betfair": {
      "requests": 15,
      "successes": 15,
      "failures": 0,
      "last_update": "2026-04-20T17:05:32.123456+00:00"
    },
    "odds_api": {
      "requests": 0,
      "successes": 0,
      "failures": 0
    },
    "api_football": {
      "requests": 5,
      "successes": 4,
      "failures": 1,
      "last_update": "2026-04-20T17:10:45.987654+00:00"
    }
  },
  "2026-04-21": {
    "betfair": {
      "requests": 15,
      "successes": 15,
      "failures": 0,
      "last_update": "2026-04-21T17:05:00+00:00"
    },
    "odds_api": {
      "requests": 15,  // Fallback due to Betfair downtime
      "successes": 15,
      "failures": 0
    },
    "api_football": {
      "requests": 0
    }
  }
}
```

### Field Meanings

- **requests**: Total API calls made to this source
- **successes**: Successful requests (status 200)
- **failures**: Failed requests (network error, auth error, etc.)
- **last_update**: ISO timestamp of last request

---

## Viewing Quota Status

### Command 1: Daily Report
```bash
python fetch_closing_odds_v2.py --quota-report
```

**Output Example**:
```
{
  "today": {
    "betfair": {
      "requests": 15,
      "successes": 15,
      "failures": 0,
      "last_update": "2026-04-20T17:05:32+00:00"
    },
    "api_football": {
      "requests": 5,
      "successes": 4,
      "failures": 1
    }
  },
  "this_month": {
    "betfair": {
      "requests": 300,
      "successes": 300
    },
    "odds_api": {
      "requests": 15,
      "successes": 15
    },
    "api_football": {
      "requests": 95,
      "successes": 90
    }
  },
  "limits": {
    "betfair": "unlimited (free tier)",
    "odds_api": "500 requests/month",
    "api_football": "100 requests/day"
  },
  "timestamp": "2026-04-20T17:15:00+00:00"
}
```

### Command 2: API Health Check
```bash
python fetch_closing_odds_v2.py --health-check
```

**Output Example**:
```json
{
  "timestamp": "2026-04-20T17:15:00+00:00",
  "betfair": {
    "service": "Betfair",
    "authenticated": true,
    "message": "Login successful"
  },
  "odds_api": "N/A",
  "cache_dir": "data/cache/odds",
  "cache_exists": true
}
```

---

## Monthly Budget Tracking

### Odds API Monthly Budget: 500 requests

**Safe zones**:
- ✅ 0-300 req/month: Normal operation (60% headroom)
- ⚠️ 300-400 req/month: Caution (20% headroom, monitor)
- 🔴 400-500 req/month: Critical (approaching limit, investigate)

### API-Football Daily Budget: 100 requests

**Safe zones**:
- ✅ 0-60 req/day: Normal operation (40% headroom)
- ⚠️ 60-80 req/day: Caution (monitor usage)
- 🔴 80-100 req/day: Critical (approaching limit)

### Betfair: Unlimited
- ✅ Always safe (no quota limit)
- Monitor uptime via `--health-check`, not quota

---

## Alerts & Thresholds

### Automatic Alerts

The hybrid fetcher logs warnings when:

1. **Betfair unavailable**
   ```
   WARNING: Betfair fetch failed, falling back to The Odds API
   ```

2. **Odds API quota low**
   ```
   ERROR: Quota too low (50 remaining) — aborting to preserve budget
   ```

3. **API-Football limit reached**
   ```
   WARNING: API-Football rate limit reached; skipping enrichment
   ```

### Manual Quota Check

```python
import json
from pathlib import Path

tracker_path = Path("data/quota_tracker.json")
tracker = json.loads(tracker_path.read_text())

# Check today's Odds API usage
today = "2026-04-20"
if today in tracker and "odds_api" in tracker[today]:
    used_today = tracker[today]["odds_api"]["requests"]
    print(f"Odds API: {used_today} req today")

# Check this month's total
this_month_requests = 0
for date, sources in tracker.items():
    if date.startswith("2026-04"):  # April 2026
        this_month_requests += sources.get("odds_api", {}).get("requests", 0)

print(f"Odds API: {this_month_requests}/500 req this month")
remaining = 500 - this_month_requests
print(f"Remaining: {remaining} req")
```

---

## Source Selection Analysis

After running daily for a month, analyze which source was primary:

```python
import json
from collections import defaultdict
from pathlib import Path

tracker = json.loads(Path("data/quota_tracker.json").read_text())

sources = defaultdict(lambda: {"requests": 0, "successes": 0})

for date, date_sources in tracker.items():
    for source, data in date_sources.items():
        sources[source]["requests"] += data.get("requests", 0)
        sources[source]["successes"] += data.get("successes", 0)

print("Monthly Source Usage:")
for source, data in sources.items():
    req = data["requests"]
    success_rate = (data["successes"] / req * 100) if req > 0 else 0
    print(f"  {source:15} {req:3} requests ({success_rate:.1f}% success)")

# Analysis
betfair_requests = sources.get("betfair", {}).get("requests", 0)
odds_api_requests = sources.get("odds_api", {}).get("requests", 0)

if betfair_requests > 0 and odds_api_requests == 0:
    print("\n✅ Perfect: Betfair primary, zero fallbacks")
elif betfair_requests > 0 and odds_api_requests < 50:
    print(f"\n✅ Good: Betfair primary with {odds_api_requests} fallback days")
else:
    print(f"\n⚠️ Issue: High fallback rate ({odds_api_requests} req)")
    print("   Investigate Betfair connectivity or network issues")
```

---

## Quota Efficiency Metric

Define **Quota Efficiency** as:

```
Efficiency = (Requests via Betfair) / (Total Requests)
```

**Target**: > 99% (Betfair primary, <1% fallback)

```python
def calculate_efficiency(tracker_path="data/quota_tracker.json"):
    tracker = json.loads(Path(tracker_path).read_text())

    total_requests = 0
    betfair_requests = 0

    for date, sources in tracker.items():
        for source, data in sources.items():
            req = data.get("requests", 0)
            total_requests += req
            if source == "betfair":
                betfair_requests += req

    if total_requests == 0:
        return None

    efficiency = betfair_requests / total_requests * 100
    return efficiency

efficiency = calculate_efficiency()
print(f"Quota Efficiency: {efficiency:.1f}%")
if efficiency >= 99:
    print("✅ Excellent (Betfair primary, minimal fallback)")
elif efficiency >= 90:
    print("⚠️ Good (occasional fallback, normal)")
else:
    print("🔴 Investigate (high fallback rate)")
```

---

## Cron Monitoring

### Log Rotation

Add to crontab to archive old tracker data weekly:

```bash
# Archive quotas every Sunday
0 2 * * 0 python -c "
import json, shutil
from datetime import datetime, timedelta
from pathlib import Path

tracker_path = Path('data/quota_tracker.json')
if tracker_path.exists():
    tracker = json.loads(tracker_path.read_text())

    # Keep only last 60 days
    cutoff = (datetime.now() - timedelta(days=60)).date().isoformat()
    filtered = {d: v for d, v in tracker.items() if d >= cutoff}

    tracker_path.write_text(json.dumps(filtered, indent=2))
    print(f'Archived quota data older than {cutoff}')
"
```

### Daily Report

Add to crontab to log quota status:

```bash
# Email quota report daily at 9 AM
0 9 * * * cd /path/to/sports_predictor && \
    python fetch_closing_odds_v2.py --quota-report >> logs/quota.log 2>&1
```

### Quota Alert

Add to crontab to alert when approaching limits:

```bash
# Check Odds API quota daily
0 18 * * * python -c "
import json, sys
from pathlib import Path
from datetime import datetime

tracker = json.loads(Path('data/quota_tracker.json').read_text())

# Calculate this month's Odds API usage
this_month = datetime.now().strftime('%Y-%m')
month_usage = sum(
    tracker[d].get('odds_api', {}).get('requests', 0)
    for d in tracker if d.startswith(this_month)
)

remaining = 500 - month_usage
if remaining < 100:
    print(f'⚠️  WARNING: Odds API quota low: {remaining}/500 remaining')
    # Could send alert here (email, Slack, etc.)
    sys.exit(1)
" >> logs/quota_alerts.log 2>&1
```

---

## Troubleshooting

### Issue: Odds API Quota Burning Faster Than Expected

**Symptoms**:
- Odds API requests >20/day
- Betfair health check shows authenticated but requests still fallback

**Diagnosis**:
1. Check Betfair API status: https://status.betfair.com
2. Verify credentials: `python fetch_closing_odds_v2.py --health-check`
3. Check network connectivity to `api-au.betfair.com`

**Solution**:
1. Wait for Betfair to recover (usually < 30 min)
2. Manually restart Betfair session: Delete session token cache
3. If persistent, temporarily disable Betfair in `fetch_closing_odds_v2.py`:
   ```bash
   python fetch_closing_odds_v2.py --source odds_api
   ```

### Issue: API-Football Daily Quota Exhausted

**Symptoms**:
```
WARNING: API-Football rate limit reached; skipping enrichment
```

**Cause**: Too many matches enriched (100 req/day limit)

**Solution**:
- Enrich only pending matches (automatic in v2)
- Skip enrichment on low-value days:
  ```bash
  python fetch_closing_odds_v2.py  # Skip --enrich-soccer
  ```

### Issue: Betfair Login Fails

**Symptoms**:
```
ERROR: Betfair login failed: Invalid credentials
```

**Solution**:
1. Verify app key at https://developer.betfair.com/apps/
2. Confirm username/password correct
3. Check Delayed App Key approval status (may take 1-2 days)

---

## Sample Integration: Monitor Dashboard

Save as `monitor_quota.py`:

```python
#!/usr/bin/env python
"""Monitor and display quota usage dashboard."""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def main():
    tracker_path = Path("data/quota_tracker.json")
    if not tracker_path.exists():
        print("No quota data yet. Run fetch_closing_odds_v2.py first.")
        return

    tracker = json.loads(tracker_path.read_text())

    # Aggregate monthly
    this_month = datetime.now().strftime("%Y-%m")
    month_usage = defaultdict(lambda: {"requests": 0, "successes": 0})

    for date, sources in tracker.items():
        if date.startswith(this_month):
            for source, data in sources.items():
                month_usage[source]["requests"] += data.get("requests", 0)
                month_usage[source]["successes"] += data.get("successes", 0)

    # Print dashboard
    print(f"\n{'='*60}")
    print(f"API Quota Dashboard — {this_month}")
    print(f"{'='*60}\n")

    print("Source          Requests  Successes  Success %   Status")
    print("-" * 60)

    for source in ["betfair", "odds_api", "api_football"]:
        data = month_usage.get(source, {})
        requests = data.get("requests", 0)
        successes = data.get("successes", 0)
        success_rate = (successes / requests * 100) if requests > 0 else 0

        if source == "betfair":
            status = "✅ Unlimited"
        elif source == "odds_api":
            remaining = 500 - requests
            pct = requests / 500 * 100
            status = f"🟢 {remaining:3}/{500} remaining" if remaining > 100 else f"🔴 {remaining:3}/{500} remaining"
        else:  # api_football
            today = datetime.now().strftime("%Y-%m-%d")
            today_usage = tracker.get(today, {}).get("api_football", {}).get("requests", 0)
            remaining = 100 - today_usage
            status = f"🟢 {remaining:3}/{100} today" if remaining > 20 else f"🔴 {remaining:3}/{100} today"

        print(f"{source:15} {requests:8}  {successes:9}  {success_rate:6.1f}%  {status}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python monitor_quota.py
```

Output:
```
============================================================
API Quota Dashboard — 2026-04
============================================================

Source          Requests  Successes  Success %   Status
------------------------------------------------------------
betfair         300       300        100.0%    ✅ Unlimited
odds_api         15        15        100.0%    🟢  485/500 remaining
api_football     95        90         94.7%    🟢  5/100 today

============================================================
```

---

## References

- [Betfair API Status](https://status.betfair.com)
- [Odds API Docs](https://the-odds-api.com/liveapi)
- [API-Football Docs](https://www.api-football.com/)
