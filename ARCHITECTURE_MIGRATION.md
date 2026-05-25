# Odds API Migration: Betfair-Primary Hybrid Architecture

## Executive Summary

**Goal**: Replace The Odds API (500 req/month quota) with a multi-source hybrid that favors Betfair (unlimited free) and intelligently falls back.

**Result**: Shift from ~450 req/month (Odds API) to ~0 req/month from Odds API, with unlimited free capacity via Betfair.

---

## Architecture Overview

### Current State (Before)
```
fetch_closing_odds.py
    ↓
OddsFetcher (The Odds API)
    ├── 500 req/month quota
    ├── 15 req per daily fetch
    ├── 450 req/month burned (90% of quota)
    └── Tight for future scaling
```

### New State (After)
```
fetch_closing_odds_v2.py
    ↓
HybridOddsFetcher
    ├── Primary: BetfairFetcher (unlimited, free tier)
    ├── Fallback: OddsFetcher (500/month, backup only)
    ├── Enrichment: APIFootballEnricher (100 req/day, optional)
    └── Quota Tracker (quota_tracker.json)
        ├── Daily logging across all sources
        ├── Monthly breakdowns
        └── Alerts when approaching limits
```

---

## Data Sources & Specifications

### 1. Betfair Exchange API (Primary)

**Status**: ✅ Free for development (Delayed App Key)

**Limits**:
- Unlimited requests (free tier)
- Data delayed 15-20 minutes (acceptable for daily closing lines)
- No monthly quota

**Coverage**:
- Soccer (all major leagues + cups)
- Basketball (NBA)
- Tennis (ATP, Grand Slams)
- MLB
- NHL
- Other sports via Betfair's full market list

**Advantages**:
- Truly unlimited (no quota burnout)
- Real matched odds + liquidity data
- Delayed tier sufficient for daily snapshots (~17:00 UTC fetch, game start 19:00+ UTC)
- Scaling independent of request volume

**Setup**:
```
1. Register free account at betfair.com
2. Request Delayed App Key (1-2 days approval)
3. Set environment:
   BETFAIR_APP_KEY=<key>
   BETFAIR_USERNAME=<username>
   BETFAIR_PASSWORD=<password>
```

### 2. The Odds API (Fallback)

**Status**: ⚠️ Fallback only (not primary)

**Limits**:
- 500 requests/month
- 1 request = 1 sport-key snapshot (entire league)
- Currently: ~450 req/month (daily 15-key fetch)

**Usage Strategy**:
- Only triggered if Betfair unavailable
- Typical fallback frequency: <1% of requests
- Preserves quota for emergency/backtest scenarios

**Implementation**:
```python
HybridOddsFetcher.fetch_odds()
    ├── Try BetfairFetcher.fetch_odds()
    │   └── If successful → return (quota cost: 0)
    ├── If failed → Try OddsFetcher.fetch_odds()
    │   └── If successful → return (quota cost: 1-15 per fetch)
    └── If all failed → Use disk cache (cost: 0)
```

### 3. API-Football (Optional Enrichment)

**Status**: ⚠️ Optional (soccer only)

**Limits**:
- 100 requests/day (free tier)
- RapidAPI key required

**Enrichment Fields**:
- Team form (recent 5 results)
- Expected Goals (xG)
- Corner statistics
- Head-to-head records
- Injury/suspension news

**Usage**:
```bash
# Basic fetch (no enrichment)
python fetch_closing_odds_v2.py

# With soccer enrichment
python fetch_closing_odds_v2.py --enrich-soccer

# Enrich only pending matches (efficient)
# (fetcher will skip non-pending matches)
```

---

## Module Structure

### New Files Created

| File | Purpose |
|------|---------|
| `src/data/betfair_fetcher.py` | Betfair API wrapper (unlimited free) |
| `src/data/hybrid_odds_fetcher.py` | Multi-source with intelligent fallback |
| `src/data/api_football_enricher.py` | Soccer stats enrichment (optional) |
| `fetch_closing_odds_v2.py` | Refactored main script (backward-compatible) |
| `ARCHITECTURE_MIGRATION.md` | This document |
| `QUOTA_TRACKING.md` | Quota monitoring guide |

### Modified Files

| File | Changes |
|------|---------|
| `settle.py` | Add fallback to hybrid fetcher (optional) |
| `daily_scan.py` | Update to use hybrid fetcher (optional) |

### Backward Compatibility

- Old `fetch_closing_odds.py` still works (uses Odds API directly)
- New `fetch_closing_odds_v2.py` is drop-in replacement
- Both write to same `data/cache/odds/` directory (compatible snapshots)

---

## Cost Analysis

### Scenario 1: Daily Closing-Line Snapshot (Current + New)

**Current (Odds API only)**:
```
Fetch frequency: 1 × daily @ 17:00 UTC
Sport keys per fetch: 15 (soccer leagues + other sports)
Requests per fetch: 15
Monthly burn: 15 × 30 = 450 req/month
Quota utilization: 450/500 = 90%
Headroom: 50 req/month (almost none)
```

**New (Hybrid with Betfair primary)**:
```
Betfair fetch: 15 requests/day @ 0 cost = 0 req/month
Fallback frequency: <1% (estimated 1 day/month)
Fallback cost: 15 req × 1 day = 15 req/month
API-Football enrichment (optional): 5 req/day on pending-only basis = ~100 req/month
Total Odds API consumption: ~15 req/month (3% utilization)
Headroom: 485 req/month (97%+ unused!)
Scaling: Unlimited scaling potential
```

### Scenario 2: Historical Backtesting (1000 matches)

**Current**:
```
Fetch all 12 soccer leagues + 4 other sports = 16 requests
Backtest over 100 days: 16 × 100 = 1,600 requests
Problem: Exceeds monthly quota (500 req/month)
Solution: Pay for API upgrade OR slow backtesting (5-6 days spread)
```

**New**:
```
Fetch via Betfair for all 100 days = 0 cost
No quota impact, can run full backtest in 1 day
Fallback + enrichment combined: still <200 req/month
Result: Backtest freely without quota concerns
```

---

## Migration Roadmap

### Phase 1: Setup & Testing (1-2 hours)
- [ ] Task #2: Register Betfair Delayed App Key
- [ ] Set environment variables
- [ ] Test BetfairFetcher.login() + basic fetch
- [ ] Validate odds output format matches OddsFetcher

### Phase 2: Hybrid Implementation (2-3 hours)
- [ ] Task #3: Complete HybridOddsFetcher
- [ ] Add rate limiting + quota tracking
- [ ] Implement fallback logic
- [ ] Test failover scenarios

### Phase 3: Enrichment (1-2 hours)
- [ ] Task #4: Deploy APIFootballEnricher
- [ ] Test soccer enrichment
- [ ] Validate API-Football quota tracking

### Phase 4: Integration (1-2 hours)
- [ ] Task #5: Refactor fetch_closing_odds.py → v2
- [ ] Update cron jobs to use v2
- [ ] Update settle.py to use hybrid (optional)

### Phase 5: Monitoring & Optimization (1 hour)
- [ ] Task #6: Deploy quota dashboard
- [ ] Set up alerts for quota limits
- [ ] Monitor source selection ratios

---

## Quota Tracking

### Tracker File Format
```json
{
  "2026-04-20": {
    "betfair": {
      "requests": 15,
      "successes": 15,
      "failures": 0,
      "last_update": "2026-04-20T17:05:00+00:00"
    },
    "odds_api": {
      "requests": 0,
      "successes": 0,
      "failures": 0
    },
    "api_football": {
      "requests": 5,
      "successes": 5,
      "failures": 0
    }
  }
}
```

### Viewing Quota Status
```bash
# Show today's usage
python fetch_closing_odds_v2.py --quota-report

# Show health of all APIs
python fetch_closing_odds_v2.py --health-check

# Dry-run (no API calls, plan only)
python fetch_closing_odds_v2.py --dry-run
```

### Monthly Budget Alert
When Odds API consumption approaches 400/500 (80% of quota):
```
⚠️  WARNING: The Odds API quota at 80%. Monthly fallback activity high.
    Suggestion: Review Betfair health; if fallback frequent, investigate.
    See QUOTA_TRACKING.md for details.
```

---

## Failover Scenarios

### Scenario: Betfair Temporarily Down
```
1. HybridOddsFetcher.fetch_odds() → BetfairFetcher fails
2. Automatically falls back to OddsFetcher
3. Logs: "Betfair fetch failed, falling back to The Odds API"
4. Quota tracker records: odds_api +1 request
5. Return odds (source = "odds_api")
6. Alert: "Using fallback; check Betfair status"
```

### Scenario: Both APIs Down
```
1. Both BetfairFetcher and OddsFetcher fail
2. HybridOddsFetcher checks disk cache
3. Loads most recent snapshot from data/cache/odds/
4. Returns stale data with warning: "Using cached data from X hours ago"
5. Settlement.py still runs, uses cache for CLV calculation
```

### Scenario: API-Football Quota Exceeded
```
1. APIFootballEnricher rate limit hit
2. fetch_odds() returns successfully (odds still there)
3. Enrichment skipped with warning
4. Logs: "API-Football rate limit reached; skipping enrichment"
5. Next day: quota resets, enrichment resumes
```

---

## Environment Setup

### Required (Free Tier Betfair)
```bash
export BETFAIR_APP_KEY="<your_delayed_app_key>"
export BETFAIR_USERNAME="<username>"
export BETFAIR_PASSWORD="<password>"
```

### Optional (Odds API - for fallback)
```bash
export ODDS_API_KEY="<your_odds_api_key>"
```

### Optional (API-Football enrichment)
```bash
export API_SPORTS_KEY="<your_rapidapi_key>"
```

### Recommended (Local Testing)
```bash
# .env file (not committed to git)
BETFAIR_APP_KEY=...
BETFAIR_USERNAME=...
BETFAIR_PASSWORD=...
ODDS_API_KEY=...
API_SPORTS_KEY=...
```

---

## Testing & Validation

### Test 1: Betfair Connectivity
```bash
python -c "
from src.data.betfair_fetcher import BetfairFetcher
bf = BetfairFetcher(sport='soccer')
print(bf.health_check())
"
```

### Test 2: Hybrid Fallback
```bash
# Unset BETFAIR credentials temporarily
unset BETFAIR_APP_KEY
python fetch_closing_odds_v2.py --dry-run --health-check
# Should show Betfair down, Odds API ready for fallback
```

### Test 3: Quota Tracking
```bash
python fetch_closing_odds_v2.py --quota-report
# Shows daily + monthly breakdown across sources
```

### Test 4: End-to-End Fetch
```bash
python fetch_closing_odds_v2.py --all-sports --dry-run
# Verify 15+ sport keys selected, no API calls made
```

---

## Performance Metrics

### Expected Latency
| Source | Latency | Reliability |
|--------|---------|-------------|
| Betfair | ~500ms (with auth) | 99.5% (uptime SLA) |
| Odds API | ~800ms | 99.9% (uptime SLA) |
| Cache | ~50ms | 100% (if exists) |

### Expected Throughput
| Operation | Time | Requests |
|-----------|------|----------|
| Daily closing-line (15 sports) | ~10s | 15 (Betfair: 0 cost) |
| Daily with enrichment | ~30s | 15 + 5 (API-Football) |
| Fallback to Odds API | ~15s | 15 (cost: 15 req) |

---

## References

- [Betfair API Docs](https://developer.betfair.com/)
- [The Odds API Docs](https://the-odds-api.com/)
- [API-Football Docs](https://www.api-football.com/)
- [QUOTA_TRACKING.md](./QUOTA_TRACKING.md)

---

## Rollback Plan

If issues arise, rollback is trivial:
```bash
# Revert to old script (still functional)
mv fetch_closing_odds.py fetch_closing_odds_old.py
git checkout fetch_closing_odds.py
# Update cron to use old script
```

Old script continues working without changes. No data loss or downtime.

---

## Future Enhancements

1. **Live Streaming**: If Betfair live tier activated (£499 one-time), upgrade to real-time odds
2. **Pinnacle Integration**: Add highest closing lines (requires funded account)
3. **Exchange Aggregation**: Compare Betfair vs Pinnacle vs bookmaker consensus
4. **Automated Alerts**: Notify when line moves > X% from prediction

---

**Status**: ✅ Ready for implementation
**Complexity**: Medium (3-4 day project)
**Risk**: Low (fallback ensures no data loss)
**Benefit**: Unlimited scaling + 97% quota headroom
