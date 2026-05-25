# Webapp Integration: Hybrid Quota Tracking

## Overview

The webapp now has integration modules to display hybrid quota tracking on the dashboard.

**Status**: Not yet connected (optional integration)

---

## Option 1: Minimal Integration (5 minutes)

Update `webapp/app.py` to sync hybrid quota to legacy format:

```python
# In webapp/app.py, after Flask initialization:

from src.utils.quota_api_bridge import QuotaAPIBridge

# Create bridge instance
quota_bridge = QuotaAPIBridge()

# In your main route or startup, sync quota:
@app.before_request
def sync_quota():
    """Sync hybrid quota to legacy api_usage.json before each request."""
    quota_bridge.save_legacy_api_usage()
```

This keeps the existing webapp dashboard working without changes.

---

## Option 2: Full Integration (15 minutes)

Replace quota display with new hybrid endpoints:

### Step 1: Update `webapp/app.py`

```python
from webapp.quota_dashboard_integration import register_quota_endpoints

app = Flask(__name__)
# ... existing setup ...

# Register new quota endpoints
register_quota_endpoints(app)
```

### Step 2: Update `webapp/templates/index.html`

Replace the old Odds API quota display with new hybrid display:

**Before**:
```html
<div class="api-quota">
    <div class="api-quota-bar">
        <div class="api-quota-fill" style="width: 90%"></div>
    </div>
    <div class="api-quota-text">45/500</div>
</div>
```

**After**:
```html
<div class="quota-status">
    <!-- Betfair (Unlimited) -->
    <div class="quota-item">
        <span class="quota-label">Betfair:</span>
        <span class="quota-unlimited">∞</span>
    </div>

    <!-- Odds API (500/month) -->
    <div class="quota-item">
        <span class="quota-label">Odds:</span>
        <div class="quota-bar">
            <div class="quota-fill" id="odds-fill"></div>
        </div>
        <span class="quota-value"><span id="odds-remaining">500</span>/500</span>
    </div>

    <!-- API-Football (100/day) -->
    <div class="quota-item">
        <span class="quota-label">Football:</span>
        <div class="quota-bar">
            <div class="quota-fill" id="apifootball-fill"></div>
        </div>
        <span class="quota-value"><span id="apifootball-remaining">100</span>/100</span>
    </div>
</div>
```

### Step 3: Add JavaScript to Update Display

Add to `webapp/templates/index.html` (in `<script>` section):

```javascript
// Fetch and display hybrid quota status
async function updateQuotaDisplay() {
    try {
        const res = await fetch('/api/quota/status');
        const data = res.json();

        // Update Odds API
        const oddsRemaining = data.odds_api.remaining || 500;
        const oddsPct = (1 - oddsRemaining / 500) * 100;
        document.getElementById('odds-remaining').textContent = oddsRemaining;
        const oddsFill = document.getElementById('odds-fill');
        oddsFill.style.width = oddsPct + '%';
        if (oddsPct > 80) oddsFill.classList.add('red');
        else if (oddsPct > 60) oddsFill.classList.add('yellow');

        // Update API-Football
        const apifbRemaining = data.api_football.remaining || 100;
        const apifbPct = (1 - apifbRemaining / 100) * 100;
        document.getElementById('apifootball-remaining').textContent = apifbRemaining;
        const apifbFill = document.getElementById('apifootball-fill');
        apifbFill.style.width = apifbPct + '%';
        if (apifbPct > 90) apifbFill.classList.add('red');
        else if (apifbPct > 70) apifbFill.classList.add('yellow');

    } catch (err) {
        console.error('Failed to fetch quota:', err);
    }
}

// Update on page load and every 30 seconds
updateQuotaDisplay();
setInterval(updateQuotaDisplay, 30000);
```

---

## New API Endpoints

If you enable webapp integration, these new endpoints become available:

### `/api/quota/status` (Full Status)
```bash
curl http://localhost:5000/api/quota/status
```

Returns:
```json
{
  "timestamp": "2026-04-20T17:15:00+00:00",
  "betfair": {
    "status": "active",
    "requests_today": 15,
    "requests_month": 300,
    "limit": "unlimited",
    "utilization_pct": 0
  },
  "odds_api": {
    "status": "active",
    "requests_today": 0,
    "requests_month": 15,
    "limit": 500,
    "remaining": 485,
    "utilization_pct": 3.0
  },
  "api_football": {
    "status": "idle",
    "requests_today": 0,
    "limit": 100,
    "remaining": 100,
    "utilization_pct": 0
  }
}
```

### `/api/quota/simple` (Backward Compatible)
```bash
curl http://localhost:5000/api/quota/simple
```

Returns:
```json
{
  "odds_remaining": 485,
  "odds_used_total": 15,
  "odds_start": 500,
  "betfair": "active",
  "betfair_requests": 300,
  "api_football": "0/100",
  "api_football_remaining": 100
}
```

### `/api/quota/warning` (UI Color)
```bash
curl http://localhost:5000/api/quota/warning
```

Returns:
```json
{
  "level": "green"  // or "yellow" / "red"
}
```

### `/api/quota/sync-legacy` (Backward Compatibility)
```bash
curl http://localhost:5000/api/quota/sync-legacy
```

Syncs hybrid quota data to `data/api_usage.json` for backward compatibility.

---

## Current Status

### ✅ What's Ready
- [x] `src/utils/quota_api_bridge.py` (QuotaAPIBridge class)
- [x] `webapp/quota_dashboard_integration.py` (Flask integration)
- [x] New Flask endpoints (quota status, simple, warning)
- [x] Backward compatibility with api_usage.json
- [x] Auto-sync to legacy format

### ⏳ What Needs Manual Update
- [ ] Update `webapp/app.py` to import and register endpoints (15 min)
- [ ] Update `webapp/templates/index.html` with new quota display (10 min)
- [ ] Add JavaScript to fetch and update quota display (5 min)
- [ ] Test in browser (5 min)

**Total Time**: ~35 minutes to fully integrate

---

## Integration Steps (Detailed)

### Step 1: Update Flask App (5 min)

Edit `webapp/app.py`:

```python
# Add imports at top
from webapp.quota_dashboard_integration import register_quota_endpoints

# After app = Flask(__name__), add:
register_quota_endpoints(app)

# Optional: Auto-sync on each request
from src.utils.quota_api_bridge import QuotaAPIBridge
quota_bridge = QuotaAPIBridge()

@app.before_request
def sync_quota():
    try:
        quota_bridge.save_legacy_api_usage()
    except Exception:
        pass  # Don't block request if sync fails
```

### Step 2: Update HTML Template (10 min)

Find the quota display section in `webapp/templates/index.html` (around line 100-150).

Replace:
```html
<div class="api-quota">
    <!-- OLD: Odds API only -->
</div>
```

With (from `QUOTA_DISPLAY_HTML` in `quota_dashboard_integration.py`):
```html
<div class="quota-status">
    <!-- NEW: Betfair + Odds API + API-Football -->
</div>
```

### Step 3: Add CSS (Already included)

The CSS is in the `QUOTA_DISPLAY_HTML` constant. Add to `<style>` section if not present.

### Step 4: Add JavaScript (5 min)

Add the fetch script (from `QUOTA_DISPLAY_HTML`) to the main `<script>` section.

### Step 5: Test

```bash
python run_webapp.py
# Open http://localhost:5000
# Check dashboard header for Betfair ∞, Odds API, API-Football quota bars
```

---

## Backward Compatibility

✅ **All changes are backward compatible**:
- Old `api_usage.json` still works
- Old webapp still displays quota (just reads from old file)
- New endpoints co-exist with old code
- Can enable integration gradually

**No action needed** to maintain compatibility. The hybrid fetcher automatically logs to `data/quota_tracker.json` and the bridge can sync to `data/api_usage.json` on demand.

---

## Optional: Add to Daily Sync

If you want quota synced daily without manual intervention:

Add to `fetch_closing_odds_v2.py`:

```python
# At end of main()
try:
    from src.utils.quota_api_bridge import QuotaAPIBridge
    bridge = QuotaAPIBridge()
    bridge.save_legacy_api_usage()
    logger.info("Synced quota to api_usage.json for webapp")
except Exception as exc:
    logger.warning(f"Could not sync quota to webapp: {exc}")
```

This ensures `data/api_usage.json` is updated after each fetch.

---

## Decision: Enable or Skip?

### Enable If:
- You want to see Betfair + Odds API + API-Football quotas in dashboard
- You want real-time quota monitoring in the webapp
- You want to customize quota display colors/format

### Skip If:
- Current dashboard is sufficient
- Don't need to monitor quota in real-time
- Can check quota via CLI: `python fetch_closing_odds_v2.py --quota-report`

**Recommendation**: Start with **Option 1** (minimal integration, 5 min). Upgrade to **Option 2** later if needed (another 15 min).

---

## Files Created

- `src/utils/quota_api_bridge.py` (195 lines) — Bridge between hybrid tracker and webapp
- `webapp/quota_dashboard_integration.py` (175 lines) — Flask integration + HTML example
- `WEBAPP_INTEGRATION.md` — This guide

---

## Questions?

- **Technical details**: See docstrings in `quota_api_bridge.py`
- **Integration help**: See examples in `quota_dashboard_integration.py`
- **Deployment**: See MIGRATION_SUMMARY.md
