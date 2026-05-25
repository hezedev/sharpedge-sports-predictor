"""
Quota Dashboard Integration for Webapp

This module provides Flask endpoints for the webapp to display
hybrid quota tracking. Drop-in replacement for the old api_usage.json approach.

Usage in Flask app:

    from webapp.quota_dashboard_integration import register_quota_endpoints

    app = Flask(__name__)
    register_quota_endpoints(app)

Then the webapp dashboard will show:
- Betfair status (unlimited, uptime)
- Odds API quota (X/500 per month)
- API-Football quota (X/100 per day)
"""

import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify

from src.utils.quota_api_bridge import QuotaAPIBridge


def register_quota_endpoints(app: Flask) -> None:
    """
    Register Flask endpoints for quota tracking.

    Adds:
    - /api/quota/status — Full quota status across all sources
    - /api/quota/simple — Simplified format for dashboard
    - /api/quota/warning — Warning level (green/yellow/red)
    """
    bridge = QuotaAPIBridge()

    @app.route("/api/quota/status")
    def quota_status():
        """
        Get full quota status.

        Returns:
        {
            "betfair": { "status": "healthy", "requests_month": 300, ... },
            "odds_api": { "remaining": 485, "limit": 500, "utilization_pct": 3, ... },
            "api_football": { "remaining": 95, "limit": 100, ... }
        }
        """
        return jsonify(bridge.get_quota_status())

    @app.route("/api/quota/simple")
    def quota_simple():
        """
        Get simplified quota format (backward compatible with old api_usage.json).

        Returns:
        {
            "odds_remaining": 485,
            "odds_used_total": 15,
            "betfair": "healthy",
            "api_football": "5/100"
        }
        """
        return jsonify(bridge.get_simple_quota_for_webapp())

    @app.route("/api/quota/warning")
    def quota_warning():
        """
        Get warning level for UI color coding.

        Returns: { "level": "green|yellow|red" }
        """
        return jsonify({"level": bridge.get_warning_level()})

    @app.route("/api/quota/sync-legacy")
    def quota_sync_legacy():
        """
        Sync quota data to legacy api_usage.json for backward compatibility.

        Useful if other code still reads api_usage.json directly.
        """
        try:
            bridge.save_legacy_api_usage()
            return jsonify({"status": "synced"})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500


def update_dashboard_endpoint(app: Flask) -> None:
    """
    Update the existing /api/dashboard endpoint to include hybrid quota info.

    Call this to enhance the existing dashboard endpoint:

        app = Flask(__name__)
        # ... existing setup ...
        update_dashboard_endpoint(app)
    """
    bridge = QuotaAPIBridge()

    # Get the existing dashboard function
    dashboard_fn = app.view_functions.get("api_dashboard")

    def enhanced_api_dashboard():
        """Enhanced dashboard with hybrid quota info."""
        # Call original dashboard function if it exists
        original_data = {}
        if dashboard_fn:
            response = dashboard_fn()
            if hasattr(response, "get_json"):
                original_data = response.get_json() or {}
            else:
                original_data = response

        # Add hybrid quota info
        quota_status = bridge.get_quota_status()
        original_data.update({
            "quota": {
                "betfair": quota_status["betfair"],
                "odds_api": quota_status["odds_api"],
                "api_football": quota_status["api_football"],
                "warning_level": bridge.get_warning_level(),
            }
        })

        return jsonify(original_data)

    # Replace the endpoint
    if dashboard_fn:
        app.view_functions["api_dashboard"] = enhanced_api_dashboard


# Example HTML snippet for displaying in the webapp template:
QUOTA_DISPLAY_HTML = """
<!-- Quota Status Display for Dashboard Header -->
<style>
.quota-status {
    display: flex;
    gap: 15px;
    font-size: 0.85rem;
}

.quota-item {
    display: flex;
    align-items: center;
    gap: 5px;
}

.quota-bar {
    width: 60px;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.quota-fill {
    height: 100%;
    background: var(--green);
    transition: width 0.3s, background 0.3s;
}

.quota-fill.yellow { background: var(--yellow); }
.quota-fill.red { background: var(--red); }

.quota-label {
    color: var(--text2);
    font-weight: 500;
}

.quota-value {
    color: var(--text1);
    font-family: monospace;
}

.quota-unlimited {
    color: var(--green);
    font-weight: bold;
}
</style>

<div class="quota-status">
    <!-- Betfair (Unlimited) -->
    <div class="quota-item">
        <span class="quota-label">Betfair:</span>
        <span class="quota-unlimited">∞</span>
    </div>

    <!-- Odds API (500/month) -->
    <div class="quota-item">
        <span class="quota-label">Odds API:</span>
        <div class="quota-bar">
            <div class="quota-fill" id="odds-fill"></div>
        </div>
        <span class="quota-value"><span id="odds-remaining">500</span>/500</span>
    </div>

    <!-- API-Football (100/day) -->
    <div class="quota-item">
        <span class="quota-label">API-Football:</span>
        <div class="quota-bar">
            <div class="quota-fill" id="apifootball-fill"></div>
        </div>
        <span class="quota-value"><span id="apifootball-remaining">100</span>/100</span>
    </div>
</div>

<script>
// Fetch and display quota status
async function updateQuotaDisplay() {
    try {
        const res = await fetch('/api/quota/status');
        const data = await res.json();

        // Update Odds API
        const oddsRemaining = data.odds_api.remaining || 500;
        const oddsPct = (1 - oddsRemaining / 500) * 100;
        document.getElementById('odds-remaining').textContent = oddsRemaining;
        const oddsFill = document.getElementById('odds-fill');
        oddsFill.style.width = oddsPct + '%';
        oddsFill.className = 'quota-fill';
        if (oddsPct > 80) oddsFill.classList.add('red');
        else if (oddsPct > 60) oddsFill.classList.add('yellow');

        // Update API-Football
        const apifbRemaining = data.api_football.remaining || 100;
        const apifbPct = (1 - apifbRemaining / 100) * 100;
        document.getElementById('apifootball-remaining').textContent = apifbRemaining;
        const apifbFill = document.getElementById('apifootball-fill');
        apifbFill.style.width = apifbPct + '%';
        apifbFill.className = 'quota-fill';
        if (apifbPct > 90) apifbFill.classList.add('red');
        else if (apifbPct > 70) apifbFill.classList.add('yellow');

    } catch (err) {
        console.error('Failed to fetch quota status:', err);
    }
}

// Update on page load and every 30 seconds
updateQuotaDisplay();
setInterval(updateQuotaDisplay, 30000);
</script>
"""
