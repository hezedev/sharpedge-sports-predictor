// ═══════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════
const state = {
  currentPage: 'dashboard',
  filters: { sport: 'all', sports: [], market: 'all', policy: 'all', window: 'all', context: 'all', date: '' },
  picksSort: 'confidence',
  allGames: [],
  scanSports: [],
  scanMarket: 'all',
  scanRetrain: false,
  scanOfflineOdds: false,
  scanForceFreshOdds: false,
  scanLeanContext: false,
  scanContextReferee: false,
  scanFullSoccerScope: true,
  soccerGames: [],
  soccerSelected: new Set(),
  parlayLegs: [],
  parlayPanelOpen: false,
  allBets: [],
  reviewBets: [],
  lastGoodBets: [],
  lastGoodReviewBets: [],
  lastGoodGames: [],
  parlayTab: 'value',
  allParlays: [],
  manualParlays: [],
  aiValueParlays: [],
  aiLongshotParlays: [],
  lastSavedParlayId: null,
  gameSlateExpanded: {},
  pnlChart: null,
  marketChart: null,
  latestAnalysis: null,
  reasoningCandidates: [],
  marketPolicy: { preferred: [], experimental: [], disabled: [] },
  focusedPredictionLanes: { primary: [], secondary: [], controlled: [], disabled: [] },
  resultsData: null,
  selectedLane: null,
  selectedReplaySlateDate: null,
  selectedParlayCohort: null,
  activePicksSummaryDate: '',
  worldCupTeams: [],
  worldCupMeta: null,
};

// ═══════════════════════════════════════════════════════════
// NAV
// ═══════════════════════════════════════════════════════════
function showPage(page, triggerEl) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  const navTrigger = triggerEl || (typeof event !== 'undefined' ? event.currentTarget : null);
  if (navTrigger && navTrigger.classList) navTrigger.classList.add('active');

  const titles = {
    dashboard: 'Dashboard', picks: 'Picks Board', parlays: 'Parlay Desk',
    performance: 'Performance', results: 'Results Desk',
    reasoning: 'Reasoning', worldcup: 'World Cup Predictor', scan: 'Run Scan', apis: 'API Manager',
  };
  document.getElementById('page-title').textContent = titles[page] || page;
  state.currentPage = page;

  // Close mobile sidebar when navigating
  closeSidebar();

  if (page === 'dashboard')   loadDashboard();
  if (page === 'picks')       { loadPicks(); startLiveScorePolling(); }
  if (page !== 'picks')       stopLiveScorePolling();
  if (page === 'results')     { _resultsTab === 'my' ? loadMySelections() : loadResults(); }
  if (page === 'parlays')     loadParlays();
  if (page === 'performance') loadPerformance();
  if (page === 'reasoning')   loadReasoningCandidates();
  if (page === 'worldcup')    loadWorldCupPage();
  if (page === 'apis')        loadApis();
  if (page === 'scan')        { syncScanOptionAvailability(); loadScanApiUsage(); checkScanStatus(); }
}

// ── Mobile sidebar ────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebar-overlay').classList.toggle('open');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebar-overlay').classList.remove('open');
}

async function checkScanStatus() {
  // When navigating to the scan page, sync button state with server
  try {
    const r = await fetch('/api/scan/status');
    const d = await r.json();
    const btn = document.getElementById('btn-scan');
    const stopBtn = document.getElementById('btn-stop');
    const badge = document.getElementById('scan-status-badge');
    const logEl = document.getElementById('scan-log');
    if (d.running) {
      // Scan is running — attach to the live stream
      btn.disabled = true;
      btn.classList.add('running');
      btn.innerHTML = '<div class="spinner"></div> Scanning…';
      stopBtn.style.display = 'inline-flex';
      badge.textContent = '● Running';
      badge.style.color = 'var(--yellow)';
      // Replay existing log
      logEl.innerHTML = '';
      (d.log || []).forEach(line => {
        const div = document.createElement('div');
        div.className = 'log-line' + (line.includes('ERROR') ? ' err' : line.includes('WARNING') ? ' warn' : line.includes('Finished') ? ' done' : '');
        div.textContent = line;
        logEl.appendChild(div);
      });
      logEl.scrollTop = logEl.scrollHeight;
      // Re-attach SSE stream from current position
      const es = new EventSource('/api/scan/stream');
      es.onmessage = e => {
        if (e.data === '__DONE__') { es.close(); resetScanBtn(); loadDashboard(); loadScanApiUsage(); loadPicks(); badge.textContent = '✓ Complete'; badge.style.color = 'var(--green)'; return; }
        const line = JSON.parse(e.data);
        const div = document.createElement('div');
        div.className = 'log-line' + (line.includes('ERROR') ? ' err' : line.includes('WARNING') ? ' warn' : line.includes('Finished') ? ' done' : '');
        div.textContent = line;
        logEl.appendChild(div);
        logEl.scrollTop = logEl.scrollHeight;
      };
      es.onerror = () => { es.close(); resetScanBtn(); };
    } else {
      resetScanBtn();
    }
  } catch(e) { /* ignore */ }
}

// ═══════════════════════════════════════════════════════════
// DASHBOARD / HEADER
// ═══════════════════════════════════════════════════════════
async function loadDashboard() {
  try {
    const [r, myR] = await Promise.all([
      fetch('/api/dashboard'),
      fetch('/api/my-selections/results'),
    ]);
    const d   = await r.json();
    const myD = myR.ok ? await myR.json() : null;
    d._myStats = myD ? (myD.overall || myD) : null;
    document.getElementById('hdr-bankroll').textContent = '£' + d.bankroll.toLocaleString();
    document.getElementById('hdr-bets').textContent = d.total_bets;
    const t = d.scan_time ? new Date(d.scan_time).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '—';
    document.getElementById('hdr-time').textContent = t;

    // Quota topbar pill
    const rem = d.odds_remaining;
    const start = d.odds_start || rem || 1;
    const pct = start > 0 ? rem / start * 100 : 0;
    const quotaPill = document.getElementById('quota-topbar-pill');
    const quotaText = document.getElementById('quota-topbar-text');
    if (quotaText) quotaText.textContent = `${rem ?? '—'} remaining`;
    if (quotaPill) {
      quotaPill.className = 'stat-pill quota-pill' + (pct <= 10 ? ' low' : pct <= 25 ? ' warn' : '');
    }
    document.getElementById('hdr-dot').className = 'dot' + (pct > 20 ? '' : ' red');

    // Legacy sidebar elements (may not exist in new layout — safe no-ops)
    const sbBar  = document.getElementById('sb-api-bar');
    const sbText = document.getElementById('sb-api-text');
    if (sbBar)  { sbBar.style.width = pct + '%'; sbBar.style.background = pct > 50 ? 'var(--green)' : pct > 20 ? 'var(--yellow)' : 'var(--red)'; }
    if (sbText) sbText.textContent = `${rem ?? '—'} remaining`;

    // Hybrid quota status
    if (d.quota) updateHybridQuotaDisplay(d.quota);

    renderPicksDashboard(d);
    renderDashboardPage(d);
  } catch(e) { console.error(e); }
}

function renderPicksDashboard(d) {
  const el = document.getElementById('dashboard-hero');
  if (!el) return;

  const mode = d.quota_mode || 'healthy';
  const modeLabel = {
    healthy: 'Healthy',
    caution: 'Caution',
    critical: 'Critical',
  }[mode] || 'Healthy';
  const modeCopy = {
    healthy: 'All loaded Odds API keys are available for normal scanning and rotation.',
    caution: 'The active key is getting lower, but the scanner can continue using other usable keys in the pool.',
    critical: 'The active key is exhausted or nearly exhausted. Add or rotate to another usable key to keep live odds flowing.',
  }[mode] || '';

  const sports = Object.entries(d.by_sport || {})
    .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
    .slice(0, 5);
  const maxSport = sports.length ? Math.max(...sports.map(([, count]) => Number(count || 0)), 1) : 1;
  const sportMarkup = sports.length
    ? sports.map(([sport, count]) => `
        <div class="dash-sport-row">
          <div class="dash-sport-name">${escapeHtml(String(sport))}</div>
          <div class="dash-sport-track"><div class="dash-sport-fill" style="width:${(Number(count || 0) / maxSport) * 100}%"></div></div>
          <div class="dash-sport-count">${count}</div>
        </div>
      `).join('')
    : '<div class="dash-copy" style="margin-top:10px">No picks have been recorded yet for today.</div>';

  const topSport = sports.length ? sports[0][0] : '—';
  const lastScan = d.scan_time
    ? new Date(d.scan_time).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
    : '—';
  const quotaLeft = `${d.odds_remaining ?? '—'} remaining`;
  const bankroll = '£' + Number(d.bankroll || 0).toLocaleString();
  const dailyAllowance = d.odds_daily_allowance ?? '—';
  const reserveRemaining = d.odds_remaining_after_reserve ?? '—';
  const daysLeft = d.odds_days_left_in_cycle ?? '—';
  const process = d.process_summary || {};
  const avgClv = process.avg_clv == null ? '—' : `${process.avg_clv >= 0 ? '+' : ''}${(process.avg_clv * 100).toFixed(1)}%`;
  const clvWin = process.clv_positive_pct == null ? '—' : `${Math.round(process.clv_positive_pct * 100)}%`;
  const processBets = process.n_bets ?? 0;

  el.innerHTML = `
    <div class="dash-panel">
      <div class="dash-heading">
        <div>
          <div class="dash-eyebrow">System Dashboard</div>
          <div class="dash-title">Today's trading picture at a glance</div>
          <div class="dash-copy">${modeCopy}</div>
        </div>
        <div class="dash-mode ${mode}">${modeLabel}</div>
      </div>
      <div class="dash-kpi-grid">
        <div class="dash-kpi">
          <div class="dash-kpi-label">Bankroll</div>
          <div class="dash-kpi-value">${bankroll}</div>
          <div class="dash-kpi-meta">Current operating bankroll in the app.</div>
        </div>
        <div class="dash-kpi">
          <div class="dash-kpi-label">Picks Today</div>
          <div class="dash-kpi-value">${d.total_bets ?? 0}</div>
          <div class="dash-kpi-meta">Value bets currently in today's report.</div>
        </div>
        <div class="dash-kpi">
          <div class="dash-kpi-label">Odds Quota</div>
          <div class="dash-kpi-value">${quotaLeft}</div>
          <div class="dash-kpi-meta">Monthly live odds credits still available.</div>
        </div>
        <div class="dash-kpi">
          <div class="dash-kpi-label">Used Today</div>
          <div class="dash-kpi-value">${d.odds_used_today ?? 0}</div>
          <div class="dash-kpi-meta">Live odds calls recorded on the active key today.</div>
        </div>
        <div class="dash-kpi">
          <div class="dash-kpi-label">Avg CLV</div>
          <div class="dash-kpi-value">${avgClv}</div>
          <div class="dash-kpi-meta">Closing-line value across ${processBets} resolved tracked bets.</div>
        </div>
        <div class="dash-kpi">
          <div class="dash-kpi-label">CLV Win Rate</div>
          <div class="dash-kpi-value">${clvWin}</div>
          <div class="dash-kpi-meta">Share of tracked bets that beat the close.</div>
        </div>
      </div>
    </div>
    <div class="dash-panel">
      <div class="dash-eyebrow">Operational Detail</div>
      <div class="dash-title" style="font-size:1rem">Where today's activity is concentrated</div>
      <div class="dash-sports">${sportMarkup}</div>
      <div class="dash-op-list">
        <div class="dash-op-item">
          <div>
            <div class="dash-op-title">Top Sport</div>
            <div class="dash-op-copy">Sport currently contributing the most value bets.</div>
          </div>
          <div class="dash-op-val">${escapeHtml(String(topSport))}</div>
        </div>
        <div class="dash-op-item">
          <div>
            <div class="dash-op-title">Last Scan</div>
            <div class="dash-op-copy">Latest completed scan time from today's summary.</div>
          </div>
          <div class="dash-op-val">${lastScan}</div>
        </div>
        <div class="dash-op-item">
          <div>
            <div class="dash-op-title">Runtime Remaining</div>
            <div class="dash-op-copy">Remaining requests reported by the currently selected key.</div>
          </div>
          <div class="dash-op-val">${reserveRemaining}</div>
        </div>
        <div class="dash-op-item">
          <div>
            <div class="dash-op-title">Rotation Model</div>
            <div class="dash-op-copy">The scanner uses loaded keys until they are exhausted, then rotates to other usable keys.</div>
          </div>
          <div class="dash-op-val">Live Pool</div>
        </div>
      </div>
    </div>
  `;
}

function updateHybridQuotaDisplay(quota) {
  // Update Betfair display (should show ∞)
  const betfairEl = document.getElementById('quota-betfair');
  if (betfairEl) {
    const status = quota.betfair.status === 'active' ? '✓' : '—';
    betfairEl.textContent = status + ' Betfair';
    betfairEl.title = `Betfair: ${quota.betfair.status} (${quota.betfair.requests_month} req/month)`;
  }

  // Update Odds API quota bar
  const oddsEl = document.getElementById('quota-odds');
  if (oddsEl) {
    const remaining = quota.odds_api.remaining || 500;
    const pct = (500 - remaining) / 500 * 100;
    oddsEl.textContent = `Odds API: ${remaining}/500`;
    oddsEl.title = `The Odds API: ${remaining}/500 remaining (${pct.toFixed(1)}%)`;
  }

  // Update API-Football quota
  const apifbEl = document.getElementById('quota-apifootball');
  if (apifbEl) {
    const remaining = quota.api_football.remaining || 100;
    apifbEl.textContent = `Football: ${remaining}/100`;
    apifbEl.title = `API-Football: ${remaining}/100 today`;
  }

  // Update warning level (color indicator)
  if (quota.warning_level) {
    const dotEl = document.getElementById('quota-warning-dot');
    if (dotEl) {
      dotEl.style.background =
        quota.warning_level === 'red' ? 'var(--red)' :
        quota.warning_level === 'yellow' ? 'var(--yellow)' :
        'var(--green)';
      dotEl.title = `Quota warning: ${quota.warning_level}`;
    }
  }
}

function renderScanNotes(notes) {
  const el = document.getElementById('scan-notes-banner');
  if (!el) return;
  const items = Array.isArray(notes) ? notes : [];
  if (!items.length) {
    el.style.display = 'none';
    el.innerHTML = '';
    return;
  }
  el.style.display = '';
  el.innerHTML = items.map(note => {
    if (note.type === 'market_coverage') {
      const rows = note.by_sport && typeof note.by_sport === 'object' ? note.by_sport : {};
      const order = ['soccer', 'basketball', 'mlb', 'nhl', 'tennis', 'tennis_wta'];
      const labels = {
        soccer: 'Soccer',
        basketball: 'NBA',
        mlb: 'MLB',
        nhl: 'NHL',
        tennis: 'ATP',
        tennis_wta: 'WTA',
      };
      const prettyMarket = (key) => String(key || '').replaceAll('_', ' ');
      const parts = order
        .filter(key => Array.isArray(rows[key]) && rows[key].length)
        .map(key => `${labels[key] || key}: ${rows[key].map(prettyMarket).join(', ')}`);
      return `
        <div class="review-info-item">
          <span class="review-info-tag preferred">Markets</span>
          ${escapeHtml(note.reason || 'Markets seen in the latest scan.')}
          ${parts.length ? `<div style="margin-top:4px;color:var(--text3);font-size:.74rem">${escapeHtml(parts.join(' || '))}</div>` : ''}
        </div>`;
    }
    if (note.type === 'sport_funnel') {
      const rows = note.by_sport && typeof note.by_sport === 'object' ? note.by_sport : {};
      const order = ['soccer', 'basketball', 'mlb', 'nhl', 'tennis', 'tennis_wta'];
      const labels = {
        soccer: 'Soccer',
        basketball: 'NBA',
        mlb: 'MLB',
        nhl: 'NHL',
        tennis: 'ATP',
        tennis_wta: 'WTA',
      };
      const parts = order
        .filter(key => rows[key])
        .map(key => {
          const row = rows[key] || {};
          const noCandidate = row.no_candidate_reason_breakdown && typeof row.no_candidate_reason_breakdown === 'object'
            ? Object.entries(row.no_candidate_reason_breakdown)
                .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
                .slice(0, 2)
                .map(([reason, count]) => `${String(reason).replaceAll('_', ' ')} ${count}`)
                .join(', ')
            : '';
          return `${labels[key] || key}: scanned ${row.scanned_games || 0}, published ${row.published_games || 0}, review ${row.review_games || 0}, suppressed ${row.suppressed_games || 0}, no candidate ${row.no_candidate_games || 0}${noCandidate ? ` [top: ${noCandidate}]` : ''}`;
        });
      return `
        <div class="review-info-item">
          <span class="review-info-tag limited">Funnel</span>
          ${escapeHtml(note.reason || 'How each sport moved through the scan funnel.')}
          ${parts.length ? `<div style="margin-top:4px;color:var(--text3);font-size:.74rem">${escapeHtml(parts.join(' || '))}</div>` : ''}
        </div>`;
    }
    if (note.type === 'full_soccer_scope') {
      return `
        <div class="review-info-item">
          <span class="review-info-tag production">Full scope</span>
          ${escapeHtml(note.reason || 'Full soccer scope override kept review-only leagues in the live scan.')}
        </div>`;
    }
    if (note.type === 'sport_scan_counts') {
      const counts = note.counts && typeof note.counts === 'object' ? note.counts : {};
      const order = ['soccer', 'basketball', 'mlb', 'nhl', 'tennis', 'tennis_wta'];
      const labels = {
        soccer: 'Soccer',
        basketball: 'NBA',
        mlb: 'MLB',
        nhl: 'NHL',
        tennis: 'ATP',
        tennis_wta: 'WTA',
      };
      const parts = order
        .filter(key => counts[key] != null)
        .map(key => `${labels[key] || key}: ${counts[key]}`);
      return `
        <div class="review-info-item">
          <span class="review-info-tag production">Coverage</span>
          ${escapeHtml(note.reason || 'Current full-game scan coverage by supported sport lane.')}
          ${parts.length ? `<div style="margin-top:4px;color:var(--text3);font-size:.74rem">${escapeHtml(parts.join(' | '))}</div>` : ''}
        </div>`;
    }
    if (note.type === 'deferred_leagues') {
      const leagues = Array.isArray(note.leagues) ? note.leagues : [];
      const preview = leagues.slice(0, 6).join(', ');
      const suffix = leagues.length > 6 ? ' …' : '';
      return `
        <div class="review-info-item">
          <span class="review-info-tag stale">Speed mode</span>
          Deferred ${note.count || 0} ${escapeHtml(note.sport || 'league')} review-only league${Number(note.count || 0) === 1 ? '' : 's'} to keep the live scan faster and protect quota.
          ${preview ? `<div style="margin-top:4px;color:var(--text3);font-size:.74rem">Deferred: ${escapeHtml(preview + suffix)}</div>` : ''}
        </div>`;
    }
    if (note.type === 'discovered_soccer_leagues') {
      const leagues = Array.isArray(note.leagues) ? note.leagues : [];
      const preview = leagues.slice(0, 6).join(', ');
      const suffix = leagues.length > 6 ? ' …' : '';
      return `
        <div class="review-info-item">
          <span class="review-info-tag limited">Discovered</span>
          ${escapeHtml(note.reason || 'Active soccer markets discovered from the odds feed were scanned as review-only coverage.')}
          ${preview ? `<div style="margin-top:4px;color:var(--text3);font-size:.74rem">Leagues: ${escapeHtml(preview + suffix)}</div>` : ''}
        </div>`;
    }
    return `
      <div class="review-info-item">
        <span class="review-info-tag stale">Scan note</span>
        ${escapeHtml(note.reason || 'A scan-time optimization was applied.')}
      </div>`;
  }).join('');
}

function formatPicksReportDate(dateText) {
  const raw = String(dateText || '').trim();
  if (!raw) return '';
  const parsed = new Date(`${raw}T12:00:00`);
  if (Number.isNaN(parsed.getTime())) return raw;
  return new Intl.DateTimeFormat('de-AT', {
    timeZone: 'Europe/Vienna',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  }).format(parsed);
}

function formatAustrianEventDateTime(item) {
  const text = String(item.commence_local || item.commence || item.commence_time || '').trim();
  if (!text) return '';
  const parsed = new Date(text);
  if (Number.isNaN(parsed.getTime())) return '';
  return new Intl.DateTimeFormat('de-AT', {
    timeZone: 'Europe/Vienna',
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(parsed).replace(',', ' ·');
}

function updatePicksSummaryCopy() {
  const eyebrowEl = document.getElementById('picks-summary-eyebrow');
  const titleEl = document.getElementById('picks-summary-title');
  if (!eyebrowEl || !titleEl) return;
  const reportDate = formatPicksReportDate(state.activePicksSummaryDate || state.filters.date || '');
  eyebrowEl.textContent = reportDate ? `Picks Focus · Report ${reportDate}` : 'Picks Focus';
  titleEl.textContent = reportDate
    ? 'Production picks from the selected report date, with review candidates shown separately when they do not clear the live gates.'
    : 'Production picks first, with held or review candidates shown separately when they do not clear the live gates.';
}

// ═══════════════════════════════════════════════════════════
// PICKS
// ═══════════════════════════════════════════════════════════
async function loadPicks() {
  const params = new URLSearchParams();
  const selectedSports = Array.isArray(state.filters.sports) ? state.filters.sports : [];
  if (selectedSports.length) {
    selectedSports.forEach(sport => params.append('sport', sport));
  } else if (state.filters.sport !== 'all') {
    params.append('sport', state.filters.sport);
  }
  if (state.filters.market !== 'all') params.append('market', state.filters.market);
  if (state.filters.window !== 'all') params.append('window', state.filters.window);
  if (state.filters.date) params.append('date', state.filters.date);
  const dateInput = document.getElementById('picks-date');
  if (dateInput) dateInput.value = state.filters.date || '';
  updatePicksWindowLabel();
  updatePicksTierLabel();
  updatePicksSummaryCopy();

  loadAllGames();

  // Soccer detail panel stays folded until the user opens it
  const showSoccer = selectedSports.length ? selectedSports.includes('soccer') : state.filters.sport === 'all' || state.filters.sport === 'soccer';
  const soccerSection = document.getElementById('soccer-games-section');
  soccerSection.style.display = showSoccer ? 'block' : 'none';
  if (!showSoccer) soccerSection.classList.remove('open');
  if (showSoccer && soccerSection.classList.contains('open')) loadSoccerGames();

  const fetchPicksPayload = async (queryParams, allowLatestFallback = true) => {
    const suffix = queryParams.toString();
    const r = await fetch('/api/picks' + (suffix ? `?${suffix}` : ''));
    if (!r.ok) throw new Error(`picks_http_${r.status}`);
    const d = await r.json();
    if (d && !d.error) return d;
    if (allowLatestFallback && queryParams.get('date')) {
      const retryParams = new URLSearchParams(queryParams);
      retryParams.delete('date');
      const retry = await fetch('/api/picks?' + retryParams.toString());
      if (!retry.ok) throw new Error(`picks_http_${retry.status}`);
      const retryJson = await retry.json();
      if (retryJson && !retryJson.error) return retryJson;
      throw new Error(retryJson?.error || d?.error || 'picks_unavailable');
    }
    throw new Error(d?.error || 'picks_unavailable');
  };

  try {
    const d = await fetchPicksPayload(params, true);
    state.activePicksSummaryDate = String(d.summary_date || state.filters.date || '');
    updatePicksWindowLabel();
    updatePicksSummaryCopy();
    const incoming = d.bets || [];
    const incomingReview = d.review_bets || [];
    renderScanNotes(d.scan_notes || []);
    state.allBets = applyPicksBoardFilters(incoming);
    state.reviewBets = applyPicksBoardFilters(incomingReview);
    state.lastGoodBets = Array.isArray(incoming) ? incoming.slice() : [];
    state.lastGoodReviewBets = Array.isArray(incomingReview) ? incomingReview.slice() : [];
    state.marketPolicy = d.market_policy || state.marketPolicy;
    state.focusedPredictionLanes = d.focused_prediction_lanes || state.focusedPredictionLanes;
    renderMarketPolicy(state.marketPolicy, state.focusedPredictionLanes);
    try {
      renderBets(state.allBets);
    } catch (e) {
      console.error('renderBets failed', e, state.allBets);
      document.getElementById('bets-list').innerHTML = '<div class="empty-state"><p>Could not render value bets.</p></div>';
    }
    try {
      renderReviewBets(state.reviewBets);
    } catch (e) {
      console.error('renderReviewBets failed', e, state.reviewBets);
      document.getElementById('review-bets-list').innerHTML = '<div class="empty-state"><p>Could not render review queue.</p></div>';
    }
  } catch(e) {
    console.error('loadPicks failed', e);
    if (state.lastGoodBets.length || state.lastGoodReviewBets.length) {
      state.allBets = applyPicksBoardFilters(state.lastGoodBets);
      state.reviewBets = applyPicksBoardFilters(state.lastGoodReviewBets);
      renderBets(state.allBets);
      renderReviewBets(state.reviewBets);
      const betsList = document.getElementById('bets-list');
      const reviewList = document.getElementById('review-bets-list');
      if (betsList) betsList.insertAdjacentHTML('afterbegin', '<div class="empty-state" style="margin-bottom:10px"><p>Live picks refresh failed, showing the last good board.</p></div>');
      if (reviewList) reviewList.insertAdjacentHTML('afterbegin', '<div class="empty-state" style="margin-bottom:10px"><p>Live review refresh failed, showing the last good queue.</p></div>');
      return;
    }
    state.activePicksSummaryDate = '';
    updatePicksSummaryCopy();
    document.getElementById('bets-list').innerHTML = '<div class="empty-state"><p>Failed to load picks. Is the scan run today?</p></div>';
    document.getElementById('review-bets-list').innerHTML = '<div class="empty-state"><p>Failed to load review queue.</p></div>';
  }
}

function hasContextSignal(bet, signalName) {
  if (signalName === 'all') return true;
  const adjustments = Array.isArray(bet.context_adjustments) ? bet.context_adjustments : [];
  return adjustments.some(item => String(item.name || '') === signalName);
}

function kickoffSortValue(bet) {
  const text = String(bet.commence_time || bet.commence || '').trim();
  if (!text) return Number.MAX_SAFE_INTEGER;
  const ts = Date.parse(text);
  return Number.isFinite(ts) ? ts : Number.MAX_SAFE_INTEGER;
}

function eventDisplayMeta(item) {
  const status = String(item.status || '').toLowerCase();
  const statusLabel = String(item.status_label || '').trim().toUpperCase();
  const window = String(item.window || '').toLowerCase();
  const timeLabel = String(item.time_label || item.kick_off || '').trim();
  const liveLike = status === 'live';
  const playedLike = status === 'played';
  let bucket = 'upcoming';
  if (liveLike) bucket = 'live';
  else if (playedLike) bucket = 'played';
  else if (window === 'today') bucket = 'today';
  else if (window === 'tomorrow') bucket = 'tomorrow';
  else if (window === 'day_after') bucket = 'day_after';
  return {
    bucket,
    status,
    statusLabel: statusLabel || (liveLike ? 'LIVE' : playedLike ? 'PLAYED' : ''),
    timeLabel: timeLabel && timeLabel !== statusLabel ? timeLabel : '',
  };
}

function eventBadgeHtml(item) {
  const meta = eventDisplayMeta(item);
  if (meta.bucket === 'live') return `<span class="tag tag-window" style="background:rgba(239,68,68,.12);color:#f87171">LIVE</span>`;
  if (meta.bucket === 'played') return `<span class="tag tag-window" style="background:rgba(148,163,184,.14);color:var(--text2)">Played</span>`;
  if (meta.bucket === 'today') return `<span class="tag tag-window">Today</span>`;
  if (meta.bucket === 'tomorrow') return `<span class="tag tag-window tomorrow">Tomorrow</span>`;
  if (meta.bucket === 'day_after') return `<span class="tag tag-window tomorrow">Day After</span>`;
  return `<span class="tag tag-window tomorrow">Upcoming</span>`;
}

function launchBadgeHtml(item) {
  const label = String(item.launch_label || '').trim();
  if (!label) return '';
  const tone = label === 'Production'
    ? 'background:rgba(16,185,129,.12);color:#34d399'
    : label === 'Limited'
      ? 'background:rgba(245,158,11,.12);color:#fbbf24'
      : 'background:rgba(148,163,184,.14);color:var(--text2)';
  const note = escapeHtml(String(item.launch_note || ''));
  return `<span class="tag" style="${tone}"${note ? ` title="${note}"` : ''}>${escapeHtml(label)}</span>`;
}

function contextSortScore(bet) {
  const adjustments = Array.isArray(bet.context_adjustments) ? bet.context_adjustments : [];
  return adjustments.filter(item => item && ['lineup', 'schedule', 'matchup', 'coaching', 'environment', 'motivation'].includes(item.category)).length;
}

function evidenceStatusSortScore(bet) {
  const status = String(
    bet?.committee?.research_mind?.evidence_status ||
    bet?.research_mind_evidence_status ||
    ''
  ).toUpperCase();
  if (status === 'COMPLETE') return 4;
  if (status === 'ACCEPTABLE') return 3;
  if (status === 'PARTIAL') return 2;
  if (status === 'INSUFFICIENT') return 1;
  return 0;
}

function concreteEvidenceSortScore(bet) {
  const concrete = Number(
    bet?.committee?.research_mind?.concrete_info_score ??
    bet?.research_mind_concrete_info_score ??
    -1
  );
  const sources = Number(
    bet?.committee?.research_mind?.source_count ??
    bet?.research_mind_source_count ??
    0
  );
  return { concrete, evidence: evidenceStatusSortScore(bet), sources, context: contextSortScore(bet) };
}

function applyPicksBoardFilters(bets) {
  const filtered = (state.filters.policy === 'all'
    ? bets
    : bets.filter(b => (b.market_status || 'experimental') === state.filters.policy)
  ).filter(b => hasContextSignal(b, state.filters.context));

  const sorted = filtered.slice();
  if (state.picksSort === 'confidence') {
    sorted.sort((a, b) => Number(b.ml_prob || 0) - Number(a.ml_prob || 0) || Number(b.edge || 0) - Number(a.edge || 0));
  } else if (state.picksSort === 'evidence') {
    sorted.sort((a, b) => {
      const aEvidence = concreteEvidenceSortScore(a);
      const bEvidence = concreteEvidenceSortScore(b);
      return bEvidence.concrete - aEvidence.concrete ||
        bEvidence.evidence - aEvidence.evidence ||
        bEvidence.sources - aEvidence.sources ||
        bEvidence.context - aEvidence.context ||
        Number(b.edge || 0) - Number(a.edge || 0) ||
        Number(b.ml_prob || 0) - Number(a.ml_prob || 0);
    });
  } else if (state.picksSort === 'kickoff') {
    sorted.sort((a, b) => kickoffSortValue(a) - kickoffSortValue(b) || Number(b.edge || 0) - Number(a.edge || 0));
  } else if (state.picksSort === 'context') {
    sorted.sort((a, b) => contextSortScore(b) - contextSortScore(a) || Number(b.edge || 0) - Number(a.edge || 0));
  } else {
    sorted.sort((a, b) => Number(b.edge || 0) - Number(a.edge || 0) || Number(b.ml_prob || 0) - Number(a.ml_prob || 0));
  }
  return sorted;
}

function renderMarketPolicy(policy, focusedLanes) {
  const el = document.getElementById('policy-strip');
  if (!el) return;
  const focusedItems = [
    ...(focusedLanes?.primary || []),
    ...(focusedLanes?.secondary || []),
    ...(focusedLanes?.controlled || []),
  ].slice(0, 8);
  const focusCard = `
    <div class="policy-card">
      <div class="policy-card-head">
        <div class="policy-card-title">Focused Prediction Lanes</div>
        <div class="policy-card-count">${focusedItems.length}</div>
      </div>
      <div class="policy-list">
        ${focusedItems.length
          ? focusedItems.map(item => `
            <div class="policy-item">
              <div class="policy-item-dot preferred"></div>
              <div class="policy-item-main">
                <div class="policy-item-name">${escapeHtml(item.sport.toUpperCase())} · ${escapeHtml(item.market.replaceAll('_', ' '))}</div>
                <div class="policy-item-copy">${escapeHtml(item.label || item.status || 'Focus')} · Quality floor ${escapeHtml(String(item.quality_floor || ''))}</div>
              </div>
            </div>`).join('')
          : '<div class="policy-item-copy">No focused lanes configured yet.</div>'}
      </div>
    </div>`;
  const sections = [
    { key: 'preferred', title: 'Preferred Markets' },
    { key: 'experimental', title: 'Experimental Markets' },
    { key: 'disabled', title: 'Suppressed Markets' },
  ];
  el.innerHTML = focusCard + sections.map(section => {
    const items = (policy?.[section.key] || []).slice(0, 4);
    const rows = items.length
      ? items.map(item => `
          <div class="policy-item">
            <div class="policy-item-dot ${section.key}"></div>
            <div class="policy-item-main">
              <div class="policy-item-name">${escapeHtml(item.sport.toUpperCase())} · ${escapeHtml(item.market.replaceAll('_', ' '))}</div>
              <div class="policy-item-copy">${escapeHtml(item.reason || '')}</div>
            </div>
          </div>`).join('')
      : '<div class="policy-item-copy">No markets in this bucket yet.</div>';
    return `
      <div class="policy-card">
        <div class="policy-card-head">
          <div class="policy-card-title">${section.title}</div>
          <div class="policy-card-count">${(policy?.[section.key] || []).length}</div>
        </div>
        <div class="policy-list">${rows}</div>
      </div>`;
  }).join('');
}

async function loadAllGames() {
  const params = new URLSearchParams();
  const selectedSports = Array.isArray(state.filters.sports) ? state.filters.sports : [];
  if (selectedSports.length) {
    selectedSports.forEach(sport => params.append('sport', sport));
  } else if (state.filters.sport !== 'all') {
    params.append('sport', state.filters.sport);
  }
  if (state.filters.market !== 'all') params.append('market', state.filters.market);
  if (state.filters.window !== 'all') params.append('window', state.filters.window);
  if (state.filters.date) params.append('date', state.filters.date);
  // Show loading state immediately so user sees the list is refreshing
  const el = document.getElementById('all-games-list');
  if (el) el.innerHTML = '<div style="color:var(--text3);font-size:.8rem;padding:8px;opacity:.6">Loading…</div>';
  const countEl = document.getElementById('games-count');
  if (countEl) countEl.textContent = '…';
  const fetchGamesPayload = async (queryParams, allowLatestFallback = true) => {
    const suffix = queryParams.toString();
    const r = await fetch('/api/games' + (suffix ? `?${suffix}` : ''));
    if (!r.ok) throw new Error(`games_http_${r.status}`);
    const d = await r.json();
    if (d && !d.error) return d;
    if (allowLatestFallback && queryParams.get('date')) {
      const retryParams = new URLSearchParams(queryParams);
      retryParams.delete('date');
      const retry = await fetch('/api/games?' + retryParams.toString());
      if (!retry.ok) throw new Error(`games_http_${retry.status}`);
      const retryJson = await retry.json();
      if (retryJson && !retryJson.error) return retryJson;
      throw new Error(retryJson?.error || d?.error || 'games_unavailable');
    }
    throw new Error(d?.error || 'games_unavailable');
  };

  try {
    const d = await fetchGamesPayload(params, true);
    if (!state.activePicksSummaryDate) {
      state.activePicksSummaryDate = String(d.summary_date || '');
      updatePicksWindowLabel();
      updatePicksSummaryCopy();
    }
    state.allGames = d.games || [];
    state.lastGoodGames = Array.isArray(d.games) ? d.games.slice() : [];
    renderAllGames(state.allGames);
  } catch(e) {
    if (state.lastGoodGames.length) {
      state.allGames = state.lastGoodGames.slice();
      renderAllGames(state.allGames);
      if (el) el.insertAdjacentHTML('afterbegin', '<div style="color:var(--text3);font-size:.8rem;padding:8px">Live games refresh failed, showing the last good slate.</div>');
      return;
    }
    if (el) el.innerHTML = '<div style="color:var(--text3);font-size:.8rem;padding:8px">Could not load games.</div>';
  }
}

const SPORT_ICONS = { soccer:'⚽', basketball:'🏀', mlb:'⚾', nhl:'🏒', tennis:'🎾' };

function updatePicksWindowLabel() {
  const labels = {
    all: 'All windows',
    today: 'Today only',
    tomorrow: 'Tomorrow only',
    day_after: 'Day after',
  };
  const el = document.getElementById('picks-window-label');
  if (el) {
    if (state.filters.date) {
      const selected = formatPicksReportDate(state.filters.date);
      el.textContent = selected || state.filters.date;
    } else {
      const base = labels[state.filters.window] || state.filters.window || 'All windows';
      const reportDateLabel = formatPicksReportDate(state.activePicksSummaryDate);
      const reportDate = reportDateLabel ? ` · report ${reportDateLabel}` : '';
      el.textContent = `${base}${reportDate}`;
    }
  }
}

function changePicksDate(value) {
  state.filters.date = String(value || '').trim();
  state.activePicksSummaryDate = state.filters.date || '';
  updatePicksSummaryCopy();
  loadPicks();
}

function clearPicksDate() {
  state.filters.date = '';
  state.activePicksSummaryDate = '';
  updatePicksSummaryCopy();
  const el = document.getElementById('picks-date');
  if (el) el.value = '';
  loadPicks();
}

function updatePicksTierLabel() {
  const labels = {
    all: 'All active',
    preferred: 'Preferred only',
    experimental: 'Experimental only',
  };
  const el = document.getElementById('picks-tier-label');
  if (el) el.textContent = labels[state.filters.policy] || 'All active';
}

function setPicksCount(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(value);
}

function toggleFoldPanel(sectionId) {
  const el = document.getElementById(sectionId);
  if (!el) return;
  const opening = !el.classList.contains('open');
  el.classList.toggle('open', opening);
  if (opening && sectionId === 'soccer-games-section') {
    loadSoccerGames();
  }
}

function toggleSlateDay(groupKey) {
  state.gameSlateExpanded[groupKey] = !state.gameSlateExpanded[groupKey];
  renderAllGames(state.allGames || []);
}

function renderAllGames(games) {
  const el = document.getElementById('all-games-list');
  setPicksCount('games-count', games.length);
  setPicksCount('games-count-pill', games.length);
  if (!games.length) {
    el.innerHTML = '<div style="color:var(--text3);font-size:.8rem;padding:8px">No games found for these filters.</div>';
    return;
  }

  // Group by window
  const groups = { live: [], today: [], tomorrow: [], day_after: [], upcoming: [], played: [] };
  games.forEach(g => {
    groups[eventDisplayMeta(g).bucket].push(g);
  });

  const groupMeta = {
    live: { title: 'Live', copy: 'Games that should be in progress right now.' },
    today: { title: 'Today', copy: 'Closest kickoff window for the current filters.' },
    tomorrow: { title: 'Tomorrow', copy: 'Next-day matches still on the slate.' },
    day_after: { title: 'Day After', copy: 'Further-out matches inside the saved scan window.' },
    upcoming: { title: 'Upcoming', copy: 'Other matches outside the named day buckets.' },
    played: { title: 'Played', copy: 'Games whose scheduled live window has already passed.' },
  };
  let html = '';
  for (const key of ['live', 'today', 'tomorrow', 'day_after', 'upcoming', 'played']) {
    const list = groups[key];
    if (!list.length) continue;
    const isOpen = !!state.gameSlateExpanded[key];
    const rows = list.map(g => {
      const icon = SPORT_ICONS[g.sport] || '🎯';
      const leagueTxt = g.league || g.sport.toUpperCase();
      const timing = eventDisplayMeta(g);
      const badge = eventBadgeHtml(g);
      const launchBadge = launchBadgeHtml(g);
      const time = timing.timeLabel || '';
      const austrianDateTime = formatAustrianEventDateTime(g);
      const pickBadge = g.model_pick ? `<span class="game-row-pick">★ ${g.model_pick}</span>` : '';
      const valueDot = g.has_value ? `<span title="Value bet found" style="width:7px;height:7px;border-radius:50%;background:#34d399;flex-shrink:0"></span>` : '';
      const playoffBadge = g.is_playoff ? `<span class="tag tag-playoff" style="font-size:.58rem;padding:1px 6px">🏆 PO</span>` : '';
      const abstainBadge = g.abstain ? `<span class="tag tag-abstain" style="font-size:.58rem;padding:1px 6px" title="Line moved sharply — abstain">⚡ Abstain</span>` : '';
      const basketballProbDebug = g.basketball_probability_debug || null;
      const nhlProbDebug = g.nhl_probability_debug || null;
      const mlbProbDebug = g.mlb_probability_debug || null;
      const basketballProbSummary = basketballProbDebug && g.sport === 'basketball'
        ? (() => {
            const pickKey = g.model_pick && String(g.model_pick).toLowerCase() === String(g.home || '').toLowerCase() ? 'home' : 'away';
            const fmtProb = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
            const regime = basketballProbDebug.regime ? ` · ${String(basketballProbDebug.regime).replace(/_/g, ' ')}` : '';
            const gap = basketballProbDebug.disagreement_pp != null ? ` · gap ${basketballProbDebug.disagreement_pp}pp` : '';
            return `Model check · classifier ${fmtProb(basketballProbDebug.classifier_probs?.[pickKey])} · structural ${fmtProb(basketballProbDebug.structural_probs?.[pickKey])} · final ${fmtProb(basketballProbDebug.final_probs?.[pickKey])} · market ${fmtProb(basketballProbDebug.market_probs?.[pickKey])}${regime}${gap}`;
          })()
        : '';
      const nhlProbSummary = nhlProbDebug && g.sport === 'nhl'
        ? (() => {
            const pickKey = g.model_pick && String(g.model_pick).toLowerCase() === String(g.home || '').toLowerCase() ? 'home' : 'away';
            const fmtProb = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
            const regime = nhlProbDebug.regime ? ` · ${String(nhlProbDebug.regime).replace(/_/g, ' ')}` : '';
            const gap = nhlProbDebug.disagreement_pp != null ? ` · gap ${nhlProbDebug.disagreement_pp}pp` : '';
            return `Model check · classifier ${fmtProb(nhlProbDebug.classifier_probs?.[pickKey])} · structural ${fmtProb(nhlProbDebug.structural_probs?.[pickKey])} · final ${fmtProb(nhlProbDebug.final_probs?.[pickKey])} · market ${fmtProb(nhlProbDebug.market_probs?.[pickKey])}${regime}${gap}`;
          })()
        : '';
      const mlbProbSummary = mlbProbDebug && g.sport === 'mlb'
        ? (() => {
            const pickKey = g.model_pick && String(g.model_pick).toLowerCase() === String(g.home || '').toLowerCase() ? 'home' : 'away';
            const fmtProb = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
            const regime = mlbProbDebug.regime ? ` · ${String(mlbProbDebug.regime).replace(/_/g, ' ')}` : '';
            const gap = mlbProbDebug.disagreement_pp != null ? ` · gap ${mlbProbDebug.disagreement_pp}pp` : '';
            return `Model check · classifier ${fmtProb(mlbProbDebug.classifier_probs?.[pickKey])} · structural ${fmtProb(mlbProbDebug.structural_probs?.[pickKey])} · final ${fmtProb(mlbProbDebug.final_probs?.[pickKey])} · market ${fmtProb(mlbProbDebug.market_probs?.[pickKey])}${regime}${gap}`;
          })()
        : '';
      const boardStatus = String(g.board_status || 'no_candidate');
      const boardLabels = {
        published: 'On board',
        review: 'Review',
        suppressed: 'Suppressed',
        bankroll_blocked: 'Stake blocked',
        no_candidate: 'Passed',
      };
      const boardClass = boardStatus === 'published'
        ? 'production'
        : boardStatus === 'review' || boardStatus === 'bankroll_blocked'
          ? 'limited'
          : 'review';
      const bestCandidate = g.best_candidate || {};
      const fmtSignedPct = (v) => {
        if (v === null || v === undefined || v === '') return '';
        const n = Number(v);
        return Number.isFinite(n) ? `${n >= 0 ? '+' : ''}${(n * 100).toFixed(1)}pp` : '';
      };
      const blockerParts = Array.isArray(g.missing_to_promote) ? g.missing_to_promote.slice(0, 3) : [];
      const boardReason = String(g.board_reason || '').trim();
      const boardSummaryBits = [];
      if (bestCandidate.team) boardSummaryBits.push(String(bestCandidate.team));
      if (bestCandidate.market) boardSummaryBits.push(String(bestCandidate.market).replace(/_/g, ' '));
      const bestEdge = fmtSignedPct(bestCandidate.edge);
      if (bestEdge) boardSummaryBits.push(`edge ${bestEdge}`);
      if (boardReason) boardSummaryBits.push(boardReason);
      if (blockerParts.length) boardSummaryBits.push(`needs ${blockerParts.join(' | ')}`);
      const boardSummary = boardSummaryBits.join(' · ');
      // Format odds
      let oddsTxt = '';
      if (g.sport === 'soccer' && g.home_odds) {
        const drawPart = g.draw_odds ? ` · D ${g.draw_odds.toFixed(2)}` : '';
        oddsTxt = `H ${g.home_odds.toFixed(2)}${drawPart} · A ${g.away_odds ? g.away_odds.toFixed(2) : '—'}`;
      } else if (g.home_odds || g.away_odds) {
        const h = g.home_odds ? g.home_odds.toFixed(2) : '—';
        const a = g.away_odds ? g.away_odds.toFixed(2) : '—';
        oddsTxt = `${h} / ${a}`;
      }
      return `<div class="game-row">
        <span class="game-row-sport">${icon}</span>
          <div class="game-row-teams">
          <div class="game-row-match">${g.home} <span style="color:var(--text3);font-weight:400">vs</span> ${g.away}</div>
          <div class="game-row-meta"><span style="color:var(--text3)">${leagueTxt}</span>${launchBadge ? ' ' + launchBadge : ''}${badge ? ' · ' + badge : ''}${time ? ' · ' + time : ''}${austrianDateTime ? ' · ' + austrianDateTime + ' AT' : ''}${playoffBadge ? ' ' : ''}${playoffBadge}${abstainBadge ? ' ' : ''}${abstainBadge}</div>
          ${basketballProbSummary ? `<div class="game-row-meta" style="font-size:.68rem;color:var(--text3)">${escapeHtml(basketballProbSummary)}</div>` : ''}
          ${nhlProbSummary ? `<div class="game-row-meta" style="font-size:.68rem;color:var(--text3)">${escapeHtml(nhlProbSummary)}</div>` : ''}
          ${mlbProbSummary ? `<div class="game-row-meta" style="font-size:.68rem;color:var(--text3)">${escapeHtml(mlbProbSummary)}</div>` : ''}
          ${boardSummary ? `<div class="game-row-meta" style="font-size:.68rem;color:var(--text3)"><span class="launch-support-tag ${boardClass}" style="font-size:.58rem;padding:1px 6px">${escapeHtml(boardLabels[boardStatus] || boardStatus)}</span> ${escapeHtml(boardSummary)}</div>` : ''}
        </div>
        ${pickBadge}
        ${oddsTxt ? `<span class="game-row-odds">${oddsTxt}</span>` : ''}
        ${valueDot}
      </div>`;
    }).join('');
    html += `
      <div class="slate-day${isOpen ? ' open' : ''}">
        <button class="slate-day-head" onclick="toggleSlateDay('${key}')">
          <div>
            <div class="slate-day-title">${groupMeta[key].title}</div>
            <div class="slate-day-copy">${groupMeta[key].copy}</div>
          </div>
          <div class="slate-day-meta">
            <span class="slate-day-count">${list.length}</span>
            <span class="slate-day-chevron">▶</span>
          </div>
        </button>
        <div class="slate-day-body">${rows}</div>
      </div>`;
  }
  el.innerHTML = html;
}

async function loadSoccerGames() {
  const params = new URLSearchParams();
  if (state.filters.window !== 'all') params.append('window', state.filters.window);
  try {
    const r = await fetch('/api/soccer/games?' + params);
    const d = await r.json();
    renderSoccerGames(d.games || []);
  } catch(e) {
    document.getElementById('soccer-games-list').innerHTML = '<div style="color:var(--text3);font-size:.8rem;padding:8px">Could not load soccer games.</div>';
  }
}

function renderSoccerGames(games) {
  const el = document.getElementById('soccer-games-list');
  setPicksCount('soccer-games-count', games.length);
  if (!games.length) {
    el.innerHTML = '<div style="color:var(--text3);font-size:.8rem;padding:8px">No soccer games found. Run the scan first.</div>';
    return;
  }
  el.innerHTML = games.map((g, gi) => {
    const timing = eventDisplayMeta(g);
    const badge = eventBadgeHtml(g);
    const leagueBadge = g.league ? `<span class="tag" style="background:rgba(139,92,246,.15);color:#a78bfa;font-size:.62rem">${g.league}</span>` : '';
    const launchBadge = launchBadgeHtml(g);
    const modelBadge = g.model_available
      ? `<span class="tag" style="background:rgba(16,185,129,.12);color:#34d399;font-size:.6rem">ML</span>`
      : `<span class="tag" style="background:rgba(100,116,139,.12);color:var(--text3);font-size:.6rem">odds only</span>`;

    // Find the model's top pick — only from the 3 core single outcomes
    // (Home Win, Draw, Away Win), not double-chance combos which always dominate.
    const SINGLE_OUTCOMES = ['Home Win', 'Draw', 'Away Win'];
    let modelPickIdx = -1;
    if (g.model_available) {
      let bestProb = -1;
      (g.outcomes || []).forEach((o, oi) => {
        if (SINGLE_OUTCOMES.includes(o.label) && o.ml_prob != null && o.ml_prob > bestProb && o.odds && o.odds >= 1.01) {
          bestProb = o.ml_prob;
          modelPickIdx = oi;
        }
      });
    }
    const soccerProbDebug = g.soccer_probability_debug || null;
    const debugPickKey = modelPickIdx >= 0
      ? (g.outcomes?.[modelPickIdx]?.label === 'Draw'
          ? 'draw'
          : g.outcomes?.[modelPickIdx]?.label === 'Away Win'
            ? 'away'
            : 'home')
      : 'home';
    const fmtProb = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
    const soccerProbSummary = soccerProbDebug
      ? (() => {
          const regime = soccerProbDebug.regime ? ` · ${String(soccerProbDebug.regime).replace(/_/g, ' ')}` : '';
          const gap = soccerProbDebug.disagreement_pp != null ? ` · gap ${soccerProbDebug.disagreement_pp}pp` : '';
          return `Model check · classifier ${fmtProb(soccerProbDebug.classifier_probs?.[debugPickKey])} · structural ${fmtProb(soccerProbDebug.structural_probs?.[debugPickKey])} · final ${fmtProb(soccerProbDebug.final_probs?.[debugPickKey])} · market ${fmtProb(soccerProbDebug.market_probs?.[debugPickKey])}${regime}${gap}`;
        })()
      : '';

    const outcomes = (g.outcomes || []).map((o, oi) => {
      if (!o.odds || o.odds < 1.01) return '';  // skip if no odds
      const edgePct = o.edge != null ? (o.edge * 100).toFixed(1) : null;
      const edgeClass = o.has_value ? 'pos' : (o.edge != null && o.edge < -0.05 ? 'neg' : 'nil');
      const edgeTxt = edgePct != null ? (o.edge >= 0 ? '+' : '') + edgePct + '%' : '';
      const probTxt = o.ml_prob != null ? Math.round(o.ml_prob * 100) + '%' : '';
      const isModelPick = oi === modelPickIdx;
      const selKey = `${gi}_${oi}`;
      const isSelected = state.soccerSelected.has(selKey);
      const outcomeTrackId = `soccer|${g.home}|${g.away}|${o.label}`;
      const outcomeTracked = isTracked(outcomeTrackId);
      return `<div class="outcome-row">
        <button class="outcome-btn${o.has_value ? ' value' : ''}${isModelPick ? ' model-pick' : ''}${isSelected ? ' selected' : ''}"
          onclick="toggleSoccerOutcome(${gi}, ${oi}, this)"
          data-game="${gi}" data-outcome="${oi}">
          <span class="outcome-label">${o.label}${isModelPick ? ' <span class="pick-star">★ Pick</span>' : ''}</span>
          <span class="outcome-prob">${probTxt}</span>
          ${edgeTxt ? `<span class="outcome-edge ${edgeClass}">${edgeTxt}</span>` : ''}
          <span class="outcome-odds">${o.odds ? o.odds.toFixed(2) : '—'}</span>
          <span class="outcome-add">${isSelected ? '✓ Added' : '+ Parlay'}</span>
        </button>
        <button class="track-btn${outcomeTracked?' tracked':''}"
          onclick="trackSoccerOutcome(state.soccerGames[${gi}], state.soccerGames[${gi}].outcomes[${oi}])"
          title="${outcomeTracked?'Remove from Results':'Track in Results'}">
          ${outcomeTracked ? '★' : '☆'}
        </button>
      </div>`;
    }).join('');

    return `<div class="soccer-game-card" id="sgc-${gi}">
      <div class="soccer-game-header" onclick="toggleSoccerGame(${gi})">
        <span style="font-size:1.2rem">⚽</span>
        <div style="flex:1;min-width:0">
          <div class="soccer-teams">${g.home} <span style="color:var(--text3);font-weight:400">vs</span> ${g.away}</div>
          <div class="soccer-meta">${badge} ${timing.timeLabel ? '· ' + escapeHtml(timing.timeLabel) : ''} · ${leagueBadge} ${launchBadge} ${modelBadge}</div>
          ${soccerProbSummary ? `<div class="soccer-meta" style="font-size:.68rem;color:var(--text3);margin-top:2px">${escapeHtml(soccerProbSummary)}</div>` : ''}
        </div>
        <span class="soccer-chevron">▼</span>
      </div>
      <div class="soccer-outcomes">${outcomes}</div>
    </div>`;
  }).join('');

  // Re-apply selected state from parlay legs
  state.soccerGames = games;
}

function toggleSoccerGame(gi) {
  const card = document.getElementById('sgc-' + gi);
  card.classList.toggle('open');
}

function toggleSoccerOutcome(gi, oi, btn) {
  const game = state.soccerGames[gi];
  const outcome = game.outcomes[oi];
  if (!outcome.odds || outcome.odds < 1.01) return;

  const selKey = `${gi}_${oi}`;
  const legId = `soccer|${game.home}|${game.away}|${outcome.label}`;

  if (state.soccerSelected.has(selKey)) {
    // Remove from parlay
    state.soccerSelected.delete(selKey);
    state.parlayLegs = state.parlayLegs.filter(l => l._id !== legId);
    btn.classList.remove('selected');
    btn.querySelector('.outcome-add').textContent = '+ Parlay';
    showToast(`Removed: ${outcome.label}`, 'info');
  } else {
    // Add to parlay
    state.soccerSelected.add(selKey);
    state.parlayLegs.push({
      _id:      legId,
      sport:    'soccer',
      home:     game.home,
      away:     game.away,
      team:     outcome.label === 'Draw' ? 'Draw' : outcome.team,
      match:    `${game.home} vs ${game.away}`,
      kick_off: game.kick_off || '',
      label:    outcome.label,
      odds:     outcome.odds,
      ml_prob:  outcome.ml_prob || 0.5,
      edge:     outcome.edge,
      league:   game.league,
      commence: game.commence,
    });
    btn.classList.add('selected');
    btn.querySelector('.outcome-add').textContent = '✓ Added';
    showToast(`Added: ${outcome.label} @ ${outcome.odds.toFixed(2)}`, 'success');
  }
  updateParlayPanel();
}

function renderBetGrid(bets, opts = {}) {
  const {
    listId,
    countId,
    pillId = null,
    source = 'allBets',
    emptyIcon = '🔍',
    emptyCopy = 'No picks match the selected filters.<br>Try running the scan or changing filters.',
    title = 'Reviewed single-bet candidate',
    edgeLabel = null,
  } = opts;
  const el = document.getElementById(listId);
  setPicksCount(countId, bets.length);
  if (pillId) setPicksCount(pillId, bets.length);
  if (!bets.length) {
    el.innerHTML = `<div class="empty-state"><div class="big-icon">${emptyIcon}</div><p>${emptyCopy}</p></div>`;
    return;
  }
  const importantContextNames = new Set([
    'rotation_quality_edge',
    'goalie_quality',
    'goalie_stability',
    'starter_confirmation',
    'lineup_confirmation',
    'lineup_shape_uncertainty',
    'tactical_matchup',
    'style_clash',
    'h2h_tactical_history',
    'pace_control',
    'venue_comfort',
    'closing_execution',
    'pitcher_command',
    'venue_split_edge',
    'special_teams_edge',
    'system_stability',
    'xg_structure',
    'weather_environment',
    'travel_fatigue',
  ]);
  const getResearchView = (bet) => {
    const committee = bet.committee || {};
    const research = committee.research_mind || {};
    const arbiter = committee.arbiter || {};
    const asArray = (value) => Array.isArray(value) ? value : [];
    return {
      evidenceStatus: String(research.evidence_status || bet.research_mind_evidence_status || '').toUpperCase(),
      concreteInfoScore: research.concrete_info_score ?? bet.research_mind_concrete_info_score ?? null,
      sourceCount: research.source_count ?? bet.research_mind_source_count ?? null,
      sourceQualitySummary: String(research.source_quality_summary || bet.research_mind_source_quality_summary || ''),
      marketAvailabilityStatus: String(research.market_availability_status || bet.research_mind_market_availability_status || ''),
      oddsAgeMinutes: research.odds_age_minutes ?? bet.research_mind_odds_age_minutes ?? null,
      oddsFreshnessStatus: String(research.odds_freshness_status || bet.research_mind_odds_freshness_status || ''),
      lineupStatus: String(research.lineup_status || bet.research_mind_lineup_status || ''),
      injuryStatus: String(research.injury_status || bet.research_mind_injury_status || ''),
      motivationStatus: String(research.motivation_status || bet.research_mind_motivation_status || ''),
      rotationStatus: String(research.rotation_status || bet.research_mind_rotation_status || ''),
      missingEvidence: asArray(research.missing_evidence || bet.research_mind_missing_evidence),
      sportSpecificMissingEvidence: asArray(research.sport_specific_missing_evidence || bet.research_mind_sport_specific_missing_evidence),
      conflictingEvidence: asArray(research.conflicting_evidence || bet.research_mind_conflicting_evidence),
      mainRisks: asArray(research.main_risks || bet.research_mind_main_risks),
      confidence: String(research.confidence || bet.research_mind_confidence || ''),
      parlaySuitability: String(arbiter.parlay_suitability || bet.committee_parlay_suitability || ''),
      finalDecision: String(arbiter.final_decision || bet.committee_final_decision || ''),
      vetoFlags: asArray(arbiter.veto_flags || bet.committee_veto_flags),
      enrichment: committee.evidence_enrichment || bet.committee_enrichment || {},
    };
  };
  const evidenceClassFor = (status) => {
    const key = String(status || '').toUpperCase();
    if (key === 'COMPLETE') return 'tag-evidence-complete';
    if (key === 'ACCEPTABLE') return 'tag-evidence-acceptable';
    if (key === 'PARTIAL') return 'tag-evidence-partial';
    return 'tag-evidence-insufficient';
  };
  const prettyStatus = (value) => {
    const raw = String(value || '').trim();
    if (!raw) return 'n/a';
    return raw.replace(/_/g, ' ');
  };
  const cards = [];
  bets.forEach((b, i) => {
    try {
    const market = b.market || 'moneyline';
    const icon = SPORT_ICONS[b.sport] || '🎲';
    const timing = eventDisplayMeta(b);
    const statusTag = eventBadgeHtml(b);
    const isSelected = state.parlayLegs.some(l => l._id === betId(b));
    const edge_pct = (b.edge * 100).toFixed(1);
    const effectiveStakeAbs = b.committee_effective_stake_abs ?? b.stake_abs;
    const effectiveKellyPct = b.committee_effective_kelly_pct ?? b.kelly_stake_pct;
    const kelly = effectiveStakeAbs ? '£' + Number(effectiveStakeAbs).toFixed(0) : (effectiveKellyPct ? Number(effectiveKellyPct).toFixed(1) + '% = £' + (Number(effectiveKellyPct) / 100 * 1000).toFixed(0) : '—');
    const match = b.home && b.away ? b.home + ' vs ' + b.away : '';
    const tid = b.pred_id || `${b.sport}|${b.team}|${b.home}|${b.away}`;
    const tracked = isTracked(tid);
    const policyStatus = b.market_status || 'experimental';
    const policyLabel = b.market_policy_label || 'Experimental';
    const policyReason = b.market_policy_reason || 'No market policy note available.';
    const policyTag = `<span class="tag tag-policy-${policyStatus}">${policyLabel}</span>`;
    const launchTag = launchBadgeHtml(b);
    const quickProb = `${Math.round((b.ml_prob || 0) * 100)}%`;
    const quickMarket = `${Math.round((b.market_implied_prob || 0) * 100)}%`;
    const quickFair = b.fair_odds ? b.fair_odds.toFixed(2) : '—';
    const quickMinOdds = b.minimum_acceptable_odds ? Number(b.minimum_acceptable_odds).toFixed(2) : '—';
    const refereeDecision = String(b.context_referee_decision || '').toUpperCase();
    const refereeReason = b.context_referee_reason || '';
    const refereeClass = refereeDecision === 'APPROVE'
      ? 'tag-ref-approve'
      : refereeDecision === 'REVIEW'
        ? 'tag-ref-review'
        : refereeDecision === 'VETO'
          ? 'tag-ref-veto'
          : refereeDecision
            ? 'tag-ref-error'
            : '';
    const refereeTag = refereeDecision
      ? `<span class="tag ${refereeClass}" title="${escapeHtml(refereeReason || 'Context referee decision')}">🧠 ${escapeHtml(refereeDecision)}</span>`
      : '';
    const oddsGateTag = b.odds_recheck_status
      ? `<span class="tag" title="Minimum acceptable odds: ${escapeHtml(String(quickMinOdds))}">📉 ${escapeHtml(String(b.odds_recheck_status).replace(/_/g,' '))}</span>`
      : '';

    // --- new upgrade signals ---
    // flagged = edge > 7% (was 12%); rest_advantage: +N home, -N away; is_playoff; abstain
    const highEdgeTag   = b.flagged ? `<span class="tag tag-flagged">⚠ Review Edge</span>` : '';
    const playoffTag    = b.is_playoff  ? `<span class="tag tag-playoff">🏆 Playoff</span>` : '';
    const restAdv = b.rest_advantage;
    let restTag = '';
    if (restAdv && restAdv !== 0) {
      const restDays = Math.abs(restAdv);
      const restDir  = restAdv > 0 ? 'Home' : 'Away';
      restTag = `<span class="tag tag-rest">💤 +${restDays}d ${restDir}</span>`;
    }
    const abstainTag = b.abstain ? `<span class="tag tag-abstain">⚡ Line Move</span>` : '';
    // Line movement signal from opening odds store
    let lineMovTag = '';
    if (b.line_movement) {
      const lm = b.line_movement;
      const pct = Math.round(Math.abs(lm.move_pct) * 100);
      if (lm.direction === 'shortened' && pct >= 2) {
        lineMovTag = `<span class="tag tag-sharp" title="Odds shortened ${pct}% from opening ${lm.opening_odds} → ${lm.current_odds} (sharp money signal)">📈 Sharp +${pct}%</span>`;
      } else if (lm.direction === 'drifted' && pct >= 2) {
        lineMovTag = `<span class="tag tag-fade" title="Odds drifted ${pct}% from opening ${lm.opening_odds} → ${lm.current_odds} (fade signal — market moving against)">📉 Fade −${pct}%</span>`;
      }
    }
    const availabilityTag = b.availability_summary
      ? `<span class="tag" title="${escapeHtml(b.availability_source || 'availability context')}">🚑 ${escapeHtml(b.availability_summary)}</span>`
      : '';
    const contextTags = (Array.isArray(b.context_adjustments) ? b.context_adjustments : [])
      .filter(item => importantContextNames.has(String(item.name || '')))
      .slice(0, 2)
      .map(item => {
        const labelMap = {
          rotation_quality_edge: '🏀 Rotation quality',
          goalie_quality: '🥅 Goalie quality',
          goalie_stability: '🥅 Goalie stability',
          starter_confirmation: '⚾ Starter confirmed',
          lineup_confirmation: '⚽ XI confirmed',
          lineup_shape_uncertainty: '⚽ XI shape risk',
          tactical_matchup: '⚽ Tactical edge',
          style_clash: '⚽ Style clash',
          h2h_tactical_history: '⚽ H2H pattern',
          pace_control: '🏀 Pace control',
          venue_comfort: '🏀 Venue comfort',
          closing_execution: '🏀 Closing edge',
          pitcher_command: '⚾ Pitcher command',
          venue_split_edge: '⚾ Venue split',
          special_teams_edge: '🥅 Special teams',
          system_stability: '🥅 System stability',
          xg_structure: '🥅 xG structure',
          weather_environment: '🌦 Weather',
          travel_fatigue: '✈️ Travel fatigue',
        };
        const label = labelMap[item.name] || item.name;
        return `<span class="tag" title="${escapeHtml(item.summary || item.name || 'context signal')}">${label}</span>`;
      })
      .join('');
    const scraperTags = (Array.isArray(b.scraped_context_highlights) ? b.scraped_context_highlights : [])
      .slice(0, 2)
      .map(item => `<span class="tag tag-context" title="Scraper-fed context">${escapeHtml(String(item))}</span>`)
      .join('');
    // Dynamic Kelly note: if kelly_dynamic_pct present and differs from kelly_stake_pct
    let kellyNote = '';
    if (b.kelly_dynamic_pct != null && b.kelly_stake_pct) {
      const diff = Math.abs(b.kelly_dynamic_pct - b.kelly_stake_pct);
      if (diff > 0.1) {
        kellyNote = `<span style="font-size:.65rem;color:var(--yellow)" title="Kelly reduced due to drawdown">↓ DD-adj</span>`;
      }
    }
    const edgeNote = edgeLabel || `${policyLabel} signal`;
    const reviewNote = b.review_reason ? `<div class="bet-policy-note" style="color:#fbbf24">${b.review_reason}</div>` : '';
    const researchView = getResearchView(b);
    const soccerProbDebug = b.soccer_probability_debug || null;
    const mlbProbDebug = b.mlb_probability_debug || null;
    const basketballProbDebug = b.basketball_probability_debug || null;
    const nhlProbDebug = b.nhl_probability_debug || null;
    const soccerProbDebugNote = soccerProbDebug && b.sport === 'soccer'
      ? (() => {
          const teamText = String(b.team || '').toLowerCase();
          const homeText = String(b.home || '').toLowerCase();
          const awayText = String(b.away || '').toLowerCase();
          const pickKey = teamText === 'draw'
            ? 'draw'
            : teamText && homeText && teamText.includes(homeText)
              ? 'home'
              : teamText && awayText && teamText.includes(awayText)
                ? 'away'
                : 'home';
          const fmt = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
          const regime = soccerProbDebug.regime ? ` · ${String(soccerProbDebug.regime).replace(/_/g, ' ')}` : '';
          const gap = soccerProbDebug.disagreement_pp != null ? ` · gap ${soccerProbDebug.disagreement_pp}pp` : '';
          return `Soccer model view: classifier ${fmt(soccerProbDebug.classifier_probs?.[pickKey])} · structural ${fmt(soccerProbDebug.structural_probs?.[pickKey])} · pre-market ${fmt(soccerProbDebug.pre_market_blend_probs?.[pickKey])} · final ${fmt(soccerProbDebug.final_probs?.[pickKey])} · market ${fmt(soccerProbDebug.market_probs?.[pickKey])}${regime}${gap}`;
        })()
      : '';
    const mlbProbDebugNote = mlbProbDebug && b.sport === 'mlb'
      ? (() => {
          const teamText = String(b.team || '').toLowerCase();
          const homeText = String(b.home || '').toLowerCase();
          const pickKey = teamText && homeText && teamText.includes(homeText) ? 'home' : 'away';
          const fmt = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
          const regime = mlbProbDebug.regime ? ` · ${String(mlbProbDebug.regime).replace(/_/g, ' ')}` : '';
          const gap = mlbProbDebug.disagreement_pp != null ? ` · gap ${mlbProbDebug.disagreement_pp}pp` : '';
          const starter = (() => {
            const h = mlbProbDebug.home_starter_confirmed;
            const a = mlbProbDebug.away_starter_confirmed;
            if (h == null && a == null) return '';
            const hs = h ? 'H✓' : 'H?';
            const as = a ? 'A✓' : 'A?';
            return ` · starters ${hs}/${as}`;
          })();
          const lineup = (() => {
            const hc = mlbProbDebug.home_lineup_confirmed;
            const ac = mlbProbDebug.away_lineup_confirmed;
            const hl = mlbProbDebug.home_likely_starters_count;
            const al = mlbProbDebug.away_likely_starters_count;
            if (hc == null && ac == null && !hl && !al) return '';
            const hs = hc ? `H XI ${hl || 9}` : (hl ? `H XI ${hl}` : 'H XI ?');
            const as = ac ? `A XI ${al || 9}` : (al ? `A XI ${al}` : 'A XI ?');
            return ` · ${hs}/${as}`;
          })();
          const weather = mlbProbDebug.weather_risk ? ` · weather risk${mlbProbDebug.wind_mph != null ? ` ${mlbProbDebug.wind_mph}mph` : ''}` : '';
          return `MLB model view: classifier ${fmt(mlbProbDebug.classifier_probs?.[pickKey])} · structural ${fmt(mlbProbDebug.structural_probs?.[pickKey])} · pre-market ${fmt(mlbProbDebug.pre_market_blend_probs?.[pickKey])} · final ${fmt(mlbProbDebug.final_probs?.[pickKey])} · market ${fmt(mlbProbDebug.market_probs?.[pickKey])}${regime}${gap}${starter}${lineup}${weather}`;
        })()
      : '';
    const basketballProbDebugNote = basketballProbDebug && b.sport === 'basketball'
      ? (() => {
          const teamText = String(b.team || '').toLowerCase();
          const homeText = String(b.home || '').toLowerCase();
          const pickKey = teamText && homeText && teamText.includes(homeText) ? 'home' : 'away';
          const fmt = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
          const regime = basketballProbDebug.regime ? ` · ${String(basketballProbDebug.regime).replace(/_/g, ' ')}` : '';
          const gap = basketballProbDebug.disagreement_pp != null ? ` · gap ${basketballProbDebug.disagreement_pp}pp` : '';
          const rest = basketballProbDebug.rest_advantage
            ? ` · rest ${Number(basketballProbDebug.rest_advantage) > 0 ? 'home+' : 'away+'}${Math.abs(Number(basketballProbDebug.rest_advantage))}`
            : '';
          const starters = (basketballProbDebug.home_projected_starters_count || basketballProbDebug.away_projected_starters_count)
            ? ` · starters ${basketballProbDebug.home_projected_starters_count || 0}/${basketballProbDebug.away_projected_starters_count || 0}`
            : '';
          return `Basketball model view: classifier ${fmt(basketballProbDebug.classifier_probs?.[pickKey])} · structural ${fmt(basketballProbDebug.structural_probs?.[pickKey])} · pre-market ${fmt(basketballProbDebug.pre_market_blend_probs?.[pickKey])} · final ${fmt(basketballProbDebug.final_probs?.[pickKey])} · market ${fmt(basketballProbDebug.market_probs?.[pickKey])}${regime}${gap}${rest}${starters}`;
        })()
      : '';
    const nhlProbDebugNote = nhlProbDebug && b.sport === 'nhl'
      ? (() => {
          const teamText = String(b.team || '').toLowerCase();
          const homeText = String(b.home || '').toLowerCase();
          const pickKey = teamText && homeText && teamText.includes(homeText) ? 'home' : 'away';
          const fmt = (v) => (v == null ? '—' : `${Math.round(Number(v) * 100)}%`);
          const regime = nhlProbDebug.regime ? ` · ${String(nhlProbDebug.regime).replace(/_/g, ' ')}` : '';
          const gap = nhlProbDebug.disagreement_pp != null ? ` · gap ${nhlProbDebug.disagreement_pp}pp` : '';
          const goalie = nhlProbDebug.goalie_status ? ` · goalie ${String(nhlProbDebug.goalie_status).replace(/_/g, ' ')}` : '';
          const rest = nhlProbDebug.rest_advantage
            ? ` · rest ${Number(nhlProbDebug.rest_advantage) > 0 ? 'home+' : 'away+'}${Math.abs(Number(nhlProbDebug.rest_advantage))}`
            : '';
          return `NHL model view: classifier ${fmt(nhlProbDebug.classifier_probs?.[pickKey])} · structural ${fmt(nhlProbDebug.structural_probs?.[pickKey])} · pre-market ${fmt(nhlProbDebug.pre_market_blend_probs?.[pickKey])} · final ${fmt(nhlProbDebug.final_probs?.[pickKey])} · market ${fmt(nhlProbDebug.market_probs?.[pickKey])}${regime}${gap}${goalie}${rest}`;
        })()
      : '';
    const evidenceTags = [];
    if (researchView.evidenceStatus) {
      evidenceTags.push(`<span class="tag ${evidenceClassFor(researchView.evidenceStatus)}">Evidence ${escapeHtml(researchView.evidenceStatus)}</span>`);
    }
    if (researchView.concreteInfoScore != null && researchView.concreteInfoScore !== '') {
      evidenceTags.push(`<span class="tag tag-evidence-score">Concrete ${escapeHtml(String(researchView.concreteInfoScore))}/100</span>`);
    }
    if (researchView.sourceCount != null && researchView.sourceCount !== '') {
      const quality = researchView.sourceQualitySummary ? ` · ${escapeHtml(researchView.sourceQualitySummary)}` : '';
      evidenceTags.push(`<span class="tag tag-evidence-source">Sources ${escapeHtml(String(researchView.sourceCount))}${quality}</span>`);
    }
    if (researchView.oddsAgeMinutes != null && researchView.oddsAgeMinutes !== '') {
      const freshness = researchView.oddsFreshnessStatus ? ` · ${escapeHtml(prettyStatus(researchView.oddsFreshnessStatus))}` : '';
      evidenceTags.push(`<span class="tag tag-evidence-odds">Odds age ${escapeHtml(String(researchView.oddsAgeMinutes))}m${freshness}</span>`);
    }
    if (researchView.marketAvailabilityStatus) {
      evidenceTags.push(`<span class="tag tag-evidence-source">Market ${escapeHtml(prettyStatus(researchView.marketAvailabilityStatus))}</span>`);
    }
    const evidencePrimaryNote = researchView.conflictingEvidence[0]
      || researchView.sportSpecificMissingEvidence[0]
      || researchView.missingEvidence[0]
      || (researchView.mainRisks.find(item => String(item || '').toLowerCase() !== 'no major risks detected from available evidence'))
      || '';
    const evidenceFallbackNote = researchView.mainRisks[0] || '';
    const blockerSummary = String(b.committee_blocker_summary || '').trim();
    const blockerList = Array.isArray(b.committee_blockers) ? b.committee_blockers.filter(item => String(item || '').trim()) : [];
    const evidenceNote = blockerSummary || evidencePrimaryNote || evidenceFallbackNote;
    const vetoFlagNote = researchView.vetoFlags.length ? `Arbiter vetoes: ${researchView.vetoFlags.map(prettyStatus).join(', ')}` : '';
    const enrichment = researchView.enrichment || {};
    const enrichmentTriggered = Boolean(enrichment.triggered);
    const enrichmentDetails = [
      ['Fixture', enrichment.fixture_status],
      ['Probable lineup', enrichment.probable_lineup_status],
      ['Lineup', enrichment.lineup_status],
      ['Injuries', enrichment.injury_status],
      ['Suspensions', enrichment.suspension_status],
      ['Goalkeeper', enrichment.goalkeeper_status],
      ['Motivation', enrichment.motivation_status],
      ['Rotation', enrichment.rotation_status],
      ['Congestion', enrichment.fixture_congestion_status],
      ['Home/away form', enrichment.home_away_form_status],
      ['xG context', enrichment.xg_context_status],
      ['Market fit', enrichment.market_fit_status],
      ['Probable pitchers', enrichment.probable_pitcher_status],
      ['Pitcher change', enrichment.pitcher_change_status],
      ['Home pitcher', enrichment.home_pitcher],
      ['Away pitcher', enrichment.away_pitcher],
      ['Pitcher handedness', enrichment.pitcher_handedness_status],
      ['Bullpen', enrichment.bullpen_status],
      ['Weather', enrichment.weather_status],
      ['Park factor', enrichment.park_factor_status],
      ['Travel/rest', enrichment.travel_rest_status],
      ['Surface', enrichment.surface_status],
      ['Ranking/Elo', enrichment.ranking_elo_status],
      ['Injury/retirement', enrichment.injury_retirement_status],
      ['Fatigue', enrichment.fatigue_status],
      ['Tournament context', enrichment.tournament_context_status],
      ['Style matchup', enrichment.style_matchup_status],
    ].filter(([, value]) => String(value || '').trim());
    const enrichmentBlock = enrichmentTriggered ? `
      <div class="bet-evidence-note">
        Evidence enrichment: triggered
        ${Array.isArray(enrichment.trigger_reason) && enrichment.trigger_reason.length ? ` · ${escapeHtml(enrichment.trigger_reason.join('; '))}` : ''}
      </div>
      <div class="bet-evidence-note">
        Before/after: ${escapeHtml(String(enrichment.evidence_before || 'n/a'))} → ${escapeHtml(String(enrichment.evidence_after || 'n/a'))}
        · Concrete ${escapeHtml(String(enrichment.concrete_score_before ?? 'n/a'))} → ${escapeHtml(String(enrichment.concrete_score_after ?? 'n/a'))}
      </div>
      ${Array.isArray(enrichment.sources_found) && enrichment.sources_found.length ? `<div class="bet-evidence-note">Sources found: ${escapeHtml(enrichment.sources_found.join(', '))}</div>` : ''}
      ${enrichment.source_quality ? `<div class="bet-evidence-note">Source quality: ${escapeHtml(prettyStatus(enrichment.source_quality))}</div>` : ''}
      ${enrichmentDetails.length ? `<div class="bet-evidence-note">Enrichment details: ${escapeHtml(enrichmentDetails.map(([label, value]) => `${label} ${prettyStatus(value)}`).join(' · '))}</div>` : ''}
      ${Array.isArray(enrichment.remaining_missing_evidence) && enrichment.remaining_missing_evidence.length ? `<div class="bet-evidence-note">Remaining gaps: ${escapeHtml(enrichment.remaining_missing_evidence.join(', '))}</div>` : ''}
      ${enrichment.final_arbiter_decision ? `<div class="bet-evidence-note">Final Arbiter: ${escapeHtml(prettyStatus(enrichment.final_arbiter_decision))}</div>` : ''}
    ` : '';
    const evidencePanel = evidenceTags.length || researchView.marketAvailabilityStatus || researchView.lineupStatus || researchView.injuryStatus || researchView.motivationStatus || researchView.rotationStatus || evidenceNote || researchView.parlaySuitability || vetoFlagNote
      ? `
        <div class="bet-evidence-panel">
          <div class="bet-evidence-tags">${evidenceTags.join('')}</div>
          <div class="bet-evidence-grid">
            <div class="bet-evidence-item"><span class="bet-evidence-label">Market</span><span class="bet-evidence-value">${escapeHtml(prettyStatus(researchView.marketAvailabilityStatus))}</span></div>
            <div class="bet-evidence-item"><span class="bet-evidence-label">Lineup</span><span class="bet-evidence-value">${escapeHtml(prettyStatus(researchView.lineupStatus))}</span></div>
            <div class="bet-evidence-item"><span class="bet-evidence-label">Injuries</span><span class="bet-evidence-value">${escapeHtml(prettyStatus(researchView.injuryStatus))}</span></div>
            <div class="bet-evidence-item"><span class="bet-evidence-label">Motivation</span><span class="bet-evidence-value">${escapeHtml(prettyStatus(researchView.motivationStatus))}</span></div>
            <div class="bet-evidence-item"><span class="bet-evidence-label">Rotation</span><span class="bet-evidence-value">${escapeHtml(prettyStatus(researchView.rotationStatus))}</span></div>
          </div>
          ${researchView.parlaySuitability ? `<div class="bet-evidence-note">Parlay lane: ${escapeHtml(prettyStatus(researchView.parlaySuitability))}</div>` : ''}
          ${vetoFlagNote ? `<div class="bet-evidence-note">${escapeHtml(vetoFlagNote)}</div>` : ''}
          ${blockerList.length ? `<div class="bet-evidence-note">Hold blockers: ${escapeHtml(blockerList.join(' | '))}</div>` : ''}
          ${soccerProbDebugNote ? `<div class="bet-evidence-note">${escapeHtml(soccerProbDebugNote)}</div>` : ''}
          ${mlbProbDebugNote ? `<div class="bet-evidence-note">${escapeHtml(mlbProbDebugNote)}</div>` : ''}
          ${basketballProbDebugNote ? `<div class="bet-evidence-note">${escapeHtml(basketballProbDebugNote)}</div>` : ''}
          ${nhlProbDebugNote ? `<div class="bet-evidence-note">${escapeHtml(nhlProbDebugNote)}</div>` : ''}
          ${evidenceNote ? `<div class="bet-evidence-note">${escapeHtml(String(evidenceNote))}</div>` : ''}
          ${enrichmentBlock}
        </div>
      `
      : '';

    cards.push(`
      <div class="bet-card${isSelected?' selected':''}" title="${title}" data-sport="${escapeHtml(b.sport||'')}" data-home="${escapeHtml(b.home||'')}" data-away="${escapeHtml(b.away||'')}">
      <div class="bet-sport-icon" onclick="toggleBetInParlayFrom('${source}', ${i})">${icon}</div>
      <div class="bet-main" onclick="toggleBetInParlayFrom('${source}', ${i})">
        <div class="bet-main-top">
          <div>
            <div class="bet-match">${match || b.league || ''} <span class="bet-live-score"></span></div>
            <div class="bet-pick">${b.team}</div>
          </div>
          <div class="bet-market-score">
            <span class="tag tag-market">${market}</span>
            ${policyTag}
          </div>
        </div>
        <div class="bet-tags">
          ${statusTag.replace('class="tag tag-window"', `class="tag tag-window${timing.bucket === 'tomorrow' || timing.bucket === 'day_after' || timing.bucket === 'upcoming' ? ' tomorrow' : ''}"`)}
          ${timing.timeLabel ? `<span class="tag">${escapeHtml(timing.timeLabel)}</span>` : ''}
          ${launchTag}
          ${b.bookmaker ? `<span class="tag tag-book">${b.bookmaker}</span>` : ''}
          ${highEdgeTag}${playoffTag}${restTag}${abstainTag}${lineMovTag}${availabilityTag}${refereeTag}${oddsGateTag}${contextTags}${scraperTags}
        </div>
        <div class="bet-quick-stats">
          <div class="bet-quick-stat">
            <div class="bet-quick-label">Model</div>
            <div class="bet-quick-value">${quickProb}</div>
          </div>
          <div class="bet-quick-stat">
            <div class="bet-quick-label">Market</div>
            <div class="bet-quick-value">${quickMarket}</div>
          </div>
          <div class="bet-quick-stat">
            <div class="bet-quick-label">Fair Odds</div>
            <div class="bet-quick-value">${quickFair}</div>
          </div>
          <div class="bet-quick-stat">
            <div class="bet-quick-label">Min Odds</div>
            <div class="bet-quick-value">${quickMinOdds}</div>
          </div>
          <div class="bet-quick-stat">
            <div class="bet-quick-label">Edge</div>
            <div class="bet-quick-value">+${edge_pct}%</div>
          </div>
        </div>
        ${evidencePanel}
      </div>
      <div class="bet-right">
        <div class="bet-odds">${b.odds}</div>
        <div class="bet-edge">${edgeNote}</div>
        <div class="bet-kelly">Quarter Kelly: ${kelly} ${kellyNote}</div>
        ${reviewNote}
        ${b.launch_note ? `<div class="bet-policy-note">${escapeHtml(b.launch_note)}</div>` : ''}
        <div class="bet-policy-note">${policyReason}</div>
        <button class="track-btn${tracked?' tracked':''}" onclick="event.stopPropagation();trackBetFrom('${source}', ${i})" title="${tracked?'Remove from Results':'Track in Results'}">
          ${tracked ? '★ Tracking' : '☆ Track'}
        </button>
        <div class="bet-verify-btns">
          <button class="bet-verify-btn" id="bvb-g-${i}-${source}" onclick="event.stopPropagation();runBetReasoning(this,'${escapeHtml([b.sport||'',b.market||'',b.home||'',b.away||'',b.team||''].join('|'))}','guarded')" title="Quick reasoning scan — uses cached data (~5 s)">🧠 Scan</button>
          <button class="bet-verify-btn" id="bvb-l-${i}-${source}" onclick="event.stopPropagation();runBetReasoning(this,'${escapeHtml([b.sport||'',b.market||'',b.home||'',b.away||'',b.team||''].join('|'))}','full_live')" title="Full live verification — fetches fresh data (~30 s)">🔬 Live</button>
        </div>
      </div>
    </div>`);
    } catch (e) {
      console.error('renderBetGrid card failed', e, b);
      cards.push(`
        <div class="bet-card">
          <div class="bet-main">
            <div class="bet-main-top">
              <div>
                <div class="bet-match">${escapeHtml(String(b.home || b.league || 'Candidate'))}</div>
                <div class="bet-pick">${escapeHtml(String(b.team || b.market || 'Could not render candidate'))}</div>
              </div>
            </div>
            <div class="bet-policy-note" style="color:#fbbf24">This candidate could not be fully rendered in the browser.</div>
          </div>
        </div>`);
    }
  });
  el.innerHTML = cards.join('');
}

function trackBetFrom(source, idx) {
  const bucket = state[source] || [];
  const bet = bucket[idx];
  if (!bet) return;
  trackBet(bet);
}

function renderBets(bets) {
  renderBetGrid(bets, {
    listId: 'bets-list',
    countId: 'bets-count',
    pillId: 'bets-count-pill',
    source: 'allBets',
    emptyIcon: '🔍',
    emptyCopy: 'No picks match the selected filters.<br>Try running the scan or changing filters.',
    title: 'Reviewed single-bet candidate',
  });
}

function renderReviewBets(bets) {
  renderBetGrid(bets, {
    listId: 'review-bets-list',
    countId: 'review-bets-count',
    source: 'reviewBets',
    emptyIcon: '🧠',
    emptyCopy: 'No bets are waiting for manual review right now.',
    title: 'Manual review candidate',
    edgeLabel: 'Held for review',
  });
}

function betId(b) { return `${b.sport}|${b.team}|${b.odds}`; }

function toggleBetInParlayFrom(source, idx) {
  const bucket = state[source] || [];
  const bet = bucket[idx];
  if (!bet) return;
  const id  = betId(bet);
  const existIdx = state.parlayLegs.findIndex(l => l._id === id);
  if (existIdx >= 0) {
    state.parlayLegs.splice(existIdx, 1);
    showToast('Removed from parlay', 'success');
  } else {
    state.parlayLegs.push({
      _id:     id,
      team:    bet.team,
      sport:   bet.sport,
      odds:    bet.odds,
      ml_prob: bet.ml_prob,
      edge:    bet.edge,
      match:   bet.home && bet.away ? bet.home + ' vs ' + bet.away : '',
      kick_off: bet.kick_off || '',
    });
    showToast('Added to parlay ✓', 'success');
  }
  renderBets(state.allBets);
  renderReviewBets(state.reviewBets);
  updateParlayPanel();
}

function toggleBetInParlay(idx) {
  toggleBetInParlayFrom('allBets', idx);
}

// ═══════════════════════════════════════════════════════════
// FILTERS
// ═══════════════════════════════════════════════════════════
function toggleFilter(btn) {
  const filter = btn.dataset.filter;
  const val    = btn.dataset.val;
  if (filter === 'sport') {
    const buttons = Array.from(document.querySelectorAll('[data-filter="sport"]'));
    if (val === 'all') {
      state.filters.sports = [];
      state.filters.sport = 'all';
      buttons.forEach(b => b.classList.toggle('active', b.dataset.val === 'all'));
    } else {
      const selected = new Set(Array.isArray(state.filters.sports) ? state.filters.sports : []);
      if (selected.has(val)) selected.delete(val);
      else selected.add(val);
      state.filters.sports = Array.from(selected);
      state.filters.sport = state.filters.sports.length === 0
        ? 'all'
        : state.filters.sports.length === 1
          ? state.filters.sports[0]
          : state.filters.sports.join(',');
      buttons.forEach(b => {
        const buttonVal = b.dataset.val;
        b.classList.toggle('active', buttonVal === 'all' ? state.filters.sports.length === 0 : selected.has(buttonVal));
      });
    }
    state.gameSlateExpanded = {};
    loadPicks();
    return;
  }
  document.querySelectorAll(`[data-filter="${filter}"]`).forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  state.filters[filter] = val;
  state.gameSlateExpanded = {};
  loadPicks();
}

function changePicksSort(value) {
  state.picksSort = value || 'confidence';
  loadPicks();
}

// ═══════════════════════════════════════════════════════════
// PARLAY PANEL
// ═══════════════════════════════════════════════════════════
function toggleParlayPanel() {
  state.parlayPanelOpen = !state.parlayPanelOpen;
  document.getElementById('parlay-panel').classList.toggle('open', state.parlayPanelOpen);
}

function updateParlayPanel() {
  const legs = state.parlayLegs;
  const count = legs.length;
  const progressFill = document.getElementById('pb-progress-fill');
  const progressLabel = document.getElementById('pb-progress-label');
  document.getElementById('pb-count').textContent = count;
  document.getElementById('fab-count').textContent = count;
  document.getElementById('parlay-fab').style.display = count > 0 ? 'flex' : 'none';
  document.getElementById('pb-calc-btn').disabled = count < 2;
  if (progressFill) progressFill.style.width = `${Math.min(100, count * 34)}%`;
  if (progressLabel) {
    progressLabel.textContent = count < 2
      ? `Add ${2 - count} more leg${2 - count === 1 ? '' : 's'} to calculate`
      : `Ready to calculate with ${count} legs`;
  }

  const legsEl = document.getElementById('pb-legs');
  if (!count) {
    legsEl.innerHTML = '<div class="pb-empty"><div class="icon">🔗</div><p>Click any bet card to add it to your parlay</p><div style="margin-top:8px;font-size:.75rem">Use the strongest daily value bets or soccer outcome lines to start the build.</div></div>';
    document.getElementById('pb-stats').style.display = 'none';
    return;
  }

  legsEl.innerHTML = legs.map((l, i) => {
    // For soccer outcomes, show "Home Win — Arsenal vs Chelsea"
    // For regular bets, show team name
    const displayName = l.label ? `${l.label}` : l.team;
    const matchLine = l.match || (l.home && l.away ? `${l.home} vs ${l.away}` : l.team);
    const sportIcon = {soccer:'⚽',basketball:'🏀',mlb:'⚾',nhl:'🏒',tennis:'🎾'}[l.sport] || '🎯';
    return `<div class="pb-leg">
      <div class="pb-leg-info">
        <div class="pb-leg-team">${sportIcon} ${displayName}</div>
        <div class="pb-leg-meta">${matchLine}${l.kick_off ? ' · ' + l.kick_off : ''}</div>
      </div>
      <div class="pb-leg-odds">${typeof l.odds === 'number' ? l.odds.toFixed(2) : l.odds}</div>
      <button class="pb-leg-remove" onclick="removeParlayLeg(${i})">✕</button>
    </div>`;
  }).join('');

  document.getElementById('pb-stats').style.display = count >= 2 ? 'grid' : 'none';
}

function removeParlayLeg(idx) {
  const leg = state.parlayLegs[idx];
  state.parlayLegs.splice(idx, 1);
  // If it was a soccer outcome leg, clear its selected state and re-render the card
  if (leg && leg._id && leg._id.startsWith('soccer|')) {
    // Remove from soccerSelected: find the matching selKey
    for (const [key] of state.soccerSelected) {
      const [gi, oi] = key.split('_').map(Number);
      const g = state.soccerGames[gi];
      if (g) {
        const o = g.outcomes[oi];
        const checkId = `soccer|${g.home}|${g.away}|${o.label}`;
        if (checkId === leg._id) {
          state.soccerSelected.delete(key);
          break;
        }
      }
    }
    // Re-render soccer games to uncheck the button
    if (state.soccerGames.length) renderSoccerGames(state.soccerGames);
  }
  updateParlayPanel();
  renderBets(state.allBets);
  renderReviewBets(state.reviewBets);
}

function clearParlay() {
  state.parlayLegs = [];
  state.soccerSelected.clear();
  updateParlayPanel();
  renderBets(state.allBets);
  renderReviewBets(state.reviewBets);
  if (state.soccerGames.length) renderSoccerGames(state.soccerGames);
  document.getElementById('pb-stats').style.display = 'none';
  document.getElementById('pb-name-row').style.display = 'none';
  document.getElementById('pb-name-input').value = '';
  const saveBtn = document.getElementById('pb-save-btn');
  saveBtn.style.display = 'none';
  saveBtn.disabled = false;
  saveBtn.textContent = '💾 Save Parlay';
  saveBtn.style.background = '#7c3aed';
}

async function calculateParlay() {
  if (state.parlayLegs.length < 2) return;
  try {
    const r = await fetch('/api/parlay/calculate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ legs: state.parlayLegs }),
    });
    const d = await r.json();
    if (d.error) { showToast(d.error, 'error'); return; }
    document.getElementById('pb-odds').textContent = d.combined_odds + 'x';
    const wpStr = String(d.win_prob);
    document.getElementById('pb-winprob').textContent = wpStr.startsWith('1 in') ? wpStr : wpStr + '%';
    document.getElementById('pb-ev').textContent = d.ev + 'x';
    document.getElementById('pb-kelly').textContent = '£' + d.kelly_stake;
    document.getElementById('pb-stats').style.display = 'grid';
    document.getElementById('pb-name-row').style.display = 'flex';
    document.getElementById('pb-save-btn').style.display = 'block';
    showToast('Calculated ✓', 'success');
  } catch(e) { showToast('Calculation failed', 'error'); }
}

async function saveParlay() {
  if (state.parlayLegs.length < 2) return;
  const btn = document.getElementById('pb-save-btn');
  const nameVal = (document.getElementById('pb-name-input').value || '').trim();
  btn.disabled = true;
  btn.textContent = 'Saving…';
  try {
    const r = await fetch('/api/parlays/manual', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ legs: state.parlayLegs, name: nameVal }),
    });
    const d = await r.json();
    if (d.error) { showToast(d.error, 'error'); btn.disabled = false; btn.textContent = '💾 Save Parlay'; return; }
    btn.textContent = '✓ Saved';
    btn.style.background = '#059669';
    const savedName = d.parlay.name || `${d.parlay.n_legs}-leg Parlay`;
    state.lastSavedParlayId = d.parlay.id;
    state.parlayTab = 'custom';
    document.querySelectorAll('.ptab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.ptab').forEach(b => { if ((b.textContent || '').includes('Custom')) b.classList.add('active'); });
    showToast(`"${savedName}" saved (${d.parlay.combined_odds}x) — visible in Results Desk`, 'success');
    if (state.currentPage === 'parlays') loadParlays();
  } catch(e) {
    showToast('Failed to save parlay', 'error');
    btn.textContent = '💾 Save Parlay';
    btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════
// PARLAYS PAGE
// ═══════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════
// LIVE SCORES
// ═══════════════════════════════════════════════════════════

let _liveScoreTimer = null;

function startLiveScorePolling() {
  _fetchAndApplyLiveScores();
  if (_liveScoreTimer) return;
  _liveScoreTimer = setInterval(() => {
    if (state.currentPage === 'picks') _fetchAndApplyLiveScores();
  }, 30000);
}

function stopLiveScorePolling() {
  if (_liveScoreTimer) { clearInterval(_liveScoreTimer); _liveScoreTimer = null; }
}

async function _fetchAndApplyLiveScores() {
  try {
    const r = await fetch('/api/live-scores');
    const d = await r.json();
    _applyLiveScores(d.events || []);
  } catch(e) {}
}

function _normTeam(s) {
  return (s || '').toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ').trim();
}

function _teamsMatch(a, b) {
  a = _normTeam(a); b = _normTeam(b);
  if (!a || !b) return false;
  if (a === b) return true;
  const minLen = Math.min(5, Math.min(a.length, b.length));
  if (minLen < 3) return false;
  return a.startsWith(b.slice(0, minLen)) || b.startsWith(a.slice(0, minLen))
      || a.includes(b.slice(0, minLen))   || b.includes(a.slice(0, minLen));
}

function _findScoreEvent(events, sport, home, away) {
  const normSport = s => s === 'tennis_wta' ? 'tennis' : s;
  const targetSport = normSport(sport);
  return events.find(ev =>
    normSport(ev.sport) === targetSport &&
    _teamsMatch(ev.home, home) &&
    _teamsMatch(ev.away, away)
  ) || null;
}

function _scoreBadgeHtml(ev) {
  const st = ev.status_type || '';
  const hs = ev.home_score ?? null;
  const as_ = ev.away_score ?? null;
  const score = (hs !== null && as_ !== null) ? `${hs}–${as_}` : '';
  const FINISHED = new Set(['finished','ended','afterextratime','afterpenalties']);
  if (FINISHED.has(st)) {
    const lbl = st === 'afterextratime' ? 'AET' : st === 'afterpenalties' ? 'Pens' : 'FT';
    return `<span class="live-score-badge ended">${lbl}${score ? ' · ' + score : ''}</span>`;
  }
  if (st === 'halftime') {
    return `<span class="live-score-badge halftime">HT${score ? ' · ' + score : ''}</span>`;
  }
  if (st === 'inprogress' || st === 'live') {
    let time = '';
    if (ev.minutes != null) time = ev.injury_time ? `${ev.minutes}+${ev.injury_time}'` : `${ev.minutes}'`;
    const inner = [time, score].filter(Boolean).join(' · ');
    return `<span class="live-score-badge live">🔴 ${inner || 'LIVE'}</span>`;
  }
  if (st === 'postponed') return `<span class="live-score-badge postponed">PPND</span>`;
  return '';
}

function _applyLiveScores(events) {
  // Pre-index by sport to avoid O(N) scan per card
  const bySport = {};
  const normSport = s => s === 'tennis_wta' ? 'tennis' : s;
  events.forEach(ev => {
    const k = normSport(ev.sport || '');
    if (!bySport[k]) bySport[k] = [];
    bySport[k].push(ev);
  });

  document.querySelectorAll('.bet-card[data-home]').forEach(card => {
    const el = card.querySelector('.bet-live-score');
    if (!el) return;
    const sportKey = normSport(card.dataset.sport || '');
    const bucket = bySport[sportKey] || [];
    const ev = bucket.find(e => _teamsMatch(e.home, card.dataset.home) && _teamsMatch(e.away, card.dataset.away)) || null;
    el.innerHTML = ev ? _scoreBadgeHtml(ev) : '';
    card.classList.toggle('bet-card-live', !!(ev && (ev.status_type === 'inprogress' || ev.status_type === 'live')));
    card.classList.toggle('bet-card-ended', !!(ev && ['finished','ended','afterextratime','afterpenalties'].includes(ev.status_type)));
  });
}

// ═══════════════════════════════════════════════════════════
// AI PARLAY BUILDER
// ═══════════════════════════════════════════════════════════

let _aiParlayEs = null;

function buildAiParlay() {
  const modal   = document.getElementById('ai-parlay-modal');
  const log     = document.getElementById('ai-parlay-log');
  const result  = document.getElementById('ai-parlay-result');
  const footer  = document.getElementById('ai-parlay-footer');
  const subtitle = document.getElementById('ai-parlay-subtitle');
  const picker  = document.getElementById('ai-parlay-window-picker');

  // Reset to picker state
  log.innerHTML = '';
  result.style.display = 'none';
  result.innerHTML = '';
  footer.style.display = 'none';
  subtitle.textContent = 'Choose which games to build from…';
  if (picker) picker.style.display = 'block';
  modal.style.display = 'flex';

  if (_aiParlayEs) { try { _aiParlayEs.close(); } catch(e) {} _aiParlayEs = null; }
}

function startAiParlayBuild(window, pickerBtn) {
  // Highlight selected window button
  document.querySelectorAll('.ai-window-btn').forEach(b => b.classList.remove('active'));
  if (pickerBtn) pickerBtn.classList.add('active');

  const log      = document.getElementById('ai-parlay-log');
  const result   = document.getElementById('ai-parlay-result');
  const footer   = document.getElementById('ai-parlay-footer');
  const subtitle = document.getElementById('ai-parlay-subtitle');
  const picker   = document.getElementById('ai-parlay-window-picker');
  const btn      = document.getElementById('btn-ai-parlay');

  const windowLabel = window === 'today' ? "today's" : window === 'tomorrow' ? "tomorrow's" : "today's + tomorrow's";
  log.innerHTML = '';
  result.style.display = 'none';
  result.innerHTML = '';
  footer.style.display = 'none';
  subtitle.textContent = `Running live verification on ${windowLabel} active bets…`;
  if (picker) picker.style.display = 'none';

  btn.disabled = true;
  btn.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px"></div> Building…';

  if (_aiParlayEs) { try { _aiParlayEs.close(); } catch(e) {} }
  _aiParlayEs = new EventSource(`/api/parlays/ai-build?window=${encodeURIComponent(window)}`);

  _aiParlayEs.onmessage = (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch(e) { return; }

    if (d.stage === 'start') {
      subtitle.textContent = `Evaluating ${d.total} active bet${d.total !== 1 ? 's' : ''} (preferred + experimental)…`;

    } else if (d.stage === 'evaluating') {
      const row = document.createElement('div');
      row.id = `ai-log-row-${d.n}`;
      row.className = 'ai-log-row';
      const tierBadge = d.tier === 'preferred'
        ? '<span class="ai-tier-badge preferred">Preferred</span>'
        : '<span class="ai-tier-badge experimental">Experimental</span>';
      row.innerHTML = `
        <div class="ai-log-spinner"></div>
        <div class="ai-log-body">
          <div class="ai-log-label">${escapeHtml(d.label)} ${tierBadge}</div>
        </div>
        <div class="ai-log-badge pending">…</div>`;
      log.appendChild(row);
      log.scrollTop = log.scrollHeight;

    } else if (d.stage === 'evaluated') {
      const row = document.getElementById(`ai-log-row-${d.n}`);
      if (row) {
        const icon = d.decision === 'APPROVE' ? '✅' : d.decision === 'VETO' ? '❌' : d.decision === 'DATA_THIN' ? '⚠️' : '🔍';
        const cls  = d.decision === 'APPROVE' ? 'approve' : d.decision === 'VETO' ? 'veto' : 'review';
        const note = d.why_for || d.reasoning || '';
        const tierBadge = d.tier === 'preferred'
          ? '<span class="ai-tier-badge preferred">Preferred</span>'
          : '<span class="ai-tier-badge experimental">Experimental</span>';
        row.innerHTML = `
          <div class="ai-log-icon">${icon}</div>
          <div class="ai-log-body">
            <div class="ai-log-label">${escapeHtml(d.label)} ${tierBadge}</div>
            ${note ? `<div class="ai-log-note">${escapeHtml(note.slice(0, 160))}</div>` : ''}
          </div>
          <div class="ai-log-badge ${cls}">${d.decision}</div>`;
        row.classList.toggle('approved', d.decision === 'APPROVE');
      }

    } else if (d.stage === 'done') {
      _aiParlayEs.close();
      _aiParlayEs = null;
      const built = [d.value_parlay, d.longshot_parlay].filter(Boolean).length;
      subtitle.textContent = `Done — evaluated ${d.evaluated_count}, approved ${d.approved_count}, built ${built} parlay${built !== 1 ? 's' : ''}`;
      result.innerHTML = '';
      if (d.value_parlay) {
        result.innerHTML += _renderAiParlayCard(d.value_parlay, '🎯 Value Parlay', '#10b981');
      }
      if (d.longshot_parlay) {
        result.innerHTML += _renderAiParlayCard(d.longshot_parlay, '⚡ Longshot Parlay', '#f59e0b');
      }
      result.style.display = 'block';
      footer.style.display = 'flex';
      btn.disabled = false;
      btn.innerHTML = '🤖 AI Parlay';

    } else if (d.stage === 'error') {
      _aiParlayEs.close();
      _aiParlayEs = null;
      subtitle.textContent = 'Could not build parlay';
      log.innerHTML += `<div style="color:var(--red);font-size:.85rem;padding:8px 0">${escapeHtml(d.message)}</div>`;
      footer.style.display = 'flex';
      document.getElementById('ai-parlay-view-btn').style.display = 'none';
      btn.disabled = false;
      btn.innerHTML = '🤖 AI Parlay';
    }
  };

  _aiParlayEs.onerror = () => {
    _aiParlayEs.close();
    _aiParlayEs = null;
    subtitle.textContent = 'Connection error — is the server running?';
    footer.style.display = 'flex';
    document.getElementById('ai-parlay-view-btn').style.display = 'none';
    btn.disabled = false;
    btn.innerHTML = '🤖 AI Parlay';
  };
}

function _renderAiParlayCard(p, title, accentColor) {
  const edgeColor = p.edge > 0 ? 'var(--green)' : 'var(--red)';
  const legs = (p.legs || []).map((leg, i) => {
    const tierDot = leg.market_status === 'preferred'
      ? `<span style="width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block;flex-shrink:0"></span>`
      : `<span style="width:6px;height:6px;border-radius:50%;background:#f59e0b;display:inline-block;flex-shrink:0"></span>`;
    return `
    <div class="ai-leg-row">
      <div class="ai-leg-num" style="background:${accentColor}">${i + 1}</div>
      <div class="ai-leg-body">
        <div class="ai-leg-pick">
          <span class="ai-leg-sport">${leg.sport.toUpperCase()}</span>
          ${tierDot}
          <strong>${escapeHtml(leg.team)}</strong>
          <span class="ai-leg-market">${leg.market || ''}</span>
        </div>
        <div class="ai-leg-match">${escapeHtml(leg.match)}</div>
        ${leg.why_for ? `<div class="ai-leg-why">${escapeHtml(leg.why_for.slice(0, 180))}</div>` : ''}
      </div>
      <div class="ai-leg-odds" style="color:${accentColor}">@${leg.odds.toFixed(2)}</div>
    </div>`;
  }).join('');

  return `
    <div class="ai-parlay-card" style="border-color:${accentColor}33;margin-bottom:14px">
      <div class="ai-parlay-card-title" style="color:${accentColor}">${title}</div>
      <div class="ai-parlay-summary">
        <div class="ai-summary-stat"><div class="ai-summary-val">${p.combined_odds.toFixed(2)}x</div><div class="ai-summary-lbl">Combined odds</div></div>
        <div class="ai-summary-stat"><div class="ai-summary-val">${p.win_prob}</div><div class="ai-summary-lbl">Win probability</div></div>
        <div class="ai-summary-stat"><div class="ai-summary-val" style="color:${edgeColor}">${p.edge > 0 ? '+' : ''}${p.edge}%</div><div class="ai-summary-lbl">Model edge</div></div>
        <div class="ai-summary-stat"><div class="ai-summary-val">$${p.kelly_stake.toFixed(0)}</div><div class="ai-summary-lbl">Kelly stake</div></div>
      </div>
      <div class="ai-legs-list">${legs}</div>
      <div style="font-size:.72rem;color:var(--text3);margin-top:8px">Saved · ID ${p.id}</div>
    </div>`;
}

function closeAiParlayModal() {
  if (_aiParlayEs) { try { _aiParlayEs.close(); } catch(e) {} _aiParlayEs = null; }
  const btn = document.getElementById('btn-ai-parlay');
  btn.disabled = false;
  btn.innerHTML = '🤖 AI Parlay';
  document.getElementById('ai-parlay-modal').style.display = 'none';
  document.getElementById('ai-parlay-view-btn').style.display = '';
}

function viewAiParlay() {
  closeAiParlayModal();
  switchParlayTab('custom', document.querySelector('.ptab:nth-child(3)'));
  loadParlays();
}

async function runBetReasoning(btn, candidateId, mode) {
  const orig = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner" style="width:10px;height:10px;border-width:1.5px;margin:0 auto"></div>';
  try {
    const r = await fetch('/api/reasoning/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_id: candidateId, mode }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'Failed');
    const decision = String((d.llm_reasoning?.content?.decision) || 'REVIEW').toUpperCase();
    let verdict, color;
    if (decision === 'APPROVE')        { verdict = '✅ YES'; color = 'var(--green)'; }
    else if (decision === 'VETO')      { verdict = '❌ NO';  color = 'var(--red)';   }
    else if (decision === 'DATA_THIN') { verdict = '⚠ THIN'; color = 'var(--yellow)';}
    else                               { verdict = '🔍 REVIEW'; color = 'var(--yellow)'; }
    btn.textContent = verdict;
    btn.style.cssText += `;color:${color};border-color:${color};font-weight:700`;
    setTimeout(() => {
      btn.innerHTML = orig;
      btn.style.cssText = btn.style.cssText.replace(/color:[^;]+;border-color:[^;]+;font-weight:700/g, '');
      btn.disabled = false;
    }, 15000);
  } catch(e) {
    btn.textContent = '⚠ err';
    btn.style.color = 'var(--red)';
    setTimeout(() => {
      btn.innerHTML = orig;
      btn.style.color = '';
      btn.disabled = false;
    }, 4000);
  }
}

async function checkParlayResults() {
  const btn = document.getElementById('btn-parlay-check');
  const status = document.getElementById('parlay-settle-status');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Checking…';
  status.style.display = 'none';
  try {
    const r = await fetch('/api/settle-all', { method: 'POST',
      headers: {'Content-Type':'application/json'}, body: JSON.stringify({}) });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'Settlement failed');

    const bets    = Number(d.bets_settled || 0);
    const parlays = Number(d.parlays_settled || 0);
    const total   = bets + parlays;
    const profit  = Number(d.bets_profit || 0);
    const unresolvedSummary = Array.isArray(d.unresolved_summary) ? d.unresolved_summary : [];

    // Build leg-level breakdown lines
    const legLines = (d.parlay_results || []).map(p => {
      const statusIcon = p.status === 'won' ? '✓' : p.status === 'lost' ? '✗' : '⏳';
      return `${statusIcon} ${p.name}: ${p.legs_won}W · ${p.legs_lost}L · ${p.legs_pending} pending`;
    });

    let msg;
    if (total > 0) {
      const parts = [];
      if (bets > 0)    parts.push(`${bets} bet${bets !== 1 ? 's' : ''} settled`);
      if (parlays > 0) parts.push(`${parlays} parlay${parlays !== 1 ? 's' : ''} updated`);
      if (bets > 0) parts.push(`P&L: ${profit >= 0 ? '+' : ''}${profit.toFixed(3)}u`);
      msg = '✅ ' + parts.join(' · ');
      if (legLines.length) msg += '\n' + legLines.join('\n');
    } else {
      msg = legLines.length
        ? 'Leg results updated:\n' + legLines.join('\n')
        : `No new results — ${d.still_pending || 0} still pending`;
      if (unresolvedSummary.length) {
        msg += '\n' + unresolvedSummary.slice(0, 3).map(item => `• ${item.scope}: ${item.reason} (${item.count})`).join('\n');
      }
    }

    status.innerHTML = msg.replace(/\n/g, '<br>');
    status.style.display = 'block';
    status.style.background = total > 0 ? 'rgba(16,185,129,.15)' : 'rgba(100,116,139,.12)';
    status.style.color = total > 0 ? 'var(--green)' : 'var(--text2)';
    loadParlays();
    if (bets > 0) { loadMySelections(); loadResults(); }
  } catch(e) {
    status.textContent = '⚠ ' + String(e);
    status.style.display = 'block';
    status.style.background = 'rgba(239,68,68,.12)';
    status.style.color = 'var(--red)';
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⚡</span> Check Results';
  }
}

async function sweepSettlementBacklog(btn) {
  const trigger = btn || document.getElementById('btn-backlog-sweep');
  const settlePanel = document.getElementById('settle-result-panel');
  const original = trigger ? trigger.innerHTML : '';
  if (trigger) {
    trigger.disabled = true;
    trigger.innerHTML = '<div class="spinner"></div> Sweeping…';
  }
  if (settlePanel) {
    settlePanel.style.display = 'none';
  }
  try {
    const r = await fetch('/api/settle-all', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ mode: 'backlog' }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'Backlog sweep failed');

    const settled = Number(d.bets_settled || 0) + Number(d.parlays_settled || 0);
    const scanDays = Number(d.target_dates_scanned || 0);
    const msg = settled > 0
      ? `✅ Backlog sweep settled ${settled} item${settled !== 1 ? 's' : ''} across ${scanDays} date${scanDays !== 1 ? 's' : ''}. ${d.still_pending || 0} still pending.`
      : `No overdue results found in backlog sweep — ${d.still_pending || 0} still pending after scanning ${scanDays} date${scanDays !== 1 ? 's' : ''}.`;

    if (settlePanel) {
      const errors = Array.isArray(d.errors) ? d.errors : [];
      settlePanel.innerHTML = msg + (errors.length ? `<br><span style="color:var(--orange)">Sources: ${escapeHtml(errors.join(' | '))}</span>` : '');
      settlePanel.style.display = 'block';
      settlePanel.style.background = settled > 0 ? 'rgba(16,185,129,.15)' : 'rgba(100,116,139,.12)';
      settlePanel.style.color = settled > 0 ? 'var(--green)' : 'var(--text2)';
    } else {
      showToast(msg, settled > 0 ? 'success' : 'info');
    }

    loadResults(_resultDate !== 'all' ? _resultDate : undefined);
    loadMySelections();
    if (state.currentPage === 'parlays') loadParlays();
  } catch (e) {
    const message = '⚠ ' + String(e);
    if (settlePanel) {
      settlePanel.textContent = message;
      settlePanel.style.display = 'block';
      settlePanel.style.background = 'rgba(239,68,68,.12)';
      settlePanel.style.color = 'var(--red)';
    } else {
      showToast(message, 'error');
    }
  } finally {
    if (trigger) {
      trigger.disabled = false;
      trigger.innerHTML = original || 'Sweep Backlog';
    }
  }
}

async function loadParlays() {
  try {
    const [systemR, manualR] = await Promise.all([
      fetch('/api/parlays'),
      fetch('/api/parlays/manual'),
    ]);
    const [systemD, manualD] = await Promise.all([systemR.json(), manualR.json()]);
    const allManual = (manualD.parlays || []).slice().sort((a, b) => String(b.saved_at || b.date || '').localeCompare(String(a.saved_at || a.date || '')));
    state.allParlays      = systemD.parlays || [];
    state.aiValueParlays    = allManual.filter(p => p.type === 'ai_value');
    state.aiLongshotParlays = allManual.filter(p => p.type === 'ai_longshot');
    state.manualParlays     = allManual.filter(p => p.type !== 'ai_value' && p.type !== 'ai_longshot');
    renderParlays(state.parlayTab);
  } catch(e) {
    document.getElementById('parlay-content').innerHTML = '<div class="empty-state"><p>Failed to load parlays.</p></div>';
  }
}

function switchParlayTab(tab, btn) {
  document.querySelectorAll('.ptab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  state.parlayTab = tab;
  renderParlays(tab);
}

function reopenParlayBuilder() {
  if (!state.parlayPanelOpen) toggleParlayPanel();
}

function _parlayTypeCopy(tab) {
  if (tab === 'value') return {
    kicker: 'System Value Parlays',
    title: 'Higher-discipline combinations built from the approved live board.',
    copy: 'These are the tighter system combinations meant to preserve signal quality while still offering upside.'
  };
  return {
    kicker: 'System Longshot Parlays',
    title: 'Longer-shot combinations for higher payout targets.',
    copy: 'These are the more aggressive system builds. Use them when you want upside and can tolerate more volatility.'
  };
}

function _renderAiTabParlay(p) {
  const legs = Array.isArray(p.legs) ? p.legs : [];
  const status = p.status || 'pending';
  const statusBadge = status === 'won'
    ? '<span class="ai-tab-status won">✓ Won</span>'
    : status === 'lost'
      ? '<span class="ai-tab-status lost">✗ Lost</span>'
      : '<span class="ai-tab-status pending">⏳ Pending</span>';
  const edgeColor = (p.edge || 0) > 0 ? 'var(--green)' : 'var(--red)';
  return `
    <div class="ai-tab-parlay-card">
      <div class="ai-tab-parlay-head">
        <div class="ai-tab-parlay-name">
          🤖 ${escapeHtml(p.name || `${legs.length}-Leg AI Parlay`)}
          ${statusBadge}
        </div>
        <div class="ai-tab-parlay-stats">
          <div class="ai-summary-stat"><div class="ai-summary-val">${p.combined_odds != null ? p.combined_odds + 'x' : '—'}</div><div class="ai-summary-lbl">Odds</div></div>
          <div class="ai-summary-stat"><div class="ai-summary-val">${p.win_prob || '—'}</div><div class="ai-summary-lbl">Win prob</div></div>
          <div class="ai-summary-stat"><div class="ai-summary-val" style="color:${edgeColor}">${(p.edge || 0) > 0 ? '+' : ''}${p.edge ?? '—'}%</div><div class="ai-summary-lbl">Edge</div></div>
          <div class="ai-summary-stat"><div class="ai-summary-val">$${p.kelly_stake != null ? p.kelly_stake.toFixed(0) : '—'}</div><div class="ai-summary-lbl">Kelly</div></div>
        </div>
      </div>
      <div class="ai-legs-list" style="margin-top:10px">
        ${legs.map((l, i) => {
          const tierDot = l.market_status === 'preferred'
            ? `<span style="width:6px;height:6px;border-radius:50%;background:#10b981;display:inline-block;flex-shrink:0"></span>`
            : `<span style="width:6px;height:6px;border-radius:50%;background:#f59e0b;display:inline-block;flex-shrink:0"></span>`;
          const lr = l.result || l.status || '';
          const legBadge = lr === 'won'  ? `<span class="leg-result-badge win">W</span>`
                         : lr === 'lost' ? `<span class="leg-result-badge loss">L</span>`
                         : `<span class="leg-result-badge pending">–</span>`;
          return `
            <div class="ai-leg-row">
              <div class="ai-leg-num">${i + 1}</div>
              <div class="ai-leg-body">
                <div class="ai-leg-pick">
                  <span class="ai-leg-sport">${(l.sport || '').toUpperCase()}</span>
                  ${tierDot}
                  <strong>${escapeHtml(l.team || '')}</strong>
                  <span class="ai-leg-market">${l.market || ''}</span>
                </div>
                <div class="ai-leg-match">${escapeHtml(l.match || '')}</div>
                ${l.upset_catalyst ? `<div class="ai-leg-catalyst">⚡ ${escapeHtml(l.upset_catalyst.slice(0, 140))}</div>` : (l.why_for ? `<div class="ai-leg-why">${escapeHtml(l.why_for.slice(0, 160))}</div>` : '')}
              </div>
              <div style="display:flex;align-items:center;gap:6px">
                <div class="ai-leg-odds">@${Number(l.odds).toFixed(2)}</div>
                ${legBadge}
              </div>
            </div>`;
        }).join('')}
      </div>
      <div class="ai-tab-parlay-footer">
        <span style="font-size:.72rem;color:var(--text3)">AI · ${p.date || ''} · ID ${p.id}</span>
        <div style="display:flex;gap:6px">
          <button class="saved-parlay-btn success" onclick="setManualParlayStatus('${p.id}','won','')">✓ Won</button>
          <button class="saved-parlay-btn danger"  onclick="setManualParlayStatus('${p.id}','lost','')">✗ Lost</button>
          <button class="saved-parlay-btn primary" onclick="setManualParlayStatus('${p.id}','pending','')">⏳</button>
          <button class="saved-parlay-btn" style="background:rgba(100,116,139,.12);color:var(--text3)" onclick="deleteManualParlay('${p.id}', event)">🗑</button>
        </div>
      </div>
    </div>`;
}

function _renderSavedParlayCard(p) {
  const legs = Array.isArray(p.legs) ? p.legs : [];
  const status = p.status || 'pending';
  const statusBadge = status === 'won' ? '✓ Won' : status === 'lost' ? '✗ Lost' : '⏳ Pending';
  const combinedOdds = p.combined_odds != null ? `${p.combined_odds}x` : '—';
  const winProb = p.win_prob != null ? `${p.win_prob}` : '—';
  const ev = p.ev != null ? `${p.ev}x` : '—';
  const savedDate = p.date || (p.saved_at ? String(p.saved_at).slice(0, 10) : '—');
  return `
    <div class="saved-parlay-card">
      <div class="saved-parlay-head">
        <div>
          <div class="saved-parlay-name">${p.name || `${legs.length}-leg Parlay`}</div>
          <div class="saved-parlay-meta">
            <span class="saved-parlay-pill">${legs.length} legs</span>
            <span class="saved-parlay-pill">Odds ${combinedOdds}</span>
            <span class="saved-parlay-pill">Win ${winProb}</span>
            <span class="saved-parlay-pill">${statusBadge}</span>
            <span class="saved-parlay-pill">${savedDate}</span>
          </div>
        </div>
      </div>
      <div class="saved-parlay-edit">
        <input id="saved-parlay-name-${p.id}" class="saved-parlay-input" type="text" maxlength="60" value="${(p.name || `${legs.length}-leg Parlay`).replace(/"/g, '&quot;')}" />
        <button class="saved-parlay-btn ghost" onclick="renameManualParlay('${p.id}')">Rename</button>
      </div>
      <div class="saved-parlay-actions">
        <button class="saved-parlay-btn success" onclick="setManualParlayStatus('${p.id}','won')">✓ Won</button>
        <button class="saved-parlay-btn danger"  onclick="setManualParlayStatus('${p.id}','lost')">✗ Lost</button>
        <button class="saved-parlay-btn primary" onclick="setManualParlayStatus('${p.id}','pending')">⏳ Pending</button>
        <button class="saved-parlay-btn" style="background:rgba(100,116,139,.12);color:var(--text3)" onclick="deleteManualParlay('${p.id}', event)">🗑</button>
      </div>
      <div class="saved-parlay-legs">
        ${legs.map(l => {
          const lr = l.result || l.status || '';
          const legBadge = lr === 'won'  ? `<span class="leg-result-badge win">W</span>`
                         : lr === 'lost' ? `<span class="leg-result-badge loss">L</span>`
                         : `<span class="leg-result-badge pending">–</span>`;
          return `
          <div class="saved-parlay-leg">
            <span class="tag tag-market">${l.sport || 'bet'}</span>
            <span>${l.team || l.match || 'Leg'}</span>
            <span style="color:var(--text3);font-size:.8rem">${l.match || ''}</span>
            <span>@ ${l.odds}</span>
            ${legBadge}
          </div>`;
        }).join('')}
      </div>
      <div class="saved-parlay-meta" style="margin-top:12px">
        <span class="saved-parlay-pill">EV ${ev}</span>
        <span class="saved-parlay-pill">Kelly £${p.kelly_stake != null ? p.kelly_stake : '—'}</span>
      </div>
    </div>`;
}

function renderParlays(tab) {
  const el = document.getElementById('parlay-content');

  if (tab === 'custom') {
    const hasLegs = state.parlayLegs.length > 0;
    const combined = hasLegs ? state.parlayLegs.reduce((acc, l) => acc * Number(l.odds || 1), 1) : 0;
    const winProb = hasLegs ? state.parlayLegs.reduce((acc, l) => acc * Number(l.ml_prob || 0), 1) : 0;
    const avgEdge = hasLegs ? state.parlayLegs.reduce((acc, l) => acc + Number(l.edge || 0), 0) / state.parlayLegs.length : 0;
    const savedParlays = state.manualParlays || [];
    el.innerHTML = `
      <div class="parlay-workspace">
        <div class="parlay-stack">
          <div class="parlay-panel">
            <div class="parlay-panel-body">
              <div class="parlay-workspace-head">
                <div>
                  <div class="parlay-kicker">Custom Builder</div>
                  <div class="parlay-title">Build your own daily parlay and keep it attached to the results desk.</div>
                  <div class="parlay-copy">Add legs from Today's Picks, calculate the combination, then save it so it shows up in Results alongside the system parlays.</div>
                </div>
                <div class="parlay-toolbar">
                  <span class="parlay-chip">${state.parlayLegs.length} active legs</span>
                  <span class="parlay-chip">${savedParlays.length} saved manual parlays</span>
                </div>
              </div>
              <div class="parlay-summary-grid">
                <div class="parlay-summary-stat">
                  <div class="parlay-summary-label">Current Legs</div>
                  <div class="parlay-summary-value">${state.parlayLegs.length}</div>
                </div>
                <div class="parlay-summary-stat">
                  <div class="parlay-summary-label">Projected Odds</div>
                  <div class="parlay-summary-value blue">${hasLegs ? combined.toFixed(2) + 'x' : '—'}</div>
                </div>
                <div class="parlay-summary-stat">
                  <div class="parlay-summary-label">Model Win Chance</div>
                  <div class="parlay-summary-value">${hasLegs ? (winProb * 100).toFixed(1) + '%' : '—'}</div>
                </div>
                <div class="parlay-summary-stat">
                  <div class="parlay-summary-label">Average Edge</div>
                  <div class="parlay-summary-value green">${hasLegs ? ((avgEdge || 0) * 100).toFixed(1) + '%' : '—'}</div>
                </div>
              </div>
              <div class="parlay-action-row">
                <button class="parlay-action-btn primary" onclick="reopenParlayBuilder()">Open Builder Rail</button>
                <button class="parlay-action-btn secondary" onclick="calculateParlay();reopenParlayBuilder()" ${state.parlayLegs.length < 2 ? 'disabled' : ''}>Calculate & Review</button>
                <button class="parlay-action-btn ghost" onclick="showPage('picks')">Add Legs from Picks</button>
              </div>
              ${state.lastSavedParlayId && savedParlays.some(p => p.id === state.lastSavedParlayId) ? `
                <div class="parlay-save-banner">
                  <div>Latest save is now in your manual parlay library and Results.</div>
                  <button class="saved-parlay-btn ghost" onclick="document.getElementById('saved-parlay-name-${state.lastSavedParlayId}')?.focus()">Rename Latest</button>
                </div>
              ` : ''}
              ${hasLegs ? `
                <div class="parlay-mini-list">
                  ${state.parlayLegs.map((l, i) => `
                    <div class="parlay-mini-row">
                      <div class="parlay-mini-index">${i + 1}</div>
                      <div class="parlay-mini-main">
                        <div class="parlay-mini-team">${l.label || l.team}</div>
                        <div class="parlay-mini-meta">${l.match || (l.home && l.away ? `${l.home} vs ${l.away}` : '')}${l.kick_off ? ' · ' + l.kick_off : ''}</div>
                      </div>
                      <div class="parlay-mini-side">
                        <strong>@ ${typeof l.odds === 'number' ? l.odds.toFixed(2) : l.odds}</strong>
                        <span style="color:${Number(l.edge || 0) >= 0 ? 'var(--green)' : 'var(--red)'}">${Number(l.edge || 0) >= 0 ? '+' : ''}${(Number(l.edge || 0) * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  `).join('')}
                </div>
              ` : `
                <div class="parlay-empty-note">
                  <div class="big-icon">✏️</div>
                  <div>No legs added yet.</div>
                  <div style="margin-top:6px">Go to Today's Picks, add daily bets to the parlay rail, and come back here to review or save them.</div>
                </div>
              `}
            </div>
          </div>
        </div>
        <div class="parlay-stack">
          <div class="parlay-panel">
            <div class="parlay-panel-body">
              <div class="parlay-library-head">
                <div>
                  <div class="parlay-kicker">Manual Parlay Library</div>
                  <div class="parlay-title" style="font-size:1rem">Saved parlays tied to the results page.</div>
                  <div class="parlay-library-copy">These are your saved manual combinations. You can settle or delete them here without leaving the parlay workspace.</div>
                </div>
                <span class="parlay-chip">${savedParlays.length} saved</span>
              </div>
              <div class="parlay-library-list">
                ${savedParlays.length ? savedParlays.map(_renderSavedParlayCard).join('') : `
                  <div class="parlay-empty-note">
                    <div class="big-icon">💾</div>
                    <div>No saved manual parlays yet.</div>
                    <div style="margin-top:6px">Once you calculate and save a custom build, it will show up here and in Results.</div>
                  </div>
                `}
              </div>
            </div>
          </div>
        </div>
      </div>`;
    return;
  }

  const filtered = state.allParlays.filter(p => p.type === tab);
  const aiParlays = tab === 'value' ? state.aiValueParlays : state.aiLongshotParlays;
  const copy = _parlayTypeCopy(tab);
  const totalCount = filtered.length + aiParlays.length;

  if (!totalCount) {
    el.innerHTML = `<div class="parlay-panel"><div class="parlay-panel-body"><div class="parlay-workspace-head"><div><div class="parlay-kicker">${copy.kicker}</div><div class="parlay-title">${copy.title}</div><div class="parlay-copy">${copy.copy}</div></div><div class="parlay-toolbar"><span class="parlay-chip">0 parlays loaded</span></div></div><div class="parlay-empty-note"><div class="big-icon">🔗</div><div>No ${tab} parlays found for today's scan.</div><div style="margin-top:6px">Run a fresh scan to generate them, or use the 🤖 AI Parlay button to build one now.</div></div></div></div>`;
    return;
  }

  let html = `<div class="parlay-panel"><div class="parlay-panel-body"><div class="parlay-workspace-head"><div><div class="parlay-kicker">${copy.kicker}</div><div class="parlay-title">${copy.title}</div><div class="parlay-copy">${copy.copy}</div></div><div class="parlay-toolbar"><span class="parlay-chip">${totalCount} parlays</span>${aiParlays.length ? `<span class="parlay-chip">🤖 ${aiParlays.length} AI</span>` : ''}</div></div>`;

  // System parlays (from scan) grouped by bracket
  if (filtered.length) {
    const brackets = ['5x','10x','20x'];
    for (const bracket of brackets) {
      const group = filtered.filter(p => p.bracket === bracket);
      if (!group.length) continue;
      html += `<div class="section-title">${bracket === '5x' ? '🟢' : bracket === '10x' ? '🟡' : '🔴'} ${bracket} Target</div>`;
      html += '<div class="parlay-cards">';
      html += group.map(p => `
        <div class="parlay-card">
          <div class="parlay-card-header">
            <span class="parlay-badge ${p.type}">${p.type}</span>
            <span class="parlay-bracket">${p.legs.length} legs</span>
            <div class="parlay-stats">
              <div class="parlay-stat"><div class="parlay-stat-label">Odds</div><div class="parlay-stat-val blue">${p.combined_odds}x</div></div>
              <div class="parlay-stat"><div class="parlay-stat-label">Win %</div><div class="parlay-stat-val">${p.win_prob}%</div></div>
              <div class="parlay-stat"><div class="parlay-stat-label">EV</div><div class="parlay-stat-val green">${p.ev}x</div></div>
              <div class="parlay-stat"><div class="parlay-stat-label">Kelly</div><div class="parlay-stat-val">${p.kelly_stake}</div></div>
            </div>
          </div>
          <div class="parlay-legs-list">
            ${p.legs.map(l => {
              const lr = l.result || l.status || '';
              const legBadge = lr === 'won'  ? `<span class="leg-result-badge win">W</span>`
                             : lr === 'lost' ? `<span class="leg-result-badge loss">L</span>`
                             : '';
              return `
              <div class="parlay-leg-row">
                <span class="parlay-leg-sport">${l.sport}</span>
                <span class="parlay-leg-team">${l.team}</span>
                <span class="parlay-leg-odds">@ ${l.odds}</span>
                <span class="parlay-leg-edge${l.edge < 0?' neg':''}">${l.edge >= 0?'+':''}${l.edge.toFixed(1)}%</span>
                ${legBadge}
              </div>`;
            }).join('')}
          </div>
        </div>`).join('');
      html += '</div><div class="divider"></div>';
    }
  }

  // AI-generated parlays section
  if (aiParlays.length) {
    html += `<div class="section-title" style="margin-top:${filtered.length ? 8 : 0}px">🤖 AI Generated</div>`;
    html += `<div class="parlay-cards">${aiParlays.map(_renderAiTabParlay).join('')}</div>`;
  }

  html += '</div></div>';
  el.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════
// RESULTS — tab switcher
// ═══════════════════════════════════════════════════════════
let _resultsTab = 'my';

function switchResultsTab(tab, btn) {
  _resultsTab = tab;
  document.querySelectorAll('#page-results .ptab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('res-view-my').style.display  = tab === 'my'  ? '' : 'none';
  document.getElementById('res-view-all').style.display = tab === 'all' ? '' : 'none';
  if (tab === 'my')  loadMySelections();
  if (tab === 'all') loadResults();
}

// ═══════════════════════════════════════════════════════════
// MY SELECTIONS
// ═══════════════════════════════════════════════════════════
let _myDate = 'all';
let _mySelections = [];

function selectMyDate(btn, date) {
  _myDate = date === 'today' ? new Date().toISOString().slice(0,10) : date;
  document.querySelectorAll('#my-date-chips .chip').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const select = document.getElementById('my-date-select');
  if (select) select.value = _myDate === 'all' ? '' : _myDate;
  loadMySelections();
}

function selectMyDateSelect(date) {
  _myDate = date || 'all';
  document.querySelectorAll('#my-date-chips .chip').forEach(b => b.classList.remove('active'));
  if (_myDate === 'all') {
    document.querySelector('#my-date-chips .chip')?.classList.add('active');
  }
  loadMySelections();
}

async function loadMySelections() {
  const url = _myDate && _myDate !== 'all'
    ? `/api/my-selections/results?date=${_myDate}`
    : '/api/my-selections/results';
  try {
    const r = await fetch(url);
    const d = await r.json();
    _mySelections = d.selections || [];

    // Populate date chips from unique selection dates
    const dates = [...new Set(_mySelections.map(s => s.date).filter(Boolean))].sort().reverse();
    const chipsEl = document.getElementById('my-date-chips');
    const todayStr = new Date().toISOString().slice(0,10);
    let html = `<button class="chip${_myDate==='all'?' active':''}" onclick="selectMyDate(this,'all')">All Time</button>`;
    html += `<button class="chip${_myDate===todayStr?' active':''}" onclick="selectMyDate(this,'today')">Today</button>`;
    chipsEl.innerHTML = html;
    const myDateSelect = document.getElementById('my-date-select');
    if (myDateSelect) {
      myDateSelect.min = dates.length ? dates[dates.length - 1] : '';
      myDateSelect.max = dates.length ? dates[0] : '';
      myDateSelect.value = _myDate !== 'all' && _myDate !== todayStr ? _myDate : '';
    }

    // Stats
    const o = d.overall || {};
    document.getElementById('my-total').textContent = o.total || 0;
    document.getElementById('my-wr').textContent    = (o.win_rate || 0) + '%';
    document.getElementById('my-pnl').textContent   = (o.pnl >= 0 ? '+' : '') + (o.pnl || 0);
    document.getElementById('my-roi').textContent   = (o.roi >= 0 ? '+' : '') + (o.roi || 0) + '%';
    document.getElementById('my-wr-card').className  = 'stat-card ' + ((o.win_rate||0) >= 50 ? 'green' : (o.settled > 0 ? 'red' : ''));
    document.getElementById('my-pnl-card').className = 'stat-card ' + ((o.pnl||0) >= 0 ? 'green' : 'red');
    document.getElementById('my-roi-card').className = 'stat-card ' + ((o.roi||0) >= 0 ? 'green' : 'red');

    // Pending
    const pending = d.pending || [];
    const pendingPanel = document.getElementById('my-pending-panel');
    if (pending.length) {
      pendingPanel.style.display = '';
      document.getElementById('my-pending-count').textContent = pending.length + ' bet' + (pending.length !== 1 ? 's' : '');
      document.getElementById('my-pending-tbody').innerHTML = pending.map(s => `
        <tr>
          <td>${SPORT_ICONS[s.sport]||''} ${s.sport}<div style="margin-top:4px">${launchBadgeHtml(s)}</div></td>
          <td style="color:var(--text);font-weight:600">
            ${s.team}
            ${s.settlement_note ? `<div style="font-size:.68rem;color:var(--text3);font-weight:500;margin-top:3px">${escapeHtml(s.settlement_note)}</div>` : ''}
          </td>
          <td style="color:var(--text2);font-size:.75rem">${s.match || '—'}</td>
          <td>${s.odds ? Number(s.odds).toFixed(2) : '—'}</td>
          <td style="color:var(--green)">${s.edge != null ? '+' + (s.edge*100).toFixed(1) + '%' : '—'}</td>
          <td style="color:var(--text2);font-size:.75rem">${eventBadgeHtml(s)} ${escapeHtml(s.time_label || s.kick_off || '—')}</td>
          <td style="display:flex;gap:5px;align-items:center">
            <button class="settle-btn won"  onclick="settleMySelection('${s.id}', true)"  title="Mark as Won">✓ Won</button>
            <button class="settle-btn lost" onclick="settleMySelection('${s.id}', false)" title="Mark as Lost">✗ Lost</button>
            <button class="settle-btn" style="background:rgba(100,116,139,.15);color:var(--text3)" onclick="removeMySelection('${s.id}')" title="Remove">✕</button>
          </td>
        </tr>`).join('');
    } else {
      pendingPanel.style.display = 'none';
    }

    // Settled
    const settled = d.settled || [];
    document.getElementById('my-settled-count').textContent = settled.length + ' bet' + (settled.length !== 1 ? 's' : '');
    const mySettledGroups = document.getElementById('my-settled-groups');
    if (mySettledGroups) {
      if (!settled.length) {
        mySettledGroups.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text3)">No settled bets yet. Add picks from the Picks tab.</div>';
      } else {
        mySettledGroups.innerHTML = _renderDateGroups(settled, b => {
          const won = b.result === 'won';
          const pnl = b.profit;
          return `<tr>
            <td>${SPORT_ICONS[b.sport]||''} ${b.sport}<div style="margin-top:4px">${launchBadgeHtml(b)}</div></td>
            <td style="color:var(--text);font-weight:600">${b.team}</td>
            <td style="color:var(--text2);font-size:.75rem">${b.match || '—'}</td>
            <td>${b.odds ? Number(b.odds).toFixed(2) : '—'}</td>
            <td style="color:var(--green)">${b.edge != null ? '+' + (b.edge*100).toFixed(1) + '%' : '—'}</td>
            <td><span class="won-badge ${won ? 'win' : 'loss'}">${won ? '✓ Won' : '✗ Lost'}</span></td>
            <td style="color:${pnl>=0?'var(--green)':'var(--red)'}">${pnl!=null?(pnl>=0?'+':'')+(pnl).toFixed(3):'—'}</td>
            <td style="display:flex;gap:5px;align-items:center">
              <button class="settle-btn won" onclick="settleMySelection('${b.id}', true)" title="Won">✓</button>
              <button class="settle-btn lost" onclick="settleMySelection('${b.id}', false)" title="Lost">✗</button>
            </td>
          </tr>`;
        }, ['Sport','Pick','Match','Odds','Edge','Result','P&L','Correct']);
      }
    }

    // Re-render picks to reflect current tracking state
    renderBets(state.allBets);
    renderReviewBets(state.reviewBets);
    if (state.soccerGames.length) renderSoccerGames(state.soccerGames);

  } catch(e) { console.error('loadMySelections error', e); }
}

async function trackBet(bet) {
  const id = bet.pred_id || `${bet.sport}|${bet.team}|${bet.home}|${bet.away}`;
  const eventDate = bet.commence ? String(bet.commence).slice(0,10) : new Date().toISOString().slice(0,10);
  const already = _mySelections.some(s => s.id === id);
  if (already) {
    await removeMySelection(id);
    return;
  }
  await fetch('/api/my-selections', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      id,
      sport:    bet.sport,
      team:     bet.team,
      match:    `${bet.home} vs ${bet.away}`,
      odds:     bet.odds,
      edge:     bet.edge,
      ml_prob:  bet.ml_prob,
      kick_off: bet.kick_off,
      commence: bet.commence || bet.commence_time || '',
      date:     eventDate,
      source:   'pick',
      pred_id:  bet.pred_id,
    }),
  });
  await loadMySelections();
  showToast(`Tracking: ${bet.team} @ ${Number(bet.odds).toFixed(2)}`, 'success');
}

async function trackSoccerOutcome(game, outcome) {
  const id = `soccer|${game.home}|${game.away}|${outcome.label}`;
  const eventDate = game.commence ? String(game.commence).slice(0,10) : new Date().toISOString().slice(0,10);
  const already = _mySelections.some(s => s.id === id);
  if (already) {
    await removeMySelection(id);
    showToast(`Removed from tracking`, 'info');
    return;
  }
  await fetch('/api/my-selections', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      id,
      sport:    'soccer',
      team:     outcome.label,
      match:    `${game.home} vs ${game.away}`,
      odds:     outcome.odds,
      edge:     outcome.edge,
      ml_prob:  outcome.ml_prob,
      kick_off: game.kick_off,
      commence: game.commence || '',
      date:     eventDate,
      source:   'soccer_outcome',
    }),
  });
  await loadMySelections();
  showToast(`Tracking: ${outcome.label} @ ${Number(outcome.odds).toFixed(2)}`, 'success');
}

async function settleMySelection(id, won) {
  await fetch(`/api/my-selections/${encodeURIComponent(id)}/settle`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ won }),
  });
  loadMySelections();
}

async function removeMySelection(id) {
  await fetch(`/api/my-selections/${encodeURIComponent(id)}`, { method: 'DELETE' });
  loadMySelections();
}

function isTracked(id) {
  return _mySelections.some(s => s.id === id);
}

// ═══════════════════════════════════════════════════════════
// RESULTS — system bets (all)
// ═══════════════════════════════════════════════════════════
let _resultDate = 'all';

function _laneSelectionMatches(row, selection) {
  if (!row || !selection) return false;
  if (String(row.sport || '') !== String(selection.sport || '')) return false;
  if (String(row.market || row.market_type || '') !== String(selection.market || '')) return false;
  if (selection.tier == null) return true;
  return String(row.tier || '') === String(selection.tier || '');
}

function _parlaySelectionMatches(row, selection) {
  if (!row || !selection) return false;
  return (
    String(row.source || '') === String(selection.source || '') &&
    String(row.style || '') === String(selection.style || '') &&
    String(row.bracket || '') === String(selection.bracket || '') &&
    Number(row.n_legs || 0) === Number(selection.n_legs || 0) &&
    String(row.sport_mix || '') === String(selection.sport_mix || '')
  );
}

function _renderParlayDrilldown(selection) {
  const panel = document.getElementById('parlay-drilldown-panel');
  const caption = document.getElementById('res-parlay-caption');
  const data = state.resultsData;
  if (!panel || !caption) return;
  if (!selection || !data) {
    caption.textContent = 'Select a parlay cohort above';
    panel.innerHTML = `<div style="color:var(--text3);font-size:.82rem">
      Click a row in Parlay Performance Matrix to inspect the settled parlays behind that cohort.
    </div>`;
    return;
  }

  const rows = (data.parlay_performance?.detail_rows || []).filter(row => _parlaySelectionMatches(row, selection));
  caption.textContent = `${selection.source} · ${selection.style} · ${selection.bracket} · ${selection.n_legs} legs · ${selection.sport_mix}`;
  if (!rows.length) {
    panel.innerHTML = '<div style="color:var(--text3);font-size:.82rem">No settled parlays matched this cohort.</div>';
    return;
  }

  const wins = rows.filter(row => row.won).length;
  const pnl = rows.reduce((sum, row) => sum + Number(row.profit_units || 0), 0);
  const avgOdds = rows.reduce((sum, row) => sum + Number(row.combined_odds || 0), 0) / rows.length;
  const tableRows = rows.slice(0, 20).map(row => {
    const pnlColor = Number(row.profit_units || 0) >= 0 ? 'var(--green)' : 'var(--red)';
    let legs = [];
    try { legs = JSON.parse(row.legs_json || '[]'); } catch {}
    const legsLabel = legs.map(leg => `${leg.team || '?'} (${leg.sport || '?'})`).join(', ');
    return `<tr>
      <td>${escapeHtml(row.recorded_at || '')}</td>
      <td>${row.combined_odds}x</td>
      <td>${row.ev}x</td>
      <td>${row.edge >= 0 ? '+' : ''}${row.edge}%</td>
      <td>${row.won ? 'Won' : 'Lost'}</td>
      <td style="color:${pnlColor}">${Number(row.profit_units || 0) >= 0 ? '+' : ''}${Number(row.profit_units || 0).toFixed(4)}</td>
      <td style="color:var(--text2);font-size:.76rem">${escapeHtml(legsLabel)}</td>
    </tr>`;
  }).join('');

  panel.innerHTML = `
    <div class="results-card-grid" style="margin-bottom:14px">
      <div class="stat-card"><div class="stat-label">Parlays</div><div class="stat-value">${rows.length}</div><div class="stat-sub">Settled parlays in this cohort.</div></div>
      <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">${((wins / rows.length) * 100).toFixed(1)}%</div><div class="stat-sub">${wins} of ${rows.length} won.</div></div>
      <div class="stat-card"><div class="stat-label">Avg Odds</div><div class="stat-value">${avgOdds.toFixed(2)}x</div><div class="stat-sub">Combined odds across the cohort.</div></div>
      <div class="stat-card"><div class="stat-label">P&amp;L</div><div class="stat-value" style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}</div><div class="stat-sub">Total cohort profit in units.</div></div>
    </div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>Saved</th><th>Odds</th><th>EV</th><th>Edge</th><th>Result</th><th>P&amp;L</th><th>Legs</th></tr></thead>
        <tbody>${tableRows}</tbody>
      </table>
    </div>`;
}

function showParlayDrilldown(source, style, bracket, nLegs, sportMix) {
  state.selectedParlayCohort = {
    source: decodeURIComponent(String(source || '')),
    style: decodeURIComponent(String(style || '')),
    bracket: decodeURIComponent(String(bracket || '')),
    n_legs: Number(nLegs || 0),
    sport_mix: decodeURIComponent(String(sportMix || '')),
  };
  _renderParlayDrilldown(state.selectedParlayCohort);
}

function _renderReplaySlateDetail(date) {
  const panel = document.getElementById('replay-slate-detail-panel');
  const caption = document.getElementById('res-replay-slate-caption');
  const data = state.resultsData;
  if (!panel || !caption) return;
  if (!date || !data) {
    caption.textContent = 'Select a replay date above';
    panel.innerHTML = `<div style="color:var(--text3);font-size:.82rem">
      Click a row in Replay Slates to inspect the historical replay events behind that date and see what current policy would have published or held out.
    </div>`;
    return;
  }

  const rows = (data.replay_slate_events || []).filter(row => String(row.date || '') === String(date));
  caption.textContent = date;
  if (!rows.length) {
    panel.innerHTML = '<div style="color:var(--text3);font-size:.82rem">No replay event rows are available for this date.</div>';
    return;
  }

  const counts = {
    preferred_live: rows.filter(row => row.policy_bucket === 'preferred_live').length,
    limited_live: rows.filter(row => row.policy_bucket === 'limited_live').length,
    held_out: rows.filter(row => row.policy_bucket === 'held_out').length,
  };

  const cards = `
    <div class="results-card-grid" style="margin-bottom:14px">
        <div class="stat-card"><div class="stat-label">Events</div><div class="stat-value">${rows.length}</div><div class="stat-sub">Replay event rows shown for this slate.</div></div>
        <div class="stat-card"><div class="stat-label">Preferred</div><div class="stat-value">${counts.preferred_live}</div><div class="stat-sub">Would publish from a preferred lane.</div></div>
        <div class="stat-card"><div class="stat-label">Limited</div><div class="stat-value">${counts.limited_live}</div><div class="stat-sub">Would stay live but constrained.</div></div>
        <div class="stat-card"><div class="stat-label">Held Out</div><div class="stat-value">${counts.held_out}</div><div class="stat-sub">Would remain off the live board.</div></div>
    </div>`;

  const table = `
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>Sport</th><th>Match</th><th>Market</th><th>Bucket</th><th>Policy</th><th>Confidence</th><th>Correct</th><th>Log Loss</th></tr></thead>
        <tbody>
          ${rows.map(row => {
            const bucketClass = row.policy_bucket === 'preferred_live' ? 'production' : row.policy_bucket === 'limited_live' ? 'limited' : 'review';
            const decisionClass = row.publish_decision === 'publish' ? 'production' : row.publish_decision === 'review' ? 'limited' : 'review';
            const conf = row.pred_confidence == null ? '—' : `${(Number(row.pred_confidence) * 100).toFixed(1)}%`;
            const correct = row.correct == null ? '—' : (row.correct ? 'Yes' : 'No');
            const correctColor = row.correct == null ? 'var(--text3)' : (row.correct ? 'var(--green)' : 'var(--red)');
            return `<tr>
              <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
              <td style="color:var(--text)">${escapeHtml(row.match_id || '')}</td>
              <td><span class="tag tag-market">${escapeHtml(row.market || '')}</span></td>
              <td><span class="launch-support-tag ${bucketClass}">${escapeHtml(row.policy_bucket)}</span></td>
              <td>${escapeHtml(row.policy_label || '')}<div style="margin-top:4px"><span class="launch-support-tag ${decisionClass}">${escapeHtml(row.publish_decision || '')}</span></div></td>
              <td>${conf}</td>
              <td style="color:${correctColor};font-weight:700">${correct}</td>
              <td>${row.event_log_loss == null ? '—' : row.event_log_loss}<div style="color:var(--text3);font-size:.72rem;margin-top:4px">${escapeHtml(row.publish_reason || '')}</div></td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>`;

  panel.innerHTML = cards + table;
}

function showReplaySlateDetail(date) {
  state.selectedReplaySlateDate = String(date || '');
  _renderReplaySlateDetail(state.selectedReplaySlateDate);
}

function _renderLaneDrilldown(selection) {
  const panel = document.getElementById('lane-drilldown-panel');
  const caption = document.getElementById('res-lane-caption');
  const data = state.resultsData;
  if (!panel || !caption) return;
  if (!selection || !data) {
    caption.textContent = 'Select a lane from the tables above';
    panel.innerHTML = `<div style="color:var(--text3);font-size:.82rem">
      Click a row in Performance Matrix or Replay Support to inspect the underlying settled bets, pending backlog, replay support, and governor rationale for that lane.
    </div>`;
    return;
  }

  const settled = (data.settled || []).filter(row => _laneSelectionMatches(row, selection));
  const pending = (data.pending || []).filter(row => _laneSelectionMatches(row, selection));
  const matrixRows = (data.performance_matrix || []).filter(row => _laneSelectionMatches(row, selection));
  const replayRows = (data.replay_support_matrix || []).filter(row => _laneSelectionMatches(row, { sport: selection.sport, market: selection.market }));
  const govRows = (data.governor_recommendations || []).filter(row => _laneSelectionMatches(row, selection));
  const scopeLabel = selection.tier == null
    ? `${selection.sport} · ${selection.market}`
    : `${selection.sport} · ${selection.market} · ${selection.tier}`;

  caption.textContent = scopeLabel;

  const laneSummary = matrixRows.length ? matrixRows.map(row => {
    const clv = row.avg_clv == null ? '—' : `${row.avg_clv >= 0 ? '+' : ''}${row.avg_clv}%`;
    return `
      <div class="launch-support-item" style="margin-bottom:8px">
        <span class="launch-support-tag ${row.tier_status === 'preferred' ? 'production' : 'limited'}">${escapeHtml(row.tier)}</span>
        <div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;width:100%">
          <div><div style="color:var(--text3);font-size:.72rem">Bets</div><div style="color:var(--text);font-weight:700">${row.bets}</div></div>
          <div><div style="color:var(--text3);font-size:.72rem">Settled%</div><div style="color:var(--text);font-weight:700">${row.settlement_coverage_pct}%</div></div>
          <div><div style="color:var(--text3);font-size:.72rem">ROI</div><div style="color:${row.roi >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700">${row.roi >= 0 ? '+' : ''}${row.roi}%</div></div>
          <div><div style="color:var(--text3);font-size:.72rem">Avg CLV</div><div style="color:var(--text);font-weight:700">${clv}</div></div>
          <div><div style="color:var(--text3);font-size:.72rem">Signal</div><div style="color:var(--text);font-weight:700">${escapeHtml(row.clv_signal)}</div></div>
        </div>
      </div>`;
  }).join('') : '<div style="color:var(--text3);font-size:.82rem">No live lane summary found for this selection.</div>';

  const replaySummary = replayRows.length ? replayRows.map(row => `
    <div class="launch-support-item" style="margin-bottom:8px">
      <span class="launch-support-tag ${row.support_level === 'strong' ? 'production' : row.support_level === 'mixed' ? 'limited' : 'review'}">${escapeHtml(row.support_level)}</span>
      <div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;width:100%">
        <div><div style="color:var(--text3);font-size:.72rem">Rank</div><div style="color:var(--text);font-weight:700">#${row.rank_within_sport}</div></div>
        <div><div style="color:var(--text3);font-size:.72rem">Specs</div><div style="color:var(--text);font-weight:700">${row.spec_count}</div></div>
        <div><div style="color:var(--text3);font-size:.72rem">Games</div><div style="color:var(--text);font-weight:700">${row.games_scored}</div></div>
        <div><div style="color:var(--text3);font-size:.72rem">Accuracy</div><div style="color:var(--text);font-weight:700">${row.avg_accuracy}%</div></div>
        <div><div style="color:var(--text3);font-size:.72rem">Log Loss / ECE</div><div style="color:var(--text);font-weight:700">${row.avg_log_loss} / ${row.avg_ece}</div></div>
      </div>
    </div>`).join('') : '<div style="color:var(--text3);font-size:.82rem">No replay summary is available for this sport/market lane.</div>';

  const govSummary = govRows.length ? govRows.map(rec => `
    <div class="launch-support-item" style="margin-bottom:8px;align-items:flex-start">
      <span class="launch-support-tag ${rec.action === 'promote' ? 'production' : rec.action === 'watch' ? 'limited' : 'review'}">${escapeHtml(rec.action)}</span>
      <div>
        <div style="color:var(--text);font-weight:700">${escapeHtml(rec.tier)} · ${escapeHtml(rec.confidence)} confidence</div>
        <div style="color:var(--text2);font-size:.8rem;margin-top:3px">${escapeHtml(rec.reason)}</div>
        ${rec.replay_note ? `<div style="color:var(--text3);font-size:.74rem;margin-top:4px">${escapeHtml(rec.replay_note)}</div>` : ''}
      </div>
    </div>`).join('') : '<div style="color:var(--text3);font-size:.82rem">No governor recommendation is active for this lane.</div>';

  const settledRows = settled.slice(0, 12).map(row => {
    const pnl = row.profit_units == null ? '—' : `${row.profit_units >= 0 ? '+' : ''}${Number(row.profit_units).toFixed(3)}`;
    const clv = row.clv == null ? '—' : `${Number(row.clv) >= 0 ? '+' : ''}${(Number(row.clv) * 100).toFixed(2)}%`;
    return `<tr>
      <td>${escapeHtml(row.date || '')}</td>
      <td style="color:var(--text)">${escapeHtml(row.team_or_player || '')}</td>
      <td>${row.bet_odds ? Number(row.bet_odds).toFixed(2) : '—'}</td>
      <td style="color:${row.won ? 'var(--green)' : 'var(--red)'}">${row.won ? 'Won' : 'Lost'}</td>
      <td>${clv}</td>
      <td style="color:${row.profit_units >= 0 ? 'var(--green)' : 'var(--red)'}">${pnl}</td>
    </tr>`;
  }).join('');

  const pendingRows = pending.slice(0, 12).map(row => `<tr>
      <td>${escapeHtml(row.date || '')}</td>
      <td style="color:var(--text)">${escapeHtml(row.team_or_player || '')}</td>
      <td>${row.bet_odds ? Number(row.bet_odds).toFixed(2) : '—'}</td>
      <td>${row.edge == null ? '—' : `${(Number(row.edge) * 100).toFixed(1)}%`}</td>
      <td>${escapeHtml(row.bookmaker || '—')}</td>
    </tr>`).join('');

  panel.innerHTML = `
    <div class="grid2" style="margin-bottom:14px">
      <div>
        <div class="segment-kicker" style="margin-bottom:8px">Live Lane Summary</div>
        ${laneSummary}
      </div>
      <div>
        <div class="segment-kicker" style="margin-bottom:8px">Historical Replay Support</div>
        ${replaySummary}
      </div>
    </div>
    <div style="margin-bottom:14px">
      <div class="segment-kicker" style="margin-bottom:8px">Governor View</div>
      ${govSummary}
    </div>
    <div class="grid2">
      <div>
        <div class="segment-kicker" style="margin-bottom:8px">Settled Bets (${settled.length})</div>
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>Date</th><th>Pick</th><th>Odds</th><th>Result</th><th>CLV</th><th>P&amp;L</th></tr></thead>
            <tbody>${settledRows || '<tr><td colspan="6" style="color:var(--text3);padding:18px">No settled bets for this lane yet.</td></tr>'}</tbody>
          </table>
        </div>
      </div>
      <div>
        <div class="segment-kicker" style="margin-bottom:8px">Pending Backlog (${pending.length})</div>
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>Date</th><th>Pick</th><th>Odds</th><th>Edge</th><th>Book</th></tr></thead>
            <tbody>${pendingRows || '<tr><td colspan="5" style="color:var(--text3);padding:18px">No pending bets for this lane.</td></tr>'}</tbody>
          </table>
        </div>
      </div>
    </div>`;
}

function showResultsLaneDetail(source, sport, market, tier = null) {
  state.selectedLane = {
    source,
    sport: decodeURIComponent(String(sport || '')),
    market: decodeURIComponent(String(market || '')),
    tier: tier == null ? null : decodeURIComponent(String(tier || '')),
  };
  _renderLaneDrilldown(state.selectedLane);
}

function _renderVersionSummary(versionSummary) {
  const panel = document.getElementById('version-summary-panel');
  const countEl = document.getElementById('res-version-count');
  if (!panel) return;

  const latest = (versionSummary && versionSummary.latest) || {};
  const modelRows = (versionSummary && versionSummary.model_rows) || [];
  const rowCounts = (versionSummary && versionSummary.row_counts) || [];
  const latestModels = modelRows.length
    ? modelRows.map(row => `
        <tr>
          <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
          <td><code>${escapeHtml(row.tag || '—')}</code></td>
          <td>${row.calibrator_present ? 'yes' : 'no'}</td>
        </tr>
      `).join('')
    : '<tr><td colspan="3" style="color:var(--text3);padding:18px">No scan snapshot has been recorded yet.</td></tr>';

  const countsMarkup = rowCounts.length
    ? rowCounts.slice(0, 6).map(row => `
        <div class="launch-support-item">
          <span class="launch-support-tag ${row.source === 'settled' ? 'production' : 'limited'}">${escapeHtml(row.source)}</span>
          <code>${escapeHtml(row.policy_hash)}</code> on ${escapeHtml(row.scan_date || 'unknown')} • ${row.count} row${row.count !== 1 ? 's' : ''}
        </div>
      `).join('')
    : '<div style="color:var(--text3);font-size:.82rem">No tracked rows carry version metadata yet. New scans will start filling this in.</div>';

  if (countEl) {
    countEl.textContent = modelRows.length
      ? `${versionSummary.pending_with_snapshot || 0} pending • ${versionSummary.settled_with_snapshot || 0} settled`
      : 'No snapshot yet';
  }

  panel.innerHTML = `
    <div class="results-card-grid" style="margin-bottom:14px">
      <div class="stat-card blue">
        <div class="stat-label">Policy Hash</div>
        <div class="stat-value" style="font-size:1.15rem">${escapeHtml(latest.policy_hash || '—')}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Scan Date</div>
        <div class="stat-value" style="font-size:1.15rem">${escapeHtml(latest.scan_date || '—')}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Pending Tagged</div>
        <div class="stat-value" style="font-size:1.15rem">${versionSummary.pending_with_snapshot || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Settled Tagged</div>
        <div class="stat-value" style="font-size:1.15rem">${versionSummary.settled_with_snapshot || 0}</div>
      </div>
    </div>
    <div class="grid2">
      <div>
        <div class="segment-kicker" style="margin-bottom:8px">Latest Scan Models</div>
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>Lane</th><th>Tag</th><th>Calibrator</th></tr></thead>
            <tbody>${latestModels}</tbody>
          </table>
        </div>
      </div>
      <div>
        <div class="segment-kicker" style="margin-bottom:8px">Tracked Row Coverage</div>
        <div class="launch-support-strip compact">${countsMarkup}</div>
      </div>
    </div>`;
}

function _renderRetrainTriggers(retrainTriggers) {
  const panel = document.getElementById('retrain-triggers-panel');
  const countEl = document.getElementById('res-retrain-count');
  if (!panel) return;
  const summary = (retrainTriggers && retrainTriggers.summary) || {};
  const rows = (retrainTriggers && retrainTriggers.rows) || [];
  if (countEl) {
    countEl.textContent = rows.length
      ? `${summary.retrain || 0} retrain • ${summary.watch || 0} watch`
      : 'No sports evaluated';
  }
  if (!rows.length) {
    panel.innerHTML = '<div style="color:var(--text3);font-size:.82rem">No settled sport-level evidence yet.</div>';
    return;
  }

  const cards = `
    <div class="results-card-grid" style="margin-bottom:14px">
      <div class="stat-card ${summary.retrain ? 'red' : ''}">
        <div class="stat-label">Retrain</div>
        <div class="stat-value">${summary.retrain || 0}</div>
      </div>
      <div class="stat-card ${summary.watch ? '' : 'green'}">
        <div class="stat-label">Watch</div>
        <div class="stat-value">${summary.watch || 0}</div>
      </div>
      <div class="stat-card green">
        <div class="stat-label">Hold</div>
        <div class="stat-value">${summary.hold || 0}</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-label">Sports</div>
        <div class="stat-value">${summary.sports || 0}</div>
      </div>
    </div>`;

  const table = `
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Sport</th>
            <th>Action</th>
            <th>Confidence</th>
            <th>Tag</th>
            <th>Bets</th>
            <th>Settled%</th>
            <th>CLV%</th>
            <th>ROI</th>
            <th>Avg CLV</th>
            <th>Weak Lanes</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => {
            const actionClass = row.action === 'retrain' ? 'experimental' : row.action === 'watch' ? 'limited' : 'production';
            const roiColor = row.roi >= 0 ? 'var(--green)' : 'var(--red)';
            const clvLabel = row.avg_clv == null ? '—' : `${row.avg_clv >= 0 ? '+' : ''}${row.avg_clv}%`;
            return `
              <tr>
                <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
                <td><span class="launch-support-tag ${actionClass}">${escapeHtml(row.action)}</span></td>
                <td>${escapeHtml(row.confidence)}</td>
                <td><code>${escapeHtml(row.latest_tag || '—')}</code></td>
                <td>${row.bets}</td>
                <td>${row.settlement_coverage_pct}%</td>
                <td>${row.clv_coverage_pct}%</td>
                <td style="color:${roiColor}">${row.roi >= 0 ? '+' : ''}${row.roi}%</td>
                <td>${clvLabel}</td>
                <td>${row.weak_lanes}/${row.lanes}</td>
                <td style="max-width:360px">${escapeHtml(row.reason)}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>`;

  panel.innerHTML = cards + table;
}

function _renderRebuildCandidates(rebuildCandidates) {
  const panel = document.getElementById('rebuild-candidates-panel');
  const countEl = document.getElementById('res-rebuild-count');
  if (!panel) return;
  const summary = (rebuildCandidates && rebuildCandidates.summary) || {};
  const rows = (rebuildCandidates && rebuildCandidates.rows) || [];
  if (countEl) {
    countEl.textContent = rows.length
      ? `${summary.candidates || 0} candidate${summary.candidates === 1 ? '' : 's'}`
      : 'No rebuild candidates';
  }
  if (!rows.length) {
    panel.innerHTML = '<div style="color:var(--text3);font-size:.82rem">No weak lanes are mature enough for a formal rebuild action yet.</div>';
    return;
  }

  const cards = `
    <div class="results-card-grid" style="margin-bottom:14px">
      <div class="stat-card ${summary.retrain ? 'red' : ''}">
        <div class="stat-label">Retrain</div>
        <div class="stat-value">${summary.retrain || 0}</div>
      </div>
      <div class="stat-card ${summary.policy ? '' : 'green'}">
        <div class="stat-label">Policy Moves</div>
        <div class="stat-value">${summary.policy || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Watch</div>
        <div class="stat-value">${summary.watch || 0}</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-label">Candidates</div>
        <div class="stat-value">${summary.candidates || 0}</div>
      </div>
    </div>`;

  panel.innerHTML = cards + rows.map(row => {
    const actionClass = (
      row.action === 'retrain' ? 'experimental' :
      row.action === 'watch' ? 'limited' :
      row.action === 'policy_pause' ? 'review' :
      row.action === 'policy_tighten' ? 'limited' :
      'production'
    );
    const roiColor = row.roi >= 0 ? 'var(--green)' : 'var(--red)';
    const clvLabel = row.avg_clv == null ? '—' : `${row.avg_clv >= 0 ? '+' : ''}${row.avg_clv}%`;
    const template = row.policy_template || null;
    const templateMarkup = template ? `
      <div style="margin-top:6px;color:var(--text2);font-size:.8rem">
        <strong>Suggested policy values:</strong>
        <code>status=${escapeHtml(String(template.status))}</code>,
        <code>score=${escapeHtml(String(template.score))}</code>,
        <code>stake_multiplier=${escapeHtml(String(template.stake_multiplier))}</code>
      </div>` : '';
    return `
      <div class="launch-support-item" style="margin-bottom:10px;align-items:flex-start">
        <span class="launch-support-tag ${actionClass}" style="min-width:110px;text-align:center">${escapeHtml(row.action)}</span>
        <div style="width:100%">
          <div style="font-weight:700;color:var(--text);margin-bottom:4px">${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)} · ${escapeHtml(row.market)} · ${escapeHtml(row.tier)}</div>
          <div style="color:var(--text2);font-size:.82rem">${escapeHtml(row.rationale)}</div>
          <div style="color:var(--text3);font-size:.75rem;margin-top:4px">
            Trigger: ${escapeHtml(row.trigger)} · Confidence: ${escapeHtml(row.confidence)} · Replay: ${escapeHtml(row.replay_support || 'missing')} · Model: <code>${escapeHtml(row.model_tag || '—')}</code>
          </div>
          <div style="color:var(--text3);font-size:.75rem;margin-top:4px">
            Bets ${row.bets}/${row.tracked_total} tracked · Settled ${row.settlement_coverage_pct}% · CLV ${row.clv_coverage_pct}% · ROI <span style="color:${roiColor}">${row.roi >= 0 ? '+' : ''}${row.roi}%</span> · Avg CLV ${clvLabel}
          </div>
          ${(row.calibration_gap_pp != null || row.calibration_brier != null || row.calibration_log_loss != null) ? `
            <div style="color:var(--text3);font-size:.75rem;margin-top:4px">
              Calibration gap ${row.calibration_gap_pp != null ? `${row.calibration_gap_pp}pp` : '—'} · Brier ${row.calibration_brier != null ? row.calibration_brier : '—'} · Log loss ${row.calibration_log_loss != null ? row.calibration_log_loss : '—'}
            </div>` : ''}
          ${row.draft_command ? `<div style="margin-top:6px;color:var(--text2);font-size:.8rem"><strong>Draft command:</strong> <code>${escapeHtml(row.draft_command)}</code></div>` : ''}
          ${row.draft_policy ? `<div style="margin-top:6px;color:var(--text2);font-size:.8rem"><strong>Draft policy move:</strong> ${escapeHtml(row.draft_policy)}</div>` : ''}
          ${templateMarkup}
          <div style="margin-top:6px;color:var(--text2);font-size:.8rem"><strong>Next step:</strong> ${escapeHtml(row.next_step || 'Review the lane in context before acting.')}</div>
        </div>
      </div>`;
  }).join('');
}

function _renderSettlementReliability(reliability) {
  const panel = document.getElementById('settlement-reliability-panel');
  const countEl = document.getElementById('res-settle-count');
  if (!panel) return;
  const summary = (reliability && reliability.summary) || {};
  const rows = (reliability && reliability.rows) || [];
  const reasonRows = (reliability && reliability.reason_rows) || [];
  const pendingSamples = Array.isArray(reliability && reliability.pending_samples) ? reliability.pending_samples : [];
  const lastAttempt = (reliability && reliability.last_attempt) || {};
  const unresolvedReasonRows = Array.isArray(lastAttempt.unresolved_by_reason) ? lastAttempt.unresolved_by_reason : [];
  const unresolvedSportRows = Array.isArray(lastAttempt.unresolved_by_sport) ? lastAttempt.unresolved_by_sport : [];
  const unresolvedSamples = Array.isArray(lastAttempt.unresolved_samples) ? lastAttempt.unresolved_samples : [];
  if (countEl) {
    countEl.textContent = rows.length
      ? `${summary.coverage_pct || 0}% covered • ${summary.overdue_total || 0} overdue`
      : 'No tracked bets yet';
  }
  if (!rows.length) {
    panel.innerHTML = '<div style="color:var(--text3);font-size:.82rem">No tracked settlement data yet.</div>';
    return;
  }

  const cards = `
    <div class="results-card-grid" style="margin-bottom:14px">
      <div class="stat-card blue">
        <div class="stat-label">Tracked</div>
        <div class="stat-value">${summary.tracked_total || 0}</div>
      </div>
      <div class="stat-card green">
        <div class="stat-label">Settled</div>
        <div class="stat-value">${summary.settled_total || 0}</div>
      </div>
      <div class="stat-card ${summary.overdue_total ? 'red' : ''}">
        <div class="stat-label">Overdue</div>
        <div class="stat-value">${summary.overdue_total || 0}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Manual Parlays</div>
        <div class="stat-value">${summary.manual_parlays_pending || 0}</div>
        <div class="stat-sub">${summary.manual_legs_pending || 0} unresolved legs</div>
      </div>
    </div>`;

  const reasons = reasonRows.length
    ? `<div class="launch-support-strip compact" style="margin-bottom:14px">${reasonRows.map(row => `
        <div class="launch-support-item">
          <span class="launch-support-tag ${String(row.reason_key).startsWith('overdue') ? 'experimental' : 'limited'}">${escapeHtml(row.reason_label)}</span>
          ${row.count}
        </div>`).join('')}</div>`
    : '<div style="color:var(--text3);font-size:.82rem;margin-bottom:14px">No unresolved tracked bets right now.</div>';

  const lastAttemptMarkup = lastAttempt.attempted_at
    ? `<div style="color:var(--text3);font-size:.8rem;margin-bottom:10px">
         Last bulk settle: ${escapeHtml(lastAttempt.attempted_at)} • sources: ${escapeHtml((lastAttempt.score_sources || []).join(', ') || 'none')} • errors: ${escapeHtml((lastAttempt.errors || []).join(' | ') || 'none')}
         ${(Array.isArray(lastAttempt.unresolved_summary) && lastAttempt.unresolved_summary.length)
           ? `<br>Unresolved: ${escapeHtml(lastAttempt.unresolved_summary.slice(0, 4).map(item => `${item.scope}:${item.reason} (${item.count})`).join(' · '))}`
           : ''}
       </div>`
    : '<div style="color:var(--text3);font-size:.8rem;margin-bottom:10px">No bulk settle attempt has been recorded since the app started.</div>';

  const unresolvedBreakdown = unresolvedReasonRows.length
    ? `
      <div style="margin-bottom:14px">
        <div style="font-size:.8rem;font-weight:700;color:var(--text2);margin-bottom:8px">Unresolved Breakdown</div>
        <div class="launch-support-strip compact" style="margin-bottom:10px">
          ${unresolvedReasonRows.map(row => `
            <div class="launch-support-item">
              <span class="launch-support-tag ${String(row.reason).includes('mismatch') ? 'limited' : String(row.reason).includes('no_score') ? 'experimental' : ''}">${escapeHtml(String(row.reason).replaceAll('_', ' '))}</span>
              ${row.count}
            </div>`).join('')}
        </div>
        ${unresolvedSportRows.length ? `
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Sport</th>
                  <th>Reason</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                ${unresolvedSportRows.slice(0, 10).map(row => `
                  <tr>
                    <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport || 'unknown')}</td>
                    <td>${escapeHtml(String(row.reason || '').replaceAll('_', ' '))}</td>
                    <td>${row.count}</td>
                  </tr>`).join('')}
              </tbody>
            </table>
          </div>` : ''}
      </div>`
    : '';

  const unresolvedSampleMarkup = unresolvedSamples.length
    ? `
      <div style="margin-bottom:14px">
        <div style="font-size:.8rem;font-weight:700;color:var(--text2);margin-bottom:8px">Latest Unresolved Examples</div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Scope</th>
                <th>Sport</th>
                <th>Match</th>
                <th>Pick</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              ${unresolvedSamples.slice(0, 6).map(row => `
                <tr>
                  <td>${escapeHtml(row.scope || 'unknown')}</td>
                  <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport || 'unknown')}</td>
                  <td>${escapeHtml(row.match || '—')}</td>
                  <td>${escapeHtml(row.pick || '—')}</td>
                  <td>${escapeHtml(String(row.reason || '').replaceAll('_', ' '))}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`
    : '';

  const pendingSampleMarkup = pendingSamples.length
    ? `
      <div style="margin-bottom:14px">
        <div style="font-size:.8rem;font-weight:700;color:var(--text2);margin-bottom:8px">Oldest Pending Tracker Rows</div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>
                <th>Sport</th>
                <th>Match</th>
                <th>Pick</th>
                <th>Reason</th>
                <th>Age</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              ${pendingSamples.slice(0, 8).map(row => `
                <tr>
                  <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport || 'unknown')}</td>
                  <td>${escapeHtml(row.match || '—')}</td>
                  <td>${escapeHtml(row.pick || '—')}</td>
                  <td>${escapeHtml(row.reason_text || row.reason || '—')}</td>
                  <td>${row.age_days == null || Number(row.age_days) < 0 ? '—' : `${Math.round(Number(row.age_days))}d`}</td>
                  <td>${escapeHtml(row.event_date || '—')}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>`
    : '';

  const table = `
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Sport</th>
            <th>Tracked</th>
            <th>Settled</th>
            <th>Pending</th>
            <th>Settled%</th>
            <th>Overdue</th>
            <th>Awaiting</th>
            <th>Missing Time</th>
            <th>Oldest Pending</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(row => `
            <tr>
              <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
              <td>${row.tracked_total}</td>
              <td>${row.settled_count}</td>
              <td>${row.pending_count}</td>
              <td>${row.settlement_coverage_pct}%</td>
              <td style="color:${row.overdue_count ? 'var(--red)' : 'var(--text)'}">${row.overdue_count}</td>
              <td>${row.awaiting_count}</td>
              <td>${row.missing_time_count}</td>
              <td>${row.oldest_pending_days == null ? '—' : `${row.oldest_pending_days}d`}</td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`;

  panel.innerHTML = cards + lastAttemptMarkup + unresolvedBreakdown + unresolvedSampleMarkup + pendingSampleMarkup + reasons + table;
}

async function loadResults(date) {
  if (date !== undefined) _resultDate = date === 'today' ? new Date().toISOString().slice(0,10) : date;
  const url = _resultDate && _resultDate !== 'all' ? `/api/results?date=${_resultDate}` : '/api/results';
  try {
    // Load date chips first time
    const datesR = await fetch('/api/results/dates');
    const datesD = await datesR.json();
    const chipsEl = document.getElementById('date-chips');
    const todayStr = new Date().toISOString().slice(0,10);
    if (chipsEl && datesD.dates) {
      let html = `<button class="chip${_resultDate==='all'?' active':''}" data-date="all" onclick="selectResultDate(this,'all')">All Time</button>`;
      html += `<button class="chip${_resultDate===todayStr?' active':''}" data-date="today" onclick="selectResultDate(this,'today')">Today</button>`;
      chipsEl.innerHTML = html;
    }
    const resultDateSelect = document.getElementById('result-date-select');
    if (resultDateSelect && datesD.dates) {
      resultDateSelect.min = datesD.dates.length ? datesD.dates[datesD.dates.length - 1] : '';
      resultDateSelect.max = datesD.dates.length ? datesD.dates[0] : '';
      resultDateSelect.value = _resultDate !== 'all' && _resultDate !== todayStr ? _resultDate : '';
    }

    const r = await fetch(url);
    const d = await r.json();
    state.resultsData = d;
    if (d.error && !d.pending?.length) {
      document.getElementById('res-total').textContent = '0';
      document.getElementById('res-wr').textContent = '—';
      document.getElementById('res-pnl').textContent = '—';
      document.getElementById('res-roi').textContent = '—';
    }

    // Stat cards
    const valueSeg = (d.segments && d.segments.value_bets) || {bets:0, win_rate:0, pnl:0, roi:0};
    const parlaySeg = (d.segments && d.segments.parlays) || {bets:0, win_rate:0, pnl:0, roi:0};
    document.getElementById('res-value-total').textContent = valueSeg.bets;
    document.getElementById('res-value-wr').textContent    = valueSeg.win_rate + '%';
    document.getElementById('res-value-pnl').textContent   = (valueSeg.pnl >= 0 ? '+' : '') + valueSeg.pnl;
    document.getElementById('res-value-roi').textContent   = (valueSeg.roi >= 0 ? '+' : '') + valueSeg.roi + '%';
    document.getElementById('res-value-wr-card').className  = 'stat-card ' + (valueSeg.bets ? (valueSeg.win_rate >= 50 ? 'green' : 'red') : '');
    document.getElementById('res-value-pnl-card').className = 'stat-card ' + (valueSeg.pnl >= 0 ? 'green' : 'red');
    document.getElementById('res-value-roi-card').className = 'stat-card ' + (valueSeg.roi >= 0 ? 'green' : 'red');
    document.getElementById('res-parlay-total').textContent = parlaySeg.bets;
    document.getElementById('res-parlay-wr').textContent    = parlaySeg.win_rate + '%';
    document.getElementById('res-parlay-pnl').textContent   = (parlaySeg.pnl >= 0 ? '+' : '') + parlaySeg.pnl;
    document.getElementById('res-parlay-roi').textContent   = (parlaySeg.roi >= 0 ? '+' : '') + parlaySeg.roi + '%';
    document.getElementById('res-parlay-wr-card').className  = 'stat-card ' + (parlaySeg.bets ? (parlaySeg.win_rate >= 50 ? 'green' : 'red') : '');
    document.getElementById('res-parlay-pnl-card').className = 'stat-card ' + (parlaySeg.pnl >= 0 ? 'green' : 'red');
    document.getElementById('res-parlay-roi-card').className = 'stat-card ' + (parlaySeg.roi >= 0 ? 'green' : 'red');
    _renderVersionSummary(d.version_summary || {});
    _renderSettlementReliability(d.settlement_reliability || {});
    _renderRetrainTriggers(d.retrain_triggers || {});
    _renderRebuildCandidates(d.rebuild_candidates || {});

    // Charts are nice-to-have. If Chart.js is blocked/offline, keep the tables working.
    if (window.Chart) {
      try {
        if (state.pnlChart) state.pnlChart.destroy();
        const pnlCtx = document.getElementById('pnl-chart')?.getContext('2d');
        if (pnlCtx) {
          state.pnlChart = new Chart(pnlCtx, {
            type: 'line',
            data: {
              labels: (d.pnl||[]).map(r => r.date),
              datasets: [{
                label: 'Cumulative P&L (units)',
                data: (d.pnl||[]).map(r => r.cumulative_pnl),
                borderColor: d.pnl?.length && d.pnl[d.pnl.length-1].cumulative_pnl >= 0 ? '#10b981' : '#ef4444',
                backgroundColor: 'rgba(59,130,246,.06)',
                borderWidth: 2, pointRadius: 3, tension: 0.3, fill: true,
              }],
            },
            options: {
              responsive: true, maintainAspectRatio: false,
              plugins: { legend: { display: false } },
              scales: {
                x: { ticks: { color: '#64748b', maxTicksLimit: 8 }, grid: { color: '#1e2d45' } },
                y: { ticks: { color: '#64748b' }, grid: { color: '#1e2d45' } },
              },
            },
          });
        }

        if (state.marketChart) state.marketChart.destroy();
        const mCtx = document.getElementById('market-chart')?.getContext('2d');
        const mLabels = Object.keys(d.by_market || {});
        const mRois   = mLabels.map(k => d.by_market[k].roi);
        if (mCtx) {
          state.marketChart = new Chart(mCtx, {
            type: 'bar',
            data: {
              labels: mLabels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
              datasets: [{
                label: 'ROI %', data: mRois,
                backgroundColor: mRois.map(v => v >= 0 ? 'rgba(16,185,129,.7)' : 'rgba(239,68,68,.7)'),
                borderRadius: 6,
              }],
            },
            options: {
              responsive: true, maintainAspectRatio: false,
              plugins: { legend: { display: false } },
              scales: {
                x: { ticks: { color: '#64748b' }, grid: { display: false } },
                y: { ticks: { color: '#64748b', callback: v => v + '%' }, grid: { color: '#1e2d45' } },
              },
            },
          });
        }
      } catch(chartError) {
        console.warn('Results charts unavailable:', chartError);
      }
    }

    // Sport bars
    const barsEl = document.getElementById('sport-bars');
    barsEl.innerHTML = Object.entries(d.by_sport || {}).map(([sport, s]) => {
      const pct = s.win_rate;
      const roiColor = s.roi >= 0 ? 'var(--green)' : 'var(--red)';
      return `
        <div class="sport-bar-row">
          <div class="sport-bar-label">${SPORT_ICONS[sport]||''} ${sport}</div>
          <div class="sport-bar-track"><div class="sport-bar-fill${s.roi >= 0 ? ' green' : ' red'}" style="width:${pct}%"></div></div>
          <div class="sport-bar-pct" style="color:${pct>=50?'var(--green)':'var(--red)'}">${pct}%</div>
          <div class="sport-bar-roi" style="color:${roiColor}">${s.roi >= 0 ? '+' : ''}${s.roi}% ROI</div>
        </div>`;
    }).join('') || '<div style="color:var(--text3);padding:8px">No settled data for this period.</div>';

    // Performance matrix
    const matrixRows = d.performance_matrix || [];
    const matrixBody = document.getElementById('performance-matrix-tbody');
    const matrixCount = document.getElementById('res-matrix-count');
    if (matrixCount) {
      matrixCount.textContent = matrixRows.length ? `${matrixRows.length} lane${matrixRows.length !== 1 ? 's' : ''}` : 'No settled lanes yet';
    }
    if (matrixBody) {
      matrixBody.innerHTML = matrixRows.length ? matrixRows.map(row => {
        const roiColor = row.roi >= 0 ? 'var(--green)' : 'var(--red)';
        const pnlColor = row.pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const tierClass = row.tier_status === 'preferred' ? 'production' : 'limited';
        const signalColor = (
          row.clv_signal === 'confirmed' ? 'var(--green)' :
          row.clv_signal === 'variance' ? 'var(--blue)' :
          row.clv_signal === 'lucky' ? 'var(--amber)' :
          row.clv_signal === 'weak' ? 'var(--red)' :
          'var(--text3)'
        );
        const clvLabel = row.avg_clv == null ? '—' : `${row.avg_clv >= 0 ? '+' : ''}${row.avg_clv}%`;
        const clvCoverageLabel = row.clv_covered ? `${row.clv_covered} / ${row.bets}` : '0 / ' + row.bets;
        const clvPositiveLabel = row.clv_positive_pct == null ? `— (${clvCoverageLabel})` : `${row.clv_positive_pct}% (${clvCoverageLabel})`;
        return `
          <tr onclick="showResultsLaneDetail('performance', '${encodeURIComponent(row.sport)}', '${encodeURIComponent(row.market)}', '${encodeURIComponent(row.tier)}')" style="cursor:pointer">
            <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
            <td><span class="tag tag-market">${escapeHtml(row.market)}</span></td>
            <td><span class="launch-support-tag ${tierClass}">${escapeHtml(row.tier)}</span></td>
            <td>${row.bets}</td>
            <td>${row.settlement_coverage_pct}%${row.pending_count ? ` <span style="color:var(--text3)">(${row.bets}/${row.tracked_total})</span>` : ''}</td>
            <td>${row.win_rate}%</td>
            <td style="color:${roiColor}">${row.roi >= 0 ? '+' : ''}${row.roi}%</td>
            <td style="color:${pnlColor}">${row.pnl >= 0 ? '+' : ''}${row.pnl}</td>
            <td>${row.avg_edge >= 0 ? '+' : ''}${row.avg_edge}%</td>
            <td>${clvLabel}</td>
            <td>${clvPositiveLabel}</td>
            <td style="color:${signalColor};font-weight:700">${escapeHtml(row.clv_signal)}</td>
          </tr>`;
      }).join('') : '<tr><td colspan="12" style="color:var(--text3);padding:18px">No settled lane data for this period.</td></tr>';
    }

    // Parlay performance matrix
    const parlayPerf = d.parlay_performance || {};
    const parlaySummaryCards = parlayPerf.summary_cards || [];
    const parlayMatrixRows = parlayPerf.matrix_rows || [];
    const parlayMatrixCount = document.getElementById('res-parlay-matrix-count');
    const parlayPerfCards = document.getElementById('parlay-performance-cards');
    const parlayPerfBody = document.getElementById('parlay-performance-tbody');
    if (parlayMatrixCount) {
      parlayMatrixCount.textContent = parlayMatrixRows.length ? `${parlayMatrixRows.length} group${parlayMatrixRows.length !== 1 ? 's' : ''}` : 'No tracked parlays yet';
    }
    if (parlayPerfCards) {
      parlayPerfCards.innerHTML = parlaySummaryCards.length ? parlaySummaryCards.map(card => `
        <div class="stat-card">
          <div class="stat-label">${escapeHtml(card.source)} · ${escapeHtml(card.style)}</div>
          <div class="stat-value">${card.bets}</div>
          <div class="stat-sub">WR ${card.win_rate}% · ROI ${card.roi >= 0 ? '+' : ''}${card.roi}% · Odds ${card.avg_odds}x</div>
          <div class="stat-sub" style="margin-top:4px;color:${card.pnl >= 0 ? 'var(--green)' : 'var(--red)'}">P&amp;L ${card.pnl >= 0 ? '+' : ''}${card.pnl}</div>
        </div>
      `).join('') : '<div style="color:var(--text3);font-size:.82rem">No tracked system parlays have settled yet.</div>';
    }
    if (parlayPerfBody) {
      parlayPerfBody.innerHTML = parlayMatrixRows.length ? parlayMatrixRows.map(row => {
        const roiColor = row.roi >= 0 ? 'var(--green)' : 'var(--red)';
        const pnlColor = row.pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const styleClass = row.style === 'Value' ? 'production' : 'limited';
        return `
          <tr onclick="showParlayDrilldown('${encodeURIComponent(row.source)}', '${encodeURIComponent(row.style)}', '${encodeURIComponent(row.bracket)}', '${row.n_legs}', '${encodeURIComponent(row.sport_mix)}')" style="cursor:pointer">
            <td>${escapeHtml(row.source)}</td>
            <td><span class="launch-support-tag ${styleClass}">${escapeHtml(row.style)}</span></td>
            <td>${escapeHtml(row.bracket)}</td>
            <td>${row.n_legs}</td>
            <td>${escapeHtml(row.sport_mix)}</td>
            <td>${row.bets}</td>
            <td>${row.win_rate}%</td>
            <td style="color:${roiColor}">${row.roi >= 0 ? '+' : ''}${row.roi}%</td>
            <td style="color:${pnlColor}">${row.pnl >= 0 ? '+' : ''}${row.pnl}</td>
            <td>${row.avg_odds}x</td>
            <td>${row.avg_edge >= 0 ? '+' : ''}${row.avg_edge}%</td>
            <td>${row.avg_ev}x</td>
          </tr>`;
      }).join('') : '<tr><td colspan="12" style="color:var(--text3);padding:18px">No tracked or saved settled parlays are available yet.</td></tr>';
    }

    // Replay support matrix
    const replayRows = d.replay_support_matrix || [];
    const replayBody = document.getElementById('replay-support-tbody');
    const replayCount = document.getElementById('res-replay-count');
    if (replayCount) {
      replayCount.textContent = replayRows.length ? `${replayRows.length} lane${replayRows.length !== 1 ? 's' : ''}` : 'No replay summaries yet';
    }
    if (replayBody) {
      replayBody.innerHTML = replayRows.length ? replayRows.map(row => {
        const supportClass = (
          row.support_level === 'strong' ? 'production' :
          row.support_level === 'mixed' ? 'limited' :
          'review'
        );
        return `
          <tr onclick="showResultsLaneDetail('replay', '${encodeURIComponent(row.sport)}', '${encodeURIComponent(row.market)}')" style="cursor:pointer">
            <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
            <td><span class="tag tag-market">${escapeHtml(row.market)}</span></td>
            <td><span class="launch-support-tag ${supportClass}">${escapeHtml(row.support_level)}</span></td>
            <td>#${row.rank_within_sport}</td>
            <td>${row.spec_count}</td>
            <td>${row.games_scored}</td>
            <td>${row.avg_accuracy}%</td>
            <td>${row.avg_log_loss}</td>
            <td>${row.avg_ece}</td>
          </tr>`;
      }).join('') : '<tr><td colspan="9" style="color:var(--text3);padding:18px">No replay summaries are available yet.</td></tr>';
    }

    // Replay policy audit
    const auditRows = d.replay_policy_audit || [];
    const auditBody = document.getElementById('replay-policy-audit-tbody');
    const auditCount = document.getElementById('res-replay-audit-count');
    if (auditCount) {
      const misaligned = auditRows.filter(row => row.alignment !== 'aligned').length;
      auditCount.textContent = auditRows.length ? `${misaligned} mismatch${misaligned !== 1 ? 'es' : ''}` : 'No replay audit rows';
    }
    if (auditBody) {
      auditBody.innerHTML = auditRows.length ? auditRows.map(row => {
        const currentClass = row.current_status === 'preferred' ? 'production' : row.current_status === 'experimental' ? 'limited' : 'review';
        const replayClass = row.support_level === 'strong' ? 'production' : row.support_level === 'mixed' ? 'limited' : 'review';
        const recommendedClass = row.recommended_status === 'preferred' ? 'production' : row.recommended_status === 'experimental' ? 'limited' : 'review';
        const alignmentColor = (
          row.alignment === 'aligned' ? 'var(--green)' :
          row.alignment === 'underpromoted' ? 'var(--blue)' :
          'var(--amber)'
        );
        return `
          <tr onclick="showResultsLaneDetail('audit', '${encodeURIComponent(row.sport)}', '${encodeURIComponent(row.market)}')" style="cursor:pointer">
            <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
            <td><span class="tag tag-market">${escapeHtml(row.market)}</span></td>
            <td><span class="launch-support-tag ${currentClass}">${escapeHtml(row.current_label)}</span></td>
            <td><span class="launch-support-tag ${replayClass}">${escapeHtml(row.support_level)}</span></td>
            <td><span class="launch-support-tag ${recommendedClass}">${escapeHtml(row.recommended_label)}</span></td>
            <td style="color:${alignmentColor};font-weight:700">${escapeHtml(row.alignment)}</td>
            <td>#${row.rank_within_sport}</td>
            <td>${row.games_scored}</td>
          </tr>`;
      }).join('') : '<tr><td colspan="8" style="color:var(--text3);padding:18px">No replay policy audit rows are available yet.</td></tr>';
    }

    // Replay portfolio simulator
    const portfolio = d.replay_portfolio || {};
    const scenarios = d.replay_scenarios || {};
    const portfolioRows = portfolio.lane_rows || [];
    const portfolioCount = document.getElementById('res-replay-portfolio-count');
    const scenarioCards = document.getElementById('replay-scenario-cards');
    const portfolioCards = document.getElementById('replay-portfolio-cards');
    const portfolioBody = document.getElementById('replay-portfolio-tbody');
    if (portfolioCount) {
      portfolioCount.textContent = portfolioRows.length ? `${portfolioRows.length} lane${portfolioRows.length !== 1 ? 's' : ''}` : 'No replay lanes';
    }
    if (scenarioCards) {
      const currentScenario = scenarios.current_policy || {};
      const alignedScenario = scenarios.replay_aligned_policy || {};
      scenarioCards.innerHTML = [
        ['Current Policy', currentScenario, 'Current live publish set over the replay universe.'],
        ['Replay-Aligned', alignedScenario, 'Counterfactual publish set if strong/mixed replay lanes were the policy anchor.'],
        ['Scenario Delta', {
          games_scored: scenarios.delta_games ?? 0,
          lanes: scenarios.delta_lanes ?? 0,
          avg_accuracy: null,
          avg_log_loss: null,
        }, `Promote ${scenarios.promoted_by_alignment ?? 0} lane(s), hold out ${scenarios.held_out_by_alignment ?? 0} lane(s).`],
      ].map(([label, bucket, copy]) => {
        const games = bucket.games_scored ?? 0;
        const lanes = bucket.lanes ?? 0;
        const acc = bucket.avg_accuracy == null ? '—' : `${bucket.avg_accuracy}%`;
        const ll = bucket.avg_log_loss == null ? '—' : bucket.avg_log_loss;
        const deltaLabel = label === 'Scenario Delta'
          ? `${games >= 0 ? '+' : ''}${games} games`
          : `${games}`;
        const laneLabel = label === 'Scenario Delta'
          ? `${lanes >= 0 ? '+' : ''}${lanes} lanes`
          : `${lanes} lane${Math.abs(Number(lanes || 0)) === 1 ? '' : 's'}`;
        return `
          <div class="stat-card">
            <div class="stat-label">${escapeHtml(label)}</div>
            <div class="stat-value">${deltaLabel}</div>
            <div class="stat-sub">${laneLabel} · Acc ${acc} · LL ${ll}</div>
            <div class="stat-sub" style="margin-top:4px">${escapeHtml(copy)}</div>
          </div>`;
      }).join('');
    }
    const bucketMeta = [
      ['preferred_live', 'Preferred Live'],
      ['limited_live', 'Limited Live'],
      ['held_out', 'Held Out'],
      ['published_total', 'Published Total'],
    ];
    if (portfolioCards) {
      portfolioCards.innerHTML = bucketMeta.map(([key, label]) => {
        const bucket = portfolio[key] || {};
        const acc = bucket.avg_accuracy == null ? '—' : `${bucket.avg_accuracy}%`;
        const logLoss = bucket.avg_log_loss == null ? '—' : bucket.avg_log_loss;
        return `
          <div class="stat-card">
            <div class="stat-label">${escapeHtml(label)}</div>
            <div class="stat-value">${bucket.games_scored || 0}</div>
            <div class="stat-sub">${bucket.lanes || 0} lane${Number(bucket.lanes || 0) === 1 ? '' : 's'} · Acc ${acc} · LL ${logLoss}</div>
          </div>`;
      }).join('');
    }
    if (portfolioBody) {
      portfolioBody.innerHTML = portfolioRows.length ? portfolioRows.map(row => {
        const bucketClass = row.bucket === 'preferred_live' ? 'production' : row.bucket === 'limited_live' ? 'limited' : 'review';
        const replayClass = row.support_level === 'strong' ? 'production' : row.support_level === 'mixed' ? 'limited' : 'review';
        return `
          <tr onclick="showResultsLaneDetail('portfolio', '${encodeURIComponent(row.sport)}', '${encodeURIComponent(row.market)}')" style="cursor:pointer">
            <td>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)}</td>
            <td><span class="tag tag-market">${escapeHtml(row.market)}</span></td>
            <td><span class="launch-support-tag ${bucketClass}">${escapeHtml(row.bucket)}</span></td>
            <td><span class="launch-support-tag ${bucketClass}">${escapeHtml(row.label)}</span></td>
            <td><span class="launch-support-tag ${replayClass}">${escapeHtml(row.support_level)}</span></td>
            <td>${row.games_scored}</td>
            <td>${(row.avg_accuracy * 100).toFixed(1)}%</td>
            <td>${row.avg_log_loss.toFixed(4)}</td>
            <td>${row.avg_ece.toFixed(4)}</td>
          </tr>`;
      }).join('') : '<tr><td colspan="9" style="color:var(--text3);padding:18px">No replay portfolio rows are available yet.</td></tr>';
    }

    // Replay slates
    const replaySlates = d.replay_slates || {};
    const replaySlateRows = replaySlates.rows || [];
    const replaySlatesCount = document.getElementById('res-replay-slates-count');
    const replaySlatesCards = document.getElementById('replay-slates-cards');
    const replaySlatesBody = document.getElementById('replay-slates-tbody');
    if (replaySlatesCount) {
      replaySlatesCount.textContent = replaySlateRows.length ? `${replaySlateRows.length} date${replaySlateRows.length !== 1 ? 's' : ''}` : 'No replay slate rows';
    }
    if (replaySlatesCards) {
      const summary = replaySlates.summary || {};
      replaySlatesCards.innerHTML = [
        ['Replay Dates', summary.dates ?? 0, 'Historical dates with event-level replay output.'],
        ['Replay Events', summary.events ?? 0, 'Total replay events currently available in the slate simulator.'],
        ['Published', summary.published_events ?? 0, 'Events that current policy would have allowed live.'],
        ['Held Out', summary.held_out_events ?? 0, 'Events that current policy would have suppressed.'],
      ].map(([label, value, copy]) => `
        <div class="stat-card">
          <div class="stat-label">${escapeHtml(label)}</div>
          <div class="stat-value">${value}</div>
          <div class="stat-sub">${escapeHtml(copy)}</div>
        </div>
      `).join('');
    }
    if (replaySlatesBody) {
      replaySlatesBody.innerHTML = replaySlateRows.length ? replaySlateRows.map(row => {
        const pubAcc = row.published_accuracy == null ? '—' : `${row.published_accuracy}%`;
        const ll = row.published_log_loss == null ? '—' : row.published_log_loss;
        const heldAcc = row.held_out_accuracy == null ? '—' : `${row.held_out_accuracy}%`;
        return `
          <tr onclick="showReplaySlateDetail('${escapeHtml(row.date)}')" style="cursor:pointer">
            <td>${escapeHtml(row.date)}</td>
            <td>${row.events}</td>
            <td>${row.published_events}</td>
            <td>${row.held_out_events}</td>
            <td>${row.preferred_events}</td>
            <td>${row.limited_events}</td>
            <td>${pubAcc}</td>
            <td>${ll}</td>
            <td>${heldAcc}</td>
            <td style="color:var(--text2);font-size:.78rem">${escapeHtml((row.top_lanes || []).join(', '))}</td>
          </tr>`;
      }).join('') : '<tr><td colspan="10" style="color:var(--text3);padding:18px">No event-level replay slates are available yet. Run market backtests to populate them.</td></tr>';
    }

    // Replay publish audit
    const replayPublish = d.replay_publish_audit || {};
    const replayPublishRows = replayPublish.rows || [];
    const replayPublishSummary = replayPublish.summary || {};
    const replayPublishCount = document.getElementById('res-replay-publish-count');
    const replayPublishCards = document.getElementById('replay-publish-cards');
    const replayPublishBody = document.getElementById('replay-publish-tbody');
    if (replayPublishCount) {
      replayPublishCount.textContent = replayPublishRows.length ? `${replayPublishRows.length} date${replayPublishRows.length !== 1 ? 's' : ''}` : 'No publish audit rows';
    }
    if (replayPublishCards) {
      replayPublishCards.innerHTML = [
        ['Publish', replayPublishSummary.publish ?? 0, 'Historical replay events the current stack would likely publish.'],
        ['Review', replayPublishSummary.review ?? 0, 'Historical replay events that would likely need manual review.'],
        ['Hold Out', replayPublishSummary.hold_out ?? 0, 'Historical replay events the current stack would suppress.'],
        ['Replay Dates', replayPublishSummary.dates ?? 0, 'Dates covered by the event-level publish audit.'],
      ].map(([label, value, copy]) => `
        <div class="stat-card">
          <div class="stat-label">${escapeHtml(label)}</div>
          <div class="stat-value">${value}</div>
          <div class="stat-sub">${escapeHtml(copy)}</div>
        </div>
      `).join('');
    }
    if (replayPublishBody) {
      replayPublishBody.innerHTML = replayPublishRows.length ? replayPublishRows.map(row => `
        <tr onclick="showReplaySlateDetail('${escapeHtml(row.date)}')" style="cursor:pointer">
          <td>${escapeHtml(row.date)}</td>
          <td>${row.events}</td>
          <td>${row.publish}</td>
          <td>${row.review}</td>
          <td>${row.hold_out}</td>
          <td>${row.publish_rate == null ? '—' : `${row.publish_rate}%`}</td>
        </tr>
      `).join('') : '<tr><td colspan="6" style="color:var(--text3);padding:18px">No replay publish audit rows are available yet.</td></tr>';
    }

    // Governor recommendations
    const govRows = d.governor_recommendations || [];
    const govCount = document.getElementById('res-governor-count');
    const govPanel = document.getElementById('governor-recommendations');
    if (govCount) {
      govCount.textContent = govRows.length ? `${govRows.length} recommendation${govRows.length !== 1 ? 's' : ''}` : 'No recommendations yet';
    }
    if (govPanel) {
      govPanel.innerHTML = govRows.length ? govRows.map(rec => {
        const actionClass = (
          rec.action === 'promote' ? 'production' :
          rec.action === 'demote' ? 'review' :
          rec.action === 'pause' ? 'review' :
          'limited'
        );
        return `
          <div class="launch-support-item" style="margin-bottom:10px;align-items:flex-start;cursor:pointer" onclick="showResultsLaneDetail('governor', '${encodeURIComponent(rec.sport)}', '${encodeURIComponent(rec.market)}', '${encodeURIComponent(rec.tier)}')">
            <span class="launch-support-tag ${actionClass}" style="min-width:76px;text-align:center">${escapeHtml(rec.action)}</span>
            <div>
              <div style="font-weight:700;color:var(--text);margin-bottom:4px">${SPORT_ICONS[rec.sport] || ''} ${escapeHtml(rec.sport)} · ${escapeHtml(rec.market)} · ${escapeHtml(rec.tier)}</div>
              <div style="color:var(--text2);font-size:.82rem">${escapeHtml(rec.reason)}</div>
              <div style="color:var(--text3);font-size:.75rem;margin-top:4px">Confidence: ${escapeHtml(rec.confidence)} · Replay: ${escapeHtml(rec.replay_support || 'missing')}</div>
              ${rec.replay_note ? `<div style="color:var(--text3);font-size:.74rem;margin-top:3px">${escapeHtml(rec.replay_note)}</div>` : ''}
            </div>
          </div>`;
      }).join('') : '<div style="color:var(--text3);font-size:.82rem">No strong promotion or demotion signals yet.</div>';
    }

    // Draft policy preview
    const previewRows = d.governor_change_preview || [];
    const previewCount = document.getElementById('res-preview-count');
    const previewPanel = document.getElementById('governor-preview-panel');
    if (previewCount) {
      previewCount.textContent = previewRows.length ? `${previewRows.length} draft change${previewRows.length !== 1 ? 's' : ''}` : 'No draft changes';
    }
    if (previewPanel) {
      previewPanel.innerHTML = previewRows.length ? previewRows.map(preview => {
        const actionClass = (
          preview.action === 'promote' ? 'production' :
          preview.action === 'demote' ? 'limited' :
          'review'
        );
        const changed = (preview.changed_fields || []).map(field => {
          const before = preview.current ? preview.current[field] : null;
          const after = preview.draft ? preview.draft[field] : null;
          return `<div style="color:var(--text2);font-size:.8rem;margin-top:3px"><code>${escapeHtml(field)}</code>: ${escapeHtml(String(before))} → ${escapeHtml(String(after))}</div>`;
        }).join('');
        return `
          <div class="launch-support-item" style="margin-bottom:10px;align-items:flex-start">
            <span class="launch-support-tag ${actionClass}" style="min-width:76px;text-align:center">${escapeHtml(preview.action)}</span>
            <div style="width:100%">
              <div style="font-weight:700;color:var(--text);margin-bottom:4px">${SPORT_ICONS[preview.sport] || ''} ${escapeHtml(preview.sport)} · ${escapeHtml(preview.market)} · ${escapeHtml(preview.tier)}</div>
              <div style="color:var(--text2);font-size:.82rem">${escapeHtml(preview.summary)}</div>
              <div style="color:var(--text3);font-size:.75rem;margin-top:4px">Target file: ${escapeHtml(preview.file)} · Confidence: ${escapeHtml(preview.confidence)} · Replay: ${escapeHtml(preview.replay_support || 'missing')}</div>
              <div style="margin-top:6px">${changed}</div>
            </div>
          </div>`;
      }).join('') : '<div style="color:var(--text3);font-size:.82rem">No governor actions are mature enough to draft into market policy changes yet.</div>';
    }

    // System parlays
    const sysParlays = d.system_parlays || [];
    const sysParlaysPanel = document.getElementById('sys-parlays-panel');
    if (sysParlays.length) {
      sysParlaysPanel.style.display = '';
      document.getElementById('res-sys-parlays-count').textContent = sysParlays.length + ' parlay' + (sysParlays.length>1?'s':'');
      document.getElementById('sys-parlays-list').innerHTML = sysParlays.map((p, i) => _renderResParlay(p, 'sp-' + i, false)).join('');
    } else {
      sysParlaysPanel.style.display = 'none';
    }

    // Manual parlays
    const manualParlays = d.manual_parlays || [];
    const manualParlaysPanel = document.getElementById('manual-parlays-panel');
    if (manualParlays.length) {
      manualParlaysPanel.style.display = '';
      document.getElementById('res-manual-parlays-count').textContent = manualParlays.length + ' saved';
      document.getElementById('manual-parlays-list').innerHTML = manualParlays.map((p, i) => _renderResParlay(p, 'mp-' + i, true)).join('');
    } else {
      manualParlaysPanel.style.display = 'none';
    }

    // Pending bets
    const pendingPanel = document.getElementById('pending-panel');
    const pending = d.pending || [];
    const singlePending = pending.filter(b => !b.is_parlay_leg);
    if (singlePending.length) {
      pendingPanel.style.display = '';
      document.getElementById('res-pending-count').textContent = singlePending.length + ' bet' + (singlePending.length>1?'s':'');
      document.getElementById('pending-tbody').innerHTML = singlePending.map(b => {
        const pid = b.pred_id || '';
        const pidAttr = pid ? `data-pred-id="${pid}"` : '';
        return `
        <tr id="prow-${pid}" ${pidAttr}>
          <td>${SPORT_ICONS[b.sport]||''} ${b.sport}<div style="margin-top:4px">${launchBadgeHtml(b)}</div></td>
          <td style="color:var(--text);font-weight:600">${b.team_or_player}</td>
          <td style="color:var(--text2);font-size:.75rem">${b.match_id||'—'}${b.kick_off ? `<div style="margin-top:4px">${eventBadgeHtml(b)} ${escapeHtml(b.kick_off)}</div>` : ''}</td>
          <td>${b.bet_odds ? Number(b.bet_odds).toFixed(2) : '—'}</td>
          <td style="color:var(--green)">+${((b.edge||0)*100).toFixed(1)}%</td>
          <td><span class="tag tag-market">${b.market_type||'ml'}</span></td>
          <td>${b.stake_units ? Number(b.stake_units).toFixed(3) : '—'}</td>
          <td style="display:flex;gap:5px;align-items:center">
            ${pid ? `
              <button class="settle-btn won"  id="sw-${pid}" onclick="settleBet('${pid}', true)"  title="Mark as Won">✓ Won</button>
              <button class="settle-btn lost" id="sl-${pid}" onclick="settleBet('${pid}', false)" title="Mark as Lost">✗ Lost</button>
            ` : '<span style="color:var(--text3);font-size:.72rem">—</span>'}
          </td>
        </tr>`;
      }).join('');
    } else {
      pendingPanel.style.display = 'none';
    }

    // Settled table
    const settled = d.settled || [];
    document.getElementById('res-settled-count').textContent = settled.length + ' bet' + (settled.length!==1?'s':'');
    const settledGroups = document.getElementById('settled-groups');
    if (settledGroups) {
      if (!settled.length) {
        settledGroups.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text3)">No settled bets for this period.</div>';
      } else {
        settledGroups.innerHTML = _renderDateGroups(settled, b => {
          const pnl    = b.profit_units;
          const status = b.won ? 'win' : (b.status === 'pending' ? 'pending' : 'loss');
          const label  = b.won ? '✓ Won' : (b.status === 'pending' ? '⏳' : '✗ Lost');
          return `<tr${b.is_parlay_leg ? ' style="opacity:.65"' : ''}>
            <td>${SPORT_ICONS[b.sport]||''} ${b.sport}<div style="margin-top:4px">${launchBadgeHtml(b)}</div></td>
            <td style="color:var(--text)">${b.is_parlay_leg ? '🔗 ' : ''}${b.team_or_player}${b.kick_off ? `<div style="margin-top:4px;color:var(--text3);font-size:.72rem">${eventBadgeHtml(b)} ${escapeHtml(b.kick_off)}</div>` : ''}</td>
            <td>${b.bet_odds ? Number(b.bet_odds).toFixed(2) : '—'}</td>
            <td style="color:var(--green)">+${((b.edge||0)*100).toFixed(1)}%</td>
            <td><span class="tag tag-market">${b.market||b.market_type||'ml'}</span></td>
            <td>${b.stake_units ? Number(b.stake_units).toFixed(3) : '—'}</td>
            <td><span class="won-badge ${status}">${label}</span></td>
            <td style="color:${pnl>=0?'var(--green)':'var(--red)'}">${pnl!=null?(pnl>=0?'+':'')+(pnl).toFixed(3):'—'}</td>
          </tr>`;
        }, ['Sport','Pick','Odds','Edge','Market','Stake','Result','P&L']);
      }
    }

    if (state.selectedLane) {
      _renderLaneDrilldown(state.selectedLane);
    } else {
      _renderLaneDrilldown(null);
    }
    if (state.selectedParlayCohort) {
      _renderParlayDrilldown(state.selectedParlayCohort);
    } else {
      _renderParlayDrilldown(null);
    }
    if (state.selectedReplaySlateDate) {
      _renderReplaySlateDetail(state.selectedReplaySlateDate);
    } else {
      _renderReplaySlateDetail(null);
    }

  } catch(e) {
    console.error('loadResults error:', e);
    const pendingPanel = document.getElementById('pending-panel');
    if (pendingPanel) {
      pendingPanel.style.display = '';
      document.getElementById('res-pending-count').textContent = 'load error';
      document.getElementById('pending-tbody').innerHTML =
        `<tr><td colspan="8" style="color:var(--red);padding:18px">Could not render results: ${escapeHtml(e.message || String(e))}</td></tr>`;
    }
    showToast(`Could not render results: ${e.message || e}`, 'error');
  }
}

// ── Date-grouped collapsible renderer ─────────────────────────────────────
function _renderDateGroups(bets, rowFn, headers) {
  // Group by date field (date or commence_time prefix)
  const groups = {};
  bets.forEach(b => {
    const d = b.date || (b.commence_time || b.settled_at || '').slice(0,10) || 'Unknown';
    if (!groups[d]) groups[d] = [];
    groups[d].push(b);
  });
  const sortedDates = Object.keys(groups).sort().reverse();
  const today = new Date().toISOString().slice(0,10);
  const recentCutoff = new Date(Date.now() - 7*86400000).toISOString().slice(0,10);

  return sortedDates.map((d, i) => {
    const dayBets = groups[d];
    const pnl = dayBets.reduce((s, b) => s + (b.profit_units ?? b.profit ?? 0), 0);
    const won  = dayBets.filter(b => b.won || b.result === 'won').length;
    const isOpen = d >= recentCutoff; // last 7 days open by default

    const dateLabel = d === today ? 'Today' : d === new Date(Date.now()-86400000).toISOString().slice(0,10) ? 'Yesterday' : d;
    const pnlColor  = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    const groupId   = `dg-${d.replace(/-/g,'')}`;

    const rowsHtml = `<div class="table-wrap"><table class="data-table">
      <thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead>
      <tbody>${dayBets.map(rowFn).join('')}</tbody>
    </table></div>`;

    return `<div class="date-group">
      <div class="date-group-hdr${isOpen?' open':''}" onclick="toggleDateGroup(this,'${groupId}')">
        <span class="date-group-chevron">▶</span>
        <span class="date-group-date">${escapeHtml(dateLabel)}</span>
        <span class="date-group-meta">${dayBets.length} bet${dayBets.length!==1?'s':''} · ${won}W/${dayBets.length-won}L</span>
        <span class="date-group-pnl" style="color:${pnlColor}">${pnl>=0?'+':''}${pnl.toFixed(2)}u</span>
      </div>
      <div class="date-group-body" id="${groupId}" style="display:${isOpen?'block':'none'}">${rowsHtml}</div>
    </div>`;
  }).join('');
}

function toggleDateGroup(hdr, groupId) {
  const body = document.getElementById(groupId);
  if (!body) return;
  const open = body.style.display !== 'none';
  body.style.display = open ? 'none' : 'block';
  hdr.classList.toggle('open', !open);
}

// ── Clear history ──────────────────────────────────────────────────────────
function confirmClearHistory() {
  const modal = document.getElementById('clear-history-modal');
  if (modal) { modal.style.display = 'flex'; document.getElementById('clear-confirm-input').value = ''; }
}
function closeClearModal() {
  const modal = document.getElementById('clear-history-modal');
  if (modal) modal.style.display = 'none';
}
async function executeClearHistory() {
  const input = document.getElementById('clear-confirm-input');
  if (!input || input.value.trim().toUpperCase() !== 'DELETE') {
    showToast('Type DELETE to confirm', 'error'); return;
  }
  closeClearModal();
  try {
    const r = await fetch('/api/results/clear', { method: 'POST' });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'Clear failed');
    showToast('History cleared and archived.', 'success');
    loadResults();
    loadMySelections();
    loadDashboard();
  } catch(e) { showToast('Clear failed: ' + e, 'error'); }
}

function selectResultDate(btn, date) {
  document.querySelectorAll('#date-chips .chip').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const select = document.getElementById('result-date-select');
  const normalized = date === 'today' ? new Date().toISOString().slice(0,10) : date;
  if (select) select.value = normalized === 'all' ? '' : normalized;
  loadResults(date);
}

function selectResultDateSelect(date) {
  _resultDate = date || 'all';
  document.querySelectorAll('#date-chips .chip').forEach(b => b.classList.remove('active'));
  if (_resultDate === 'all') {
    document.querySelector('#date-chips .chip[data-date="all"]')?.classList.add('active');
  }
  loadResults(_resultDate);
}

function toggleResParlay(id) {
  const el = document.getElementById('rp-' + id);
  el.classList.toggle('open');
}

function _renderResParlay(p, id, isManual) {
  const typeTag = p.type === 'value' ? '<span class="tag tag-flagged" style="font-size:.65rem">★ Value</span>'
                : p.type === 'speculative' ? '<span class="tag" style="font-size:.65rem;background:rgba(245,158,11,.15);color:var(--yellow)">⚡ Spec</span>'
                : '<span class="tag" style="font-size:.65rem;background:rgba(124,58,237,.15);color:#a78bfa">✍️ Manual</span>';
  const bracketTag = p.bracket ? `<span style="font-size:.7rem;color:var(--text3)">${p.bracket}</span>` : '';
  const deleteBtn = isManual && p.id
    ? `<button onclick="deleteManualParlay('${p.id}', event)" style="background:none;border:none;color:var(--text3);cursor:pointer;padding:2px 6px;font-size:.85rem" title="Delete">🗑</button>`
    : '';
  const nLegs = p.n_legs || (p.legs && p.legs.length) || '?';
  const winProb = p.win_prob != null ? (typeof p.win_prob === 'number' ? p.win_prob.toFixed(2) + '%' : p.win_prob) : '—';
  const ev = p.ev != null ? p.ev + 'x' : '—';
  const kelly = p.kelly_stake != null ? '£' + p.kelly_stake : (p.kelly_stake === 0 ? '£0' : '—');

  // For manual parlays: show custom name + status badge + status toggle buttons
  const displayName = isManual && p.name ? p.name : `${nLegs}-leg Parlay`;
  const status = p.status || 'pending';
  const statusBadge = isManual ? `
    <span class="won-badge ${status==='won'?'win':status==='lost'?'loss':'pending'}" style="flex-shrink:0;font-size:.7rem">
      ${status==='won'?'✓ Won':status==='lost'?'✗ Lost':'⏳ Pending'}
    </span>` : '';
  const statusBtns = isManual && p.id ? `
    <div onclick="event.stopPropagation()" style="display:flex;gap:5px;flex-shrink:0;margin-left:4px">
      <button class="status-btn${status==='pending'?' active-pending':''}" onclick="setManualParlayStatus('${p.id}','pending','${id}')" title="Pending">⏳</button>
      <button class="status-btn${status==='won'?' active-won':''}" onclick="setManualParlayStatus('${p.id}','won','${id}')" title="Won">✓</button>
      <button class="status-btn${status==='lost'?' active-lost':''}" onclick="setManualParlayStatus('${p.id}','lost','${id}')" title="Lost">✗</button>
    </div>` : '';

  // Show settle buttons if parlay hasn't been settled yet
  const alreadySettled = status === 'won' || status === 'lost';
  const source = p.source || (isManual ? 'manual' : 'system');
  const parlayIdForSettle = p.id || '';
  // Encode parlay data for the settle call (JSON in a data attribute to avoid inline quoting issues)
  const parlayDataAttr = `data-parlay='${JSON.stringify({
    combined_odds: p.combined_odds,
    kelly_stake:   p.kelly_stake || p.kelly_stake_units || 0,
    ev:            p.ev,
    win_prob:      p.win_prob,
    n_legs:        nLegs,
    name:          displayName,
    date:          p.date || '',
    edge:          p.edge || 0,
  }).replace(/'/g, "&#39;")}'`;

  const settleRow = !alreadySettled ? `
    <div style="padding:10px 14px 12px;display:flex;gap:8px;align-items:center;border-top:1px solid var(--border);margin-top:4px">
      <span style="font-size:.72rem;color:var(--text3);flex:1">Settle this parlay:</span>
      <button class="settle-btn won"  id="spw-${id}" onclick="settleParlayCard('${id}','${parlayIdForSettle}','${source}',true)"  ${parlayDataAttr}>✓ Won</button>
      <button class="settle-btn lost" id="spl-${id}" onclick="settleParlayCard('${id}','${parlayIdForSettle}','${source}',false)" ${parlayDataAttr}>✗ Lost</button>
    </div>` : '';

  return `
    <div class="res-parlay" id="rp-${id}">
      <div class="res-parlay-header" onclick="toggleResParlay('${id}')">
        <div class="res-parlay-title">${typeTag} ${bracketTag} <span style="color:var(--text)">${displayName}</span></div>
        <div class="res-parlay-stats">
          <span>Odds: <span class="hi">${p.combined_odds}x</span></span>
          <span>Win: <span class="hi">${winProb}</span></span>
          <span>EV: <span class="hi">${ev}</span></span>
          ${kelly !== '—' ? `<span>Kelly: <span class="hi">${kelly}</span></span>` : ''}
        </div>
        ${statusBadge}
        ${statusBtns}
        ${deleteBtn}
        <span class="res-parlay-chevron">▶</span>
      </div>
      <div class="res-parlay-legs">
        ${(p.legs || []).map(l => `
          <div class="res-parlay-leg">
            <span class="tag tag-market" style="flex-shrink:0">${l.sport}</span>
            <span class="res-parlay-leg-team">${l.team}</span>
            <span class="res-parlay-leg-meta">${l.match || ''} ${l.kick_off || ''}</span>
            <span style="font-weight:600;flex-shrink:0">@ ${l.odds}</span>
            <span style="color:${l.edge>=0?'var(--green)':'var(--red)'};flex-shrink:0;font-size:.75rem">${l.edge>=0?'+':''}${(l.edge*(l.edge>1?1:100)).toFixed(1)}%</span>
            ${l.status ? `<span class="won-badge ${l.status==='pending'?'pending':l.status==='won'||l.status==='win'?'win':'loss'}" style="flex-shrink:0">
              ${l.status==='pending'?'⏳':l.status==='won'||l.status==='win'?'✓ Won':'✗ Lost'}</span>` : ''}
          </div>`).join('')}
        ${settleRow}
      </div>
    </div>`;
}

async function setManualParlayStatus(parlayId, status, renderId) {
  try {
    const r = await fetch('/api/parlays/manual/' + parlayId, {
      method: 'PATCH',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ status }),
    });
    const d = await r.json();
    if (!d.updated) { showToast('Update failed', 'error'); return; }
    const label = status === 'won' ? '✓ Won' : status === 'lost' ? '✗ Lost' : '⏳ Pending';
    showToast(`Status set to ${label}`, 'success');
    if (state.currentPage === 'parlays') loadParlays();
    if (state.currentPage === 'results') loadResults(_resultDate !== 'all' ? _resultDate : undefined);
  } catch(e) { showToast('Failed to update status', 'error'); }
}

async function renameManualParlay(parlayId) {
  const input = document.getElementById('saved-parlay-name-' + parlayId);
  const name = (input?.value || '').trim();
  if (!name) {
    showToast('Enter a name first', 'error');
    input?.focus();
    return;
  }
  try {
    const r = await fetch('/api/parlays/manual/' + parlayId, {
      method: 'PATCH',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ name }),
    });
    const d = await r.json();
    if (!d.updated) { showToast('Rename failed', 'error'); return; }
    showToast('Parlay renamed', 'success');
    if (state.currentPage === 'parlays') loadParlays();
    if (state.currentPage === 'results') loadResults(_resultDate !== 'all' ? _resultDate : undefined);
  } catch(e) {
    showToast('Failed to rename parlay', 'error');
  }
}

async function deleteManualParlay(id, event) {
  event.stopPropagation();
  if (!confirm('Delete this saved parlay?')) return;
  try {
    await fetch('/api/parlays/manual/' + id, { method: 'DELETE' });
    showToast('Parlay deleted', 'success');
    if (state.currentPage === 'parlays') loadParlays();
    if (state.currentPage === 'results') loadResults(_resultDate !== 'all' ? _resultDate : undefined);
  } catch(e) { showToast('Failed to delete', 'error'); }
}

async function settleBet(predId, won) {
  // Disable both buttons on this row immediately
  const btnW = document.getElementById('sw-' + predId);
  const btnL = document.getElementById('sl-' + predId);
  if (btnW) { btnW.disabled = true; btnW.textContent = won ? 'Saving…' : '✓ Won'; }
  if (btnL) { btnL.disabled = true; btnL.textContent = !won ? 'Saving…' : '✗ Lost'; }

  try {
    const r = await fetch('/api/results/settle', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ pred_id: predId, won }),
    });
    const d = await r.json();
    if (d.error) {
      showToast(d.error, 'error');
      if (btnW) { btnW.disabled = false; btnW.textContent = '✓ Won'; }
      if (btnL) { btnL.disabled = false; btnL.textContent = '✗ Lost'; }
      return;
    }

    const profit = d.profit >= 0 ? `+${d.profit.toFixed(3)}` : d.profit.toFixed(3);
    const label  = won ? '✓ Won' : '✗ Lost';
    const color  = won ? 'var(--green)' : 'var(--red)';
    showToast(`${label}: ${d.pick} — P&L: ${profit} units`, won ? 'success' : 'error');

    // Replace the row with a settled indicator then reload results
    const row = document.getElementById('prow-' + predId);
    if (row) {
      row.style.opacity = '.5';
      row.style.transition = 'opacity .4s';
      setTimeout(() => loadResults(_resultDate !== 'all' ? _resultDate : undefined), 500);
    }
  } catch(e) {
    showToast('Failed to settle bet', 'error');
    if (btnW) { btnW.disabled = false; btnW.textContent = '✓ Won'; }
    if (btnL) { btnL.disabled = false; btnL.textContent = '✗ Lost'; }
  }
}

async function settleParlayCard(renderId, parlayId, source, won) {
  const btnW = document.getElementById('spw-' + renderId);
  const btnL = document.getElementById('spl-' + renderId);
  if (btnW) { btnW.disabled = true; btnW.textContent = won ? 'Saving…' : '✓ Won'; }
  if (btnL) { btnL.disabled = true; btnL.textContent = !won ? 'Saving…' : '✗ Lost'; }

  // Read parlay data from the button's data-parlay attribute
  let parlayData = {};
  try {
    const btn = document.getElementById('spw-' + renderId) || document.getElementById('spl-' + renderId);
    if (btn) parlayData = JSON.parse(btn.getAttribute('data-parlay') || '{}');
  } catch(e) {}

  try {
    const r = await fetch('/api/results/settle-parlay', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ won, source, parlay_id: parlayId, parlay_data: parlayData }),
    });
    const d = await r.json();
    if (d.error) {
      showToast(d.error, 'error');
      if (btnW) { btnW.disabled = false; btnW.textContent = '✓ Won'; }
      if (btnL) { btnL.disabled = false; btnL.textContent = '✗ Lost'; }
      return;
    }

    const profit = d.profit >= 0 ? `+${d.profit.toFixed(3)}` : d.profit.toFixed(3);
    showToast(`${won ? '✓ Won' : '✗ Lost'}: ${d.name} — P&L: ${profit} units`, won ? 'success' : 'error');

    // Fade card and reload
    const card = document.getElementById('rp-' + renderId);
    if (card) { card.style.opacity = '.4'; card.style.transition = 'opacity .4s'; }
    setTimeout(() => loadResults(_resultDate !== 'all' ? _resultDate : undefined), 500);
  } catch(e) {
    showToast('Failed to settle parlay', 'error');
    if (btnW) { btnW.disabled = false; btnW.textContent = '✓ Won'; }
    if (btnL) { btnL.disabled = false; btnL.textContent = '✗ Lost'; }
  }
}

// ═══════════════════════════════════════════════════════════
// AUTO SETTLE
// ═══════════════════════════════════════════════════════════

async function autoSettleAll(btn) {
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Checking…';
  const panel = document.getElementById('settle-result-panel');
  if (panel) { panel.style.display = 'none'; panel.innerHTML = ''; }
  try {
    const r = await fetch('/api/settle-all', { method: 'POST',
      headers: {'Content-Type':'application/json'}, body: JSON.stringify({}) });
    const raw = await r.text();
    let d = {};
    try { d = raw ? JSON.parse(raw) : {}; } catch(e) { throw new Error(raw.slice(0, 220) || `HTTP ${r.status}`); }
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);

    const bets    = Number(d.bets_settled || 0);
    const parlays = Number(d.parlays_settled || 0);
    const profit  = Number(d.bets_profit || 0);
    const scorePool = Number(d.score_pool || 0);
    const scoreSources = Array.isArray(d.score_sources) ? d.score_sources : [];
    const errors = Array.isArray(d.errors) ? d.errors : [];
    const unresolvedSummary = Array.isArray(d.unresolved_summary) ? d.unresolved_summary : [];
    const profitStr = (profit >= 0 ? '+' : '') + profit.toFixed(3);
    const anyNew  = bets > 0 || parlays > 0;

    // Build lines for the inline panel
    const lines = [];
    if (bets > 0)    lines.push(`<b>${bets} bet${bets !== 1 ? 's' : ''} settled</b> — P&L: ${profitStr} units`);
    if (parlays > 0) lines.push(`<b>${parlays} parlay${parlays !== 1 ? 's' : ''} finalised</b>`);
    if (!anyNew) {
      if (scorePool === 0) {
        const sourceLabel = scoreSources.length ? ` via ${escapeHtml(scoreSources.join(', '))}` : '';
        lines.push(`<span style="color:var(--text3)">No score data was fetched${sourceLabel} — ${d.still_pending || 0} still pending</span>`);
      } else {
        lines.push(`<span style="color:var(--text3)">No new results yet — ${d.still_pending || 0} still pending</span>`);
      }
    }
    if (unresolvedSummary.length) {
      lines.push(`<span style="color:var(--text3)">Unresolved: ${escapeHtml(unresolvedSummary.slice(0, 4).map(item => `${item.scope}:${item.reason} (${item.count})`).join(' · '))}</span>`);
    }
    if (!anyNew && errors.length) {
      lines.push(`<span style="color:var(--orange)">Sources: ${escapeHtml(errors.join(' | '))}</span>`);
    }

    for (const p of (d.parlay_results || [])) {
      const icon = p.status === 'won' ? '✅' : p.status === 'lost' ? '❌' : '⏳';
      lines.push(`${icon} <b>${escapeHtml(p.name)}</b>: ${p.legs_won}W · ${p.legs_lost}L · ${p.legs_pending} pending → <em>${p.status}</em>`);
    }

    if (panel) {
      panel.innerHTML = lines.join('<br>');
      panel.style.display = 'block';
      panel.style.background  = anyNew ? 'rgba(16,185,129,.07)' : 'rgba(100,116,139,.07)';
      panel.style.borderColor = anyNew ? 'rgba(16,185,129,.3)' : 'var(--border)';
      panel.style.color       = 'var(--text)';
    }

    showToast(anyNew ? `Settled: ${bets} bet${bets!==1?'s':''}, ${parlays} parlay${parlays!==1?'s':''} (P&L ${profitStr}u)` : 'Nothing new to settle yet', anyNew ? 'success' : 'info');

    if (anyNew) {
      loadResults(_resultDate !== 'all' ? _resultDate : undefined);
      loadMySelections(_myDate !== 'all' ? _myDate : undefined);
      loadParlays();
    }
  } catch(e) {
    if (panel) { panel.innerHTML = `⚠ ${escapeHtml(String(e.message || e))}`; panel.style.display = 'block'; panel.style.background = 'rgba(239,68,68,.08)'; panel.style.borderColor = 'rgba(239,68,68,.3)'; panel.style.color = 'var(--red)'; }
    showToast(`Failed to check results: ${e.message || e}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '✓ Check &amp; Settle';
  }
}

// ═══════════════════════════════════════════════════════════
// SCAN
// ═══════════════════════════════════════════════════════════
// Sport scanning supports a targeted multi-select. "All" is the broad scan mode.
function toggleScanSingle(btn, group) {
  const val = btn.dataset.val;
  if (group === 'sport') {
    const buttons = Array.from(document.querySelectorAll('[data-scan="sport"]'));
    if (val === 'all') {
      state.scanSports = [];
      buttons.forEach(b => b.classList.toggle('active', b.dataset.val === 'all'));
    } else {
      const selected = new Set(state.scanSports || []);
      if (selected.has(val)) selected.delete(val);
      else selected.add(val);
      state.scanSports = Array.from(selected);
      buttons.forEach(b => {
        const buttonVal = b.dataset.val;
        b.classList.toggle('active', buttonVal === 'all' ? state.scanSports.length === 0 : selected.has(buttonVal));
      });
    }
    syncScanOptionAvailability();
  } else if (group === 'market') {
    document.querySelectorAll(`[data-scan="${group}"]`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.scanMarket = val;
  }
}

// Legacy alias in case anything else calls it
function toggleScanFilter(btn) { toggleScanSingle(btn, 'sport'); }

function toggleScanOption(btn) {
  if (btn.disabled) return;
  btn.classList.toggle('active');
  const val = btn.dataset.val;
  if (val === 'retrain') {
    state.scanRetrain = btn.classList.contains('active');
  } else if (val === 'offline_odds') {
    state.scanOfflineOdds = btn.classList.contains('active');
  } else if (val === 'force_fresh_odds') {
    state.scanForceFreshOdds = btn.classList.contains('active');
  } else if (val === 'lean_context') {
    state.scanLeanContext = btn.classList.contains('active');
  } else if (val === 'context_referee') {
    state.scanContextReferee = btn.classList.contains('active');
  } else if (val === 'full_soccer_scope') {
    state.scanFullSoccerScope = btn.classList.contains('active');
  }
}

function syncScanOptionAvailability() {
  const forceFreshBtn = document.getElementById('chip-force-fresh-odds');
  if (!forceFreshBtn) return;
  const targetedSportsSelected = state.scanSports.length > 0;
  forceFreshBtn.disabled = !targetedSportsSelected;
  forceFreshBtn.title = targetedSportsSelected
    ? 'Fetch live odds for the selected sport(s).'
    : 'Select one or more specific sports first. Broad all-sport force-fresh scans are blocked to protect quota.';
  forceFreshBtn.setAttribute('aria-disabled', String(!targetedSportsSelected));
  if (!targetedSportsSelected) {
    forceFreshBtn.classList.remove('active');
    state.scanForceFreshOdds = false;
  }
}

async function startScan() {
  const btn = document.getElementById('btn-scan');
  const stopBtn = document.getElementById('btn-stop');
  const selectedSports = state.scanSports || [];
  const soccerLeagues = selectedSports.filter(sport => String(sport || '').startsWith('soccer_'));
  btn.disabled = true;
  btn.classList.add('running');
  btn.innerHTML = '<div class="spinner"></div> Scanning…';
  stopBtn.style.display = 'inline-flex';
  document.getElementById('scan-status-badge').textContent = '● Running';
  document.getElementById('scan-status-badge').style.color = 'var(--yellow)';

  const logEl = document.getElementById('scan-log');
  logEl.innerHTML = '';

  try {
    await fetch('/api/scan/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        sport:   state.scanSports.length === 1 ? state.scanSports[0] : state.scanSports.length ? 'multi' : 'all',
        sports:  state.scanSports,
        soccer_leagues: soccerLeagues,
        market:  state.scanMarket || 'all',
        retrain: state.scanRetrain,
        offline_odds: state.scanOfflineOdds,
        force_fresh_odds: state.scanForceFreshOdds,
        lean_context: state.scanLeanContext,
        context_referee: state.scanContextReferee,
        full_soccer_scope: state.scanFullSoccerScope,
        focused_lanes: false,
      }),
    });
  } catch(e) {
    logEl.innerHTML = '<div class="log-line err">Failed to start scan: ' + e + '</div>';
    resetScanBtn();
    return;
  }

  // Stream log via SSE
  const es = new EventSource('/api/scan/stream');
  es.onmessage = e => {
    if (e.data === '__DONE__') {
      es.close();
      resetScanBtn();
      loadDashboard();
      loadScanApiUsage();
      loadPicks();
      document.getElementById('scan-status-badge').textContent = '✓ Complete';
      document.getElementById('scan-status-badge').style.color = 'var(--green)';
      // Show toast with a link to navigate to picks
      showToast('Scan complete — picks updated.', 'success');
      return;
    }
    const line = JSON.parse(e.data);
    const div = document.createElement('div');
    div.className = 'log-line' + (line.includes('ERROR') || line.includes('error') ? ' err' : line.includes('WARNING') || line.includes('WARN') ? ' warn' : line.includes('Finished') || line.includes('saved') ? ' done' : '');
    div.textContent = line;
    logEl.appendChild(div);
    logEl.scrollTop = logEl.scrollHeight;
  };
  es.onerror = () => { es.close(); resetScanBtn(); };
}

function resetScanBtn() {
  const btn = document.getElementById('btn-scan');
  const stopBtn = document.getElementById('btn-stop');
  const badge = document.getElementById('scan-status-badge');
  btn.disabled = false;
  btn.classList.remove('running');
  btn.innerHTML = '<span>⚡</span> Run Scan';
  stopBtn.style.display = 'none';
  if (badge && badge.textContent === '● Running') {
    badge.textContent = 'Idle';
    badge.style.color = 'var(--text3)';
  }
}

async function stopScan() {
  const stopBtn = document.getElementById('btn-stop');
  stopBtn.disabled = true;
  stopBtn.textContent = 'Stopping…';
  try {
    await fetch('/api/scan/stop', { method: 'POST' });
  } catch(e) {
    showToast('Failed to stop scan: ' + e, 'error');
    stopBtn.disabled = false;
    stopBtn.textContent = '⏹ Stop';
  }
}

async function fetchClosingOdds() {
  const btn = document.getElementById('btn-closing');
  btn.disabled = true;
  btn.textContent = '⏳ Fetching…';
  const logEl = document.getElementById('scan-log');

  try {
    const sport = state.scanSports.length === 1 ? state.scanSports[0] : undefined;
    const r = await fetch('/api/closing-odds/fetch', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(sport ? {sport} : {}),
    });
    if (!r.ok) throw new Error(await r.text());

    // Poll status until done
    const poll = setInterval(async () => {
      const s = await fetch('/api/closing-odds/status').then(r => r.json());
      logEl.innerHTML = s.log.map(l => `<div class="log-line${l.includes('ERROR')?` err`:l.includes('Done')?` done`:``}">${l}</div>`).join('');
      logEl.scrollTop = logEl.scrollHeight;
      if (!s.running) {
        clearInterval(poll);
        btn.disabled = false;
        btn.textContent = '📉 Closing Odds';
        showToast('Closing odds saved!', 'success');
        loadScanApiUsage();
      }
    }, 800);
  } catch(e) {
    showToast('Error: ' + e, 'error');
    btn.disabled = false;
    btn.textContent = '📉 Closing Odds';
  }
}

async function runSettle() {
  const btn = document.getElementById('btn-settle');
  btn.disabled = true;
  btn.textContent = '⏳ Settling…';
  const logEl = document.getElementById('scan-log');
  logEl.innerHTML = '';

  try {
    const r = await fetch('/api/settle/run', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({}),
    });
    if (!r.ok) throw new Error(await r.text());

    const poll = setInterval(async () => {
      const s = await fetch('/api/settle/status').then(r => r.json());
      logEl.innerHTML = s.log.map(l => `<div class="log-line${l.includes('ERROR')?` err`:l.includes('Settled')||l.includes('Finished')?` done`:l.includes('WARNING')?` warn`:``}">${l}</div>`).join('');
      logEl.scrollTop = logEl.scrollHeight;
      if (!s.running) {
        clearInterval(poll);
        btn.disabled = false;
        btn.textContent = '✅ Settle Now';
        showToast('Settlement complete!', 'success');
        loadDashboard();
      }
    }, 800);
  } catch(e) {
    showToast('Error: ' + e, 'error');
    btn.disabled = false;
    btn.textContent = '✅ Settle Now';
  }
}

async function loadScanApiUsage() {
  const el = document.getElementById('scan-api-usage');
  try {
    const r = await fetch('/api/dashboard');
    const d = await r.json();
    const rem = d.odds_remaining, start = d.odds_start || d.odds_remaining || 1;
    const used = Math.max(0, start - rem);
    const pct = start > 0 ? used / start * 100 : 0;
    const color = pct < 50 ? 'var(--green)' : pct < 80 ? 'var(--yellow)' : 'var(--red)';
    const mode = d.quota_mode || 'healthy';
    const modeLabel = {
      healthy: 'Healthy',
      caution: 'Caution',
      critical: 'Critical',
    }[mode] || 'Healthy';
    const dailyAllowance = d.odds_daily_allowance;
    const reserve = d.odds_reserve ?? 0;
    const daysLeft = d.odds_days_left_in_cycle ?? '—';
    const reserveRemaining = d.odds_remaining_after_reserve;
    const modeCopy = {
      healthy: 'The scanner uses the active key normally and can rotate to other usable keys in the pool.',
      caution: 'The active key is getting lower. If needed, the scanner can keep going with other loaded usable keys.',
      critical: 'The active key is exhausted or nearly exhausted. Add a fresh key or let the scanner rotate to another usable key.',
    }[mode] || '';
    const keyPool = d.odds_key_pool || {};
    const allKeyRows = Array.isArray(keyPool.keys) ? keyPool.keys : [];
    const runtimeKeyRows = allKeyRows.filter(row => row && row.runtime_available);
    const historicalOnlyRows = allKeyRows.filter(row => row && !row.runtime_available);
    const keyRows = runtimeKeyRows.slice(0, 4);
    const hiddenHistoricalCount = historicalOnlyRows.length;
    const keySelectionReason = d.odds_selection_reason || keyPool.last_selected_reason || '';
    const runtimeExcluded = Array.isArray(keyPool.runtime_parse_excluded) ? keyPool.runtime_parse_excluded : [];
    const keyPoolHtml = keyPool.enabled ? `
            <div class="quota-item">
              <div>
                <div class="quota-item-title">Key Pool</div>
                <div class="quota-item-copy">Tracked ${keyPool.tracked_count ?? keyPool.count ?? allKeyRows.length} fingerprint${(keyPool.tracked_count ?? keyPool.count ?? allKeyRows.length) === 1 ? '' : 's'} • runtime ${keyPool.runtime_loaded_count ?? 0} • usable ${keyPool.usable_count ?? 0}${runtimeExcluded.length ? ` • ${runtimeExcluded.length} env exclusion${runtimeExcluded.length === 1 ? '' : 's'}` : ''}${hiddenHistoricalCount ? ` • ${hiddenHistoricalCount} historical raw-missing hidden below` : ''}.</div>
              </div>
              <div class="quota-item-val">${keyPool.total_remaining ?? '—'}</div>
            </div>
            <div class="quota-item">
              <div>
                <div class="quota-item-title">Last Selected Key</div>
                <div class="quota-item-copy">${keyPool.last_selected_at ? `Selected ${new Date(keyPool.last_selected_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}` : 'Latest key chosen at scan start.'}${keySelectionReason ? ` • ${escapeHtml(keySelectionReason)}` : ''}</div>
              </div>
              <div class="quota-item-val">${keyPool.last_selected_fingerprint ? `…${escapeHtml(keyPool.last_selected_fingerprint)}` : '—'}</div>
            </div>
            ${keyRows.map(row => `
            <div class="quota-item">
              <div>
                <div class="quota-item-title">${row.selected ? 'Active Runtime Key' : 'Runtime Key'}</div>
                <div class="quota-item-copy">Fingerprint …${escapeHtml(row.fingerprint || '')}${row.updated_at ? ` • updated ${new Date(row.updated_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}` : ''}${row.status === 'stale_metadata' ? ' • stale metadata' : ''}${row.low_quota ? ' • low quota' : ''}${row.runtime_available && row.usable ? ' • runtime usable' : ''}${row.status === 'runtime_only' ? ' • runtime only' : ''}${row.exclusion_reason ? ` • ${escapeHtml(row.exclusion_reason.replaceAll('_', ' '))}` : ''}</div>
              </div>
              <div class="quota-item-val">${row.remaining ?? '—'}</div>
            </div>`).join('')}
            ${hiddenHistoricalCount ? `
            <div class="quota-item">
              <div>
                <div class="quota-item-title">Historical Tracked Keys</div>
                <div class="quota-item-copy">${hiddenHistoricalCount} tracked fingerprint${hiddenHistoricalCount === 1 ? '' : 's'} are not loaded in the current runtime env, so they are hidden from the main list.</div>
              </div>
              <div class="quota-item-val">${hiddenHistoricalCount}</div>
            </div>` : ''}
    ` : '';
    el.innerHTML = `
      <div class="quota-wrap">
        <div class="quota-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap">
            <div>
              <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;margin-bottom:6px">Odds API Key Pool</div>
              <div style="font-size:1.2rem;font-weight:800">${rem}<span style="font-size:.82rem;color:var(--text3);font-weight:500"> remaining on active key</span></div>
            </div>
            <span class="quota-mode ${mode}">${modeLabel}</span>
          </div>
          <div class="quota-bar">
            <div class="quota-bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="quota-kpis">
            <div class="quota-kpi">
              <div class="quota-kpi-label">Used Today</div>
              <div class="quota-kpi-value">${d.odds_used_today ?? 0}</div>
            </div>
            <div class="quota-kpi">
              <div class="quota-kpi-label">Daily Pacing</div>
              <div class="quota-kpi-value">${dailyAllowance ?? 'Not enforced'}</div>
            </div>
            <div class="quota-kpi">
              <div class="quota-kpi-label">Protected Reserve</div>
              <div class="quota-kpi-value">${reserve}</div>
            </div>
          </div>
          <div class="quota-note">${modeCopy}</div>
        </div>
        <div class="quota-card">
          <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;margin-bottom:12px">Operational Notes</div>
          <div class="quota-list">
            <div class="quota-item">
              <div>
                <div class="quota-item-title">Days Left in Cycle</div>
                <div class="quota-item-copy">No internal monthly pacing is enforced; live key-pool mode leaves this empty.</div>
              </div>
              <div class="quota-item-val">${daysLeft}</div>
            </div>
            <div class="quota-item">
              <div>
                <div class="quota-item-title">Runtime Remaining</div>
                <div class="quota-item-copy">Remaining requests currently available on the selected runtime key.</div>
              </div>
              <div class="quota-item-val">${reserveRemaining ?? '—'}</div>
            </div>
            <div class="quota-item">
              <div>
                <div class="quota-item-title">Last Scan</div>
                <div class="quota-item-copy">Latest completed scan time from today's summary output.</div>
              </div>
              <div class="quota-item-val">${d.scan_time ? new Date(d.scan_time).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}) : '—'}</div>
            </div>
            ${keyPoolHtml}
          </div>
        </div>
      </div>`;
  } catch(e) { el.innerHTML = '<span style="color:var(--text3)">Failed to load usage data.</span>'; }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ═══════════════════════════════════════════════════════════
// WORLD CUP
// ═══════════════════════════════════════════════════════════
async function loadWorldCupPage() {
  const groupsEl = document.getElementById('wc-groups-grid');
  if (!groupsEl) return;
  if (!state.worldCupTeams.length) {
    groupsEl.innerHTML = '<div style="color:var(--text3);font-size:.85rem">Loading tournament field…</div>';
  }
  try {
    const r = await fetch('/api/world-cup/teams');
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Failed to load World Cup teams');
    state.worldCupTeams = d.teams || [];
    state.worldCupMeta = d;
    renderWorldCupTeams();
    populateWorldCupTeamSelects();
  } catch (e) {
    groupsEl.innerHTML = `<div class="empty-state"><p>${escapeHtml(e.message || e)}</p></div>`;
  }
}

function populateWorldCupTeamSelects() {
  const a = document.getElementById('wc-team-a');
  const b = document.getElementById('wc-team-b');
  if (!a || !b) return;
  const options = state.worldCupTeams.map(team =>
    `<option value="${escapeHtml(team.team_id)}">${escapeHtml(team.group)} · ${escapeHtml(team.team_name)}</option>`
  ).join('');
  a.innerHTML = options;
  b.innerHTML = options;
  if (state.worldCupTeams.length > 1) b.selectedIndex = 1;
}

function renderWorldCupTeams() {
  const teams = state.worldCupTeams || [];
  const teamCount = document.getElementById('wc-team-count');
  const groupCount = document.getElementById('wc-group-count');
  if (teamCount) teamCount.textContent = teams.length || '—';
  const groups = [...new Set(teams.map(t => t.group))].sort();
  if (groupCount) groupCount.textContent = groups.length || '—';
  const el = document.getElementById('wc-groups-grid');
  if (!el) return;
  const byGroup = {};
  teams.forEach(team => {
    if (!byGroup[team.group]) byGroup[team.group] = [];
    byGroup[team.group].push(team);
  });
  el.innerHTML = groups.map(group => `
    <div class="wc-group-card">
      <div class="wc-group-title">Group ${escapeHtml(group)}</div>
      ${(byGroup[group] || []).map(team => `
        <div class="wc-team-row">
          <span>${escapeHtml(team.draw_position || '')}</span>
          <strong>${escapeHtml(team.team_name)}</strong>
          <em>#${escapeHtml(team.fifa_ranking)}</em>
        </div>
      `).join('')}
    </div>
  `).join('');
}

async function runWorldCupFixture() {
  const result = document.getElementById('wc-fixture-result');
  const teamA = document.getElementById('wc-team-a')?.value;
  const teamB = document.getElementById('wc-team-b')?.value;
  if (!result || !teamA || !teamB) return;
  if (teamA === teamB) {
    result.innerHTML = '<span style="color:var(--yellow)">Choose two different teams.</span>';
    return;
  }
  result.innerHTML = '<div class="spinner"></div> Running match prediction…';
  try {
    const r = await fetch('/api/world-cup/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        team_a: teamA,
        team_b: teamB,
        neutral_venue: document.getElementById('wc-neutral')?.checked !== false,
      }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Prediction failed');
    result.innerHTML = renderWorldCupFixtureResult(d);
  } catch (e) {
    result.innerHTML = `<span style="color:var(--red)">${escapeHtml(e.message || e)}</span>`;
  }
}

function renderWorldCupFixtureResult(d) {
  const topScores = (d.top_scores || []).slice(0, 4).map(score =>
    `<span class="wc-score-chip">${score.team_a_goals}-${score.team_b_goals} ${(Number(score.probability || 0) * 100).toFixed(1)}%</span>`
  ).join('');
  return `
    <div class="wc-match-title">${escapeHtml(d.team_a_name)} vs ${escapeHtml(d.team_b_name)}</div>
    <div class="wc-prob-grid">
      <div><span>${(Number(d.prob_team_a_win_90 || 0) * 100).toFixed(1)}%</span><small>${escapeHtml(d.team_a_name)} win</small></div>
      <div><span>${(Number(d.prob_draw_90 || 0) * 100).toFixed(1)}%</span><small>Draw</small></div>
      <div><span>${(Number(d.prob_team_b_win_90 || 0) * 100).toFixed(1)}%</span><small>${escapeHtml(d.team_b_name)} win</small></div>
    </div>
    <div class="wc-score-row">${topScores || '<span style="color:var(--text3)">No score matrix returned.</span>'}</div>
  `;
}

async function runWorldCupSimulation() {
  const status = document.getElementById('wc-sim-status');
  if (!status) return;
  const simulations = Number(document.getElementById('wc-simulations')?.value || 1000);
  const seed = Number(document.getElementById('wc-seed')?.value || 2026);
  status.innerHTML = '<div class="spinner"></div> Running tournament simulation…';
  try {
    const r = await fetch('/api/world-cup/simulate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ simulations, seed }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Simulation failed');
    document.getElementById('wc-last-run').textContent = d.version || 'complete';
    status.innerHTML = `
      <div><strong>Simulation complete.</strong></div>
      <div style="color:var(--text3);font-size:.78rem;margin-top:4px">${escapeHtml(d.output_dir || '')}</div>
    `;
    renderWorldCupProbabilities(d.top_probabilities || []);
  } catch (e) {
    status.innerHTML = `<span style="color:var(--red)">${escapeHtml(e.message || e)}</span>`;
  }
}

function renderWorldCupProbabilities(rows) {
  const tbody = document.getElementById('wc-probabilities-tbody');
  if (!tbody) return;
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text3)">No probability rows returned.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(row => `
    <tr>
      <td>${escapeHtml(row.team_name)}</td>
      <td>${escapeHtml(row.group)}</td>
      <td>${_wcPct(row.round_of_32_probability)}</td>
      <td>${_wcPct(row.semi_final_probability)}</td>
      <td>${_wcPct(row.final_probability)}</td>
      <td><strong>${_wcPct(row.champion_probability)}</strong></td>
    </tr>
  `).join('');
}

function _wcPct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function buildAnalysisBetDraft() {
  const market = document.getElementById('analysis-market').value;
  const selection = document.getElementById('analysis-selection').value;
  const home = document.getElementById('analysis-home').value.trim();
  const away = document.getElementById('analysis-away').value.trim();
  const price = document.getElementById('analysis-price').value.trim();
  let draft = '';
  if (market === 'h2h') {
    if (selection === 'home' && home) draft = `${home} moneyline`;
    else if (selection === 'away' && away) draft = `${away} moneyline`;
    else if (selection === 'draw') draft = `${home && away ? `${home} vs ${away}` : 'match'} draw`;
    else if (home && away) draft = `${home} vs ${away} moneyline`;
  } else if (market === 'totals') {
    if (selection === 'over') draft = `${home && away ? `${home} vs ${away}` : 'match'} total over`;
    else if (selection === 'under') draft = `${home && away ? `${home} vs ${away}` : 'match'} total under`;
    else if (home && away) draft = `${home} vs ${away} total`;
  } else if (market === 'spreads') {
    if (selection === 'home' && home) draft = `${home} spread`;
    else if (selection === 'away' && away) draft = `${away} spread`;
    else if (home && away) draft = `${home} vs ${away} spread`;
  }
  if (draft && price) draft += ` at ${price}`;
  return draft;
}

function syncAnalysisBetDraft(force = false) {
  const betEl = document.getElementById('analysis-bet');
  if (!betEl) return;
  const existing = betEl.value.trim();
  if (!force && existing) return;
  betEl.value = buildAnalysisBetDraft();
}

function clearDeepAnalysisForm() {
  document.getElementById('analysis-home').value = '';
  document.getElementById('analysis-away').value = '';
  document.getElementById('analysis-selection').value = '';
  document.getElementById('analysis-price').value = '';
  document.getElementById('analysis-bet').value = '';
  document.getElementById('analysis-status').textContent = 'Ready.';
  document.getElementById('analysis-meta').innerHTML = '';
  document.getElementById('analysis-output').innerHTML =
    '<div style="color:var(--text3)">Enter a matchup and bet, then run the deep analyst.</div>';
}

async function loadReasoningCandidates() {
  const select = document.getElementById('analysis-candidate');
  const statusEl = document.getElementById('reasoning-status');
  if (!select) return;
  try {
    const r = await fetch('/api/reasoning/candidates');
    const data = await r.json();
    state.reasoningCandidates = data.candidates || [];
    const options = [`<option value="">Choose a value bet from today's board…</option>`]
      .concat(state.reasoningCandidates.map(c => `<option value="${escapeHtml(c.id)}">${escapeHtml(c.display)}</option>`));
    select.innerHTML = options.join('');
    statusEl.textContent = state.reasoningCandidates.length
      ? `${state.reasoningCandidates.length} eligible value bets ready for reasoning review.`
      : `No eligible value bets available yet. Run today's scan first.`;
  } catch (e) {
    statusEl.textContent = `Could not load today's reasoning candidates.`;
  }
}

function selectReasoningCandidate(candidateId) {
  const candidate = state.reasoningCandidates.find(c => c.id === candidateId);
  if (!candidate) {
    renderReasoningCandidateBrief(null);
    return;
  }
  document.getElementById('analysis-sport').value = candidate.sport || 'soccer';
  document.getElementById('analysis-market').value =
    candidate.market === 'moneyline' ? 'h2h' : candidate.market || 'h2h';
  document.getElementById('analysis-home').value = candidate.home_team || '';
  document.getElementById('analysis-away').value = candidate.away_team || '';
  document.getElementById('analysis-price').value = candidate.odds != null ? candidate.odds : '';
  document.getElementById('analysis-bet').value = candidate.selection || '';
  const statusBits = [];
  if (candidate.decision_status) {
    statusBits.push(`${candidate.decision_status}: ${candidate.decision_reason || 'decision explanation unavailable'}.`);
  }
  if (candidate.context_referee_decision) {
    statusBits.push(`Context referee: ${candidate.context_referee_decision}${candidate.context_referee_reason ? ` — ${candidate.context_referee_reason}` : ''}.`);
  }
  document.getElementById('reasoning-status').textContent = statusBits.length
    ? `Selected ${candidate.display}. ${statusBits.join(' ')}`
    : `Selected ${candidate.display}.`;
  renderReasoningCandidateBrief(candidate);
}

function renderReasoningCandidateBrief(candidate) {
  const el = document.getElementById('reasoning-candidate-brief');
  if (!el) return;
  if (!candidate) {
    el.classList.remove('active');
    el.innerHTML = '';
    return;
  }
  const refDecision = String(candidate.context_referee_decision || '').toUpperCase();
  const refClass = refDecision === 'APPROVE'
    ? 'tag-ref-approve'
    : refDecision === 'REVIEW'
      ? 'tag-ref-review'
      : refDecision === 'VETO'
        ? 'tag-ref-veto'
        : refDecision
          ? 'tag-ref-error'
          : '';
  const scraperHighlights = Array.isArray(candidate.scraped_context_highlights)
    ? candidate.scraped_context_highlights.slice(0, 4)
    : [];
  const adjustmentHighlights = Array.isArray(candidate.context_adjustments)
    ? candidate.context_adjustments
        .filter(item => item && item.summary)
        .slice(0, 3)
        .map(item => item.summary)
    : [];
  const combinedHighlights = [...scraperHighlights, ...adjustmentHighlights].filter((item, idx, arr) => item && arr.indexOf(item) === idx).slice(0, 6);
  const sources = Array.isArray(candidate.scraped_context_sources) ? candidate.scraped_context_sources : [];
  const decisionClass = candidate.decision_status === 'BET'
    ? 'tag-ref-approve'
    : ['HOLD', 'WAIT FOR LINEUPS'].includes(candidate.decision_status)
      ? 'tag-ref-review'
      : ['NO BET', 'AVOID'].includes(candidate.decision_status)
        ? 'tag-ref-veto'
        : '';
  const statusTags = [
    launchBadgeHtml(candidate),
    candidate.decision_status ? `<span class="tag ${decisionClass}">${escapeHtml(candidate.decision_status)}</span>` : '',
    refDecision ? `<span class="tag ${refClass}">🧠 ${escapeHtml(refDecision)}</span>` : '',
    candidate.odds_recheck_status ? `<span class="tag">📉 ${escapeHtml(String(candidate.odds_recheck_status).replace(/_/g, ' '))}</span>` : '',
    candidate.availability_summary ? `<span class="tag">🚑 ${escapeHtml(candidate.availability_summary)}</span>` : '',
  ].filter(Boolean).join('');
  el.classList.add('active');
  el.innerHTML = `
    <div class="analysis-brief-head">
      <div>
        <div class="parlay-kicker">Selected Candidate</div>
        <div class="analysis-brief-title">${escapeHtml(candidate.display || candidate.selection || 'Value bet')}</div>
        <div class="analysis-brief-copy">${escapeHtml(candidate.home_team || '')} vs ${escapeHtml(candidate.away_team || '')}</div>
      </div>
      <div class="analysis-brief-tags">${statusTags}</div>
    </div>
    <div class="analysis-brief-grid">
      <div class="analysis-brief-stat">
        <div class="analysis-brief-label">Price</div>
        <div class="analysis-brief-value">${candidate.odds != null ? escapeHtml(String(candidate.odds)) : '—'}</div>
      </div>
      <div class="analysis-brief-stat">
        <div class="analysis-brief-label">Edge</div>
        <div class="analysis-brief-value">${candidate.edge != null ? `${candidate.edge >= 0 ? '+' : ''}${(Number(candidate.edge) * 100).toFixed(1)}%` : '—'}</div>
      </div>
      <div class="analysis-brief-stat">
        <div class="analysis-brief-label">Min Odds</div>
        <div class="analysis-brief-value">${candidate.minimum_acceptable_odds != null ? escapeHtml(String(candidate.minimum_acceptable_odds)) : '—'}</div>
      </div>
      <div class="analysis-brief-stat">
        <div class="analysis-brief-label">Source Feed</div>
        <div class="analysis-brief-value">${sources.length ? escapeHtml(sources.join(', ')) : 'feature snapshot'}</div>
      </div>
    </div>
    <div class="analysis-brief-tags">
      ${combinedHighlights.map(item => `<span class="tag tag-context">${escapeHtml(item)}</span>`).join('')}
    </div>
    ${candidate.decision_reason ? `<div class="analysis-brief-note">Decision note: ${escapeHtml(candidate.decision_reason)}</div>` : ''}
    ${candidate.context_referee_reason ? `<div class="analysis-brief-note">Referee note: ${escapeHtml(candidate.context_referee_reason)}</div>` : ''}
  `;
}

async function runReasoningScan(mode = 'guarded') {
  const candidateId = document.getElementById('analysis-candidate').value;
  const btn = document.getElementById('btn-reasoning-scan');
  const fullBtn = document.getElementById('btn-reasoning-full-live');
  const statusEl = document.getElementById('reasoning-status');
  if (!candidateId) {
    showToast(`Choose one of today's value bets first.`, 'error');
    return;
  }
  btn.disabled = true;
  if (fullBtn) fullBtn.disabled = true;
  if (mode === 'full_live') {
    if (fullBtn) fullBtn.innerHTML = '<div class="spinner"></div> Verifying…';
    if (statusEl) statusEl.textContent = 'Starting full live verification from scratch…';
  } else {
    btn.innerHTML = '<div class="spinner"></div> Reasoning…';
    if (statusEl) statusEl.textContent = 'Starting guarded reasoning review…';
  }
  let progressTimer = setInterval(async () => {
    try {
      const progressResp = await fetch('/api/reasoning/status');
      const progressData = await progressResp.json();
      if (progressData.candidate_id && progressData.candidate_id !== candidateId) return;
      if (progressData.stage && statusEl) statusEl.textContent = progressData.stage;
    } catch (_) {
      // Keep the current status text if polling fails.
    }
  }, 900);
  try {
    const r = await fetch('/api/reasoning/scan', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ candidate_id: candidateId, mode }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Reasoning scan failed');
    state.latestAnalysis = data.report || null;
    renderAnalysisResult(data);
    statusEl.textContent = 'Reasoning scan complete.';
    showToast('Reasoning scan ready.', 'success');
  } catch (e) {
    statusEl.textContent = 'Reasoning scan failed.';
    document.getElementById('analysis-output').innerHTML =
      `<div style="color:var(--red)">Failed to run reasoning scan: ${escapeHtml(String(e))}</div>`;
    document.getElementById('analysis-meta').innerHTML = '';
    showToast('Failed to run reasoning scan.', 'error');
  } finally {
    if (progressTimer) clearInterval(progressTimer);
    btn.disabled = false;
    if (fullBtn) fullBtn.disabled = false;
    btn.innerHTML = '<span>🧠</span> Run Reasoning Scan';
    if (fullBtn) fullBtn.innerHTML = 'Full Live Verification';
  }
}

function renderAnalysisResult(payload) {
  const out = document.getElementById('analysis-output');
  const meta = document.getElementById('analysis-meta');
  const report = payload.report || {};
  const candidate = payload.candidate || {};
  const candidateDecision = String(candidate.decision_status || '').toUpperCase();
  const candidateDecisionReason = String(candidate.decision_reason || candidate.review_reason || candidate.suppression_reason || '').trim();
  const refereeSystemDecision = String(payload.referee_system_decision || '').toUpperCase();
  const verdict = (report.verdict || 'pass').toLowerCase();
  const confidence = report.confidence != null ? `${Math.round(report.confidence * 100)}% confidence` : '';
  const edge = report.edge_pct != null ? `${report.edge_pct >= 0 ? '+' : ''}${(report.edge_pct * 100).toFixed(1)}% edge` : 'No edge estimate';
  const market = (report.data_points || {}).market || {};
  const marketSource = (market.event && market.event.cached) ? 'Cached odds' : (market.event ? 'Live odds' : 'Odds unavailable');
  const analysisMode = payload.analysis_mode || ((report.data_points || {}).analysis_mode) || '';
  const elapsedMs = payload.elapsed_ms != null ? payload.elapsed_ms : ((report.data_points || {}).elapsed_ms);
  const modeLabel = analysisMode === 'guarded_cached_review_plus_live_context'
    ? 'Guarded review: cached board + live web context'
    : analysisMode === 'full_live_candidate_verification'
      ? 'Full live verification: rebuild + live context'
    : analysisMode === 'full_live_manual_analysis'
      ? 'Manual sandbox: full live analyst rebuild'
      : '';
  const warnings = Array.isArray(report.warnings) ? report.warnings : [];
  const unknowns = Array.isArray(report.unknowns) ? report.unknowns : [];
  const signals = Array.isArray(report.signals) ? report.signals : [];
  const quotaWarning = warnings.find(w => w.toLowerCase().includes('budget'));
  const llmReasoning = payload.llm_reasoning || null;
  const llmContent = llmReasoning && llmReasoning.content ? llmReasoning.content : null;
  const contextAdjustments = Array.isArray(candidate.context_adjustments) ? candidate.context_adjustments : [];
  const scraperHighlights = Array.isArray(candidate.scraped_context_highlights) ? candidate.scraped_context_highlights : [];
  const scraperSources = Array.isArray(candidate.scraped_context_sources) ? candidate.scraped_context_sources : [];
  const freshNews = payload.fresh_news_context || candidate.fresh_news_context || ((report.data_points || {}).fresh_news_context) || {};
  const freshNewsHighlights = Array.isArray(freshNews.highlights) ? freshNews.highlights : [];
  const freshNewsSources = Array.isArray(freshNews.sources) ? freshNews.sources : [];
  const freshNewsItems = Array.isArray(freshNews.items) ? freshNews.items : [];
  const freshNewsWarnings = Array.isArray(freshNews.warnings) ? freshNews.warnings : [];
  const freshNewsChannels = freshNews.channels || {};
  const evidence = payload.evidence_profile || ((report.data_points || {}).evidence_profile) || {};
  const evidenceChannels = evidence.channels || {};
  const evidenceQuality = String(evidence.quality || 'unknown').toUpperCase();
  const evidenceDecision = String(evidence.decision || '').toUpperCase();
  const evidenceRisks = Array.isArray(evidence.risk_flags) ? evidence.risk_flags : [];
  const contextHighlights = contextAdjustments
    .filter(item => item && ['lineup', 'schedule', 'matchup', 'coaching', 'environment', 'motivation'].includes(item.category))
    .slice(0, 5);
  const summaryCards = [
    { label: 'System Decision', value: candidateDecision || '—', klass: candidateDecision === 'BET' ? 'green' : ['NO BET', 'AVOID'].includes(candidateDecision) ? 'red' : '' },
    { label: 'Analyst Verdict', value: (report.verdict || 'pass').toUpperCase(), klass: verdict === 'avoid' ? 'red' : verdict === 'support' ? 'green' : '' },
    { label: 'Confidence', value: report.confidence != null ? `${Math.round(report.confidence * 100)}%` : '—', klass: '' },
    { label: 'Price Used', value: report.price_used != null ? Number(report.price_used).toFixed(2) : '—', klass: '' },
    { label: 'Edge', value: report.edge_pct != null ? `${report.edge_pct >= 0 ? '+' : ''}${(report.edge_pct * 100).toFixed(1)}%` : '—', klass: report.edge_pct != null && report.edge_pct >= 0 ? 'green' : report.edge_pct != null ? 'red' : '' },
    { label: 'Evidence', value: evidenceDecision || evidenceQuality, klass: evidenceQuality === 'STRONG' ? 'green' : evidenceQuality === 'THIN' ? 'red' : '' },
  ];

  meta.innerHTML = `
    ${candidateDecision ? `<span class="analysis-pill ${candidateDecision === 'BET' ? 'support' : ['NO BET', 'AVOID'].includes(candidateDecision) ? 'avoid' : 'lean'}">${escapeHtml(candidateDecision)}</span>` : ''}
    <span class="analysis-pill ${verdict}">${verdict.toUpperCase()}</span>
    <span class="analysis-pill pass">${confidence}</span>
    <span class="analysis-pill pass">${edge}</span>
    <span class="analysis-pill pass">${marketSource}</span>
    ${evidenceDecision ? `<span class="analysis-pill ${evidenceQuality === 'THIN' ? 'avoid' : 'pass'}">${escapeHtml(evidenceDecision)}</span>` : ''}
    ${modeLabel ? `<span class="analysis-pill pass">${escapeHtml(modeLabel)}</span>` : ''}
    ${elapsedMs != null ? `<span class="analysis-pill pass">${escapeHtml(String(elapsedMs))} ms</span>` : ''}
  `;

  const matchLabel = `${candidate.home_team || report.home_team || 'Home'} vs ${candidate.away_team || report.away_team || 'Away'}`;
  const betLabel = candidate.selection || report.bet || 'Selected bet';
  const strongestSignals = signals.slice(0, 2).map(signal => signal.summary).filter(Boolean);
  const briefTone = candidateDecision === 'BET'
    ? 'This candidate cleared the production decision layer, and the reasoning scan adds context checks on top.'
    : candidateDecision === 'WAIT FOR LINEUPS'
      ? 'This candidate is being held until lineup or availability uncertainty clears.'
      : candidateDecision === 'HOLD'
        ? 'This candidate is on hold rather than approved for production. Use the notes below to decide whether it deserves manual follow-up.'
        : ['NO BET', 'AVOID'].includes(candidateDecision)
          ? 'This candidate did not clear the production decision layer. The notes below explain why it was kept off the board.'
          : 'This is a grounded analyst read on the selected model candidate.';
  const edgeText = report.edge_pct != null
    ? `${report.edge_pct >= 0 ? '+' : ''}${(report.edge_pct * 100).toFixed(1)}%`
    : 'not available';
  const cleanBrief = `
    <div class="analysis-output-section-title">Readable Brief</div>
    <div class="analysis-report">
      <div class="analysis-brief-title">${escapeHtml(matchLabel)} · ${escapeHtml(betLabel)}</div>
      <div class="analysis-brief-copy">${escapeHtml(briefTone)}</div>
      <div class="analysis-brief-list">
        <div><strong>Price and edge:</strong> ${report.price_used != null ? escapeHtml(Number(report.price_used).toFixed(2)) : 'No usable price'} with an estimated edge of ${escapeHtml(edgeText)}.</div>
        <div><strong>Confidence:</strong> ${report.confidence != null ? escapeHtml(String(Math.round(report.confidence * 100))) + '%' : 'Not available'} from the available matchup signals.</div>
        ${strongestSignals.length ? strongestSignals.map(text => `<div><strong>Signal:</strong> ${escapeHtml(text)}</div>`).join('') : '<div><strong>Signal:</strong> No strong matchup signal was available.</div>'}
        ${unknowns.length ? `<div><strong>Missing information:</strong> ${escapeHtml(unknowns[0])}</div>` : ''}
      </div>
    </div>
  `;
  const topNote = quotaWarning
    ? `<div style="margin-bottom:12px;padding:10px 12px;border:1px solid rgba(245,158,11,.35);background:rgba(245,158,11,.08);border-radius:var(--r);font-size:.8rem;color:var(--text2)">${escapeHtml(quotaWarning)}</div>`
    : '';
  const reviewBlock = candidateDecision && candidateDecision !== 'BET'
    ? `
      <div style="margin-bottom:14px;padding:12px 14px;border:1px solid rgba(245,158,11,.35);background:rgba(245,158,11,.08);border-radius:var(--r)">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">System Decision</div>
        <div style="font-weight:700;color:#fbbf24;margin-bottom:6px">${escapeHtml(candidateDecision)}</div>
        <div style="font-size:.82rem;color:var(--text2);margin-bottom:8px">${escapeHtml(candidateDecisionReason || 'Decision explanation unavailable.')}</div>
        <div style="font-size:.78rem;color:var(--text3)">${escapeHtml(candidate.market_policy_label || '')}${candidate.market_policy_reason ? ` · ${escapeHtml(candidate.market_policy_reason)}` : ''}</div>
      </div>
    `
    : '';
  const executionBlock = (candidate.minimum_acceptable_odds || candidate.odds_recheck_status || candidate.context_referee_decision)
    ? `
      <div style="margin-bottom:14px;padding:12px 14px;border:1px solid rgba(148,163,184,.2);background:rgba(148,163,184,.07);border-radius:var(--r)">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Execution Gate</div>
        ${candidate.minimum_acceptable_odds ? `<div class="analysis-warning-item">Minimum acceptable odds: ${escapeHtml(String(candidate.minimum_acceptable_odds))}</div>` : ''}
        ${candidate.odds_recheck_status ? `<div class="analysis-warning-item">Odds recheck: ${escapeHtml(String(candidate.odds_recheck_status).replace(/_/g,' '))}${candidate.odds_recheck_delta != null ? ` (${candidate.odds_recheck_delta >= 0 ? '+' : ''}${escapeHtml(String(candidate.odds_recheck_delta))})` : ''}</div>` : ''}
        ${candidate.context_referee_decision ? `<div class="analysis-warning-item">Context referee: ${escapeHtml(String(candidate.context_referee_decision))}${candidate.context_referee_reason ? ` — ${escapeHtml(candidate.context_referee_reason)}` : ''}</div>` : ''}
      </div>
    `
    : '';
  const systemContextBlock = (candidate.availability_summary || contextHighlights.length)
    ? `
      <div style="margin-bottom:14px;padding:12px 14px;border:1px solid rgba(59,130,246,.25);background:rgba(59,130,246,.06);border-radius:var(--r)">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">System Context</div>
        ${candidate.availability_summary ? `<div class="analysis-warning-item">Availability: ${escapeHtml(candidate.availability_summary)}</div>` : ''}
        ${scraperSources.length ? `<div class="analysis-warning-item">Scraper sources: ${escapeHtml(scraperSources.join(', '))}</div>` : ''}
        ${[...new Set(scraperHighlights)].map(item => `<div class="analysis-warning-item">${escapeHtml(item)}</div>`).join('')}
        ${[...new Set(contextHighlights.map(item => item.summary || item.name || 'Context signal'))].map(item => `<div class="analysis-warning-item">${escapeHtml(item)}</div>`).join('')}
      </div>
    `
    : '';
  const freshNewsBlock = (freshNewsHighlights.length || freshNewsItems.length || freshNewsWarnings.length)
    ? `
      <div style="margin-bottom:14px;padding:12px 14px;border:1px solid rgba(34,197,94,.25);background:rgba(34,197,94,.06);border-radius:var(--r)">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Fresh Web Context</div>
        ${freshNewsSources.length ? `<div class="analysis-warning-item">Sources checked: ${escapeHtml(freshNewsSources.join(', '))}</div>` : ''}
        ${Object.values(freshNewsChannels).map(ch => {
          const sources = Array.isArray(ch.sources) ? ch.sources : [];
          const trust = ch.trust ? ` · ${ch.trust}` : '';
          return `<div class="analysis-warning-item"><strong>${escapeHtml(ch.label || 'Source channel')}:</strong> ${sources.length ? escapeHtml(sources.join(', ')) : 'No snippets'}${escapeHtml(trust)}</div>`;
        }).join('')}
        ${freshNewsHighlights.slice(0, 4).map(item => `<div class="analysis-warning-item">${escapeHtml(item)}</div>`).join('')}
        ${!freshNewsHighlights.length ? freshNewsItems.slice(0, 3).map(item => `<div class="analysis-warning-item">${escapeHtml(item.title || 'Web result')}${item.source ? ` · ${escapeHtml(item.source)}` : ''}</div>`).join('') : ''}
        ${freshNewsWarnings.slice(0, 2).map(item => `<div class="analysis-warning-item">${escapeHtml(item)}</div>`).join('')}
      </div>
    `
    : '';
  const channelEntries = Object.values(evidenceChannels || {});
  const evidenceBlock = channelEntries.length
    ? `
      <div style="margin-bottom:14px;padding:12px 14px;border:1px solid rgba(245,158,11,.28);background:rgba(245,158,11,.06);border-radius:var(--r)">
        <div style="font-size:.72rem;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px">Evidence Hub</div>
        <div class="analysis-warning-item"><strong>Decision:</strong> ${escapeHtml(evidenceDecision || evidenceQuality)} · ${escapeHtml(evidenceQuality.toLowerCase())} data quality</div>
        ${channelEntries.map(ch => `<div class="analysis-warning-item"><strong>${escapeHtml(ch.label || 'Channel')}:</strong> ${escapeHtml(String(ch.status || 'unknown').toUpperCase())} — ${escapeHtml(ch.detail || '')}</div>`).join('')}
        ${evidenceRisks.slice(0, 3).map(item => `<div class="analysis-warning-item">Risk: ${escapeHtml(item)}</div>`).join('')}
      </div>
    `
    : '';
  const signalsBlock = signals.length
    ? `
      <div class="analysis-output-section-title">Top Signals</div>
      <div class="analysis-signals">
        ${signals.slice(0, 4).map(signal => `
          <div class="analysis-signal-row">
            <div class="analysis-signal-top">
              <div class="analysis-signal-title">${escapeHtml(signal.name || 'Signal')}</div>
              <div class="analysis-signal-score">score ${Number(signal.score || 0).toFixed(2)} · ${Math.round((signal.confidence || 0) * 100)}% conf</div>
            </div>
            <div style="font-size:.78rem;color:var(--text2)">${escapeHtml(signal.summary || '')}</div>
          </div>
        `).join('')}
      </div>
    `
    : '';
  const warningItems = [...warnings.filter(w => {
    const text = String(w || '').toLowerCase();
    return w !== quotaWarning && !text.includes('no reasoning returned') && !text.startsWith('fresh web context:');
  }), ...unknowns];
  const warningBlock = warningItems.length
    ? `
      <div class="analysis-output-section-title">Watchouts</div>
      <div class="analysis-warning-list">
        ${warningItems.slice(0, 4).map(item => `<div class="analysis-warning-item">${escapeHtml(item)}</div>`).join('')}
      </div>
    `
    : '';
  const llmBlock = llmContent
    ? (() => {
      const recommendation = String(llmContent.recommendation || llmContent.reasoning || '').trim();
      const emptyReasoning = !recommendation || recommendation.toLowerCase() === 'no reasoning returned.';
      if (emptyReasoning) {
        return `
          <div class="analysis-output-section-title">Reasoning Layer</div>
          <div class="analysis-llm-card">
            <div style="font-weight:700;margin-bottom:6px">AI referee unavailable</div>
            <div style="font-size:.82rem;color:var(--text2)">OpenRouter did not return a usable explanation for this run. The structured analyst signals above are still valid, but treat the AI referee as unavailable rather than supportive.</div>
          </div>
        `;
      }
      return `
      <div class="analysis-output-section-title">Context-Only Reasoning Layer</div>
      <div class="analysis-llm-card">
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap">
          <div style="font-weight:700">Independent read from evidence/news only</div>
          <div style="font-size:.74rem;color:var(--text3)">${escapeHtml(llmReasoning.model || 'openrouter/free')}</div>
        </div>
        <div class="analysis-llm-grid">
          <div class="analysis-llm-item"><strong>Agrees With System</strong>${llmContent.agrees_with_system === true ? 'Yes' : llmContent.agrees_with_system === false ? 'No / Not enough' : '—'}</div>
          <div class="analysis-llm-item"><strong>Mapped Verdict</strong>${escapeHtml(refereeSystemDecision || '—')}</div>
          <div class="analysis-llm-item"><strong>Recommendation</strong>${escapeHtml(llmContent.recommendation || '—')}</div>
          <div class="analysis-llm-item"><strong>Stake Guidance</strong>${escapeHtml(llmContent.stake_guidance || '—')}</div>
          <div class="analysis-llm-item"><strong>Why For</strong>${escapeHtml(llmContent.why_for || '—')}</div>
          <div class="analysis-llm-item"><strong>Why Against</strong>${escapeHtml(llmContent.why_against || '—')}</div>
          <div class="analysis-llm-item" style="grid-column:1 / -1"><strong>Biggest Risk</strong>${escapeHtml(llmContent.biggest_risk || '—')}</div>
        </div>
      </div>
    `;
    })()
    : (payload.llm_error
      ? `<div class="analysis-warning-item" style="margin-bottom:14px">OpenRouter reasoning layer unavailable: ${escapeHtml(payload.llm_error)}</div>`
      : '');

  out.innerHTML = `
    ${topNote}
    ${reviewBlock}
    ${executionBlock}
    ${evidenceBlock}
    ${systemContextBlock}
    ${freshNewsBlock}
    <div class="analysis-summary-grid">
      ${summaryCards.map(card => `
        <div class="analysis-summary-card">
          <div class="analysis-summary-label">${card.label}</div>
          <div class="analysis-summary-value ${card.klass}">${card.value}</div>
        </div>
      `).join('')}
    </div>
    ${llmBlock}
    ${signalsBlock}
    ${warningBlock}
    ${cleanBrief}
  `;
}

async function runDeepAnalysis() {
  const btn = document.getElementById('btn-analyze-game');
  const statusEl = document.getElementById('analysis-status');
  const payload = {
    sport: document.getElementById('analysis-sport').value,
    market: document.getElementById('analysis-market').value,
    home_team: document.getElementById('analysis-home').value.trim(),
    away_team: document.getElementById('analysis-away').value.trim(),
    bet: document.getElementById('analysis-bet').value.trim(),
    selection: document.getElementById('analysis-selection').value,
    price: document.getElementById('analysis-price').value.trim(),
  };

  if (!payload.home_team || !payload.away_team || !payload.bet) {
    showToast('Home team, away team, and bet are required.', 'error');
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Analyzing…';
  statusEl.textContent = 'Pulling market and matchup context…';

  try {
    const r = await fetch('/api/analyze-game', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Analysis failed');
    state.latestAnalysis = data.report || null;
    renderAnalysisResult(data);
    statusEl.textContent = 'Analysis complete.';
    showToast('Deep analysis ready.', 'success');
  } catch (e) {
    statusEl.textContent = 'Analysis failed.';
    document.getElementById('analysis-output').innerHTML =
      `<div style="color:var(--red)">Failed to run analysis: ${escapeHtml(String(e))}</div>`;
    document.getElementById('analysis-meta').innerHTML = '';
    showToast('Failed to run deep analysis.', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>🧠</span> Analyze Game';
  }
}

// ═══════════════════════════════════════════════════════════
// API MANAGER
// ═══════════════════════════════════════════════════════════
async function loadApis() {
  const el = document.getElementById('api-cards');
  el.innerHTML = '<div style="color:var(--text3);padding:20px">Checking APIs…</div>';
  try {
    const r = await fetch('/api/apis/status');
    const d = await r.json();
    const pool = d.odds_key_pool || {};
    const activePoolRemaining = pool.active_remaining;
    const activePoolFingerprint = pool.active_fingerprint || pool.last_selected_fingerprint || '';
    const runtimeCount = pool.runtime_loaded_count ?? 0;
    const usableCount = pool.usable_count ?? 0;
    const trackedCount = pool.tracked_count ?? pool.count ?? 0;
    const historicalMissing = pool.tracked_but_unavailable_count ?? 0;

    const apiCards = d.apis.map(api => {
      const isOddsApi = api.name === 'The Odds API';
      const dotClass = api.status === 'ok' ? 'ok' : api.status === 'warn' ? 'warn' : 'error';
      let quotaHtml = '';
      if (isOddsApi) {
        const runtimeText = `${runtimeCount} runtime key${runtimeCount === 1 ? '' : 's'} · ${usableCount} usable`;
        quotaHtml = `
          <div class="api-detail" style="margin-top:8px">${runtimeText}${historicalMissing ? ` · ${historicalMissing} historical raw-missing` : ''}</div>`;
      } else if ('used' in api) {
        const pct = api.used / api.total * 100;
        const fillClass = pct < 50 ? '' : pct < 80 ? ' yellow' : ' red';
        quotaHtml = `
          <div class="api-quota">
            <div class="api-quota-bar"><div class="api-quota-fill${fillClass}" style="width:${pct}%"></div></div>
            <div class="api-quota-text">${api.remaining}/${api.total}</div>
          </div>`;
      }
      return `
        <div class="api-card">
          <div class="api-status-dot ${dotClass}"></div>
          <div class="api-name">${api.name}</div>
          <div class="api-detail">${api.detail}</div>
          ${quotaHtml}
        </div>`;
    }).join('');
    const poolRowsAll = Array.isArray(pool.keys) ? pool.keys : [];
    const poolRows = poolRowsAll.filter(row => row && row.runtime_available).slice(0, 4);
    const selectionReason = pool.last_selected_reason ? ` · ${escapeHtml(pool.last_selected_reason)}` : '';
    const poolCard = pool.enabled ? `
        <div class="api-card">
          <div class="api-status-dot ok"></div>
          <div class="api-name">Odds API Key Pool</div>
          <div class="api-detail">${trackedCount} tracked key${trackedCount === 1 ? '' : 's'} · ${runtimeCount} runtime key${runtimeCount === 1 ? '' : 's'} · ${usableCount} usable${pool.last_selected_fingerprint ? ` · active …${escapeHtml(pool.last_selected_fingerprint)}` : ''}${selectionReason}</div>
          ${poolRows.length ? `<div class="api-detail" style="margin-top:8px">${poolRows.map(row => {
            const flags = [];
            if (row.selected) flags.push('active');
            if (row.status === 'quota_exhausted') flags.push('used up');
            if (row.status === 'stale_metadata') flags.push('stale metadata');
            if (row.low_quota) flags.push('low');
            if (row.runtime_available && row.usable) flags.push('runtime');
            if (row.status === 'runtime_only') flags.push('new / quota unknown');
            return `…${escapeHtml(row.fingerprint || '')}: ${row.remaining ?? '—'}${flags.length ? ` (${flags.join(', ')})` : ''}`;
          }).join('<br>')}</div>` : ''}
          ${historicalMissing ? `<div class="api-detail" style="margin-top:8px;color:var(--text3)">${historicalMissing} older tracked key${historicalMissing === 1 ? '' : 's'} are not loaded in the current runtime pool.</div>` : ''}
        </div>` : '';
    el.innerHTML = apiCards + poolCard;
    updateApiKeyInputHelp();
  } catch(e) {
    el.innerHTML = '<div style="color:var(--red)">Failed to check APIs.</div>';
  }
}

function updateApiKeyInputHelp() {
  const varEl = document.getElementById('key-var');
  const valueEl = document.getElementById('key-value');
  const helpEl = document.getElementById('key-value-help');
  if (!varEl || !valueEl || !helpEl) return;
  const isPool = varEl.value === 'ODDS_API_KEYS';
  valueEl.rows = isPool ? 5 : 3;
  valueEl.placeholder = isPool
    ? 'Paste one Odds API key per line, or comma-separated…'
    : 'Paste new key here…';
  helpEl.innerHTML = isPool
    ? 'Paste multiple Odds API keys here. Use <b>Add Odds Keys</b> to append without replacing the pool, or <b>Save</b> to replace <code>ODDS_API_KEYS</code>.'
    : 'For <code>ODDS_API_KEYS</code>, paste comma-separated keys or one key per line.';
}

async function saveApiKey() {
  const varName = document.getElementById('key-var').value;
  const value   = document.getElementById('key-value').value.trim();
  if (!value) { showToast('Please enter a value', 'error'); return; }
  try {
    const r = await fetch('/api/apis/update', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ var: varName, value }),
    });
    const d = await r.json();
    if (d.ok) {
      showToast(varName + ' updated ✓', 'success');
      document.getElementById('key-value').value = '';
      updateApiKeyInputHelp();
      loadApis();
      loadDashboard();
    } else {
      showToast(d.error || 'Failed', 'error');
    }
  } catch(e) { showToast('Request failed', 'error'); }
}

async function addOddsApiKeys() {
  const valueEl = document.getElementById('key-value');
  const value = valueEl.value.trim();
  if (!value) { showToast('Paste one or more Odds API keys first', 'error'); return; }
  try {
    const r = await fetch('/api/apis/odds-keys/add', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ value }),
    });
    const d = await r.json();
    if (!r.ok || !d.ok) {
      showToast(d.error || 'Failed to add Odds API keys', 'error');
      return;
    }
    showToast(`Added ${d.added_count || 0} Odds API key${(d.added_count || 0) === 1 ? '' : 's'} ✓`, 'success');
    valueEl.value = '';
    loadApis();
    loadDashboard();
  } catch(e) {
    showToast('Request failed', 'error');
  }
}

async function pruneExhaustedOddsKeys() {
  try {
    const r = await fetch('/api/apis/odds-keys/prune-exhausted', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({}),
    });
    const d = await r.json();
    if (!r.ok || !d.ok) {
      showToast(d.error || 'Failed to retire used-up keys', 'error');
      return;
    }
    showToast(`Retired ${d.removed_count || 0} used-up Odds API key${(d.removed_count || 0) === 1 ? '' : 's'}`, 'success');
    loadApis();
    loadDashboard();
  } catch(e) {
    showToast('Request failed', 'error');
  }
}

// ═══════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════
let _toastTimer;
function showToast(msg, type='success') {
  const el = document.getElementById('toast');
  // Render newlines as line breaks; escape any raw HTML to avoid XSS
  el.textContent = '';
  msg.split('\n').forEach((line, i) => {
    if (i > 0) el.appendChild(document.createElement('br'));
    el.appendChild(document.createTextNode(line));
  });
  el.className = 'show ' + type;
  clearTimeout(_toastTimer);
  // Scale timeout: ~100ms per char, min 3s, max 10s
  const ms = Math.min(10000, Math.max(3000, msg.length * 80));
  _toastTimer = setTimeout(() => el.className = '', ms);
}

// ═══════════════════════════════════════════════════════════
// DASHBOARD PAGE
// ═══════════════════════════════════════════════════════════
function renderDashboardPage(d) {
  // KPI cards
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  const bankroll = '£' + Number(d.bankroll || 0).toLocaleString();
  set('db-bankroll', bankroll);
  set('db-picks-today', d.total_bets ?? '—');
  set('db-games-scanned', (d.total_games ?? '—') + ' games scanned');
  const rem = d.odds_remaining, start = d.odds_start || 500;
  set('db-quota', `${rem ?? '—'}/${start}`);
  const qPct = rem / start * 100;
  const qCard = document.getElementById('db-quota-card');
  if (qCard) qCard.className = 'stat-card' + (qPct <= 10 ? ' red' : qPct <= 25 ? ' yellow' : ' green');

  // P&L from MY selections (personal ROI), fall back to system-wide
  const my = d._myStats || {};
  const ps = d.process_summary || {};
  const hasMyData = my.settled > 0;
  const pnlRaw = hasMyData ? my.total_profit : ps.total_profit;
  const roiRaw = hasMyData ? my.roi : ps.roi;
  const pnl = pnlRaw != null ? Number(pnlRaw).toFixed(2) : null;
  const roi = roiRaw != null ? (Number(roiRaw) * 100).toFixed(1) : null;
  set('db-pnl', pnl != null ? (pnl >= 0 ? '+' : '') + pnl + 'u' : '—');
  set('db-roi-sub', roi != null ? `My ROI ${roi >= 0 ? '+' : ''}${roi}%` : (hasMyData ? 'No settled picks yet' : 'ROI —'));
  const pCard = document.getElementById('db-pnl-card');
  if (pCard && pnl != null) pCard.className = 'stat-card ' + (pnl >= 0 ? 'green' : 'red');

  // Last scan
  const lastScan = d.scan_time
    ? new Date(d.scan_time).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})
    : '—';
  set('db-last-scan', 'Last scan: ' + lastScan);

  // Model status
  const modEl = document.getElementById('db-model-status');
  if (modEl) {
    const models = d.model_tags || {};
    if (Object.keys(models).length) {
      modEl.innerHTML = Object.entries(models).map(([sport, tag]) =>
        `<div class="status-row">
          <div class="status-dot ok"></div>
          <div class="status-label">${escapeHtml(sport)}</div>
          <div class="status-meta">${escapeHtml(String(tag))}</div>
        </div>`
      ).join('');
    } else {
      modEl.innerHTML = '<div style="color:var(--text3);font-size:.85rem">Model tag data unavailable</div>';
    }
  }

  // System status
  const sysEl = document.getElementById('db-system-status');
  if (sysEl) {
    const quota_mode = d.quota_mode || 'healthy';
    const modeBg = {healthy:'ok', caution:'warn', core_only:'warn', critical:'error'}[quota_mode] || 'ok';
    const ps2 = d.process_summary || {};
    sysEl.innerHTML = `
      <div class="status-row">
        <div class="status-dot ${modeBg}"></div>
        <div class="status-label">Quota mode</div>
        <div class="status-meta">${quota_mode}</div>
      </div>
      <div class="status-row">
        <div class="status-dot ok"></div>
        <div class="status-label">Tracked bets</div>
        <div class="status-meta">${ps2.n_bets ?? '—'} resolved</div>
      </div>
      <div class="status-row">
        <div class="status-dot ${ps2.avg_clv > 0 ? 'ok' : ps2.avg_clv != null ? 'warn' : 'idle'}"></div>
        <div class="status-label">Avg CLV</div>
        <div class="status-meta">${ps2.avg_clv != null ? ((ps2.avg_clv >= 0 ? '+' : '') + (ps2.avg_clv*100).toFixed(1) + '%') : '—'}</div>
      </div>
      <div class="status-row">
        <div class="status-dot ok"></div>
        <div class="status-label">Cycle horizon</div>
        <div class="status-meta">${d.odds_days_left_in_cycle ?? '—'} days</div>
      </div>`;
  }

  // Picks by sport bars
  const sportEl = document.getElementById('db-picks-by-sport');
  if (sportEl) {
    const sports = Object.entries(d.by_sport || {})
      .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0));
    if (sports.length) {
      const max = Math.max(...sports.map(([, c]) => Number(c || 0)), 1);
      sportEl.innerHTML = sports.map(([sp, cnt]) => `
        <div class="sport-pick-row">
          <div class="sport-pick-label">${escapeHtml(sp)}</div>
          <div class="sport-pick-bar-wrap"><div class="sport-pick-bar" style="width:${(Number(cnt)/max)*100}%"></div></div>
          <div class="sport-pick-count">${cnt}</div>
        </div>`).join('');
    } else {
      sportEl.innerHTML = '<div style="color:var(--text3);font-size:.85rem">No picks today yet</div>';
    }
  }

  // Mini P&L chart (last 14 days)
  renderDashboardMiniChart(d.daily_pnl || []);
}

let _dbMiniChart = null;
function renderDashboardMiniChart(dailyPnl) {
  const ctx = document.getElementById('db-pnl-chart');
  if (!ctx) return;
  const last14 = dailyPnl.slice(-14);
  const labels = last14.map(r => r.date ? r.date.slice(5) : '');
  const values = last14.map(r => Number(r.profit_units || 0));
  if (_dbMiniChart) { _dbMiniChart.destroy(); _dbMiniChart = null; }
  if (!last14.length) {
    ctx.parentElement.innerHTML = '<div style="color:var(--text3);font-size:.85rem;padding:20px 0">No settled bets yet — P&L chart will appear here.</div>';
    return;
  }
  _dbMiniChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: values.map(v => v >= 0 ? 'rgba(16,185,129,.55)' : 'rgba(239,68,68,.55)'),
        borderColor:     values.map(v => v >= 0 ? 'rgba(16,185,129,.9)'  : 'rgba(239,68,68,.9)'),
        borderWidth: 1, borderRadius: 3,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => (ctx.parsed.y >= 0 ? '+' : '') + ctx.parsed.y.toFixed(2) + 'u' }
      }},
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#64748b', font: { size: 11 } } },
        y: { grid: { color: 'rgba(255,255,255,.06)' }, ticks: { color: '#64748b', font: { size: 11 },
          callback: v => (v >= 0 ? '+' : '') + v.toFixed(1) + 'u' } },
      },
    },
  });
}

// ═══════════════════════════════════════════════════════════
// PERFORMANCE PAGE
// ═══════════════════════════════════════════════════════════
let _perfCharts = {};

let _perfScope = 'mine'; // 'mine' | 'all'
let _perfCache  = {};

async function loadPerformance() {
  try {
    const [allR, myR] = await Promise.all([
      fetch('/api/results'),
      fetch('/api/my-selections/results'),
    ]);
    _perfCache.all  = allR.ok  ? await allR.json()  : null;
    _perfCache.mine = myR.ok   ? await myR.json()   : null;
    _renderPerfForScope();
  } catch(e) { console.error('loadPerformance:', e); }
}

function switchPerfScope(scope, btn) {
  _perfScope = scope;
  document.querySelectorAll('#perf-tab-mine, #perf-tab-all').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  _renderPerfForScope();
}

function _renderPerfForScope() {
  if (_perfScope === 'mine' && _perfCache.mine) {
    _renderPerformanceMine(_perfCache.mine);
  } else if (_perfCache.all) {
    renderPerformance(_perfCache.all);
  }
}

function _formatPerfSummaryValue(summary) {
  if (!summary) return '—';
  return `${summary.bets || 0} bets`;
}

function _formatPerfSummarySub(summary) {
  if (!summary) return '—';
  const wr = summary.win_rate ?? 0;
  const roi = summary.roi ?? 0;
  const pnl = summary.pnl ?? 0;
  return `WR ${wr}% · ROI ${roi >= 0 ? '+' : ''}${roi}% · P&L ${pnl >= 0 ? '+' : ''}${pnl}`;
}

function _renderPerfEvaluationSummary(summary) {
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  const allBets = summary?.all_bets || null;
  const singles = summary?.singles_only || null;
  const recent = summary?.recent_singles || null;
  set('perf-eval-all', _formatPerfSummaryValue(allBets));
  set('perf-eval-all-sub', _formatPerfSummarySub(allBets));
  set('perf-eval-singles', _formatPerfSummaryValue(singles));
  set('perf-eval-singles-sub', _formatPerfSummarySub(singles));
  set('perf-eval-recent', _formatPerfSummaryValue(recent));
  set('perf-eval-recent-sub', _formatPerfSummarySub(recent));
}

function _renderPerfLaneHighlights(highlights) {
  const host = document.getElementById('perf-lane-highlights');
  if (!host) return;
  const best = highlights?.best || [];
  const worst = highlights?.worst || [];
  const renderList = (title, rows, positive) => `
    <div style="margin-bottom:12px">
      <div style="font-size:.8rem;font-weight:700;color:${positive ? 'var(--green)' : 'var(--red)'};margin-bottom:8px">${title}</div>
      ${rows.length ? rows.map(row => `
        <div style="display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);font-size:.82rem">
          <div>${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)} · <span class="tag tag-market">${escapeHtml(row.market)}</span></div>
          <div style="white-space:nowrap;color:${positive ? 'var(--green)' : 'var(--red)'}">${row.roi >= 0 ? '+' : ''}${row.roi}% ROI</div>
        </div>
      `).join('') : '<div style="color:var(--text3);font-size:.82rem">No settled lane data yet.</div>'}
    </div>
  `;
  host.innerHTML = renderList('Best settled lanes', best, true) + renderList('Weakest settled lanes', worst, false);
}

function _renderPerfOddsBuckets(rows) {
  const tbody = document.getElementById('perf-odds-bucket-tbody');
  if (!tbody) return;
  tbody.innerHTML = (rows || []).length ? rows.map(row => {
    const roiColor = row.roi >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlColor = row.pnl >= 0 ? 'var(--green)' : 'var(--red)';
    return `
      <tr>
        <td>${escapeHtml(row.bucket)}</td>
        <td>${row.bets}</td>
        <td>${row.win_rate}%</td>
        <td style="color:${roiColor}">${row.roi >= 0 ? '+' : ''}${row.roi}%</td>
        <td style="color:${pnlColor}">${row.pnl >= 0 ? '+' : ''}${row.pnl}</td>
      </tr>`;
  }).join('') : '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:18px">No settled odds-bucket data yet.</td></tr>';
}

function _renderPerfCalibration(snapshot) {
  const summaryEl = document.getElementById('perf-calibration-summary');
  const tbody = document.getElementById('perf-calibration-tbody');
  const bucketTbody = document.getElementById('perf-calibration-buckets-tbody');
  if (!summaryEl || !tbody || !bucketTbody) return;

  const summary = (snapshot && snapshot.summary) || {};
  const bySport = Array.isArray(snapshot && snapshot.by_sport) ? snapshot.by_sport : [];
  const buckets = Array.isArray(snapshot && snapshot.buckets) ? snapshot.buckets : [];

  if (!summary.bets) {
    summaryEl.innerHTML = '<span style="color:var(--text3)">No settled probability data yet.</span>';
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text3);padding:18px">No calibration rows yet.</td></tr>';
    bucketTbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:18px">No confidence-bucket data yet.</td></tr>';
    return;
  }

  summaryEl.innerHTML = `Overall: <strong>${summary.bets}</strong> settled bets · avg prob <strong>${summary.avg_prob_pct}%</strong> · actual win rate <strong>${summary.win_rate_pct}%</strong> · gap <strong>${summary.gap_pp}pp</strong> · Brier <strong>${summary.brier}</strong> · log loss <strong>${summary.log_loss}</strong> · ECE <strong>${summary.ece}</strong>`;

  tbody.innerHTML = bySport.length ? bySport.map(row => `
    <tr>
      <td>${escapeHtml(row.sport)}</td>
      <td>${row.bets}</td>
      <td>${row.avg_prob_pct}%</td>
      <td>${row.win_rate_pct}%</td>
      <td>${row.gap_pp}pp</td>
      <td>${row.brier}</td>
      <td>${row.log_loss}</td>
    </tr>
  `).join('') : '<tr><td colspan="7" style="text-align:center;color:var(--text3);padding:18px">Need at least a few settled bets per sport.</td></tr>';

  bucketTbody.innerHTML = buckets.length ? buckets.map(row => `
    <tr>
      <td>${escapeHtml(row.bucket)}</td>
      <td>${row.bets}</td>
      <td>${row.avg_prob_pct}%</td>
      <td>${row.win_rate_pct}%</td>
      <td>${row.gap_pp}pp</td>
    </tr>
  `).join('') : '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:18px">No confidence-bucket data yet.</td></tr>';
}

function _renderCalibrationGovernor(governor) {
  const summaryEl = document.getElementById('perf-calibration-governor-summary');
  const activeSummaryEl = document.getElementById('perf-active-calibration-summary');
  const panel = document.getElementById('perf-calibration-governor-panel');
  if (!summaryEl || !activeSummaryEl || !panel) return;

  const summary = (governor && governor.summary) || {};
  const rows = Array.isArray(governor && governor.rows) ? governor.rows : [];
  const activeStatus = (governor && governor.active_status) || {};
  const activeSummary = (activeStatus && activeStatus.summary) || {};
  const activeRows = Array.isArray(activeStatus && activeStatus.rows) ? activeStatus.rows : [];
  if (activeRows.length) {
    activeSummaryEl.innerHTML = `Active calibration: <strong>${activeSummary.calibrated || 0}</strong> calibrated · <strong>${activeSummary.uncalibrated || 0}</strong> uncalibrated · ` +
      activeRows.map(row => {
        const tone = row.has_calibrator ? 'production' : 'review';
        const tag = row.active_tag ? escapeHtml(row.active_tag) : '—';
        return `<span class="launch-support-tag ${tone}" style="margin-right:6px">${escapeHtml(row.sport)} ${tag}</span>`;
      }).join('');
  } else {
    activeSummaryEl.innerHTML = '';
  }
  if (!rows.length) {
    summaryEl.innerHTML = '<span style="color:var(--text3)">No sport-level calibration advisories yet.</span>';
    panel.innerHTML = '';
    return;
  }

  const worstBucket = summary.worst_bucket
    ? ` · worst bucket <strong>${escapeHtml(summary.worst_bucket)}</strong>${summary.worst_bucket_gap_pp != null ? ` (${summary.worst_bucket_gap_pp}pp gap)` : ''}`
    : '';
  summaryEl.innerHTML = `Flagged sports: <strong>${summary.sports_flagged || 0}</strong> · critical <strong>${summary.critical || 0}</strong> · moderate <strong>${summary.moderate || 0}</strong> · watch <strong>${summary.watch || 0}</strong>${worstBucket}`;

  panel.innerHTML = rows.map(row => {
    const tone = row.severity === 'critical'
      ? 'review'
      : row.severity === 'moderate'
        ? 'limited'
        : 'production';
    return `
      <div class="launch-support-item" style="margin-bottom:10px;align-items:flex-start">
        <span class="launch-support-tag ${tone}" style="min-width:120px;text-align:center">${escapeHtml(row.severity)}</span>
        <div style="width:100%">
          <div style="font-weight:700;color:var(--text);margin-bottom:4px">${SPORT_ICONS[row.sport] || ''} ${escapeHtml(row.sport)} · ${escapeHtml(row.action)}</div>
          <div style="color:var(--text2);font-size:.82rem">${escapeHtml(row.reason)}</div>
          <div style="color:var(--text3);font-size:.75rem;margin-top:4px">
            Bets ${row.bets} · Bias ${escapeHtml(row.bias)} · Avg prob ${row.avg_prob_pct}% · Win rate ${row.win_rate_pct}% · Gap ${row.gap_pp}pp · Brier ${row.brier} · Log loss ${row.log_loss}
          </div>
          <div style="margin-top:6px;color:var(--text2);font-size:.8rem"><strong>Next step:</strong> ${escapeHtml(row.next_step || 'Review this sport before expanding live exposure.')}</div>
        </div>
      </div>`;
  }).join('');
}

function _renderPerformanceMine(myD) {
  // Convert my-selections format to the same shape renderPerformance expects
  const selections = myD.selections || [];
  const resolved   = selections.filter(s => s.result === 'won' || s.result === 'lost');
  if (!resolved.length) {
    document.getElementById('perf-sport-tbody').innerHTML =
      '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:24px">No settled picks yet — track bets from the Picks page.</td></tr>';
    ['perf-total','perf-won-sub','perf-pnl','perf-roi','perf-clv'].forEach(id => {
      const el = document.getElementById(id); if (el) el.textContent = '—';
    });
    _renderPerfEvaluationSummary(null);
    _renderPerfLaneHighlights(null);
    _renderPerfOddsBuckets([]);
    _renderPerfCalibration(null);
    return;
  }
  const n    = resolved.length;
  const won  = resolved.filter(s => s.result === 'won').length;
  // Use actual odds-based profit; my-selections stores profit per bet
  const pnl  = resolved.reduce((s, b) => s + (b.profit ?? 0), 0);
  // Stake = sum of implied 1-unit stakes (we track 1u per bet)
  const roi  = n > 0 ? pnl / n : 0;
  const recentResolved = resolved.slice(-30);
  const recentWon = recentResolved.filter(s => s.result === 'won').length;
  const recentPnl = recentResolved.reduce((s, b) => s + (b.profit ?? 0), 0);
  const recentRoi = recentResolved.length > 0 ? recentPnl / recentResolved.length : 0;

  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set('perf-total',   n);
  set('perf-won-sub', `${won} won (${Math.round(won/n*100)}%)`);
  set('perf-pnl',     (pnl >= 0 ? '+' : '') + pnl.toFixed(2));
  set('perf-roi',     (roi >= 0 ? '+' : '') + (roi*100).toFixed(1) + '%');
  set('perf-clv',     '—'); // CLV not stored on my-selections
  _renderPerfEvaluationSummary({
    all_bets: { bets: n, win_rate: Math.round(won/n*100), pnl: Number(pnl.toFixed(2)), roi: Number((roi*100).toFixed(1)) },
    singles_only: { bets: n, win_rate: Math.round(won/n*100), pnl: Number(pnl.toFixed(2)), roi: Number((roi*100).toFixed(1)) },
    recent_singles: {
      bets: recentResolved.length,
      win_rate: recentResolved.length ? Math.round(recentWon / recentResolved.length * 100) : 0,
      pnl: Number(recentPnl.toFixed(2)),
      roi: Number((recentRoi * 100).toFixed(1)),
    },
  });
  const oddsBuckets = {};
  resolved.forEach(b => {
    const odds = Number(b.odds || b.bet_odds || 0);
    const bucket = odds <= 1.67 ? '≤1.67' : odds <= 2.19 ? '1.68–2.19' : odds <= 3.49 ? '2.20–3.49' : '3.50+';
    if (!oddsBuckets[bucket]) oddsBuckets[bucket] = { bucket, bets: 0, wins: 0, pnl: 0 };
    oddsBuckets[bucket].bets += 1;
    oddsBuckets[bucket].wins += b.result === 'won' ? 1 : 0;
    oddsBuckets[bucket].pnl += (b.profit ?? 0);
  });
  _renderPerfOddsBuckets(['≤1.67', '1.68–2.19', '2.20–3.49', '3.50+']
    .filter(bucket => oddsBuckets[bucket])
    .map(bucket => {
      const row = oddsBuckets[bucket];
      return {
        bucket,
        bets: row.bets,
        win_rate: row.bets ? Math.round(row.wins / row.bets * 100) : 0,
        roi: row.bets ? Number((row.pnl / row.bets * 100).toFixed(1)) : 0,
        pnl: Number(row.pnl.toFixed(2)),
      };
    }));
  _renderPerfCalibration(null);

  const pCard = document.getElementById('perf-pnl-card');
  if (pCard) pCard.className = 'stat-card ' + (pnl >= 0 ? 'green' : 'red');
  const rCard = document.getElementById('perf-roi-card');
  if (rCard) rCard.className = 'stat-card ' + (roi >= 0 ? 'green' : 'red');

  // Sport breakdown table
  const byS = {};
  resolved.forEach(b => {
    const sp = b.sport || 'unknown';
    if (!byS[sp]) byS[sp] = {n:0, won:0, pnl:0};
    byS[sp].n++; byS[sp].won += b.result === 'won' ? 1 : 0;
    byS[sp].pnl += (b.profit ?? 0);
  });
  const laneRows = Object.entries(byS).map(([sport, stats]) => ({
    sport,
    market: 'all',
    bets: stats.n,
    win_rate: stats.n ? Math.round(stats.won / stats.n * 100) : 0,
    pnl: Number(stats.pnl.toFixed(2)),
    roi: stats.n ? Number((stats.pnl / stats.n * 100).toFixed(1)) : 0,
  }));
  _renderPerfLaneHighlights({
    best: [...laneRows].sort((a, b) => (b.roi - a.roi) || (b.pnl - a.pnl)).slice(0, 5),
    worst: [...laneRows].sort((a, b) => (a.roi - b.roi) || (a.pnl - b.pnl)).slice(0, 5),
  });
  const tbody = document.getElementById('perf-sport-tbody');
  if (tbody) {
    tbody.innerHTML = Object.entries(byS).sort((a,b)=>b[1].pnl-a[1].pnl).map(([sp,s])=>{
      const wr  = s.n > 0 ? (s.won/s.n*100).toFixed(0) : '—';
      const roi = s.n > 0 ? (s.pnl/s.n*100).toFixed(1) : '—';
      const col = s.pnl >= 0 ? 'var(--green)' : 'var(--red)';
      return `<tr><td>${escapeHtml(sp)}</td><td>${s.n}</td><td>${s.won}</td>
        <td>${wr}%</td><td>${s.n}</td>
        <td style="color:${col}">${s.pnl>=0?'+':''}${s.pnl.toFixed(2)}</td>
        <td style="color:${col}">${roi}%</td><td>—</td></tr>`;
    }).join('');
  }

  // Charts — convert to the format renderPerformance charts expect
  const chartBets = resolved.map(s => ({
    profit_units:  s.profit ?? 0,
    won:           s.result === 'won',
    sport:         s.sport,
    stake_units:   1,
    commence_time: s.commence || s.date || '',
    settled_at:    s.date || '',
    clv:           null,
    market:        'moneyline',
  }));
  _renderPerfCumulativeChart(chartBets);
  _renderPerfDrawdownChart(chartBets);
  _renderPerfSportChart(byS);
  _renderPerfDailyChart(chartBets);
}

function renderPerformance(d) {
  const settled = d.settled || [];
  const resolved = settled.filter(b => b.status === 'won' || b.status === 'lost');
  _renderPerfEvaluationSummary(d.evaluation_summary);
  _renderPerfLaneHighlights(d.lane_highlights);
  _renderPerfOddsBuckets(d.odds_buckets);
  _renderPerfCalibration(d.calibration_snapshot);
  const governor = d.calibration_governor || {};
  governor.active_status = d.active_calibration_status || {};
  _renderCalibrationGovernor(governor);
  if (!resolved.length) {
    document.getElementById('perf-sport-tbody').innerHTML =
      '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:24px">No settled bets yet.</td></tr>';
    return;
  }

  // KPIs
  const n     = resolved.length;
  const won   = resolved.filter(b => b.won).length;
  const stake = resolved.reduce((s, b) => s + (b.stake_units || 0), 0);
  const pnl   = resolved.reduce((s, b) => s + (b.profit_units || 0), 0);
  const roi   = stake > 0 ? pnl / stake : 0;
  const clvArr = resolved.filter(b => b.clv != null).map(b => b.clv);
  const avgClv = clvArr.length ? clvArr.reduce((s, v) => s + v, 0) / clvArr.length : null;

  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('perf-total', n);
  set('perf-won-sub', won + ' won (' + Math.round(won/n*100) + '%)');
  set('perf-pnl', (pnl >= 0 ? '+' : '') + pnl.toFixed(2));
  set('perf-roi', (roi >= 0 ? '+' : '') + (roi*100).toFixed(1) + '%');
  set('perf-clv', avgClv != null ? (avgClv >= 0 ? '+' : '') + (avgClv*100).toFixed(2) + '%' : '—');

  // Colour KPI cards
  const pCard = document.getElementById('perf-pnl-card');
  if (pCard) pCard.className = 'stat-card ' + (pnl >= 0 ? 'green' : 'red');
  const rCard = document.getElementById('perf-roi-card');
  if (rCard) rCard.className = 'stat-card ' + (roi >= 0 ? 'green' : 'red');
  const cCard = document.getElementById('perf-clv-card');
  if (cCard && avgClv != null) cCard.className = 'stat-card ' + (avgClv >= 0 ? 'green' : 'yellow');

  // Sport breakdown table
  const byS = {};
  resolved.forEach(b => {
    const sp = b.sport || 'unknown';
    if (!byS[sp]) byS[sp] = {n:0, won:0, stake:0, pnl:0, clv:[]};
    byS[sp].n++; byS[sp].won += b.won ? 1 : 0;
    byS[sp].stake += (b.stake_units||0); byS[sp].pnl += (b.profit_units||0);
    if (b.clv != null) byS[sp].clv.push(b.clv);
  });
  const tbody = document.getElementById('perf-sport-tbody');
  if (tbody) {
    tbody.innerHTML = Object.entries(byS)
      .sort((a,b) => b[1].pnl - a[1].pnl)
      .map(([sp, s]) => {
        const wr  = s.n > 0 ? (s.won/s.n*100).toFixed(0) : '—';
        const roi = s.stake > 0 ? (s.pnl/s.stake*100).toFixed(1) : '—';
        const ac  = s.clv.length ? (s.clv.reduce((a,v)=>a+v,0)/s.clv.length*100).toFixed(2) : '—';
        const col = s.pnl >= 0 ? 'var(--green)' : 'var(--red)';
        return `<tr>
          <td>${escapeHtml(sp)}</td><td>${s.n}</td><td>${s.won}</td>
          <td>${wr}%</td><td>${s.stake.toFixed(2)}</td>
          <td style="color:${col}">${s.pnl >= 0 ? '+' : ''}${s.pnl.toFixed(2)}</td>
          <td style="color:${col}">${roi}%</td><td>${ac}%</td>
        </tr>`;
      }).join('');
  }

  // Charts
  _renderPerfCumulativeChart(resolved);
  _renderPerfDrawdownChart(resolved);
  _renderPerfSportChart(byS);
  _renderPerfMarketChart(resolved);
  _renderPerfDailyChart(resolved);
}

function _destroyChart(id) {
  if (_perfCharts[id]) { _perfCharts[id].destroy(); delete _perfCharts[id]; }
}

function _renderPerfCumulativeChart(resolved) {
  const ctx = document.getElementById('perf-cumulative-chart');
  if (!ctx) return;
  _destroyChart('cumulative');
  const sorted = [...resolved].sort((a,b) => new Date(a.commence_time||a.settled_at) - new Date(b.commence_time||b.settled_at));
  let cum = 0;
  const labels = sorted.map(b => (b.commence_time || b.settled_at || '').slice(5,10));
  const data   = sorted.map(b => { cum += (b.profit_units||0); return +cum.toFixed(3); });
  _perfCharts['cumulative'] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ data, borderColor: cum >= 0 ? 'rgba(16,185,129,.9)' : 'rgba(239,68,68,.8)',
      backgroundColor: cum >= 0 ? 'rgba(16,185,129,.08)' : 'rgba(239,68,68,.08)',
      fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 }] },
    options: { responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:'#64748b',maxTicksLimit:10,font:{size:10}}, grid:{color:'rgba(255,255,255,.04)'}},
               y:{ticks:{color:'#64748b',font:{size:10},callback:v=>(v>=0?'+':'')+v.toFixed(1)+'u'}, grid:{color:'rgba(255,255,255,.06)'}}}}
  });
}

function _renderPerfDrawdownChart(resolved) {
  const ctx = document.getElementById('perf-drawdown-chart');
  if (!ctx) return;
  _destroyChart('drawdown');
  const sorted = [...resolved].sort((a,b) => new Date(a.commence_time||a.settled_at) - new Date(b.commence_time||b.settled_at));
  let cum = 0, peak = 0;
  const labels = sorted.map(b => (b.commence_time||b.settled_at||'').slice(5,10));
  const data = sorted.map(b => {
    cum += (b.profit_units||0); peak = Math.max(peak, cum);
    return +(peak > 0 ? (cum - peak) / peak * 100 : 0).toFixed(2);
  });
  _perfCharts['drawdown'] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ data, borderColor:'rgba(239,68,68,.8)', backgroundColor:'rgba(239,68,68,.1)',
      fill:true, tension:0.3, pointRadius:0, borderWidth:2 }] },
    options: { responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:'#64748b',maxTicksLimit:10,font:{size:10}},grid:{color:'rgba(255,255,255,.04)'}},
               y:{ticks:{color:'#64748b',font:{size:10},callback:v=>v.toFixed(0)+'%'},grid:{color:'rgba(255,255,255,.06)'}}}}
  });
}

function _renderPerfSportChart(byS) {
  const ctx = document.getElementById('perf-sport-chart');
  if (!ctx) return;
  _destroyChart('sport');
  const labels = Object.keys(byS);
  const rois   = labels.map(sp => byS[sp].stake > 0 ? +(byS[sp].pnl/byS[sp].stake*100).toFixed(1) : 0);
  _perfCharts['sport'] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label:'ROI %', data: rois,
      backgroundColor: rois.map(v => v >= 0 ? 'rgba(16,185,129,.55)' : 'rgba(239,68,68,.55)'),
      borderColor:     rois.map(v => v >= 0 ? 'rgba(16,185,129,.9)'  : 'rgba(239,68,68,.9)'),
      borderWidth:1, borderRadius:4 }] },
    options: { responsive:true, plugins:{legend:{display:false}},
      scales:{ x:{ticks:{color:'#64748b',font:{size:11}},grid:{color:'rgba(255,255,255,.04)'}},
               y:{ticks:{color:'#64748b',font:{size:11},callback:v=>v+'%'},grid:{color:'rgba(255,255,255,.06)'}}}}
  });
}

function _renderPerfMarketChart(resolved) {
  const ctx = document.getElementById('perf-market-chart');
  if (!ctx) return;
  _destroyChart('market');
  const byM = {};
  resolved.forEach(b => {
    const m = b.market || 'moneyline';
    if (!byM[m]) byM[m] = {n:0, won:0};
    byM[m].n++; byM[m].won += b.won ? 1 : 0;
  });
  const labels = Object.keys(byM);
  const rates  = labels.map(m => byM[m].n > 0 ? +(byM[m].won/byM[m].n*100).toFixed(1) : 0);
  _perfCharts['market'] = new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data: rates,
      backgroundColor: ['rgba(59,130,246,.7)','rgba(16,185,129,.7)','rgba(245,158,11,.7)','rgba(139,92,246,.7)'],
      borderColor: 'var(--bg2)', borderWidth: 2 }] },
    options: { responsive:true, cutout:'60%',
      plugins:{ legend:{ position:'bottom', labels:{color:'#94a3b8',font:{size:11}}},
        tooltip:{ callbacks:{ label: c => `${c.label}: ${c.parsed.toFixed(1)}% win rate`}}}}
  });
}

function _renderPerfDailyChart(resolved) {
  const ctx = document.getElementById('perf-daily-chart');
  if (!ctx) return;
  _destroyChart('daily');
  const byDay = {};
  resolved.forEach(b => {
    const day = (b.commence_time||b.settled_at||'').slice(0,10);
    if (!byDay[day]) byDay[day] = 0;
    byDay[day] += (b.profit_units||0);
  });
  const entries = Object.entries(byDay).sort().slice(-30);
  const labels = entries.map(([d]) => d.slice(5));
  const values = entries.map(([,v]) => +v.toFixed(3));
  _perfCharts['daily'] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ data: values,
      backgroundColor: values.map(v => v >= 0 ? 'rgba(16,185,129,.55)' : 'rgba(239,68,68,.55)'),
      borderColor:     values.map(v => v >= 0 ? 'rgba(16,185,129,.85)' : 'rgba(239,68,68,.85)'),
      borderWidth:1, borderRadius:3 }] },
    options: { responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>(c.parsed.y>=0?'+':'')+c.parsed.y.toFixed(2)+'u'}}},
      scales:{ x:{ticks:{color:'#64748b',maxTicksLimit:15,font:{size:10}},grid:{color:'rgba(255,255,255,.04)'}},
               y:{ticks:{color:'#64748b',font:{size:10},callback:v=>(v>=0?'+':'')+v.toFixed(1)+'u'},grid:{color:'rgba(255,255,255,.06)'}}}}
  });
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
async function init() {
  await loadDashboard();
  // Show dashboard first, then preload picks in background
  loadPicks();
  // Refresh dashboard every 60s
  setInterval(loadDashboard, 60000);
}

init();
