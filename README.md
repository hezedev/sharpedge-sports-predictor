# SharpEdge Sports Predictor

SharpEdge is a local-first sports betting research system that combines model predictions, odds movement, bankroll controls, and sport-aware evidence review before a pick is allowed through.

It is built for the uncomfortable middle ground where a model may see value, but the real world still matters: injuries, lineups, pitcher changes, rotation risk, fixture congestion, market fit, stale odds, and source quality.

## What It Does

- Scans upcoming markets across soccer, MLB, tennis, basketball, and NHL
- Pulls odds from The Odds API with quota-aware key selection
- Builds value-bet candidates from model probability, market price, and risk rules
- Runs a multi-mind committee over each candidate:
  - Model Mind checks pricing, edge, probability bands, and risk tier
  - Research Mind checks sport-critical evidence and source quality
  - Arbiter blocks bets when evidence is insufficient or conflicting
- Enriches soccer, MLB, tennis, basketball, and NHL context with sport-specific checks
- Publishes candidates into bet, review, suppressed, and parlay workflows
- Provides a Flask webapp for scanning, reviewing, and manually analyzing picks

## Why This Exists

Most prediction tools stop at "model says edge." SharpEdge is designed to ask the next questions:

- Is the lineup or injury picture known?
- Is the market stale, thin, or outside the model's reliable lane?
- Did the odds move against us?
- Is this sport's critical evidence missing?
- Should this be a bet, a wait, a manual review item, or a hard avoid?

The goal is not to produce the most picks. The goal is to keep weak picks from looking strong.

## Sport-Aware Research

SharpEdge does not use one generic checklist for every sport.

| Sport | Critical evidence checked |
| --- | --- |
| Soccer | Lineups, injuries, suspensions, motivation, rotation, standings, home/away form, fixture congestion |
| MLB | Probable pitchers, pitcher changes, bullpen workload, lineups, weather, park factors, handedness, travel/rest |
| Basketball | Injury reports, star-player status, projected lineups, rest/back-to-back, pace, usage, motivation |
| Tennis | Surface, injury/retirement risk, recent form, fatigue, H2H style matchup, tournament context |
| NHL | Starting goalie, injuries, rest/back-to-back, travel, special teams, shots/xG, playoff motivation |

If sport-critical evidence is missing, the system downgrades confidence or outputs `HOLD`, `WAIT`, or `NO_BET`. The Arbiter is not allowed to approve a bet from insufficient or conflicting research evidence.

## Webapp

The webapp is the easiest way to work with the system locally.

```bash
.venv/bin/python run_webapp.py
```

Then open:

```text
http://localhost:5000
```

The app supports:

- daily scan controls
- candidate review
- evidence summaries
- API key management helpers
- quota and scan status views
- manual analysis workflows
- parlay builder context

## Project Layout

```text
.
|-- daily_scan.py                  # Main scanner and report generator
|-- run_webapp.py                  # Flask webapp launcher
|-- main.py                        # CLI for fetch/train/predict/backtest workflows
|-- config/                        # Settings and betting rules
|-- src/
|   |-- analysis/                  # Manual analyst and fresh news context
|   |-- committee/                 # Model Mind, Research Mind, Arbiter, enrichment
|   |-- data/                      # Odds, sport APIs, fetchers, live data
|   |-- evaluation/                # Backtesting and metrics
|   |-- features/                  # Feature engineering
|   |-- markets/                   # Availability, freshness, suitability, policies
|   |-- models/                    # Training, calibration, prediction models
|   |-- risk/                      # Kelly, bankroll, value detection, parlays
|   `-- utils/                     # Cache, quota, tracking, helpers
|-- tests/                         # Pytest coverage
|-- webapp/                        # Flask app, templates, static assets
`-- docs/                          # Design and runtime notes
```

## Quick Start

Create a virtual environment:

```bash
python3 -m venv .venv
```

Install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Add your API keys to `.env`.

Run the webapp:

```bash
.venv/bin/python run_webapp.py
```

Run tests:

```bash
.venv/bin/python -m pytest
```

## Useful Commands

Run a focused daily scan:

```bash
.venv/bin/python daily_scan.py --sport all --focused-lanes
```

Run a soccer-only scan:

```bash
.venv/bin/python daily_scan.py --sport soccer
```

Run the main CLI pipeline:

```bash
.venv/bin/python main.py pipeline --sport soccer
```

Train models:

```bash
.venv/bin/python main.py train --sport soccer
```

Backtest:

```bash
.venv/bin/python main.py backtest --sport soccer
```

## Environment Variables

Copy `.env.example` to `.env` and fill in only the services you use.

| Variable | Purpose |
| --- | --- |
| `ODDS_API_KEY` | Primary The Odds API key |
| `ODDS_API_KEYS` | Optional comma-separated pool for fresh rescans |
| `FOOTBALL_DATA_API_KEY` | Soccer fixtures, results, and standings |
| `API_SPORTS_KEY` | Direct API-Sports access for football/basketball/tennis |
| `RAPIDAPI_KEY` | RapidAPI-backed API-Football access |
| `BALLDONTLIE_API_KEY` | Optional basketball data |
| `OPENWEATHER_API_KEY` | Optional weather risk checks |
| `TELEGRAM_TOKEN` | Optional alerting |
| `TELEGRAM_CHAT_ID` | Optional alert destination |

Never commit `.env`.

## Data And Model Artifacts

This repository intentionally does not commit local runtime artifacts:

- API keys and local `.env`
- virtual environments
- generated reports
- logs
- caches
- sqlite databases
- parquet tracker files
- trained model artifacts
- odds key pools

Those files can be large, private, or account-specific. Keep them local, rebuild them, or publish them through a separate artifact store if you need reproducible deployments.

## Evidence And Safety Rules

SharpEdge is intentionally conservative around uncertain information:

- A model edge is not enough by itself.
- Missing lineup, injury, pitcher, goalie, weather, or motivation context can block a pick.
- Stale odds or stale research can force manual review.
- Conflicting evidence should not be resolved by optimism.
- Suppressed candidates are part of the safety system, not a failure mode.

This project is for research and decision support. It does not guarantee profitable betting outcomes.

## Current Status

The project is a local working system with active modules for scanning, model prediction, evidence enrichment, risk handling, web review, and tests. Some workflows depend on local data/model artifacts that are intentionally ignored by git.

## Roadmap

- Improve public setup path for rebuilding feature caches and model artifacts
- Add a first-run bootstrap command
- Add optional Docker Compose profile for the webapp
- Add screenshots or demo GIFs for the review dashboard
- Add CI once lightweight fixture data is separated from local runtime state

## License

No license has been selected yet. Add one before accepting outside contributions.
