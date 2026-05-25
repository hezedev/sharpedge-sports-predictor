# SharpEdge Sports Predictor

SharpEdge is a local-first sports analytics workspace for building, testing, and reviewing model-assisted sports predictions.

The project brings together data ingestion, feature engineering, model calibration, risk controls, and a Flask review dashboard in one place. It is designed for research workflows where predictions should be explainable, auditable, and easy to review before any real-world decision is made.

## Highlights

- Multi-sport prediction research across soccer, MLB, tennis, basketball, and NHL
- Odds ingestion and quota-aware API usage
- Feature engineering and model calibration modules
- Evidence-aware review layer for contextual checks
- Bankroll and risk-management utilities
- Local Flask webapp for scans, review, and manual analysis
- Pytest coverage for core workflows

## Web Dashboard

Run the local webapp:

```bash
.venv/bin/python run_webapp.py
```

Open:

```text
http://localhost:5000
```

The dashboard is intended for local review and research workflows, including scan controls, candidate review, API helper tools, status views, and manual analysis screens.

## Project Structure

```text
.
|-- daily_scan.py          # Scan orchestration and report generation
|-- main.py                # CLI for pipeline tasks
|-- run_webapp.py          # Webapp entrypoint
|-- config/                # Project settings
|-- docs/                  # Design and runtime notes
|-- src/                   # Core application modules
|-- tests/                 # Test suite
`-- webapp/                # Flask app, templates, and static assets
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

Create a local environment file:

```bash
cp .env.example .env
```

Add the API keys you plan to use. The application can run different workflows depending on which services are configured.

Start the webapp:

```bash
.venv/bin/python run_webapp.py
```

Run tests:

```bash
.venv/bin/python -m pytest
```

## Common Commands

Run a focused scan:

```bash
.venv/bin/python daily_scan.py --sport all --focused-lanes
```

Run a sport-specific scan:

```bash
.venv/bin/python daily_scan.py --sport soccer
```

Run the CLI pipeline:

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

## Configuration

Use `.env.example` as the template for local configuration.

Common optional variables include:

| Variable | Purpose |
| --- | --- |
| `ODDS_API_KEY` | Odds data access |
| `ODDS_API_KEYS` | Optional key pool for rescans |
| `FOOTBALL_DATA_API_KEY` | Soccer fixture and standings data |
| `API_SPORTS_KEY` | Sports data provider access |
| `RAPIDAPI_KEY` | RapidAPI provider access |
| `BALLDONTLIE_API_KEY` | Basketball data provider access |
| `OPENWEATHER_API_KEY` | Weather context |
| `TELEGRAM_TOKEN` | Optional notifications |
| `TELEGRAM_CHAT_ID` | Optional notification destination |

Never commit `.env` or real API keys.

## What Is Not Included

This repository intentionally excludes local runtime artifacts:

- `.env` files with real credentials
- virtual environments
- generated reports
- logs
- caches
- sqlite databases
- parquet tracker files
- trained model binaries
- local API quota/key-pool state

Those files are private, machine-specific, or too large for normal source control. Keep them local or manage them through a separate artifact store.

## Development Notes

The codebase is organized around clear module boundaries:

- `src/data` handles external data sources and fetchers.
- `src/features` builds model-ready features.
- `src/models` handles training, calibration, and prediction helpers.
- `src/markets` contains market context and decision-support utilities.
- `src/committee` contains the review and explanation layer.
- `src/risk` contains staking and bankroll tools.
- `webapp` provides the local review interface.

## Responsible Use

SharpEdge is a research and decision-support tool. It does not guarantee outcomes and should not be treated as financial advice. Any real-world use should include independent judgment, careful risk limits, and compliance with local laws.

## Status

This is an active local project. Some workflows depend on private runtime data or generated model artifacts that are intentionally not committed to this repository.

## Roadmap

- Add a public demo dataset
- Add first-run bootstrap tooling
- Add CI with lightweight fixtures
- Add dashboard screenshots
- Improve artifact export/import workflow

## License

No license has been selected yet. Add one before accepting outside contributions.
