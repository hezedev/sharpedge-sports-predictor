# Sports Predictor

Sports Predictor is a local betting research and prediction workspace with:

- daily multi-sport scans
- odds ingestion and value-bet reports
- model, calibration, and risk modules
- sport-aware committee review for evidence quality
- a Flask webapp for manual analysis and scan review

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your local API keys. Do not commit `.env`.

## Run The Webapp

```bash
.venv/bin/python run_webapp.py
```

## Run Tests

```bash
.venv/bin/python -m pytest
```

## Notes

Generated data, model artifacts, reports, logs, caches, and local tracker state are intentionally ignored by git. Keep those local or publish them through a separate artifact store if needed.
