# Runtime Betting Layers

This document describes the live runtime layers added on top of the base
prediction models. The goal is to improve betting discipline without breaking
existing scan commands, CLI flows, or web/API routes.

## Final Decision Layer

Every candidate now resolves to one explicit decision:

- `BET`
- `NO BET`
- `HOLD`
- `WAIT FOR LINEUPS`
- `AVOID`

The decision layer is implemented in:

- `/Users/adebara/Documents/sports_predictor/src/markets/decision_layer.py`
- `/Users/adebara/Documents/sports_predictor/daily_scan.py`

This layer exists so the bot does not force picks.

## Probability And Edge Checks

Published bets now carry:

- model probability
- market implied probability
- vig-free implied probability
- fair odds
- minimum acceptable odds
- edge
- confidence range
- lower-bound pass/fail

The lower-bound check is used to stop thin or overconfident picks from being
presented as valid bets.

## Freshness And Match-State Checks

Before publication, candidates are audited for:

- match status (`pre-match`, `live`, `finished`)
- fixture verification
- odds freshness
- lineup freshness
- injury/news freshness
- standings freshness

This logic is implemented in:

- `/Users/adebara/Documents/sports_predictor/src/markets/freshness.py`
- `/Users/adebara/Documents/sports_predictor/daily_scan.py`

## Market Suitability

The system no longer treats “which team do we like?” as the only question.
It now checks whether the chosen market fits the match:

- moneyline
- double chance
- draw no bet
- handicap / spreads
- totals
- BTTS / team-goal style alternatives
- no bet

Current suitability logic is implemented in:

- `/Users/adebara/Documents/sports_predictor/src/markets/suitability.py`

## Motivation / Rotation / End-Of-Season Layer

The runtime adjustment layer now adds bounded context for:

- relegation motivation
- title motivation
- playoff motivation
- teams with little to play for
- final-day volatility
- fixture congestion
- cup / European rotation risk
- missing or unconfirmed lineups

The adjustments stay intentionally small and modular:

- `/Users/adebara/Documents/sports_predictor/src/markets/adjustments.py`
- `/Users/adebara/Documents/sports_predictor/daily_scan.py`

## Parlay Controls

Parlays now enforce:

- conservative parlays: `3–5` legs only
- anything above `5` legs cannot be labelled conservative
- duplicate-game detection
- conflicting-pick detection
- correlation warnings
- combined probability output
- weakest-leg identification
- risk tier output
- `DO NOT BUILD` verdict support

Primary implementation:

- `/Users/adebara/Documents/sports_predictor/src/risk/parlay_builder.py`
- `/Users/adebara/Documents/sports_predictor/webapp/app.py`

## Post-Result Learning

Settlement now stores and exposes:

- result status (`won`, `lost`, `void`)
- closing odds
- closing-line value
- mistake classification
- daily mistake report
- weekly mistake report

Mistake categories currently supported:

- `normal variance`
- `overconfidence error`
- `motivation error`
- `rotation error`
- `lineup/injury error`
- `stale-data error`
- `market-selection error`
- `wrong-conversion error`
- `favourite-trap error`
- `underdog-resistance error`
- `parlay-construction error`
- `odds/value error`

Primary implementation:

- `/Users/adebara/Documents/sports_predictor/src/utils/results_tracker.py`
- `/Users/adebara/Documents/sports_predictor/settle.py`
- `/Users/adebara/Documents/sports_predictor/webapp/app.py`

## Configurable Runtime Thresholds

The main runtime knobs added in Phase 8 live in:

- `/Users/adebara/Documents/sports_predictor/config/settings.yaml`

Current config groups:

- `betting.parlay`
- `betting.post_result`
- `betting.adjustments.soccer`

These settings are designed to make the newer runtime rules tunable without
breaking existing interfaces.
