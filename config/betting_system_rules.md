Never force a pick. NO BET is a valid decision.

Never call a pick safe, locked, banker, guaranteed, or free money.

Every pick must pass:
- fixture verification
- match status check
- odds freshness check
- injury/news freshness check
- lineup/rotation check
- motivation check
- market suitability check
- probability vs vig-free market check
- minimum edge threshold
- confidence interval lower-bound check

Rejecting one side does not automatically mean backing the other side.

If the correct market is unclear, output HOLD or NO BET.

A conservative parlay must have 3-5 legs maximum.

Any parlay above 5 legs must be labelled medium-risk, high-risk, or speculative.

Do not include medium-risk picks as conservative parlay anchors.

Do not include duplicate, contradictory, or heavily correlated picks unless explicitly building a same-game parlay.

For every losing pick, classify the mistake:
- normal variance
- overconfidence
- motivation error
- rotation error
- lineup/injury error
- stale data
- market-selection error
- wrong-conversion error
- favourite trap
- underdog-resistance error
- parlay-construction error
- odds/value error

Runtime notes:
- Decision, parlay, and post-result runtime layers are documented in `/Users/adebara/Documents/sports_predictor/docs/runtime_betting_layers.md`.
- Tunable runtime thresholds should live in `/Users/adebara/Documents/sports_predictor/config/settings.yaml` under the `betting` section where possible, instead of being hardcoded directly into scan or webapp flows.
