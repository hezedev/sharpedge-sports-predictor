from src.risk.parlay_builder import ParlayBuilder, ParlayLeg


def _leg(match_id: str, team: str, odds: float, ml_prob: float, fair_prob: float, sport: str = "soccer"):
    return ParlayLeg(
        sport=sport,
        match_id=match_id,
        team=team,
        odds=odds,
        ml_prob=ml_prob,
        fair_prob=fair_prob,
        edge=ml_prob - fair_prob,
        commence="2026-04-29T18:00:00Z",
    )


def test_value_parlays_prefer_cleaner_shorter_legs():
    builder = ParlayBuilder(min_legs=2, max_legs=2, top_n=1)
    candidates = [
        _leg("A vs B", "A", odds=2.05, ml_prob=0.62, fair_prob=0.55),
        _leg("C vs D", "C", odds=2.05, ml_prob=0.60, fair_prob=0.53),
        _leg("E vs F", "E", odds=3.40, ml_prob=0.55, fair_prob=0.45),
    ]

    results = builder.build(candidates)
    top = results["value"]["5x"][0]
    teams = {leg.team for leg in top.legs}

    assert teams == {"A", "C"}


def test_longshot_parlays_require_a_true_upside_leg():
    builder = ParlayBuilder(min_legs=2, max_legs=4, top_n=1)
    candidates = [
        _leg("A vs B", "A", odds=2.40, ml_prob=0.58, fair_prob=0.51),
        _leg("C vs D", "C", odds=2.65, ml_prob=0.55, fair_prob=0.49),
        _leg("E vs F", "E", odds=3.35, ml_prob=0.43, fair_prob=0.40),
    ]

    results = builder.build(candidates)
    top = results["speculative"]["20x"][0]
    teams = {leg.team for leg in top.legs}

    assert "E" in teams
    assert top.combined_odds >= 20.0
    assert top.target_bracket == "20x"


def test_format_report_uses_longshot_label():
    builder = ParlayBuilder(min_legs=2, max_legs=4, top_n=1)
    candidates = [
        _leg("A vs B", "A", odds=1.85, ml_prob=0.62, fair_prob=0.55),
        _leg("C vs D", "C", odds=1.95, ml_prob=0.60, fair_prob=0.53),
        _leg("E vs F", "E", odds=2.60, ml_prob=0.46, fair_prob=0.44),
        _leg("G vs H", "G", odds=3.20, ml_prob=0.43, fair_prob=0.41),
    ]

    report = builder.format_report(builder.build(candidates))

    assert "Longshot Parlays" in report


def test_assess_legs_rejects_duplicate_game_and_conflict() -> None:
    builder = ParlayBuilder(min_legs=2, max_legs=3, top_n=1)
    legs = [
        _leg("A vs B", "A", odds=2.0, ml_prob=0.6, fair_prob=0.53),
        _leg("A vs B", "B", odds=2.1, ml_prob=0.45, fair_prob=0.43),
    ]

    assessment = builder.assess_legs(legs, tier="value")

    assert assessment["build_verdict"] == "DO NOT BUILD"
    assert assessment["duplicate_games"] == ["A vs B"]
    assert assessment["conflicting_picks"] == ["A vs B"]


def test_parlay_above_five_legs_cannot_be_conservative() -> None:
    builder = ParlayBuilder(min_legs=6, max_legs=6, top_n=1)
    assessment = builder.assess_legs(
        [
            _leg("A vs B", "A", odds=1.75, ml_prob=0.63, fair_prob=0.56),
            _leg("C vs D", "C", odds=1.8, ml_prob=0.61, fair_prob=0.54),
            _leg("E vs F", "E", odds=1.78, ml_prob=0.6, fair_prob=0.53),
            _leg("G vs H", "G", odds=1.82, ml_prob=0.59, fair_prob=0.52),
            _leg("I vs J", "I", odds=1.76, ml_prob=0.6, fair_prob=0.53),
            _leg("K vs L", "K", odds=1.79, ml_prob=0.58, fair_prob=0.52),
        ],
        tier="value",
    )

    assert assessment["build_verdict"] == "BUILD"
    assert assessment["risk_tier"] in {"high-risk", "speculative"}
    assert any("above 5 legs" in note for note in assessment["notes"])


def test_parlay_tracks_combined_probability_weakest_leg_and_correlation() -> None:
    builder = ParlayBuilder(min_legs=3, max_legs=3, top_n=1)
    legs = [
        _leg("A vs B", "A", odds=1.9, ml_prob=0.62, fair_prob=0.55, sport="soccer"),
        _leg("C vs D", "C", odds=1.95, ml_prob=0.59, fair_prob=0.53, sport="soccer"),
        _leg("E vs F", "E", odds=2.05, ml_prob=0.54, fair_prob=0.49, sport="soccer"),
    ]
    for leg in legs:
        leg.market = "moneyline"

    parlay = builder._create_parlay(legs, tier="value", target_bracket="5x")

    assert round(parlay.combined_prob, 6) == round(0.62 * 0.59 * 0.54, 6)
    assert parlay.weakest_leg is not None
    assert parlay.weakest_leg["team"] == "E"
    assert parlay.risk_tier in {"medium-risk", "high-risk"}
    assert parlay.correlated_pick_groups
