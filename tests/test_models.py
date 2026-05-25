"""
Tests for model training, prediction, and risk management modules.

Uses synthetic data to test the full pipeline without API dependencies.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.trainer import ModelTrainer
from src.models.calibration import ProbabilityCalibrator
from src.models.basketball_side_model import BasketballSideModel
from src.models.mlb_side_model import MLBSideModel
from src.models.nhl_side_model import NHLSideModel
from src.models.soccer_score_model import SoccerScoreModel
from src.risk.kelly import KellyCriterion
from src.risk.bankroll import BankrollManager, Bet
from src.risk.value_detector import ValueDetector
from src.evaluation.metrics import MetricsCalculator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_training_data(
    n: int = 300,
    n_features: int = 10,
    n_classes: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic classification data."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, n_classes, size=n), name="target")
    return X, y


# ------------------------------------------------------------------
# ModelTrainer Tests
# ------------------------------------------------------------------


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_train_all_algorithms(self) -> None:
        """All configured algorithms should train successfully."""
        X, y = make_training_data(200, 8, 3)
        trainer = ModelTrainer(sport="soccer")
        models = trainer.train(X, y)

        assert len(models) > 0
        for name, model in models.items():
            assert hasattr(model, "predict")
            assert hasattr(model, "predict_proba")

    def test_cross_validation(self) -> None:
        """CV should return results for each algorithm."""
        X, y = make_training_data(200, 8, 3)
        trainer = ModelTrainer(sport="soccer")
        cv_results = trainer.cross_validate(X, y)

        assert len(cv_results) > 0
        for name, metrics in cv_results.items():
            assert "mean_accuracy" in metrics
            assert "mean_log_loss" in metrics
            assert metrics["mean_accuracy"] > 0

    def test_ensemble_predictions(self) -> None:
        """Ensemble should produce valid probability predictions."""
        X, y = make_training_data(300, 8, 3)
        X_train, X_test = X[:240], X[240:]
        y_train, y_test = y[:240], y[240:]

        trainer = ModelTrainer(sport="soccer")
        trainer.train(X_train, y_train)
        ensemble = trainer.build_ensemble(X_train, y_train)

        proba = ensemble.predict_proba(X_test)
        assert proba.shape == (60, 3)
        # Probabilities should sum to ~1
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_evaluate(self) -> None:
        """Evaluation should return accuracy and log_loss."""
        X, y = make_training_data(200, 8, 3)
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]

        trainer = ModelTrainer(sport="soccer")
        trainer.train(X_train, y_train)
        results = trainer.evaluate(X_test, y_test)

        for name, metrics in results.items():
            assert "accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1


# ------------------------------------------------------------------
# Calibration Tests
# ------------------------------------------------------------------


class TestCalibration:
    """Test suite for ProbabilityCalibrator."""

    def test_isotonic_calibration(self) -> None:
        """Isotonic calibration should produce valid probabilities."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 3, size=n)
        y_proba = np.random.dirichlet(alpha=[1, 1, 1], size=n)

        cal = ProbabilityCalibrator(method="isotonic", n_classes=3)
        cal.fit(y_true, y_proba)
        calibrated = cal.calibrate(y_proba)

        assert calibrated.shape == y_proba.shape
        row_sums = calibrated.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)


class TestSoccerScoreModel:
    def test_structural_probs_use_direct_prob_columns(self) -> None:
        model = SoccerScoreModel()
        snapshot = pd.Series(
            {
                "home_dc_win_prob": 0.48,
                "dc_draw_prob": 0.27,
                "away_dc_win_prob": 0.25,
            }
        )

        probs = model.structural_probs_from_snapshot(snapshot)

        assert probs is not None
        assert probs.as_tuple() == pytest.approx((0.48, 0.27, 0.25))

    def test_combine_with_classifier_rebalances_toward_structural_view(self) -> None:
        model = SoccerScoreModel()

        combined = model.combine_with_classifier((0.60, 0.18, 0.22), (0.46, 0.28, 0.26))

        assert sum(combined.as_tuple()) == pytest.approx(1.0)
        assert combined.home < 0.60
        assert combined.draw > 0.18

    def test_combine_with_classifier_diagnostics_can_shift_to_structural_override(self) -> None:
        model = SoccerScoreModel()

        diagnostics = model.combine_with_classifier_diagnostics(
            (0.72, 0.14, 0.14),
            (0.38, 0.30, 0.32),
        )

        assert diagnostics.regime == "structural_override"
        assert diagnostics.structural_weight > diagnostics.model_weight
        assert diagnostics.combined.home < diagnostics.classifier.home
        assert diagnostics.combined.draw > diagnostics.classifier.draw


class TestMLBSideModel:
    def test_structural_probs_use_existing_mlb_snapshot_features(self) -> None:
        model = MLBSideModel()
        snapshot = pd.Series(
            {
                "elo_win_prob": 0.61,
                "sp_era_diff": -0.8,
                "sp_whip_diff": -0.2,
                "sp_k9_diff": 1.5,
                "home_win_pct_10": 0.6,
                "away_win_pct_10": 0.4,
                "home_run_diff_10": 0.7,
                "away_run_diff_10": -0.2,
                "density_diff": 0.0,
            }
        )

        probs = model.structural_probs_from_snapshot(snapshot)

        assert probs is not None
        assert probs.home > 0.61
        assert sum(probs.as_tuple()) == pytest.approx(1.0)

    def test_combine_with_classifier_diagnostics_reduces_mlb_classifier_overconfidence(self) -> None:
        model = MLBSideModel()

        diagnostics = model.combine_with_classifier_diagnostics(
            (0.68, 0.32),
            (0.54, 0.46),
        )

        assert diagnostics.combined.home < diagnostics.classifier.home
        assert diagnostics.combined.away > diagnostics.classifier.away
        assert diagnostics.regime in {"balanced", "structural_override", "classifier_lean"}

    def test_structural_probs_use_schedule_density_as_bullpen_proxy(self) -> None:
        model = MLBSideModel()
        rested = pd.Series(
            {
                "elo_win_prob": 0.54,
                "home_games_L3D": 1,
                "away_games_L3D": 3,
                "home_b2b": 0,
                "away_b2b": 1,
            }
        )
        taxed_home = pd.Series(
            {
                "elo_win_prob": 0.54,
                "home_games_L3D": 3,
                "away_games_L3D": 1,
                "home_b2b": 1,
                "away_b2b": 0,
            }
        )

        rested_probs = model.structural_probs_from_snapshot(rested)
        taxed_probs = model.structural_probs_from_snapshot(taxed_home)

        assert rested_probs is not None
        assert taxed_probs is not None
        assert rested_probs.home > taxed_probs.home


class TestBasketballSideModel:
    def test_structural_probs_use_existing_basketball_snapshot_features(self) -> None:
        model = BasketballSideModel()
        snapshot = pd.Series(
            {
                "elo_win_prob": 0.57,
                "form_diff": 0.20,
                "rest_diff": 1.0,
                "away_travel_bucket": 2.0,
                "away_cross_country": 1.0,
                "away_crossed_2tz": 1.0,
            }
        )

        probs = model.structural_probs_from_snapshot(snapshot)

        assert probs is not None
        assert probs.home > 0.57
        assert sum(probs.as_tuple()) == pytest.approx(1.0)

    def test_combine_with_classifier_diagnostics_reduces_basketball_classifier_overconfidence(self) -> None:
        model = BasketballSideModel()

        diagnostics = model.combine_with_classifier_diagnostics(
            (0.69, 0.31),
            (0.55, 0.45),
        )

        assert diagnostics.combined.home < diagnostics.classifier.home
        assert diagnostics.combined.away > diagnostics.classifier.away
        assert diagnostics.regime in {"balanced", "structural_override", "classifier_lean"}


class TestNHLSideModel:
    def test_structural_probs_use_existing_nhl_snapshot_features(self) -> None:
        model = NHLSideModel()
        snapshot = pd.Series(
            {
                "elo_win_prob": 0.55,
                "home_xg_diff_10": 0.4,
                "away_xg_diff_10": -0.2,
                "home_xgf_pg_10": 3.0,
                "away_xgf_pg_10": 2.5,
                "home_pp_pct_10": 24.0,
                "away_pp_pct_10": 18.0,
                "home_pk_pct_10": 82.0,
                "away_pk_pct_10": 77.0,
                "home_rest_days": 2.0,
                "away_rest_days": 1.0,
                "away_travel_bucket": 2.0,
            }
        )

        probs = model.structural_probs_from_snapshot(snapshot)

        assert probs is not None
        assert probs.home > 0.55
        assert sum(probs.as_tuple()) == pytest.approx(1.0)

    def test_combine_with_classifier_diagnostics_reduces_nhl_classifier_overconfidence(self) -> None:
        model = NHLSideModel()

        diagnostics = model.combine_with_classifier_diagnostics(
            (0.67, 0.33),
            (0.54, 0.46),
        )

        assert diagnostics.combined.home < diagnostics.classifier.home
        assert diagnostics.combined.away > diagnostics.classifier.away
        assert diagnostics.regime in {"balanced", "structural_override", "classifier_lean"}

    def test_platt_calibration(self) -> None:
        """Platt scaling should produce valid probabilities."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, size=n)
        y_proba = np.random.dirichlet(alpha=[2, 2], size=n)

        cal = ProbabilityCalibrator(method="platt", n_classes=2)
        cal.fit(y_true, y_proba)
        calibrated = cal.calibrate(y_proba)

        assert calibrated.shape == (n, 2)
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()


# ------------------------------------------------------------------
# Kelly Criterion Tests
# ------------------------------------------------------------------


class TestKellyCriterion:
    """Test suite for KellyCriterion."""

    def test_positive_edge(self) -> None:
        """Positive edge should return a positive stake."""
        kelly = KellyCriterion(fraction=1.0, max_bet_pct=1.0, min_edge=0.0)
        stake = kelly.calculate(model_prob=0.60, decimal_odds=2.10)
        assert stake > 0

    def test_no_edge(self) -> None:
        """No edge should return zero stake."""
        kelly = KellyCriterion(fraction=0.25, max_bet_pct=0.05, min_edge=0.03)
        # Fair odds for 50% probability
        stake = kelly.calculate(model_prob=0.50, decimal_odds=2.00)
        assert stake == 0.0

    def test_fractional_kelly_reduces_stake(self) -> None:
        """Fractional Kelly should be less than full Kelly."""
        full = KellyCriterion(fraction=1.0, max_bet_pct=1.0, min_edge=0.0)
        quarter = KellyCriterion(fraction=0.25, max_bet_pct=1.0, min_edge=0.0)

        full_stake = full.calculate(0.60, 2.10)
        quarter_stake = quarter.calculate(0.60, 2.10)

        assert quarter_stake < full_stake
        assert abs(quarter_stake - full_stake * 0.25) < 0.001

    def test_max_bet_cap(self) -> None:
        """Stake should never exceed max_bet_pct."""
        kelly = KellyCriterion(fraction=1.0, max_bet_pct=0.05, min_edge=0.0)
        stake = kelly.calculate(model_prob=0.90, decimal_odds=5.00)
        assert stake <= 0.05

    def test_multiway_kelly(self) -> None:
        """Multi-way Kelly should only bet on one outcome."""
        kelly = KellyCriterion(fraction=0.25, max_bet_pct=0.10, min_edge=0.01)
        probs = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
        odds = {"home_win": 1.90, "draw": 3.50, "away_win": 5.00}

        stakes = kelly.calculate_multiway(probs, odds)
        positive = [k for k, v in stakes.items() if v > 0]
        assert len(positive) <= 1

    def test_expected_value(self) -> None:
        """EV should be positive when model prob > implied prob."""
        kelly = KellyCriterion()
        ev = kelly.expected_value(0.60, 2.10)  # implied = 0.476
        assert ev > 0

        ev_neg = kelly.expected_value(0.40, 2.10)  # below implied
        assert ev_neg < 0


# ------------------------------------------------------------------
# BankrollManager Tests
# ------------------------------------------------------------------


class TestBankrollManager:
    """Test suite for BankrollManager."""

    def test_place_and_settle_winning_bet(self) -> None:
        """Winning bet should increase bankroll."""
        from datetime import datetime

        bm = BankrollManager(initial_bankroll=1000.0)
        bet = Bet(
            bet_id="test1", sport="soccer", match_id="m1",
            outcome="home_win", stake=50.0, odds=2.00,
            model_prob=0.55, edge=0.05, kelly_fraction=0.05,
            timestamp=datetime.now(),
        )

        assert bm.place_bet(bet) is True
        assert bm.current_bankroll == 950.0  # 1000 - 50

        pnl = bm.settle_bet("test1", "won")
        assert pnl == 50.0  # 50 * 2.0 - 50
        assert bm.current_bankroll == 1050.0

    def test_place_and_settle_losing_bet(self) -> None:
        """Losing bet should reduce bankroll."""
        from datetime import datetime

        bm = BankrollManager(initial_bankroll=1000.0)
        bet = Bet(
            bet_id="test2", sport="soccer", match_id="m2",
            outcome="away_win", stake=100.0, odds=3.00,
            model_prob=0.40, edge=0.07, kelly_fraction=0.10,
            timestamp=datetime.now(),
        )

        bm.place_bet(bet)
        pnl = bm.settle_bet("test2", "lost")

        assert pnl == -100.0
        assert bm.current_bankroll == 900.0

    def test_drawdown_pause(self) -> None:
        """Betting should pause when drawdown limit is exceeded."""
        from datetime import datetime

        bm = BankrollManager(initial_bankroll=1000.0)
        # Simulate losing enough to trigger 20% drawdown
        bm.current_bankroll = 790.0  # 21% below peak of 1000

        bet = Bet(
            bet_id="test3", sport="soccer", match_id="m3",
            outcome="draw", stake=50.0, odds=3.50,
            model_prob=0.30, edge=0.02, kelly_fraction=0.05,
            timestamp=datetime.now(),
        )

        allowed, reason = bm.can_place_bet(50.0)
        assert allowed is False
        assert "Drawdown" in reason

    def test_stats_output(self) -> None:
        """Stats should include all expected keys."""
        bm = BankrollManager(initial_bankroll=1000.0)
        stats = bm.get_stats()

        assert "initial_bankroll" in stats
        assert "current_bankroll" in stats
        assert stats["total_bets"] == 0


# ------------------------------------------------------------------
# Metrics Tests
# ------------------------------------------------------------------


class TestMetrics:
    """Test suite for MetricsCalculator."""

    def test_brier_score(self) -> None:
        """Perfect predictions should have Brier score near 0."""
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.array([
            [0.95, 0.03, 0.02],
            [0.02, 0.95, 0.03],
            [0.02, 0.03, 0.95],
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
        ])
        bs = MetricsCalculator.brier_score(y_true, y_proba)
        assert bs < 0.1

    def test_brier_skill_score(self) -> None:
        """Good predictions should have positive BSS."""
        y_true = np.array([0, 1, 2, 0, 1, 2] * 10)
        # Good predictions
        good_proba = np.zeros((60, 3))
        for i, yt in enumerate(y_true):
            good_proba[i, yt] = 0.8
            for j in range(3):
                if j != yt:
                    good_proba[i, j] = 0.1

        bss = MetricsCalculator.brier_skill_score(y_true, good_proba)
        assert bss > 0

    def test_ece(self) -> None:
        """ECE should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, size=100)
        y_proba = np.random.dirichlet(alpha=[2, 2, 2], size=100)

        ece = MetricsCalculator.expected_calibration_error(y_true, y_proba)
        assert 0 <= ece <= 1

    def test_max_drawdown(self) -> None:
        """Max drawdown should be correctly calculated."""
        history = [1000, 1100, 1050, 900, 950, 1000]
        dd, peak, trough = MetricsCalculator.max_drawdown(history)

        # Peak was 1100, trough was 900: dd = 200/1100 ≈ 0.1818
        assert abs(dd - 200 / 1100) < 0.01
        assert peak == 1  # index of 1100
        assert trough == 3  # index of 900

    def test_sharpe_ratio(self) -> None:
        """Positive consistent returns should have positive Sharpe."""
        returns = [0.05, 0.03, 0.04, 0.06, 0.02, 0.04]
        sharpe = MetricsCalculator.sharpe_ratio(returns)
        assert sharpe > 0

    def test_roi(self) -> None:
        """ROI calculation should be correct."""
        roi = MetricsCalculator.roi(total_pnl=150, total_staked=1000)
        assert roi == 0.15
