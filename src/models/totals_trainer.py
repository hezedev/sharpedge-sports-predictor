"""
totals_trainer.py
=================
Trains Over/Under (totals) and spread-cover models for each sport.

Design
------
Totals:
  Binary classification — predict whether total score EXCEEDS a reference line.
  • Soccer:     Over/Under 2.5 goals  (line is fixed at 2.5)
  • Basketball: Over/Under points     (line is market-posted; we predict direction)
  • NHL:        Over/Under 5.5 goals  (most common NHL line)
  • MLB:        Over/Under 8.5 runs   (most common MLB line)

Spreads:
  Binary classification — predict whether the home team covers the spread.
  • NBA:   Home covers -X.5 / Away covers +X.5
  • MLB:   Run-line home covers -1.5 / away +1.5
  • NHL:   Puck-line home covers -1.5 / away +1.5
  Soccer: Asian Handicap 0.5 (essentially DNW / DNL of h2h model)

The models reuse exactly the same feature columns as the h2h models, just
with a different binary target. This means no new features to engineer —
just new labels from historical scores.

Usage
-----
    from src.models.totals_trainer import TotalsTrainer
    trainer = TotalsTrainer("soccer")
    trainer.fit(X, y_over)
    trainer.save("data/models/soccer/totals_soccer.joblib")

    pred = trainer.predict_proba_over(X_inference)   # P(over line)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.models.calibration import EnsembleCalibrator
from src.models.trainer import _SoftVotingWrapper

logger = logging.getLogger(__name__)

# ── Default reference lines ────────────────────────────────────────────────────
DEFAULT_TOTALS_LINE = {
    "soccer":     2.5,    # Over/Under 2.5 goals
    "basketball": 220.0,  # placeholder; market-posted line used at inference
    "nhl":        5.5,
    "mlb":        8.5,
}

DEFAULT_SPREAD_LINE = {
    "basketball": 0.0,    # ATS (any spread; we predict home covers)
    "nhl":        1.5,    # puck line
    "mlb":        1.5,    # run line
}


# ── Ensemble wrapper (same as h2h, reuse) ─────────────────────────────────────

class TotalsTrainer:
    """
    Trains a binary Over/Under or Spread-Cover model for a given sport.

    Parameters
    ----------
    sport : str
        'soccer', 'basketball', 'nhl', 'mlb'
    market : str
        'totals' or 'spreads'
    line : float, optional
        Reference line (e.g. 2.5 for soccer totals). Defaults to sport default.
    """

    MODEL_DIR = Path("data/models")

    def __init__(self, sport: str, market: str = "totals", line: Optional[float] = None):
        self.sport  = sport
        self.market = market
        if line is not None:
            self.line = line
        elif market == "totals":
            self.line = DEFAULT_TOTALS_LINE.get(sport, 2.5)
        else:
            self.line = DEFAULT_SPREAD_LINE.get(sport, 1.5)

        self._models: dict = {}
        self._ensemble: Optional[_SoftVotingWrapper] = None
        self._calibrator: Optional[EnsembleCalibrator] = None
        self._feature_cols: list = []

    def _align_features_for_inference(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reindex inference data to the exact feature set used during training.

        This keeps saved totals/spreads models usable even when the broader
        feature engineering layer has grown new columns since the model was fit.
        """
        if not self._feature_cols:
            return X.fillna(0)
        aligned = X.copy()
        missing = [c for c in self._feature_cols if c not in aligned.columns]
        for col in missing:
            aligned[col] = 0.0
        return aligned.reindex(columns=self._feature_cols, fill_value=0.0).fillna(0)

    # ── Target builders ────────────────────────────────────────────────────────

    @staticmethod
    def make_totals_target(df: pd.DataFrame, sport: str, line: float) -> Optional[pd.Series]:
        """
        Build binary over/under target from raw score columns.
        Returns None if required columns are missing.
        """
        if sport == "soccer":
            # Soccer stores rolling avg goals, not actual match scores in parquet
            # Use home_goals + away_goals if available, else can't build totals target
            if "home_score" in df.columns and "away_score" in df.columns:
                total = df["home_score"] + df["away_score"]
            elif "home_goals" in df.columns and "away_goals" in df.columns:
                total = df["home_goals"] + df["away_goals"]
            else:
                logger.warning("Soccer: no raw score columns found — cannot build totals target. "
                               "Need home_score/away_score or home_goals/away_goals.")
                return None
            return (total > line).astype(int)

        elif sport == "basketball":
            if "home_score" in df.columns and "away_score" in df.columns:
                total = df["home_score"] + df["away_score"]
                return (total > line).astype(int)
            # BallDontLie doesn't store final scores in feature parquet; compute from margins
            # home_scoring_margin = home_score - away_score → can't get total without one absolute
            logger.warning("Basketball: home_score/away_score not in features — "
                           "totals target unavailable without raw score data.")
            return None

        elif sport in ("nhl", "mlb"):
            if "home_score" in df.columns and "away_score" in df.columns:
                total = df["home_score"] + df["away_score"]
                return (total > line).astype(int)
            return None

        return None

    @staticmethod
    def make_spreads_target(df: pd.DataFrame, sport: str, line: float) -> Optional[pd.Series]:
        """
        Build binary home-covers target from raw score columns.
        home_covers = home_score - away_score > line  (or >= line for push handling)
        """
        if sport == "soccer":
            # Asian Handicap 0.0 (Draw No Bet): home wins → 1, draw/away → 0
            if "result" in df.columns:
                return (df["result"] == "home_win").astype(int)
            return None

        if sport == "basketball":
            # Home team covers if (home_score - away_score) > spread line
            # For general spreads model we predict: home_margin > 0 (home wins outright)
            # Specific spread lines are applied at inference using the market-posted number
            if "home_score" in df.columns and "away_score" in df.columns:
                margin = df["home_score"] - df["away_score"]
                return (margin > line).astype(int)
            # Fallback: use home_scoring_margin which IS in features
            if "home_scoring_margin" in df.columns:
                # This is a rolling average, not the actual game margin — less accurate
                # but gives a directional signal
                logger.warning("Basketball: using rolling home_scoring_margin as spread proxy (lower quality)")
                return None  # Don't use rolling averages as targets — too leaky
            return None

        if sport in ("nhl", "mlb"):
            if "home_score" in df.columns and "away_score" in df.columns:
                margin = df["home_score"] - df["away_score"]
                return (margin > line).astype(int)
            return None

        return None

    # ── Model fitting ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "TotalsTrainer":
        """Train XGBoost + LightGBM + RandomForest ensemble."""
        self._feature_cols = list(X_train.columns)

        models = []

        # XGBoost
        if xgb is not None:
            xgb_m = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                early_stopping_rounds=20,
                verbosity=0,
                random_state=42,
            )
            xgb_m.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            models.append(("xgboost", xgb_m))
            p = xgb_m.predict_proba(X_val)
            logger.info(
                "%s %s xgboost: acc=%.4f  ll=%.4f",
                self.sport, self.market,
                accuracy_score(y_val, xgb_m.predict(X_val)),
                log_loss(y_val, p),
            )

        # LightGBM
        if lgb is not None:
            lgb_m = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1,
                random_state=42,
            )
            lgb_m.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(period=-1)],
            )
            models.append(("lightgbm", lgb_m))
            p = lgb_m.predict_proba(X_val)
            logger.info(
                "%s %s lightgbm: acc=%.4f  ll=%.4f",
                self.sport, self.market,
                accuracy_score(y_val, lgb_m.predict(X_val)),
                log_loss(y_val, p),
            )

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        models.append(("random_forest", rf))
        p = rf.predict_proba(X_val)
        logger.info(
            "%s %s rf: acc=%.4f  ll=%.4f",
            self.sport, self.market,
            accuracy_score(y_val, rf.predict(X_val)),
            log_loss(y_val, p),
        )

        if not models:
            raise RuntimeError("No models trained — install xgboost or lightgbm")

        self._models = dict(models)
        classes = np.array([0, 1])
        self._ensemble = _SoftVotingWrapper(models, weights=None, classes=classes)
        return self

    def fit_calibrator(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """Fit and evaluate calibrator on hold-out sets."""
        if self._ensemble is None:
            raise RuntimeError("Call fit() before fit_calibrator()")
        cal = EnsembleCalibrator()
        cal.fit(self._ensemble, X_cal, y_cal)
        self._calibrator = cal

        metrics = cal.evaluate(self._ensemble, X_test, y_test)
        logger.info(
            "%s %s calibration: ll Δ%+.4f  brier Δ%+.4f",
            self.sport, self.market,
            metrics["log_loss_improvement"], metrics["brier_improvement"],
        )
        return metrics

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba_over(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return P(over line) for each row in X.
        Applies calibration if fitted.
        """
        if self._ensemble is None:
            raise RuntimeError("Model not fitted")
        raw = self._ensemble.predict_proba(self._align_features_for_inference(X))
        if self._calibrator is not None:
            raw = self._calibrator.transform(raw)
        return raw[:, 1]  # class 1 = "over" or "home covers"

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            path = self.MODEL_DIR / self.sport / f"{self.market}_{self.sport}.joblib"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("TotalsTrainer saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> Optional["TotalsTrainer"]:
        path = Path(path)
        if not path.exists():
            logger.debug("No totals model at %s", path)
            return None
        obj = joblib.load(path)
        logger.info("TotalsTrainer loaded ← %s", path)
        return obj
