"""
Probability Calibration
=======================
Applies calibration to raw ensemble probabilities so that a stated
70% probability actually wins ~70% of the time.

Why this matters for betting:
  Uncalibrated models are typically overconfident near 0 and 1, underconfident
  near 0.5. This distorts Kelly stake sizes in both directions — you over-bet
  apparent certainties and under-bet genuine edges near the margin.

Methods (tried in cascade, best log-loss wins):
  1. Temperature scaling   (1 parameter — works with < 100 samples, best for small cal sets)
  2. Platt / sigmoid       (per-class logistic on log-odds — good for 50–500 samples)
  3. Isotonic regression   (non-parametric — best for ≥ 500 samples per class)

  Multi-class (soccer): each class calibrated independently, then renormalised.
  Binary (basketball, tennis, NHL, MLB): two calibrators.

Usage:
    cal = EnsembleCalibrator()
    cal.fit(ensemble, X_cal, y_cal)
    cal.save(path)

    cal = EnsembleCalibrator.load(path)
    p_calibrated = cal.transform(raw_proba)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# ── Temperature Scaling ───────────────────────────────────────────────────────

class TemperatureScaler:
    """
    Single-parameter calibration via temperature scaling.

    Raw logits are divided by T before softmax/sigmoid. T > 1 shrinks
    probabilities toward 0.5 (reduces overconfidence); T < 1 sharpens them.

    This is the most data-efficient calibration method — it has just ONE
    free parameter and works reliably with as few as 20 samples, unlike
    isotonic regression (needs 500+) or Platt scaling (needs 50+).

    Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks"
    (also effective for GBDT/RF ensembles).
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._is_fitted: bool = False

    def fit(self, raw_proba: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        """Find T that minimises NLL on raw_proba, y."""
        raw_proba = np.clip(raw_proba, 1e-7, 1 - 1e-7)
        n_classes = raw_proba.shape[1]

        def nll(T):
            T = max(T, 1e-3)
            # Convert probs → logits → divide by T → softmax
            logits = np.log(raw_proba)
            scaled_logits = logits / T
            # Numerically stable softmax
            exp_l = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
            p = exp_l / exp_l.sum(axis=1, keepdims=True)
            p = np.clip(p, 1e-7, 1.0)
            return log_loss(y, p)

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        self._is_fitted = True
        logger.info(
            "TemperatureScaler: T=%.4f  (T>1 = model was overconfident, T<1 = underconfident)",
            self.temperature,
        )
        return self

    def transform(self, raw_proba: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        if not self._is_fitted:
            return raw_proba
        raw_proba = np.clip(raw_proba, 1e-7, 1 - 1e-7)
        logits = np.log(raw_proba)
        scaled = logits / self.temperature
        exp_l = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        p = exp_l / exp_l.sum(axis=1, keepdims=True)
        return np.clip(p, 1e-7, 1.0)


class EnsembleCalibrator:
    """
    Isotonic regression calibrator for ensemble probability output.

    Parameters
    ----------
    n_classes : int
        Number of output classes (2 for basketball/tennis, 3 for soccer).
    """

    def __init__(self, n_classes: int = 2) -> None:
        self.n_classes = n_classes
        self._calibrators: List[IsotonicRegression] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        ensemble,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        allowed_methods: tuple[str, ...] | list[str] | None = None,
    ) -> "EnsembleCalibrator":
        """
        Fit calibrators on a held-out calibration set.

        Method cascade (tries all three, picks the lowest log-loss):
          1. Temperature scaling  — 1 parameter, works with ≥ 10 samples
          2. Platt / sigmoid      — per-class logistic, good for 50–500 samples
          3. Isotonic regression  — non-parametric, best for ≥ 500 per class

        Calibration is only applied if it reduces log_loss. If all methods
        degrade performance, the calibrator becomes a no-op (passthrough).

        Parameters
        ----------
        ensemble : _SoftVotingWrapper
        X_cal : array-like (n, n_features)
        y_cal : array-like (n,)
        """
        from sklearn.linear_model import LogisticRegression
        import pandas as pd

        if isinstance(X_cal, pd.DataFrame):
            X_cal = X_cal.values
        y_cal = np.asarray(y_cal)

        raw_proba = ensemble.predict_proba(X_cal)
        allowed = {m.lower().strip() for m in (allowed_methods or ("temperature", "sigmoid", "isotonic"))}
        n_classes = raw_proba.shape[1]
        self.n_classes = n_classes
        raw_ll = log_loss(y_cal, raw_proba)

        # ── Method 1: Temperature scaling ────────────────────────────────────
        if "temperature" in allowed:
            ts = TemperatureScaler()
            ts.fit(raw_proba, y_cal)
            ts_proba = ts.transform(raw_proba)
            ts_ll = log_loss(y_cal, ts_proba)
        else:
            ts = None
            ts_proba = None
            ts_ll = raw_ll + 1

        # ── Method 2: Sigmoid / Platt ─────────────────────────────────────────
        def _fit_sigmoid():
            cals = []
            for i in range(n_classes):
                y_b = (y_cal == i).astype(float)
                p = np.clip(raw_proba[:, i], 1e-7, 1 - 1e-7)
                log_odds = np.log(p / (1 - p)).reshape(-1, 1)
                lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
                lr.fit(log_odds, y_b)
                cals.append(lr)
            return cals

        def _apply_sigmoid(cals):
            out = np.zeros_like(raw_proba)
            for i, cal in enumerate(cals):
                p = np.clip(raw_proba[:, i], 1e-7, 1 - 1e-7)
                log_odds = np.log(p / (1 - p)).reshape(-1, 1)
                out[:, i] = cal.predict_proba(log_odds)[:, 1]
            out = np.clip(out, 1e-7, 1.0)
            return out / out.sum(axis=1, keepdims=True)

        if "sigmoid" in allowed:
            try:
                sig_cals = _fit_sigmoid()
                sig_proba = _apply_sigmoid(sig_cals)
                sig_ll = log_loss(y_cal, sig_proba)
            except Exception as exc:
                logger.warning("Sigmoid calibration failed: %s", exc)
                sig_cals, sig_proba, sig_ll = None, None, raw_ll + 1
        else:
            sig_cals, sig_proba, sig_ll = None, None, raw_ll + 1

        # ── Method 3: Isotonic ────────────────────────────────────────────────
        def _fit_isotonic():
            cals = []
            for i in range(n_classes):
                y_b = (y_cal == i).astype(float)
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_proba[:, i], y_b)
                cals.append(iso)
            return cals

        def _apply_isotonic(cals):
            out = np.zeros_like(raw_proba)
            for i, cal in enumerate(cals):
                out[:, i] = cal.transform(raw_proba[:, i])
            out = np.clip(out, 1e-7, 1.0)
            return out / out.sum(axis=1, keepdims=True)

        min_per_class = min((y_cal == i).sum() for i in range(n_classes))
        # Isotonic regression needs enough samples to avoid overfitting the cal set.
        # With < 150 per class, it typically memorises the cal set but degrades on new data.
        # Temperature scaling is strongly preferred for small datasets.
        _ISO_MIN_SAMPLES = 150
        if "isotonic" not in allowed:
            iso_cals, iso_proba, iso_ll = None, None, raw_ll + 1
            logger.info("Isotonic disabled for this calibration run")
        elif min_per_class >= _ISO_MIN_SAMPLES:
            try:
                iso_cals = _fit_isotonic()
                iso_proba = _apply_isotonic(iso_cals)
                iso_ll = log_loss(y_cal, iso_proba)
            except Exception as exc:
                logger.warning("Isotonic calibration failed: %s", exc)
                iso_cals, iso_proba, iso_ll = None, None, raw_ll + 1
        else:
            iso_cals, iso_proba, iso_ll = None, None, raw_ll + 1
            logger.info(
                "Isotonic skipped (min_per_class=%d < %d) — overfits small cal sets; "
                "temperature/sigmoid preferred",
                min_per_class, _ISO_MIN_SAMPLES,
            )

        # ── Pick best ────────────────────────────────────────────────────────
        best_ll    = raw_ll
        best_name  = "passthrough"
        best_state = None   # (method_tag, calibrator_object)

        for name, ll, state in [
            ("temperature", ts_ll,  ("temperature", ts)),
            ("sigmoid",     sig_ll, ("sigmoid", sig_cals)),
            ("isotonic",    iso_ll, ("isotonic", iso_cals)),
        ]:
            if ll < best_ll:
                best_ll   = ll
                best_name = name
                best_state = state

        if best_state is None:
            # All three methods degraded — passthrough
            self._calibrators = []
            self._cal_method = "passthrough"
            self._is_fitted = False
            logger.warning(
                "Calibration skipped (n=%d): all methods degraded vs raw log_loss=%.4f "
                "(temp=%.4f sig=%.4f iso=%.4f). "
                "This usually means the cal set is too small or perfectly separable.",
                len(y_cal), raw_ll, ts_ll, sig_ll, iso_ll,
            )
        else:
            method_tag, cal_obj = best_state
            self._cal_method = method_tag
            self._is_fitted = True
            if method_tag == "temperature":
                self._temperature_scaler = cal_obj
                self._calibrators = []
                self._is_sigmoid = False
            elif method_tag == "sigmoid":
                self._calibrators = cal_obj
                self._is_sigmoid = True
            else:
                self._calibrators = cal_obj
                self._is_sigmoid = False
            logger.info(
                "Calibration (%s, n=%d, T=%.3f): log_loss %.4f → %.4f  (Δ%+.4f ✓) "
                "[temp=%.4f sig=%.4f iso=%.4f]",
                best_name, len(y_cal),
                getattr(ts, "temperature", 1.0) if ts is not None else 1.0,
                raw_ll, best_ll, raw_ll - best_ll,
                ts_ll, sig_ll, iso_ll,
            )

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, raw_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probability matrix.

        Parameters
        ----------
        raw_proba : np.ndarray, shape (n, n_classes)

        Returns
        -------
        np.ndarray, shape (n, n_classes)  — renormalised to sum to 1.
        """
        if not self._is_fitted:
            return raw_proba

        method = getattr(self, "_cal_method", None)

        # ── Temperature scaling (new primary method) ──────────────────────────
        if method == "temperature":
            ts = getattr(self, "_temperature_scaler", None)
            if ts is not None:
                return ts.transform(raw_proba)
            return raw_proba

        # ── Sigmoid / Isotonic (legacy + fallback) ────────────────────────────
        is_sigmoid = getattr(self, "_is_sigmoid", False)
        if not self._calibrators:
            return raw_proba

        out = np.zeros_like(raw_proba)
        for i, cal in enumerate(self._calibrators):
            p = np.clip(raw_proba[:, i], 1e-7, 1 - 1e-7)
            if is_sigmoid:
                log_odds = np.log(p / (1 - p)).reshape(-1, 1)
                out[:, i] = cal.predict_proba(log_odds)[:, 1]
            else:
                out[:, i] = cal.transform(raw_proba[:, i])

        out = np.clip(out, 1e-7, 1.0)
        return out / out.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, ensemble, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compare raw vs calibrated metrics on a test set."""
        import pandas as pd
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        y_test = np.asarray(y_test)

        raw = ensemble.predict_proba(X_test)
        cal = self.transform(raw)

        raw_ll = log_loss(y_test, raw)
        cal_ll = log_loss(y_test, cal)

        if self.n_classes == 2:
            raw_bs = brier_score_loss(y_test, raw[:, 1])
            cal_bs = brier_score_loss(y_test, cal[:, 1])
        else:
            raw_bs = float(np.mean([
                brier_score_loss((y_test == i).astype(int), raw[:, i])
                for i in range(self.n_classes)
            ]))
            cal_bs = float(np.mean([
                brier_score_loss((y_test == i).astype(int), cal[:, i])
                for i in range(self.n_classes)
            ]))

        return {
            "raw_log_loss": float(raw_ll),
            "cal_log_loss": float(cal_ll),
            "raw_brier": float(raw_bs),
            "cal_brier": float(cal_bs),
            "log_loss_improvement": float(raw_ll - cal_ll),
            "brier_improvement": float(raw_bs - cal_bs),
        }

    # ------------------------------------------------------------------
    # Calibration reliability diagram data
    # ------------------------------------------------------------------

    def reliability_data(
        self, ensemble, X: np.ndarray, y: np.ndarray, n_bins: int = 10
    ) -> dict:
        """
        Return data for a reliability diagram (fraction positive vs mean predicted prob).
        Useful for visually confirming calibration quality.
        """
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        y = np.asarray(y)

        raw = ensemble.predict_proba(X)
        cal = self.transform(raw)

        bins = np.linspace(0, 1, n_bins + 1)
        result = {}

        for label, proba in [("raw", raw), ("calibrated", cal)]:
            p1 = proba[:, 1] if self.n_classes == 2 else proba.max(axis=1)
            mean_pred, frac_pos = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (p1 >= lo) & (p1 < hi)
                if mask.sum() > 0:
                    if self.n_classes == 2:
                        mean_pred.append(float(p1[mask].mean()))
                        frac_pos.append(float((y[mask] == 1).mean()))
                    else:
                        pred_cls = proba[mask].argmax(axis=1)
                        mean_pred.append(float(proba[mask].max(axis=1).mean()))
                        frac_pos.append(float((pred_cls == y[mask]).mean()))
            result[label] = {"mean_predicted": mean_pred, "fraction_positive": frac_pos}

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Calibrator saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> Optional["EnsembleCalibrator"]:
        path = Path(path)
        if not path.exists():
            logger.debug("No calibrator found at %s", path)
            return None
        obj = joblib.load(path)
        logger.info("Calibrator loaded ← %s", path)
        return obj


class ProbabilityCalibrator:
    """
    Backward-compatible probability calibrator used by older tests and callers.

    Supports direct calibration of probability matrices via:
      - ``method="isotonic"``
      - ``method="platt"`` / ``method="sigmoid"``

    For multi-class inputs, each class is calibrated one-vs-rest and then
    renormalized so rows still sum to 1.
    """

    def __init__(self, method: str = "isotonic", n_classes: int = 2) -> None:
        method = method.lower().strip()
        if method == "sigmoid":
            method = "platt"
        if method not in {"isotonic", "platt"}:
            raise ValueError(f"Unsupported calibration method: {method}")

        self.method = method
        self.n_classes = n_classes
        self._calibrators = []
        self._is_fitted = False

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray) -> "ProbabilityCalibrator":
        from sklearn.linear_model import LogisticRegression

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim != 2:
            raise ValueError("y_proba must be a 2D probability matrix")

        self.n_classes = y_proba.shape[1]
        self._calibrators = []

        for class_idx in range(self.n_classes):
            y_binary = (y_true == class_idx).astype(float)
            class_proba = np.clip(y_proba[:, class_idx], 1e-7, 1 - 1e-7)

            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(class_proba, y_binary)
            else:
                log_odds = np.log(class_proba / (1 - class_proba)).reshape(-1, 1)
                calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
                calibrator.fit(log_odds, y_binary)

            self._calibrators.append(calibrator)

        self._is_fitted = True
        return self

    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim != 2:
            raise ValueError("y_proba must be a 2D probability matrix")
        if not self._is_fitted:
            return y_proba

        out = np.zeros_like(y_proba, dtype=float)
        for class_idx, calibrator in enumerate(self._calibrators):
            class_proba = np.clip(y_proba[:, class_idx], 1e-7, 1 - 1e-7)
            if self.method == "isotonic":
                out[:, class_idx] = calibrator.transform(class_proba)
            else:
                log_odds = np.log(class_proba / (1 - class_proba)).reshape(-1, 1)
                out[:, class_idx] = calibrator.predict_proba(log_odds)[:, 1]

        out = np.clip(out, 1e-7, 1.0)
        return out / out.sum(axis=1, keepdims=True)
