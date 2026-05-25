"""
Prediction module.

Loads trained models and generates calibrated probability
predictions for upcoming matches.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from config import settings
from src.models.artifacts import get_current_model_tag
from src.models.calibration import EnsembleCalibrator as ProbabilityCalibrator

logger = logging.getLogger(__name__)


class Predictor:
    """
    Generate calibrated probability predictions from trained models.

    Loads serialized models, applies calibration, and outputs
    per-outcome probabilities suitable for the risk module.

    Parameters
    ----------
    sport : str
        Sport identifier for loading correct models.
    label_map : dict
        Mapping of result labels to integer codes.
    tag : str
        Model version tag to load.
    """

    def __init__(
        self,
        sport: str,
        label_map: Dict[str, int],
        tag: str = "latest",
    ) -> None:
        self.sport = sport
        self.label_map = label_map
        self.inverse_label_map = {v: k for k, v in label_map.items()}
        self._tag = get_current_model_tag(sport, fallback=tag) if tag == "latest" else tag

        model_dir = Path(settings.get("paths", {}).get("models", "data/models")) / sport
        self._model_dir = model_dir

        self.models: Dict[str, Any] = {}
        self.ensemble: Optional[Any] = None
        self.calibrator: Optional[ProbabilityCalibrator] = None

        self._load_models()
        logger.info("Predictor initialized for %s (tag=%s)", sport, tag)

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load all available models for the configured sport and tag."""
        algo_configs = settings.get("model", {}).get("algorithms", [])

        for algo_cfg in algo_configs:
            name = algo_cfg["name"]
            path = self._model_dir / f"{name}_{self._tag}.joblib"
            if path.exists():
                self.models[name] = joblib.load(path)
                logger.info("Loaded model: %s", name)
            else:
                logger.debug("Model not found: %s", path)

        ensemble_path = self._model_dir / f"ensemble_{self._tag}.joblib"
        if ensemble_path.exists():
            self.ensemble = joblib.load(ensemble_path)
            logger.info("Loaded ensemble model")

        calibrator_path = self._model_dir / f"calibrator_{self._tag}.joblib"
        if calibrator_path.exists():
            self.calibrator = joblib.load(calibrator_path)
            logger.info("Loaded probability calibrator")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: pd.DataFrame,
        use_ensemble: bool = True,
        calibrate: bool = True,
    ) -> pd.DataFrame:
        """
        Generate probability predictions for each outcome.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for upcoming matches.
        use_ensemble : bool
            If True and ensemble is available, use ensemble.
            Otherwise, average individual model probabilities.
        calibrate : bool
            If True and calibrator is available, apply calibration.

        Returns
        -------
        pd.DataFrame
            Columns are outcome labels, values are probabilities.
            Index matches input X.
        """
        if X.empty:
            logger.warning("Empty feature matrix passed to predictor")
            return pd.DataFrame()

        # Fill NaN with 0 (consistent with training)
        X_clean = X.fillna(0)

        if use_ensemble and self.ensemble is not None:
            raw_proba = self.ensemble.predict_proba(X_clean)
            logger.debug("Generated ensemble predictions for %d matches", len(X_clean))

        elif self.models:
            # Average probabilities from all individual models
            all_probas = []
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba(X_clean)
                    all_probas.append(proba)
                except Exception as exc:
                    logger.error("Prediction error from %s: %s", name, exc)

            if not all_probas:
                raise RuntimeError("No models produced predictions")

            raw_proba = np.mean(all_probas, axis=0)
            logger.debug(
                "Averaged predictions from %d models for %d matches",
                len(all_probas), len(X_clean),
            )
        else:
            raise RuntimeError("No models loaded. Cannot generate predictions.")

        # Apply calibration
        if calibrate and self.calibrator is not None:
            raw_proba = self.calibrator.transform(raw_proba)
            logger.debug("Applied probability calibration")

        # Build output DataFrame with label names
        columns = [
            self.inverse_label_map.get(i, f"class_{i}")
            for i in range(raw_proba.shape[1])
        ]
        proba_df = pd.DataFrame(raw_proba, index=X.index, columns=columns)

        # Ensure probabilities sum to 1.0
        row_sums = proba_df.sum(axis=1)
        proba_df = proba_df.div(row_sums, axis=0)

        return proba_df

    def predict(
        self,
        X: pd.DataFrame,
        use_ensemble: bool = True,
    ) -> pd.Series:
        """
        Generate class predictions (most probable outcome).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        use_ensemble : bool
            Whether to use ensemble model.

        Returns
        -------
        pd.Series
            Predicted outcome labels.
        """
        proba_df = self.predict_proba(X, use_ensemble=use_ensemble)
        predictions = proba_df.idxmax(axis=1)
        return predictions

    def predict_with_confidence(
        self,
        X: pd.DataFrame,
        match_info: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions with confidence scores and match context.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        match_info : pd.DataFrame, optional
            Additional match info (teams, date, etc.) to include.

        Returns
        -------
        pd.DataFrame
            Prediction summary with probabilities, predicted outcome,
            confidence, and optional match info.
        """
        proba_df = self.predict_proba(X)
        predictions = proba_df.idxmax(axis=1)
        confidence = proba_df.max(axis=1)

        result = proba_df.copy()
        result["predicted"] = predictions
        result["confidence"] = confidence

        # Model agreement (how many individual models agree)
        if len(self.models) > 1:
            agreements = []
            X_clean = X.fillna(0)
            individual_preds = []
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_clean)
                    individual_preds.append(pred)
                except Exception:
                    continue

            if individual_preds:
                pred_array = np.array(individual_preds)  # (n_models, n_samples)
                for i in range(pred_array.shape[1]):
                    col_preds = pred_array[:, i]
                    mode_val = np.bincount(col_preds.astype(int)).argmax()
                    agree_pct = np.mean(col_preds == mode_val)
                    agreements.append(agree_pct)
                result["model_agreement"] = agreements

        # Attach match info if provided
        if match_info is not None:
            for col in match_info.columns:
                if col not in result.columns:
                    result[col] = match_info[col].values

        return result

    # ------------------------------------------------------------------
    # Individual Model Predictions
    # ------------------------------------------------------------------

    def predict_per_model(
        self,
        X: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get probability predictions from each individual model.

        Useful for analyzing model disagreement.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        dict
            Per-model probability DataFrames.
        """
        X_clean = X.fillna(0)
        per_model: Dict[str, pd.DataFrame] = {}

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_clean)
                columns = [
                    self.inverse_label_map.get(i, f"class_{i}")
                    for i in range(proba.shape[1])
                ]
                per_model[name] = pd.DataFrame(
                    proba, index=X.index, columns=columns
                )
            except Exception as exc:
                logger.error("Error getting predictions from %s: %s", name, exc)

        return per_model
