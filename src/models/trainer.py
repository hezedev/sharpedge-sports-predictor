"""
Model training module.

Trains XGBoost, LightGBM, and Random Forest classifiers with
time-series cross-validation, hyperparameter configs from YAML,
and optional ensemble stacking.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

from config import settings
from src.models.artifacts import set_current_model_tag

logger = logging.getLogger(__name__)


class _SoftVotingWrapper:
    """
    Lightweight soft-voting ensemble that averages predict_proba()
    across already-trained estimators without refitting them.
    """

    def __init__(
        self,
        estimators: list,
        weights: Optional[list],
        classes: np.ndarray,
    ) -> None:
        self.estimators = estimators          # [(name, model), ...]
        self.weights = weights                 # None = equal
        self.classes_ = classes

    def predict_proba(self, X: Any) -> np.ndarray:
        probas = []
        for name, model in self.estimators:
            try:
                p = model.predict_proba(X)
                probas.append(p)
            except Exception as exc:
                logger.warning("Skipping %s in ensemble predict_proba: %s", name, exc)

        if not probas:
            raise RuntimeError("All estimators failed in soft-voting ensemble")

        if self.weights:
            w = np.array(self.weights[:len(probas)], dtype=float)
            w /= w.sum()
            avg = sum(p * wi for p, wi in zip(probas, w))
        else:
            avg = np.mean(probas, axis=0)

        return avg

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class ModelTrainer:
    """
    Sport-agnostic model training pipeline.

    Trains multiple algorithms defined in settings.yaml, evaluates
    with walk-forward time-series CV, and builds an ensemble.

    Parameters
    ----------
    sport : str
        Sport identifier (for model persistence path).
    """

    ALGO_MAP = {
        "xgboost": "XGBClassifier",
        "lightgbm": "LGBMClassifier",
        "random_forest": "RandomForestClassifier",
    }

    def __init__(self, sport: str) -> None:
        self.sport = sport
        self._model_cfg = settings.get("model", {})
        self._algo_configs = self._model_cfg.get("algorithms", [])
        self._cv_cfg = self._model_cfg.get("cv", {})
        self._ensemble_cfg = self._model_cfg.get("ensemble", {})

        self._model_dir = Path(settings.get("paths", {}).get("models", "data/models")) / sport
        self._model_dir.mkdir(parents=True, exist_ok=True)

        self.trained_models: Dict[str, Any] = {}
        self.ensemble_model: Optional[Any] = None
        self.cv_results: Dict[str, Dict[str, float]] = {}

        logger.info("ModelTrainer initialized for %s", sport)

    # ------------------------------------------------------------------
    # Model Factory
    # ------------------------------------------------------------------

    def _build_model(
        self,
        name: str,
        params: Dict[str, Any],
        n_classes: int = 3,
    ) -> Any:
        """
        Instantiate a model by name with given parameters.

        Parameters
        ----------
        name : str
            Algorithm name ('xgboost', 'lightgbm', 'random_forest').
        params : dict
            Hyperparameters for the model.
        n_classes : int
            Number of target classes (2 = binary, 3+ = multiclass).

        Returns
        -------
        Estimator
            Scikit-learn compatible classifier.
        """
        clean_params = {k: v for k, v in params.items() if v is not None}

        if name == "xgboost":
            if xgb is None:
                raise ImportError("xgboost is not installed")
            # In XGBoost >= 2.0, early_stopping_rounds lives in the constructor
            early_stop = clean_params.pop("early_stopping_rounds", 50)
            clean_params.pop("eval_metric", None)
            clean_params.pop("use_label_encoder", None)
            # Auto-select objective and metric based on number of classes
            if n_classes == 2:
                objective = "binary:logistic"
                eval_metric = "logloss"
            else:
                objective = "multi:softprob"
                eval_metric = "mlogloss"
            model = xgb.XGBClassifier(
                objective=objective,
                eval_metric=eval_metric,
                early_stopping_rounds=early_stop,   # constructor, not fit()
                verbosity=0,
                **clean_params,
            )
            return model

        elif name == "lightgbm":
            if lgb is None:
                raise ImportError("lightgbm is not installed")
            # verbose is a constructor param in LightGBM >= 4.x, not a fit param
            clean_params.setdefault("verbose", -1)
            model = lgb.LGBMClassifier(**clean_params)
            return model

        elif name == "random_forest":
            model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **clean_params,
            )
            return model

        else:
            raise ValueError(f"Unknown algorithm: {name}")

    # ------------------------------------------------------------------
    # Time-Series Cross-Validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform walk-forward time-series cross-validation for all
        configured algorithms.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        dict
            Per-algorithm CV results (mean accuracy, log_loss).
        """
        n_splits = self._cv_cfg.get("n_splits", 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results: Dict[str, Dict[str, List[float]]] = {}

        n_classes = y.nunique()
        for algo_cfg in self._algo_configs:
            name = algo_cfg["name"]
            params = algo_cfg.get("params", {})
            logger.info("Cross-validating %s with %d splits", name, n_splits)

            fold_accuracies: List[float] = []
            fold_losses: List[float] = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                try:
                    model = self._build_model(name, params.copy(), n_classes=n_classes)

                    # Fit with eval set for early stopping (XGB/LGB)
                    if name in ("xgboost", "lightgbm"):
                        fit_kwargs = {"eval_set": [(X_val, y_val)]}
                        if name == "lightgbm":
                            fit_kwargs["callbacks"] = [lgb.log_evaluation(period=-1)]
                        else:
                            fit_kwargs["verbose"] = False
                        model.fit(X_train, y_train, **fit_kwargs)
                    else:
                        model.fit(X_train, y_train)

                    y_pred = model.predict(X_val)
                    y_proba = model.predict_proba(X_val)

                    acc = accuracy_score(y_val, y_pred)
                    ll = log_loss(y_val, y_proba, labels=sorted(y.unique()))

                    fold_accuracies.append(acc)
                    fold_losses.append(ll)

                    logger.debug(
                        "%s fold %d: accuracy=%.4f, log_loss=%.4f",
                        name, fold, acc, ll,
                    )

                except Exception as exc:
                    logger.error("Error in %s fold %d: %s", name, fold, exc)
                    continue

            results[name] = {
                "mean_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else 0.0,
                "std_accuracy": float(np.std(fold_accuracies)) if fold_accuracies else 0.0,
                "mean_log_loss": float(np.mean(fold_losses)) if fold_losses else float("inf"),
                "std_log_loss": float(np.std(fold_losses)) if fold_losses else 0.0,
                "n_folds": len(fold_accuracies),
            }

            logger.info(
                "%s CV: accuracy=%.4f±%.4f, log_loss=%.4f±%.4f",
                name,
                results[name]["mean_accuracy"],
                results[name]["std_accuracy"],
                results[name]["mean_log_loss"],
                results[name]["std_log_loss"],
            )

        self.cv_results = results
        return results

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train all configured algorithms on the full training set.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        X_val : pd.DataFrame, optional
            Validation features (for early stopping).
        y_val : pd.Series, optional
            Validation target.

        Returns
        -------
        dict
            Trained model instances keyed by algorithm name.
        """
        n_classes = y_train.nunique()
        for algo_cfg in self._algo_configs:
            name = algo_cfg["name"]
            params = algo_cfg.get("params", {})
            logger.info("Training %s on %d samples", name, len(X_train))

            try:
                model = self._build_model(name, params.copy(), n_classes=n_classes)

                if name == "xgboost" and X_val is not None:
                    # early_stopping_rounds is in constructor; just pass eval_set
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )
                elif name == "lightgbm" and X_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False),
                                   lgb.log_evaluation(period=-1)],
                    )
                else:
                    model.fit(X_train, y_train)

                self.trained_models[name] = model
                logger.info("Successfully trained %s", name)

                # Log feature importance for tree-based models
                if hasattr(model, "feature_importances_"):
                    importances = pd.Series(
                        model.feature_importances_,
                        index=X_train.columns,
                    ).sort_values(ascending=False)
                    logger.info("Top 10 features (%s):\n%s", name, importances.head(10))

            except Exception as exc:
                logger.error("Failed to train %s: %s", name, exc)
                continue

        return self.trained_models

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def build_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """
        Build an ensemble from individually trained models.

        Uses soft voting (probability averaging) or stacking
        based on configuration.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (for stacking meta-learner).
        y_train : pd.Series
            Training target.

        Returns
        -------
        Ensemble model instance.
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Call train() first.")

        method = self._ensemble_cfg.get("method", "soft_voting")
        weights = self._ensemble_cfg.get("weights")

        estimators = list(self.trained_models.items())
        logger.info(
            "Building ensemble: method=%s, models=%s",
            method, [n for n, _ in estimators],
        )

        if method == "soft_voting":
            # Use our own lightweight wrapper to avoid refitting already-trained
            # XGB/LGB models (which require eval_set for early stopping).
            self.ensemble_model = _SoftVotingWrapper(
                estimators=estimators,
                weights=weights,
                classes=np.array(sorted(y_train.unique())),
            )

        elif method == "stacking":
            meta_learner = LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
            )
            self.ensemble_model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                n_jobs=-1,
                passthrough=False,
            )
            self.ensemble_model.fit(X_train, y_train)

        logger.info("Ensemble built successfully: %s", method)
        return self.ensemble_model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models and the ensemble on a test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target.

        Returns
        -------
        dict
            Per-model evaluation metrics.
        """
        results: Dict[str, Dict[str, float]] = {}
        all_labels = sorted(y_test.unique())

        # Individual models
        for name, model in self.trained_models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                results[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "log_loss": log_loss(y_test, y_proba, labels=all_labels),
                }
                logger.info(
                    "%s test: accuracy=%.4f, log_loss=%.4f",
                    name, results[name]["accuracy"], results[name]["log_loss"],
                )
            except Exception as exc:
                logger.error("Error evaluating %s: %s", name, exc)

        # Ensemble
        if self.ensemble_model is not None:
            try:
                y_pred = self.ensemble_model.predict(X_test)
                y_proba = self.ensemble_model.predict_proba(X_test)

                results["ensemble"] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "log_loss": log_loss(y_test, y_proba, labels=all_labels),
                }
                logger.info(
                    "Ensemble test: accuracy=%.4f, log_loss=%.4f",
                    results["ensemble"]["accuracy"],
                    results["ensemble"]["log_loss"],
                )
            except Exception as exc:
                logger.error("Error evaluating ensemble: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_models(self, tag: str = "latest") -> Dict[str, Path]:
        """
        Save all trained models and ensemble to disk.

        Parameters
        ----------
        tag : str
            Version tag for the saved models.

        Returns
        -------
        dict
            Mapping of model name -> saved file path.
        """
        paths: Dict[str, Path] = {}

        for name, model in self.trained_models.items():
            path = self._model_dir / f"{name}_{tag}.joblib"
            joblib.dump(model, path)
            paths[name] = path
            logger.info("Saved model: %s -> %s", name, path)

        if self.ensemble_model is not None:
            path = self._model_dir / f"ensemble_{tag}.joblib"
            joblib.dump(self.ensemble_model, path)
            paths["ensemble"] = path
            logger.info("Saved ensemble -> %s", path)

        # Save CV results
        if self.cv_results:
            cv_path = self._model_dir / f"cv_results_{tag}.joblib"
            joblib.dump(self.cv_results, cv_path)
            paths["cv_results"] = cv_path

        set_current_model_tag(self.sport, tag)
        return paths

    def load_models(self, tag: str = "latest") -> Dict[str, Any]:
        """
        Load previously saved models from disk.

        Parameters
        ----------
        tag : str
            Version tag to load.

        Returns
        -------
        dict
            Loaded model instances.
        """
        for algo_cfg in self._algo_configs:
            name = algo_cfg["name"]
            path = self._model_dir / f"{name}_{tag}.joblib"
            if path.exists():
                self.trained_models[name] = joblib.load(path)
                logger.info("Loaded model: %s from %s", name, path)
            else:
                logger.warning("Model file not found: %s", path)

        ensemble_path = self._model_dir / f"ensemble_{tag}.joblib"
        if ensemble_path.exists():
            self.ensemble_model = joblib.load(ensemble_path)
            logger.info("Loaded ensemble from %s", ensemble_path)

        return self.trained_models
