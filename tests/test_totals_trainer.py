from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.totals_trainer import TotalsTrainer


class _FakeBinaryEnsemble:
    def __init__(self) -> None:
        self.last_columns: list[str] | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.last_columns = list(X.columns)
        return np.tile(np.array([[0.35, 0.65]]), (len(X), 1))


def test_totals_trainer_aligns_inference_columns_to_training_shape() -> None:
    trainer = TotalsTrainer("mlb", market="spreads")
    trainer._feature_cols = ["feat_a", "feat_b"]
    trainer._ensemble = _FakeBinaryEnsemble()

    X = pd.DataFrame([{"feat_a": 1.0, "feat_b": 2.0, "new_extra": 99.0}])
    proba = trainer.predict_proba_over(X)

    assert trainer._ensemble.last_columns == ["feat_a", "feat_b"]
    assert proba.shape == (1,)
    assert float(proba[0]) == 0.65
