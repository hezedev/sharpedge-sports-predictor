from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class ProbabilityValidationReport:
    brier_score: float
    log_loss: float
    calibration_curve: list[dict[str, float]]
    roi: float | None = None
    max_drawdown: float | None = None
    avg_clv: float | None = None
    avg_edge: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def multiclass_brier_score(y_true: Sequence[int], y_proba: np.ndarray, *, labels: Sequence[int] | None = None) -> float:
    y = np.asarray(y_true)
    probs = np.asarray(y_proba, dtype=float)
    labels_arr = np.asarray(labels if labels is not None else sorted(np.unique(y)))
    encoded = np.zeros((len(y), len(labels_arr)), dtype=float)
    label_pos = {label: idx for idx, label in enumerate(labels_arr)}
    for row_idx, label in enumerate(y):
        if label in label_pos:
            encoded[row_idx, label_pos[label]] = 1.0
    return float(np.mean(np.sum((probs - encoded) ** 2, axis=1)))


def binary_or_multiclass_brier(y_true: Sequence[int], y_proba: np.ndarray) -> float:
    probs = np.asarray(y_proba, dtype=float)
    if probs.ndim == 1 or probs.shape[1] == 1:
        return float(brier_score_loss(y_true, probs.ravel()))
    if probs.shape[1] == 2:
        return float(brier_score_loss(y_true, probs[:, 1]))
    return multiclass_brier_score(y_true, probs, labels=list(range(probs.shape[1])))


def calibration_curve_data(y_true: Sequence[int], y_proba: Sequence[float], *, buckets: int = 10) -> list[dict[str, float]]:
    frame = pd.DataFrame({"actual": list(y_true), "prob": list(y_proba)})
    frame["bucket"] = pd.cut(frame["prob"], bins=np.linspace(0, 1, buckets + 1), include_lowest=True)
    rows: list[dict[str, float]] = []
    for _, grp in frame.groupby("bucket", observed=True):
        if grp.empty:
            continue
        rows.append(
            {
                "mean_predicted": round(float(grp["prob"].mean()), 6),
                "observed_rate": round(float(grp["actual"].mean()), 6),
                "count": int(len(grp)),
            }
        )
    return rows


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    equity = 1.0 + np.cumsum(np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / np.maximum(peaks, 1e-12)
    return float(np.max(drawdowns)) if len(drawdowns) else 0.0


def probability_validation_report(
    *,
    y_true: Sequence[int],
    y_proba: np.ndarray,
    positive_class_index: int | None = None,
    returns: Sequence[float] | None = None,
    clv: Sequence[float] | None = None,
    edges: Sequence[float] | None = None,
) -> ProbabilityValidationReport:
    probs = np.asarray(y_proba, dtype=float)
    labels = list(range(probs.shape[1])) if probs.ndim == 2 else None
    ll = float(log_loss(y_true, probs, labels=labels))
    brier = binary_or_multiclass_brier(y_true, probs)
    if probs.ndim == 1:
        curve_probs = probs
        curve_actual = y_true
    else:
        pos_idx = positive_class_index if positive_class_index is not None else probs.shape[1] - 1
        curve_probs = probs[:, pos_idx]
        curve_actual = [1 if int(y) == pos_idx else 0 for y in y_true]

    ret_arr = np.asarray(returns, dtype=float) if returns is not None else None
    clv_arr = np.asarray(clv, dtype=float) if clv is not None else None
    edge_arr = np.asarray(edges, dtype=float) if edges is not None else None
    return ProbabilityValidationReport(
        brier_score=round(brier, 6),
        log_loss=round(ll, 6),
        calibration_curve=calibration_curve_data(curve_actual, curve_probs),
        roi=round(float(np.mean(ret_arr)), 6) if ret_arr is not None and len(ret_arr) else None,
        max_drawdown=round(max_drawdown_from_returns(ret_arr), 6) if ret_arr is not None and len(ret_arr) else None,
        avg_clv=round(float(np.nanmean(clv_arr)), 6) if clv_arr is not None and len(clv_arr) else None,
        avg_edge=round(float(np.nanmean(edge_arr)), 6) if edge_arr is not None and len(edge_arr) else None,
    )


def basketball_spread_validation(*, actual_margins: Sequence[float], projected_margins: Sequence[float], spread_lines: Sequence[float] | None = None) -> dict[str, float]:
    actual = np.asarray(actual_margins, dtype=float)
    projected = np.asarray(projected_margins, dtype=float)
    payload = {
        "spread_mae": round(float(mean_absolute_error(actual, projected)), 6),
        "spread_rmse": round(float(mean_squared_error(actual, projected) ** 0.5), 6),
    }
    if spread_lines is not None:
        line = np.asarray(spread_lines, dtype=float)
        actual_cover = actual + line > 0
        projected_cover = projected + line > 0
        payload["ats_accuracy"] = round(float(np.mean(actual_cover == projected_cover)), 6)
    return payload


def nhl_goalie_status_performance(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    required = {"goalie_status", "won", "edge", "clv"}
    if not required.issubset(frame.columns):
        return {}
    output: dict[str, dict[str, float]] = {}
    for status, grp in frame.groupby("goalie_status"):
        output[str(status)] = {
            "count": int(len(grp)),
            "win_rate": round(float(pd.to_numeric(grp["won"], errors="coerce").mean()), 6),
            "avg_edge": round(float(pd.to_numeric(grp["edge"], errors="coerce").mean()), 6),
            "avg_clv": round(float(pd.to_numeric(grp["clv"], errors="coerce").mean()), 6),
        }
    return output
