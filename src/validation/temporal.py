from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd


DEFAULT_TIMESTAMP_COLUMNS = (
    "data_as_of_time",
    "odds_timestamp",
    "injury_report_timestamp",
    "lineup_timestamp",
    "goalie_confirmation_timestamp",
)


@dataclass(frozen=True)
class TemporalAuditResult:
    unsafe_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.unsafe_columns and not self.missing_required_columns

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_temporal_frame(
    frame: pd.DataFrame,
    *,
    prediction_time_col: str = "prediction_time",
    game_start_time_col: str = "game_start_time",
    timestamp_columns: Iterable[str] = DEFAULT_TIMESTAMP_COLUMNS,
    required_columns: Iterable[str] = ("game_start_time", "prediction_time"),
) -> TemporalAuditResult:
    missing = tuple(col for col in required_columns if col not in frame.columns)
    warnings: list[str] = []
    unsafe: list[str] = []
    if missing:
        return TemporalAuditResult((), missing, ("required temporal columns missing",))

    prediction_times = pd.to_datetime(frame[prediction_time_col], errors="coerce", utc=True)
    game_start_times = pd.to_datetime(frame[game_start_time_col], errors="coerce", utc=True)
    if bool((prediction_times >= game_start_times).fillna(False).any()):
        unsafe.append(prediction_time_col)
        warnings.append("prediction_time must be before game_start_time")

    for col in timestamp_columns:
        if col not in frame.columns:
            continue
        observed = pd.to_datetime(frame[col], errors="coerce", utc=True)
        if bool((observed > prediction_times).fillna(False).any()):
            unsafe.append(col)
            warnings.append(f"{col} occurs after prediction_time")

    return TemporalAuditResult(
        unsafe_columns=tuple(dict.fromkeys(unsafe)),
        missing_required_columns=missing,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def chronological_split_indices(n_rows: int, *, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[slice, slice, slice]:
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid chronological split ratios")
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("not enough rows for chronological train/val/test split")
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_rows)


def walk_forward_splits(
    n_rows: int,
    *,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    if n_rows <= 0 or initial_train_size <= 0 or test_size <= 0:
        raise ValueError("n_rows, initial_train_size, and test_size must be positive")
    step = step_size or test_size
    splits: list[tuple[list[int], list[int]]] = []
    train_end = initial_train_size
    while train_end < n_rows:
        test_end = min(train_end + test_size, n_rows)
        if test_end <= train_end:
            break
        train_idx = list(range(0, train_end))
        test_idx = list(range(train_end, test_end))
        if train_idx and test_idx:
            splits.append((train_idx, test_idx))
        train_end += step
    return splits
