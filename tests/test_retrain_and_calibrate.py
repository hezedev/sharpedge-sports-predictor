from __future__ import annotations

import pytest

import retrain_and_calibrate as rac


def test_tennis_uses_larger_calibration_slice() -> None:
    i_val, i_cal, i_test = rac._compute_temporal_split_indices(100, rac.SPORT_CONFIGS["tennis"])

    assert (i_val, i_cal, i_test) == (70, 80, 95)


def test_default_split_stays_backward_compatible() -> None:
    i_val, i_cal, i_test = rac._compute_temporal_split_indices(100, {})

    assert (i_val, i_cal, i_test) == (70, 85, 90)


def test_invalid_split_ratios_raise() -> None:
    with pytest.raises(ValueError):
        rac._compute_temporal_split_indices(
            100,
            {"split_ratios": {"train": 0.7, "val": 0.2, "cal": 0.2, "test": 0.1}},
        )
