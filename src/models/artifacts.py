"""Helpers for resolving the active model artifact tag per sport."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config import settings


def model_dir_for_sport(sport: str) -> Path:
    """Return the model directory for a sport."""
    return Path(settings.get("paths", {}).get("models", "data/models")) / sport


def ensemble_path_for_tag(sport: str, tag: str) -> Path:
    """Return the ensemble artifact path for a given sport/tag."""
    return model_dir_for_sport(sport) / f"ensemble_{tag}.joblib"


def calibrator_path_for_tag(sport: str, tag: str) -> Path:
    """Return the calibrator artifact path for a given sport/tag."""
    return model_dir_for_sport(sport) / f"calibrator_{tag}.joblib"


def current_tag_path(sport: str) -> Path:
    """Return the metadata path storing the active tag for a sport."""
    return model_dir_for_sport(sport) / "current_tag.txt"


def set_current_model_tag(sport: str, tag: str) -> Path:
    """Persist the currently active model tag for a sport."""
    path = current_tag_path(sport)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{tag}\n", encoding="utf-8")
    return path


def get_current_model_tag(sport: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    Resolve the current model tag for a sport.

    Order:
    1. `current_tag.txt` if it points to an existing ensemble
    2. explicit `fallback` if its ensemble exists
    3. newest `ensemble_*.joblib` in the sport directory
    4. raw fallback
    """
    meta_path = current_tag_path(sport)
    if meta_path.exists():
        tag = meta_path.read_text(encoding="utf-8").strip()
        if tag and ensemble_path_for_tag(sport, tag).exists():
            return tag

    if fallback and ensemble_path_for_tag(sport, fallback).exists():
        return fallback

    model_dir = model_dir_for_sport(sport)
    candidates = sorted(
        model_dir.glob("ensemble_*.joblib"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].stem.replace("ensemble_", "", 1)

    return fallback
