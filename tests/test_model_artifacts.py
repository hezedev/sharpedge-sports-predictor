"""Tests for model artifact resolution helpers."""

from src.models.artifacts import get_current_model_tag, set_current_model_tag


def test_get_current_model_tag_uses_metadata_when_present(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "models"
    sport_dir = model_root / "soccer"
    sport_dir.mkdir(parents=True)
    (sport_dir / "ensemble_pl_2024_25.joblib").write_text("x", encoding="utf-8")
    (sport_dir / "ensemble_old.joblib").write_text("x", encoding="utf-8")

    from src.models import artifacts as artifacts_mod

    monkeypatch.setitem(artifacts_mod.settings["paths"], "models", str(model_root))
    set_current_model_tag("soccer", "pl_2024_25")

    assert get_current_model_tag("soccer", fallback="old") == "pl_2024_25"


def test_get_current_model_tag_falls_back_to_latest_ensemble(tmp_path, monkeypatch) -> None:
    model_root = tmp_path / "models"
    sport_dir = model_root / "basketball"
    sport_dir.mkdir(parents=True)
    older = sport_dir / "ensemble_old.joblib"
    newer = sport_dir / "ensemble_nba_2024_25.joblib"
    older.write_text("x", encoding="utf-8")
    newer.write_text("x", encoding="utf-8")

    from src.models import artifacts as artifacts_mod

    monkeypatch.setitem(artifacts_mod.settings["paths"], "models", str(model_root))
    meta = sport_dir / "current_tag.txt"
    if meta.exists():
        meta.unlink()

    assert get_current_model_tag("basketball", fallback="missing") == "nba_2024_25"
