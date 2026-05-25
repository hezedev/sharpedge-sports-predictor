"""Configuration package for Sports Predictor."""

from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env file from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Load settings.yaml
_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"


def load_settings() -> dict:
    """Load and return the settings dictionary from settings.yaml."""
    with open(_SETTINGS_PATH, "r") as f:
        return yaml.safe_load(f)


settings: dict = load_settings()
