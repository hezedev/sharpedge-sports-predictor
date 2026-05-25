"""ML model training, prediction, and calibration modules."""

from src.models.trainer import ModelTrainer
from src.models.predictor import Predictor
from src.models.calibration import EnsembleCalibrator
from src.models.soccer_score_model import SoccerScoreModel, SoccerProbabilityView
from src.models.mlb_side_model import MLBSideModel, MLBProbabilityView
from src.models.basketball_side_model import BasketballSideModel, BasketballProbabilityView
from src.models.nhl_side_model import NHLSideModel, NHLProbabilityView
# Alias for backward compatibility
ProbabilityCalibrator = EnsembleCalibrator

__all__ = [
    "ModelTrainer",
    "Predictor",
    "EnsembleCalibrator",
    "ProbabilityCalibrator",
    "SoccerScoreModel",
    "SoccerProbabilityView",
    "MLBSideModel",
    "MLBProbabilityView",
    "BasketballSideModel",
    "BasketballProbabilityView",
    "NHLSideModel",
    "NHLProbabilityView",
]
