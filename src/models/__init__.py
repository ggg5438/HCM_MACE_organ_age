"""
Machine Learning Models for MACE Prediction

This module contains the machine learning models used for predicting
Major Adverse Cardiovascular Events (MACE).
"""

from .mace_predictor import MACEPredictor
from .splsda import SPLSDAClassifier
from .model_training import ModelTrainer

__all__ = ['MACEPredictor', 'SPLSDAClassifier', 'ModelTrainer']