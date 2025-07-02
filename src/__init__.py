"""
MACE Prediction using Protein Signatures and Organ Aging

This package provides tools for predicting Major Adverse Cardiovascular Events (MACE)
using protein expression profiles and organ aging signatures.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .models import MACEPredictor, SPLSDAClassifier
from .preprocessing import DataProcessor
from .analysis import FeatureImportanceAnalyzer
from .visualization import ResultsVisualizer

__all__ = [
    'MACEPredictor',
    'SPLSDAClassifier', 
    'DataProcessor',
    'FeatureImportanceAnalyzer',
    'ResultsVisualizer'
]