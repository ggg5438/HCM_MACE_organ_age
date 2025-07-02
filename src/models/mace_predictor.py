"""
MACE Predictor Module

This module provides the main MACEPredictor class for predicting
Major Adverse Cardiovascular Events using protein signatures and organ aging.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple, Union
import warnings
import contextlib
import io

try:
    from organage import OrganAge
except ImportError:
    warnings.warn("OrganAge library not found. Some functionality may be limited.")
    OrganAge = None


class MACEPredictor:
    """
    Predicts MACE (Major Adverse Cardiovascular Events) using protein profiles and organ aging.
    
    This class loads a pre-trained model and provides methods for vectorized prediction
    on new patient data. It integrates protein expression data with organ aging calculations
    to provide comprehensive MACE risk assessment.
    
    Attributes:
        model: Trained machine learning model
        model_info: Model metadata and configuration
        feature_names: List of features used by the model
        preprocessing_info: Preprocessing parameters
        organ_list: List of organ features
        protein_list: List of protein features
    """
    
    def __init__(self, models_dir: str = "saved_models"):
        """
        Initialize the MACE Predictor.
        
        Args:
            models_dir: Directory containing saved model files
            
        Raises:
            FileNotFoundError: If model files are not found
            ImportError: If required dependencies are missing
        """
        self.models_dir = models_dir
        
        # Load model and metadata
        model_info_path = os.path.join(models_dir, "RandomForest_organ_list_info.joblib")
        model_path = os.path.join(models_dir, "RandomForest_organ_list_model.joblib")
        
        if not os.path.exists(model_info_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found in {models_dir}")
            
        self.model_info = joblib.load(model_info_path)
        self.model = joblib.load(model_path)
        
        # Extract model configuration
        self.feature_names = self.model_info['feature_names']
        self.preprocessing_info = self.model_info['preprocessing_info']
        
        # Define organ and protein lists
        self.organ_list = [
            'Adipose', 'Artery', 'Brain', 'Conventional', 'Heart', 
            'Immune', 'Intestine', 'Kidney', 'Liver', 'Lung', 
            'Muscle', 'Organismal', 'Pancreas'
        ]
        
        # Load protein list from reference data
        self.protein_list = self._load_protein_list()
        
        print(f"MACE Predictor initialized with {len(self.feature_names)} features")
        
    def _load_protein_list(self) -> list:
        """
        Load the list of protein features from reference data.
        
        Returns:
            List of protein feature names
        """
        # This would typically load from a configuration file or reference data
        # For now, return empty list - should be configured based on your data
        return []
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict MACE occurrence for multiple samples.
        
        Args:
            X: DataFrame with columns ['age', 'sex'] + protein columns
               
        Returns:
            Array of predictions (0: no MACE, 1: MACE)
            
        Example:
            >>> predictor = MACEPredictor()
            >>> predictions = predictor.predict(patient_data)
            >>> print(f"MACE predictions: {predictions}")
        """
        X_processed = self._preprocess_input(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict MACE probabilities for multiple samples.
        
        Args:
            X: DataFrame with columns ['age', 'sex'] + protein columns
               
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
            
        Example:
            >>> predictor = MACEPredictor()
            >>> probabilities = predictor.predict_proba(patient_data)
            >>> mace_probabilities = probabilities[:, 1]  # Probability of MACE
        """
        X_processed = self._preprocess_input(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_input(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data for model prediction.
        
        This method:
        1. Validates required columns
        2. Prepares metadata (age, sex)
        3. Calculates organ aging using OrganAge
        4. Applies saved preprocessing transformations
        
        Args:
            X: Input DataFrame
            
        Returns:
            Preprocessed feature array
            
        Raises:
            ValueError: If required columns are missing
            RuntimeError: If OrganAge calculation fails
        """
        # Validate required columns
        required_cols = ['age', 'sex']
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Prepare metadata for OrganAge
        metadata = X[['age', 'sex']].copy()
        metadata.columns = ['Age', 'Sex_F']
        
        # Extract protein profile
        if self.protein_list:
            protein_profile = X[self.protein_list]
        else:
            # If no protein list defined, use all columns except age/sex
            protein_cols = [col for col in X.columns if col not in ['age', 'sex']]
            protein_profile = X[protein_cols]
        
        # Calculate organ aging (suppress output)
        if OrganAge is None:
            raise RuntimeError("OrganAge library not available")
            
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                organ_data = OrganAge.CreateOrganAgeObject()
                organ_data.add_data(metadata, protein_profile)
                organ_data.normalize(assay_version="v4.1")
                res = organ_data.estimate_organ_ages()
            except Exception as e:
                raise RuntimeError(f"OrganAge calculation failed: {e}")
        
        # Extract organ aging features
        gap_df = res.reset_index().pivot_table(
            index='index', columns='Organ', values='AgeGap_zscored'
        )
        
        # Select required organ features
        available_organs = [organ for organ in self.organ_list if organ in gap_df.columns]
        organ_features = gap_df[available_organs]
        
        # Apply preprocessing
        X_processed = self._apply_preprocessing(organ_features)
        
        return X_processed
    
    def _apply_preprocessing(self, organ_features: pd.DataFrame) -> np.ndarray:
        """
        Apply saved preprocessing transformations to organ features.
        
        Args:
            organ_features: DataFrame of organ aging features
            
        Returns:
            Preprocessed feature array
        """
        # Handle missing values
        organ_data = organ_features.fillna(0)
        
        # Apply saved scaler
        organ_scaler = self.preprocessing_info.get('organ_scaler')
        if organ_scaler is not None:
            organ_data_scaled = organ_scaler.transform(organ_data)
        else:
            organ_data_scaled = organ_data.values
        
        return organ_data_scaled
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores, or None if not available
            
        Example:
            >>> predictor = MACEPredictor()
            >>> importance = predictor.get_feature_importance()
            >>> print(importance.head())
        """
        if hasattr(self.model, 'feature_importances_'):
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        return None
    
    def get_model_info(self) -> dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata, performance metrics, and configuration
        """
        return {
            'model_type': self.model_info.get('model_name', 'Unknown'),
            'feature_set': self.model_info.get('feature_set', 'Unknown'),
            'n_features': len(self.feature_names),
            'training_metrics': self.model_info.get('metrics', {}),
            'preprocessing_info': self.preprocessing_info
        }