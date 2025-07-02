"""
Data Processing Module

This module handles data loading, cleaning, preprocessing, and train/test splitting
for MACE prediction analysis.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import warnings

from .organ_age_calculator import OrganAgeCalculator


class DataProcessor:
    """
    Comprehensive data processing pipeline for MACE prediction.
    
    This class handles:
    - Data loading and validation
    - Missing value treatment
    - Feature preprocessing (protein and organ features)
    - Stratified train/test splitting
    - Data export for reproducibility
    """
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing data files
            random_state: Random state for reproducible splits
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.organ_calculator = OrganAgeCalculator()
        
        # Define feature sets
        self.organ_list = [
            'Adipose', 'Artery', 'Brain', 'Conventional', 'Heart', 
            'Immune', 'Intestine', 'Kidney', 'Liver', 'Lung', 
            'Muscle', 'Organismal', 'Pancreas'
        ]
        
        self.organ_list_specific = [
            'Adipose', 'Artery', 'Brain', 'Heart', 'Immune', 
            'Intestine', 'Kidney', 'Liver', 'Lung', 'Muscle', 'Pancreas'
        ]
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the main dataset.
        
        Args:
            file_path: Path to data file. If None, uses default path.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If required columns are missing
        """
        if file_path is None:
            file_path = os.path.join(
                self.data_dir, 
                "proteomics_v4.1_HCM_CUIMC(with sample ID, with MACE).xlsx"
            )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data
        data = pd.read_excel(file_path)
        
        # Validate required columns
        required_cols = ['age', 'sex', 'mace']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Preprocess basic columns
        data = self._preprocess_basic_columns(data)
        
        return data
    
    def _preprocess_basic_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess basic demographic and outcome columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with preprocessed columns
        """
        data = data.copy()
        
        # Convert sex to binary (0: Male, 1: Female)
        if data['sex'].dtype == 'object':
            sex_mapping = {'Male': 0, 'Female': 1, 'M': 0, 'F': 1}
            data['sex'] = data['sex'].map(sex_mapping)
        
        # Ensure MACE is binary integer
        data['mace'] = data['mace'].astype(int)
        
        # Validate age values
        if data['age'].isna().any():
            warnings.warn("Missing age values found. Consider imputation.")
        
        return data
    
    def create_feature_sets(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Create different feature sets for model comparison.
        
        Args:
            data: Input DataFrame with organ age features
            
        Returns:
            Dictionary mapping feature set names to feature lists
        """
        # Get protein columns (assuming they start from column 8)
        protein_columns = self._get_protein_columns(data)
        
        # Define feature sets
        feature_sets = {
            'organ_list': self.organ_list,
            'organ_list_specific': self.organ_list_specific,
            'protein_only': protein_columns,
            'organ_protein_combined': list(self.organ_list_specific) + protein_columns
        }
        
        # Filter to only include columns that exist in the data
        filtered_sets = {}
        for name, features in feature_sets.items():
            available_features = [f for f in features if f in data.columns]
            if available_features:
                filtered_sets[name] = available_features
            else:
                warnings.warn(f"No features available for set '{name}'")
        
        return filtered_sets
    
    def _get_protein_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Identify protein columns in the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of protein column names
        """
        # Load reference protein list
        try:
            ref_path = os.path.join(
                self.data_dir,
                "proteomics_v4.0_HCM_CUIMC(with sample ID, with MACE).xlsx"
            )
            ref_data = pd.read_excel(ref_path)
            protein_columns = ref_data.columns[8:].tolist()
            
            # Filter to columns that exist in current data
            available_proteins = [col for col in protein_columns if col in data.columns]
            return available_proteins
            
        except FileNotFoundError:
            warnings.warn("Reference protein file not found. Using heuristic detection.")
            
            # Heuristic: columns that are not basic demographic/outcome columns
            exclude_cols = ['age', 'sex', 'mace', 'Age', 'Sex_F'] + self.organ_list
            protein_cols = [col for col in data.columns if col not in exclude_cols]
            return protein_cols
    
    def preprocess_features(self, data: pd.DataFrame, 
                           feature_list: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply feature-specific preprocessing.
        
        Args:
            data: Input DataFrame
            feature_list: List of features to process
            
        Returns:
            Tuple of (processed_data, preprocessing_info)
        """
        # Separate protein and organ features
        protein_features = [f for f in feature_list if f not in self.organ_list]
        organ_features = [f for f in feature_list if f in self.organ_list]
        
        processed_data = pd.DataFrame(index=data.index)
        preprocessing_info = {
            'protein_features': protein_features,
            'organ_features': organ_features,
            'protein_scaler': None,
            'organ_scaler': None
        }
        
        # Process protein features (log transformation + standardization)
        if protein_features:
            protein_data = data[protein_features].copy()
            
            # Handle missing values
            protein_data = protein_data.fillna(0)
            
            # Log transformation
            protein_data = np.log1p(protein_data.clip(lower=0))
            
            # Standardization
            protein_scaler = StandardScaler()
            protein_scaled = protein_scaler.fit_transform(protein_data)
            
            processed_data[protein_features] = protein_scaled
            preprocessing_info['protein_scaler'] = protein_scaler
        
        # Process organ features (standardization only)
        if organ_features:
            organ_data = data[organ_features].copy()
            
            # Handle missing values
            organ_data = organ_data.fillna(0)
            
            # Standardization
            organ_scaler = StandardScaler()
            organ_scaled = organ_scaler.fit_transform(organ_data)
            
            processed_data[organ_features] = organ_scaled
            preprocessing_info['organ_scaler'] = organ_scaler
        
        return processed_data, preprocessing_info
    
    def apply_preprocessing(self, data: pd.DataFrame, 
                           preprocessing_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply saved preprocessing to new data.
        
        Args:
            data: New data to preprocess
            preprocessing_info: Preprocessing configuration from training
            
        Returns:
            Preprocessed DataFrame
        """
        processed_data = pd.DataFrame(index=data.index)
        
        # Process protein features
        protein_features = preprocessing_info.get('protein_features', [])
        protein_scaler = preprocessing_info.get('protein_scaler')
        
        if protein_features and protein_scaler is not None:
            protein_data = data[protein_features].copy()
            protein_data = protein_data.fillna(0)
            protein_data = np.log1p(protein_data.clip(lower=0))
            protein_scaled = protein_scaler.transform(protein_data)
            processed_data[protein_features] = protein_scaled
        
        # Process organ features
        organ_features = preprocessing_info.get('organ_features', [])
        organ_scaler = preprocessing_info.get('organ_scaler')
        
        if organ_features and organ_scaler is not None:
            organ_data = data[organ_features].copy()
            organ_data = organ_data.fillna(0)
            organ_scaled = organ_scaler.transform(organ_data)
            processed_data[organ_features] = organ_scaled
        
        return processed_data
    
    def split_data(self, data: pd.DataFrame, target_col: str = 'mace',
                   test_size: float = 0.33, stratify_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split.
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            test_size: Proportion of test set
            stratify_cols: Additional columns for stratification
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Create stratification variable
        if stratify_cols is None:
            stratify_cols = ['sex', target_col]
        
        # Create age bins for stratification
        data_copy = data.copy()
        data_copy['age_bin'] = pd.qcut(data_copy['age'], q=4, labels=False, duplicates='drop')
        
        # Create combined stratification variable
        strata_cols = stratify_cols + ['age_bin']
        data_copy['strata'] = data_copy[strata_cols].astype(str).agg('_'.join, axis=1)
        
        # Perform split
        train_df, test_df = train_test_split(
            data_copy,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=data_copy['strata']
        )
        
        # Clean up temporary columns
        train_df = train_df.drop(columns=['age_bin', 'strata'])
        test_df = test_df.drop(columns=['age_bin', 'strata'])
        
        return train_df, test_df
    
    def prepare_full_pipeline(self, data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete data preparation pipeline.
        
        Args:
            data_file: Path to data file
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Load data
        print("Loading data...")
        data = self.load_data(data_file)
        
        # Calculate organ aging features
        print("Calculating organ aging features...")
        data_with_organs = self.organ_calculator.add_organ_features(data)
        
        # Create feature sets
        print("Creating feature sets...")
        feature_sets = self.create_feature_sets(data_with_organs)
        
        # Split data
        print("Splitting data...")
        train_df, test_df = self.split_data(data_with_organs)
        
        # Save split data
        self.save_split_data(train_df, test_df)
        
        print(f"Data preparation complete:")
        print(f"  Total samples: {len(data_with_organs)}")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        print(f"  Feature sets: {list(feature_sets.keys())}")
        
        return {
            'train_data': train_df,
            'test_data': test_df,
            'feature_sets': feature_sets,
            'full_data': data_with_organs
        }
    
    def save_split_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                       train_file: str = "train_data.xlsx", 
                       test_file: str = "test_data.xlsx"):
        """
        Save train/test split to files.
        
        Args:
            train_df: Training data
            test_df: Test data
            train_file: Training data filename
            test_file: Test data filename
        """
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        train_df.to_excel(train_path, index=False)
        test_df.to_excel(test_path, index=False)
        
        print(f"Train data saved to: {train_path}")
        print(f"Test data saved to: {test_path}")
    
    def load_split_data(self, train_file: str = "train_data.xlsx",
                       test_file: str = "test_data.xlsx") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved train/test split.
        
        Args:
            train_file: Training data filename
            test_file: Test data filename
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        train_df = pd.read_excel(train_path)
        test_df = pd.read_excel(test_path)
        
        return train_df, test_df