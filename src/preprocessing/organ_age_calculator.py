"""
Organ Age Calculator Module

This module provides functionality for calculating organ aging signatures
using the OrganAge library and integrating them with patient data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings
import contextlib
import io

try:
    from organage import OrganAge
except ImportError:
    warnings.warn("OrganAge library not found. Organ age calculations will be disabled.")
    OrganAge = None


class OrganAgeCalculator:
    """
    Calculator for organ aging signatures using the OrganAge library.
    
    This class provides methods to compute organ aging features from protein
    expression data and integrate them with patient demographic information.
    """
    
    def __init__(self, assay_version: str = "v4.1", suppress_output: bool = True):
        """
        Initialize the organ age calculator.
        
        Args:
            assay_version: SomaScan assay version for normalization
            suppress_output: Whether to suppress OrganAge output messages
        """
        self.assay_version = assay_version
        self.suppress_output = suppress_output
        
        if OrganAge is None:
            warnings.warn("OrganAge library not available. Some functionality will be limited.")
    
    def calculate_organ_ages(self, metadata: pd.DataFrame, 
                           protein_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate organ aging signatures from protein expression data.
        
        Args:
            metadata: DataFrame with columns ['Age', 'Sex_F'] (age and sex as 0/1)
            protein_data: DataFrame with protein expression values
            
        Returns:
            DataFrame with organ aging signatures (AgeGap_zscored)
            
        Raises:
            RuntimeError: If OrganAge calculation fails
            ValueError: If input data is invalid
        """
        if OrganAge is None:
            raise RuntimeError("OrganAge library not available")
        
        # Validate inputs
        self._validate_inputs(metadata, protein_data)
        
        # Calculate organ ages
        try:
            if self.suppress_output:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = self._run_organ_age_calculation(metadata, protein_data)
            else:
                result = self._run_organ_age_calculation(metadata, protein_data)
                
            return result
            
        except Exception as e:
            raise RuntimeError(f"OrganAge calculation failed: {e}")
    
    def _validate_inputs(self, metadata: pd.DataFrame, protein_data: pd.DataFrame):
        """
        Validate input data for organ age calculation.
        
        Args:
            metadata: Metadata DataFrame
            protein_data: Protein expression DataFrame
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check metadata columns
        required_meta_cols = ['Age', 'Sex_F']
        missing_meta = [col for col in required_meta_cols if col not in metadata.columns]
        if missing_meta:
            raise ValueError(f"Missing metadata columns: {missing_meta}")
        
        # Check data alignment
        if len(metadata) != len(protein_data):
            raise ValueError("Metadata and protein data must have same number of samples")
        
        # Check for missing values in critical columns
        if metadata['Age'].isna().any():
            raise ValueError("Missing values in Age column")
        
        if metadata['Sex_F'].isna().any():
            raise ValueError("Missing values in Sex_F column")
        
        # Validate age range
        age_range = (metadata['Age'].min(), metadata['Age'].max())
        if age_range[0] < 0 or age_range[1] > 120:
            warnings.warn(f"Unusual age range detected: {age_range}")
        
        # Validate sex values
        unique_sex = metadata['Sex_F'].unique()
        if not all(val in [0, 1] for val in unique_sex):
            raise ValueError("Sex_F must contain only 0 (male) and 1 (female)")
    
    def _run_organ_age_calculation(self, metadata: pd.DataFrame, 
                                 protein_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the actual OrganAge calculation.
        
        Args:
            metadata: Validated metadata
            protein_data: Validated protein data
            
        Returns:
            DataFrame with organ aging results
        """
        # Create OrganAge object
        organ_data = OrganAge.CreateOrganAgeObject()
        
        # Add data
        organ_data.add_data(metadata, protein_data)
        
        # Normalize
        organ_data.normalize(assay_version=self.assay_version)
        
        # Estimate organ ages
        results = organ_data.estimate_organ_ages()
        
        # Reshape results
        gap_df = results.reset_index().pivot_table(
            index='index', columns='Organ', values='AgeGap_zscored'
        )
        
        return gap_df
    
    def add_organ_features(self, data: pd.DataFrame, 
                          protein_start_col: int = 8) -> pd.DataFrame:
        """
        Add organ aging features to an existing dataset.
        
        Args:
            data: DataFrame with age, sex, and protein columns
            protein_start_col: Column index where protein data starts
            
        Returns:
            DataFrame with added organ aging features
        """
        # Prepare metadata
        metadata = data[['age', 'sex']].copy()
        metadata.columns = ['Age', 'Sex_F']
        
        # Extract protein data
        protein_data = data.iloc[:, protein_start_col:]
        
        # Calculate organ ages
        organ_ages = self.calculate_organ_ages(metadata, protein_data)
        
        # Combine with original data
        result = pd.concat([data, metadata, organ_ages], axis=1)
        
        return result
    
    def get_available_organs(self, data: pd.DataFrame) -> list:
        """
        Get list of organ features available in the data.
        
        Args:
            data: DataFrame potentially containing organ features
            
        Returns:
            List of available organ feature names
        """
        standard_organs = [
            'Adipose', 'Artery', 'Brain', 'Conventional', 'Heart', 
            'Immune', 'Intestine', 'Kidney', 'Liver', 'Lung', 
            'Muscle', 'Organismal', 'Pancreas'
        ]
        
        available_organs = [organ for organ in standard_organs if organ in data.columns]
        return available_organs
    
    def validate_organ_features(self, data: pd.DataFrame, 
                              required_organs: Optional[list] = None) -> Tuple[bool, list]:
        """
        Validate that required organ features are present in data.
        
        Args:
            data: DataFrame to check
            required_organs: List of required organ names. If None, checks for any organs.
            
        Returns:
            Tuple of (all_present, missing_organs)
        """
        if required_organs is None:
            required_organs = [
                'Adipose', 'Artery', 'Brain', 'Heart', 'Immune', 
                'Intestine', 'Kidney', 'Liver', 'Lung', 'Muscle', 'Pancreas'
            ]
        
        missing_organs = [organ for organ in required_organs if organ not in data.columns]
        all_present = len(missing_organs) == 0
        
        return all_present, missing_organs
    
    def create_organ_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics for organ aging features.
        
        Args:
            data: DataFrame with organ aging features
            
        Returns:
            DataFrame with summary statistics for each organ
        """
        available_organs = self.get_available_organs(data)
        
        if not available_organs:
            return pd.DataFrame()
        
        organ_data = data[available_organs]
        
        summary = pd.DataFrame({
            'Organ': available_organs,
            'Mean': organ_data.mean(),
            'Std': organ_data.std(),
            'Min': organ_data.min(),
            'Max': organ_data.max(),
            'Missing_Count': organ_data.isna().sum(),
            'Missing_Percent': (organ_data.isna().sum() / len(organ_data)) * 100
        })
        
        return summary.reset_index(drop=True)