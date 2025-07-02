"""
Adiponectin Comprehensive Analysis Module

This module provides comprehensive analysis of adiponectin and related adipokines
in the context of MACE prediction, including correlations with organ aging and
protein expression profiles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json

from ..utils.common import setup_logging
from ..visualization.network_plots import NetworkVisualizer


class AdiponectinAnalyzer:
    """
    Comprehensive adiponectin analysis for MACE prediction studies.
    
    This class provides methods to analyze:
    1. Adipokine differences between MACE vs non-MACE groups
    2. Heart organ age matching analysis for adipokines
    3. Adiponectin correlations with other adipokines
    4. Adiponectin correlations with organ ages
    5. Adiponectin correlations with organ-specific proteins
    6. Adiponectin correlations with HCM-related proteins
    7. PPI network analysis and visualization
    """
    
    def __init__(self, output_dir: str = "adiponectin_analysis_results"):
        """
        Initialize the adiponectin analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        self.setup_directories()
        
        # Define organ age columns
        self.organ_columns = [
            'Adipose', 'Artery', 'Brain', 'Conventional', 'Heart', 
            'Immune', 'Intestine', 'Kidney', 'Liver', 'Lung', 
            'Muscle', 'Organismal', 'Pancreas'
        ]
        
    def setup_directories(self):
        """Create output directories for organized results."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            'scatterplots',
            'scatterplots/adipokine_correlations',
            'scatterplots/organ_age_correlations', 
            'scatterplots/organ_protein_correlations',
            'scatterplots/hcm_protein_correlations',
            'networks',
            'tables'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def load_adipokine_list(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load adipokine list from Excel file.
        
        Args:
            file_path: Path to adipokine list file
            
        Returns:
            DataFrame with adipokine information or None if not found
        """
        try:
            adipokine_df = pd.read_excel(file_path)
            print(f"Loaded adipokine list: {adipokine_df.shape}")
            return adipokine_df
        except FileNotFoundError:
            warnings.warn(f"Adipokine list not found at {file_path}")
            return None
    
    def load_hcm_proteins(self, file_path: str) -> pd.DataFrame:
        """
        Load HCM-related proteins from CTD database.
        
        Args:
            file_path: Path to HCM proteins file
            
        Returns:
            DataFrame with HCM protein information
        """
        try:
            hcm_proteins = pd.read_csv(file_path)
            print(f"Loaded HCM proteins: {hcm_proteins.shape}")
            return hcm_proteins
        except FileNotFoundError:
            warnings.warn(f"HCM proteins file not found at {file_path}")
            return pd.DataFrame()
    
    def load_tissue_protein_mapping(self, file_path: str) -> Dict[str, List[str]]:
        """
        Load tissue-specific protein mapping.
        
        Args:
            file_path: Path to tissue protein mapping JSON file
            
        Returns:
            Dictionary mapping tissues to protein lists
        """
        try:
            with open(file_path, 'r') as f:
                tissue_mapping = json.load(f)
            print(f"Loaded tissue protein mapping for {len(tissue_mapping)} organs")
            return tissue_mapping
        except FileNotFoundError:
            warnings.warn(f"Tissue protein mapping not found at {file_path}")
            return {}
    
    def analyze_adipokine_mace_differences(self, data: pd.DataFrame, 
                                         adipokine_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze adipokine expression differences between MACE and non-MACE groups.
        
        Args:
            data: DataFrame with patient data including MACE status
            adipokine_list: List of adipokine column names
            
        Returns:
            DataFrame with statistical comparison results
        """
        if adipokine_list is None:
            # Identify adipokine columns heuristically
            adipokine_list = self._identify_adipokine_columns(data)
        
        results = []
        
        # Group data by MACE status
        mace_group = data[data['mace'] == 1]
        non_mace_group = data[data['mace'] == 0]
        
        for adipokine in adipokine_list:
            if adipokine not in data.columns:
                continue
                
            # Extract values for each group
            mace_values = mace_group[adipokine].dropna()
            non_mace_values = non_mace_group[adipokine].dropna()
            
            if len(mace_values) == 0 or len(non_mace_values) == 0:
                continue
            
            # Perform statistical tests
            t_stat, p_value = stats.ttest_ind(mace_values, non_mace_values, equal_var=False)
            u_stat, p_value_mw = stats.mannwhitneyu(mace_values, non_mace_values, alternative='two-sided')
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(mace_values) - 1) * mace_values.var() + 
                                 (len(non_mace_values) - 1) * non_mace_values.var()) / 
                                (len(mace_values) + len(non_mace_values) - 2))
            cohens_d = (mace_values.mean() - non_mace_values.mean()) / pooled_std
            
            results.append({
                'Adipokine': adipokine,
                'MACE_Mean': mace_values.mean(),
                'MACE_Std': mace_values.std(),
                'NonMACE_Mean': non_mace_values.mean(),
                'NonMACE_Std': non_mace_values.std(),
                'Mean_Difference': mace_values.mean() - non_mace_values.mean(),
                'Cohens_D': cohens_d,
                'T_Statistic': t_stat,
                'P_Value_TTest': p_value,
                'P_Value_MannWhitney': p_value_mw,
                'N_MACE': len(mace_values),
                'N_NonMACE': len(non_mace_values)
            })
        
        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction
        if len(results_df) > 0:
            from statsmodels.stats.multitest import multipletests
            _, q_values_t, _, _ = multipletests(results_df['P_Value_TTest'], method='fdr_bh')
            _, q_values_mw, _, _ = multipletests(results_df['P_Value_MannWhitney'], method='fdr_bh')
            
            results_df['Q_Value_TTest'] = q_values_t
            results_df['Q_Value_MannWhitney'] = q_values_mw
            results_df['Significant_TTest'] = results_df['Q_Value_TTest'] < 0.05
            results_df['Significant_MannWhitney'] = results_df['Q_Value_MannWhitney'] < 0.05
        
        # Sort by effect size
        results_df = results_df.sort_values('Cohens_D', key=abs, ascending=False)
        
        # Save results
        output_path = os.path.join(self.output_dir, 'adipokine_mace_differences.csv')
        results_df.to_csv(output_path, index=False)
        
        return results_df
    
    def _identify_adipokine_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Heuristically identify adipokine columns in the dataset.
        
        Args:
            data: DataFrame to search
            
        Returns:
            List of potential adipokine column names
        """
        # Common adipokine names and patterns
        adipokine_patterns = [
            'adiponectin', 'leptin', 'resistin', 'visfatin', 'omentin',
            'chemerin', 'apelin', 'vaspin', 'retinol', 'tnf', 'il6',
            'adipoq', 'retn', 'nampt'
        ]
        
        adipokine_cols = []
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in adipokine_patterns):
                adipokine_cols.append(col)
        
        return adipokine_cols
    
    def analyze_adiponectin_correlations(self, data: pd.DataFrame, 
                                       adiponectin_col: str = 'adiponectin') -> Dict[str, pd.DataFrame]:
        """
        Analyze adiponectin correlations with various molecular features.
        
        Args:
            data: DataFrame with patient data
            adiponectin_col: Column name for adiponectin
            
        Returns:
            Dictionary with correlation results for different feature types
        """
        results = {}
        
        if adiponectin_col not in data.columns:
            # Try to find adiponectin column
            potential_cols = [col for col in data.columns if 'adiponectin' in col.lower()]
            if potential_cols:
                adiponectin_col = potential_cols[0]
            else:
                raise ValueError("Adiponectin column not found in data")
        
        adiponectin_values = data[adiponectin_col].dropna()
        
        # 1. Correlations with organ ages
        organ_correlations = []
        for organ in self.organ_columns:
            if organ in data.columns:
                organ_values = data.loc[adiponectin_values.index, organ].dropna()
                
                if len(organ_values) > 10:  # Minimum sample size
                    corr_coef, p_value = pearsonr(
                        adiponectin_values.loc[organ_values.index], 
                        organ_values
                    )
                    
                    organ_correlations.append({
                        'Organ': organ,
                        'Correlation': corr_coef,
                        'P_Value': p_value,
                        'N_Samples': len(organ_values)
                    })
        
        if organ_correlations:
            organ_df = pd.DataFrame(organ_correlations)
            # Multiple testing correction
            from statsmodels.stats.multitest import multipletests
            _, q_values, _, _ = multipletests(organ_df['P_Value'], method='fdr_bh')
            organ_df['Q_Value'] = q_values
            organ_df['Significant'] = organ_df['Q_Value'] < 0.05
            
            results['organ_correlations'] = organ_df
        
        # 2. Correlations with other adipokines
        adipokine_cols = self._identify_adipokine_columns(data)
        adipokine_cols = [col for col in adipokine_cols if col != adiponectin_col]
        
        adipokine_correlations = []
        for adipokine in adipokine_cols:
            if adipokine in data.columns:
                adipokine_vals = data.loc[adiponectin_values.index, adipokine].dropna()
                
                if len(adipokine_vals) > 10:
                    corr_coef, p_value = pearsonr(
                        adiponectin_values.loc[adipokine_vals.index], 
                        adipokine_vals
                    )
                    
                    adipokine_correlations.append({
                        'Adipokine': adipokine,
                        'Correlation': corr_coef,
                        'P_Value': p_value,
                        'N_Samples': len(adipokine_vals)
                    })
        
        if adipokine_correlations:
            adipokine_df = pd.DataFrame(adipokine_correlations)
            # Multiple testing correction
            _, q_values, _, _ = multipletests(adipokine_df['P_Value'], method='fdr_bh')
            adipokine_df['Q_Value'] = q_values
            adipokine_df['Significant'] = adipokine_df['Q_Value'] < 0.05
            
            results['adipokine_correlations'] = adipokine_df
        
        # Save results
        for result_type, result_df in results.items():
            output_path = os.path.join(self.output_dir, f'adiponectin_{result_type}.csv')
            result_df.to_csv(output_path, index=False)
        
        return results
    
    def create_correlation_plots(self, data: pd.DataFrame, correlations: Dict[str, pd.DataFrame],
                               adiponectin_col: str, top_n: int = 10):
        """
        Create correlation scatter plots for top correlated features.
        
        Args:
            data: DataFrame with patient data
            correlations: Dictionary with correlation results
            adiponectin_col: Adiponectin column name
            top_n: Number of top correlations to plot
        """
        for result_type, result_df in correlations.items():
            if len(result_df) == 0:
                continue
                
            # Get top correlations
            top_correlations = result_df.nlargest(top_n, 'Correlation', keep='all')
            
            # Create plots
            n_plots = len(top_correlations)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_plots == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, (_, row) in enumerate(top_correlations.iterrows()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Get feature name
                if result_type == 'organ_correlations':
                    feature_col = row['Organ']
                    feature_name = f"Organ: {feature_col}"
                elif result_type == 'adipokine_correlations':
                    feature_col = row['Adipokine']
                    feature_name = f"Adipokine: {feature_col}"
                else:
                    continue
                
                # Create scatter plot
                if feature_col in data.columns:
                    x_data = data[adiponectin_col].dropna()
                    y_data = data.loc[x_data.index, feature_col].dropna()
                    
                    if len(y_data) > 0:
                        x_plot = x_data.loc[y_data.index]
                        
                        ax.scatter(x_plot, y_data, alpha=0.6, s=20)
                        
                        # Add regression line
                        z = np.polyfit(x_plot, y_data, 1)
                        p = np.poly1d(z)
                        ax.plot(x_plot, p(x_plot), "r--", alpha=0.8)
                        
                        # Labels and title
                        ax.set_xlabel(f'Adiponectin')
                        ax.set_ylabel(feature_name)
                        ax.set_title(f'r = {row["Correlation"]:.3f}, p = {row["P_Value"]:.3e}')
                        ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(
                self.output_dir, 'scatterplots', 
                f'adiponectin_{result_type}_top{top_n}.png'
            )
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_comprehensive_analysis(self, data_path: str, 
                                 adipokine_list_path: Optional[str] = None,
                                 hcm_proteins_path: Optional[str] = None,
                                 tissue_mapping_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive adiponectin analysis.
        
        Args:
            data_path: Path to main dataset
            adipokine_list_path: Path to adipokine list (optional)
            hcm_proteins_path: Path to HCM proteins (optional)
            tissue_mapping_path: Path to tissue protein mapping (optional)
            
        Returns:
            Dictionary with all analysis results
        """
        print("Starting comprehensive adiponectin analysis...")
        
        # Load data
        data = pd.read_excel(data_path)
        
        # Load additional data if provided
        adipokine_df = None
        if adipokine_list_path:
            adipokine_df = self.load_adipokine_list(adipokine_list_path)
        
        # Extract adipokine list
        if adipokine_df is not None:
            adipokine_list = adipokine_df['Protein_ID'].tolist() if 'Protein_ID' in adipokine_df.columns else None
        else:
            adipokine_list = None
        
        results = {}
        
        # 1. Adipokine MACE differences
        print("Analyzing adipokine differences between MACE groups...")
        mace_diff_results = self.analyze_adipokine_mace_differences(data, adipokine_list)
        results['mace_differences'] = mace_diff_results
        
        # 2. Adiponectin correlations
        print("Analyzing adiponectin correlations...")
        correlation_results = self.analyze_adiponectin_correlations(data)
        results['correlations'] = correlation_results
        
        # 3. Create correlation plots
        print("Creating correlation plots...")
        adiponectin_col = 'adiponectin'  # Adjust based on your data
        potential_adiponectin_cols = [col for col in data.columns if 'adiponectin' in col.lower()]
        if potential_adiponectin_cols:
            adiponectin_col = potential_adiponectin_cols[0]
            self.create_correlation_plots(data, correlation_results, adiponectin_col)
        
        # Save summary
        summary_data = {
            'Analysis Date': pd.Timestamp.now().isoformat(),
            'Dataset Shape': data.shape,
            'MACE Patients': data['mace'].sum() if 'mace' in data.columns else 'Unknown',
            'Adipokines Analyzed': len(mace_diff_results) if not mace_diff_results.empty else 0,
            'Significant MACE Differences': mace_diff_results['Significant_TTest'].sum() if not mace_diff_results.empty else 0
        }
        
        summary_path = os.path.join(self.output_dir, 'analysis_summary.csv')
        pd.DataFrame([summary_data]).to_csv(summary_path, index=False)
        
        print(f"Analysis completed. Results saved to: {self.output_dir}")
        return results