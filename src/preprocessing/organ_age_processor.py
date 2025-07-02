"""
OrganAge Processing Script

This module provides a standalone script for applying OrganAge calculations
to proteomics data, corresponding to the original 'apply_organage' script.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import warnings

try:
    from organage import OrganAge
except ImportError:
    warnings.warn("OrganAge library not found. Some functionality will be limited.")
    OrganAge = None

from .organ_age_calculator import OrganAgeCalculator


class OrganAgeProcessor:
    """
    Processor for applying OrganAge calculations to proteomics datasets.
    
    This class handles the complete workflow of:
    1. Loading proteomics data (v4.0 and v4.1)
    2. Preparing metadata
    3. Calculating organ aging signatures
    4. Visualizing results
    5. Statistical analysis
    """
    
    def __init__(self, base_dir: str = "/home/dongyeop/organage", 
                 output_dir: str = "results"):
        """
        Initialize the OrganAge processor.
        
        Args:
            base_dir: Base directory for data files
            output_dir: Output directory for results
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.calculator = OrganAgeCalculator()
        
        # Create output directory
        os.makedirs(os.path.join(base_dir, output_dir), exist_ok=True)
    
    def load_proteomics_data(self, version: str = "v4.1") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load proteomics data for specified version.
        
        Args:
            version: Data version ('v4.0' or 'v4.1')
            
        Returns:
            Tuple of (full_data, metadata, protein_data)
        """
        if version == "v4.1":
            file_path = os.path.join(
                self.base_dir, 
                'data/proteomics_v4.1_HCM_CUIMC(with sample ID, with MACE).xlsx'
            )
        elif version == "v4.0":
            file_path = os.path.join(
                self.base_dir,
                'data/proteomics_v4.0_HCM_CUIMC(with sample ID, with MACE).xlsx'
            )
        else:
            raise ValueError(f"Unsupported version: {version}")
        
        # Load data
        data = pd.read_excel(file_path)
        
        # Prepare metadata
        metadata = data[['age', 'sex']].copy()
        metadata['sex'] = metadata['sex'].map({'Male': 0, 'Female': 1})
        metadata.columns = ['Age', 'Sex_F']
        
        # Extract protein data (assuming starts from column 8)
        protein_data = data.iloc[:, 8:]
        
        print(f"Loaded {version} data:")
        print(f"  Total samples: {len(data)}")
        print(f"  Metadata columns: {list(metadata.columns)}")
        print(f"  Protein features: {len(protein_data.columns)}")
        
        return data, metadata, protein_data
    
    def apply_organage_calculation(self, metadata: pd.DataFrame, 
                                 protein_data: pd.DataFrame,
                                 assay_version: str = "v4.1") -> pd.DataFrame:
        """
        Apply OrganAge calculation to the data.
        
        Args:
            metadata: Metadata with Age and Sex_F columns
            protein_data: Protein expression data
            assay_version: SomaScan assay version
            
        Returns:
            DataFrame with organ aging results
        """
        if OrganAge is None:
            raise RuntimeError("OrganAge library not available")
        
        print(f"Calculating organ ages using {assay_version}...")
        
        # Calculate organ ages
        organ_ages = self.calculator.calculate_organ_ages(metadata, protein_data)
        
        print(f"Organ age calculation completed:")
        print(f"  Available organs: {list(organ_ages.columns)}")
        print(f"  Shape: {organ_ages.shape}")
        
        return organ_ages
    
    def create_organ_age_visualizations(self, organ_ages: pd.DataFrame,\n                                      metadata: pd.DataFrame,\n                                      save_plots: bool = True) -> Dict[str, Any]:\n        \"\"\"\n        Create comprehensive visualizations of organ aging results.\n        \n        Args:\n            organ_ages: Organ aging results\n            metadata: Patient metadata\n            save_plots: Whether to save plots to files\n            \n        Returns:\n            Dictionary with plotting results and statistics\n        \"\"\"\n        results = {}\n        \n        # 1. Correlation heatmap between organs\n        plt.figure(figsize=(12, 10))\n        correlation_matrix = organ_ages.corr()\n        \n        # Create custom colormap\n        from matplotlib.colors import LinearSegmentedColormap\n        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']\n        n_bins = 100\n        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)\n        \n        # Plot heatmap\n        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, center=0,\n                   square=True, fmt='.2f', cbar_kws={\"shrink\": .8})\n        plt.title('Organ Age Correlations', fontsize=16)\n        plt.tight_layout()\n        \n        if save_plots:\n            plt.savefig(os.path.join(self.base_dir, self.output_dir, 'organ_age_correlations.png'), \n                       dpi=300, bbox_inches='tight')\n        \n        results['correlation_matrix'] = correlation_matrix\n        \n        # 2. Distribution plots for each organ\n        n_organs = len(organ_ages.columns)\n        n_cols = 4\n        n_rows = (n_organs + n_cols - 1) // n_cols\n        \n        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))\n        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes\n        \n        for i, organ in enumerate(organ_ages.columns):\n            if i < len(axes):\n                ax = axes[i]\n                \n                # Plot distribution\n                organ_values = organ_ages[organ].dropna()\n                ax.hist(organ_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')\n                ax.axvline(organ_values.mean(), color='red', linestyle='--', \n                          label=f'Mean: {organ_values.mean():.2f}')\n                ax.axvline(organ_values.median(), color='orange', linestyle='--',\n                          label=f'Median: {organ_values.median():.2f}')\n                \n                ax.set_title(f'{organ} Age Gap Distribution')\n                ax.set_xlabel('Age Gap (Z-score)')\n                ax.set_ylabel('Frequency')\n                ax.legend()\n                ax.grid(True, alpha=0.3)\n        \n        # Hide unused subplots\n        for i in range(n_organs, len(axes)):\n            axes[i].set_visible(False)\n        \n        plt.tight_layout()\n        \n        if save_plots:\n            plt.savefig(os.path.join(self.base_dir, self.output_dir, 'organ_age_distributions.png'),\n                       dpi=300, bbox_inches='tight')\n        \n        # 3. Age vs Organ Age scatter plots\n        if 'Age' in metadata.columns:\n            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))\n            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes\n            \n            for i, organ in enumerate(organ_ages.columns):\n                if i < len(axes):\n                    ax = axes[i]\n                    \n                    # Scatter plot\n                    ax.scatter(metadata['Age'], organ_ages[organ], alpha=0.6, s=20)\n                    \n                    # Calculate correlation\n                    valid_idx = organ_ages[organ].notna() & metadata['Age'].notna()\n                    if valid_idx.sum() > 10:\n                        from scipy.stats import pearsonr\n                        corr, p_value = pearsonr(metadata.loc[valid_idx, 'Age'], \n                                               organ_ages.loc[valid_idx, organ])\n                        \n                        # Add regression line\n                        z = np.polyfit(metadata.loc[valid_idx, 'Age'], \n                                      organ_ages.loc[valid_idx, organ], 1)\n                        p = np.poly1d(z)\n                        ax.plot(metadata['Age'], p(metadata['Age']), \"r--\", alpha=0.8)\n                        \n                        ax.set_title(f'{organ} vs Chronological Age\\nr={corr:.3f}, p={p_value:.3e}')\n                    else:\n                        ax.set_title(f'{organ} vs Chronological Age')\n                    \n                    ax.set_xlabel('Chronological Age')\n                    ax.set_ylabel(f'{organ} Age Gap')\n                    ax.grid(True, alpha=0.3)\n            \n            # Hide unused subplots\n            for i in range(n_organs, len(axes)):\n                axes[i].set_visible(False)\n            \n            plt.tight_layout()\n            \n            if save_plots:\n                plt.savefig(os.path.join(self.base_dir, self.output_dir, 'organ_age_vs_chronological_age.png'),\n                           dpi=300, bbox_inches='tight')\n        \n        # 4. Summary statistics\n        summary_stats = organ_ages.describe()\n        results['summary_statistics'] = summary_stats\n        \n        if save_plots:\n            summary_stats.to_csv(os.path.join(self.base_dir, self.output_dir, 'organ_age_summary_statistics.csv'))\n        \n        print(\"Visualization completed.\")\n        return results\n    \n    def perform_statistical_analysis(self, organ_ages: pd.DataFrame,\n                                   metadata: pd.DataFrame,\n                                   clinical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:\n        \"\"\"\n        Perform statistical analysis on organ aging results.\n        \n        Args:\n            organ_ages: Organ aging results\n            metadata: Patient metadata\n            clinical_data: Additional clinical data (optional)\n            \n        Returns:\n            Dictionary with statistical analysis results\n        \"\"\"\n        from scipy.stats import pearsonr, spearmanr\n        from statsmodels.stats.multitest import multipletests\n        \n        results = {}\n        \n        # 1. Age and sex associations\n        age_correlations = []\n        for organ in organ_ages.columns:\n            if metadata is not None and 'Age' in metadata.columns:\n                valid_idx = organ_ages[organ].notna() & metadata['Age'].notna()\n                if valid_idx.sum() > 10:\n                    corr, p_value = pearsonr(metadata.loc[valid_idx, 'Age'],\n                                           organ_ages.loc[valid_idx, organ])\n                    age_correlations.append({\n                        'Organ': organ,\n                        'Correlation': corr,\n                        'P_Value': p_value,\n                        'N_Samples': valid_idx.sum()\n                    })\n        \n        if age_correlations:\n            age_corr_df = pd.DataFrame(age_correlations)\n            \n            # Multiple testing correction\n            _, q_values, _, _ = multipletests(age_corr_df['P_Value'], method='fdr_bh')\n            age_corr_df['Q_Value'] = q_values\n            age_corr_df['Significant'] = age_corr_df['Q_Value'] < 0.05\n            \n            results['age_correlations'] = age_corr_df\n        \n        # 2. Sex differences\n        if metadata is not None and 'Sex_F' in metadata.columns:\n            from scipy.stats import ttest_ind\n            \n            sex_differences = []\n            for organ in organ_ages.columns:\n                male_values = organ_ages.loc[metadata['Sex_F'] == 0, organ].dropna()\n                female_values = organ_ages.loc[metadata['Sex_F'] == 1, organ].dropna()\n                \n                if len(male_values) > 5 and len(female_values) > 5:\n                    t_stat, p_value = ttest_ind(male_values, female_values)\n                    \n                    sex_differences.append({\n                        'Organ': organ,\n                        'Male_Mean': male_values.mean(),\n                        'Male_Std': male_values.std(),\n                        'Female_Mean': female_values.mean(),\n                        'Female_Std': female_values.std(),\n                        'Mean_Difference': female_values.mean() - male_values.mean(),\n                        'T_Statistic': t_stat,\n                        'P_Value': p_value,\n                        'N_Male': len(male_values),\n                        'N_Female': len(female_values)\n                    })\n            \n            if sex_differences:\n                sex_diff_df = pd.DataFrame(sex_differences)\n                \n                # Multiple testing correction\n                _, q_values, _, _ = multipletests(sex_diff_df['P_Value'], method='fdr_bh')\n                sex_diff_df['Q_Value'] = q_values\n                sex_diff_df['Significant'] = sex_diff_df['Q_Value'] < 0.05\n                \n                results['sex_differences'] = sex_diff_df\n        \n        return results\n    \n    def run_complete_analysis(self, version: str = \"v4.1\", \n                            save_results: bool = True) -> Dict[str, Any]:\n        \"\"\"\n        Run the complete OrganAge analysis pipeline.\n        \n        Args:\n            version: Data version to analyze\n            save_results: Whether to save results to files\n            \n        Returns:\n            Dictionary with all analysis results\n        \"\"\"\n        print(f\"Starting complete OrganAge analysis for {version}...\")\n        \n        # 1. Load data\n        full_data, metadata, protein_data = self.load_proteomics_data(version)\n        \n        # 2. Calculate organ ages\n        organ_ages = self.apply_organage_calculation(metadata, protein_data, version)\n        \n        # 3. Combine with original data\n        combined_data = pd.concat([full_data, metadata, organ_ages], axis=1)\n        \n        # 4. Create visualizations\n        viz_results = self.create_organ_age_visualizations(organ_ages, metadata, save_results)\n        \n        # 5. Perform statistical analysis\n        stat_results = self.perform_statistical_analysis(organ_ages, metadata)\n        \n        # 6. Save combined results\n        if save_results:\n            output_path = os.path.join(self.base_dir, self.output_dir, f'{version}', 'md_hot.xlsx')\n            os.makedirs(os.path.dirname(output_path), exist_ok=True)\n            combined_data.to_excel(output_path)\n            print(f\"Combined data saved to: {output_path}\")\n            \n            # Save statistical results\n            if 'age_correlations' in stat_results:\n                stat_results['age_correlations'].to_csv(\n                    os.path.join(self.base_dir, self.output_dir, f'{version}', 'age_correlations.csv'),\n                    index=False\n                )\n            \n            if 'sex_differences' in stat_results:\n                stat_results['sex_differences'].to_csv(\n                    os.path.join(self.base_dir, self.output_dir, f'{version}', 'sex_differences.csv'),\n                    index=False\n                )\n        \n        print(\"Complete OrganAge analysis finished.\")\n        \n        return {\n            'full_data': full_data,\n            'organ_ages': organ_ages,\n            'combined_data': combined_data,\n            'visualizations': viz_results,\n            'statistics': stat_results\n        }