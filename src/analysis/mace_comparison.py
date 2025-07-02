"""
MACE vs Non-MACE Comparison Analysis Module

This module provides comprehensive comparison analysis between MACE and non-MACE
patient groups across organ aging signatures and protein expression profiles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Any
import warnings


class MACEComparisonAnalyzer:
    """
    Analyzer for comparing MACE vs non-MACE patient groups.
    
    This class provides methods to:
    1. Compare organ aging signatures between groups
    2. Perform statistical testing with multiple comparison correction
    3. Create comprehensive visualizations
    4. Calculate effect sizes and confidence intervals
    """
    
    def __init__(self, output_dir: str = "mace_comparison_results"):
        """
        Initialize the MACE comparison analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        self.organ_list = [
            'Adipose', 'Artery', 'Brain', 'Conventional', 'Heart', 
            'Immune', 'Intestine', 'Kidney', 'Liver', 'Lung', 
            'Muscle', 'Organismal', 'Pancreas'
        ]
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def compare_organ_ages(self, data: pd.DataFrame, 
                          target_col: str = 'mace') -> pd.DataFrame:
        """
        Compare organ aging signatures between MACE and non-MACE groups.
        
        Args:
            data: DataFrame with patient data including MACE status and organ ages
            target_col: Column name for MACE status (0/1)
            
        Returns:
            DataFrame with statistical comparison results
        """
        results = []
        
        # Ensure target column exists
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        for organ in self.organ_list:
            if organ not in data.columns:
                warnings.warn(f"Organ '{organ}' not found in data, skipping")
                continue
            
            # Separate groups
            mace_values = data[data[target_col] == 1][organ].dropna()
            non_mace_values = data[data[target_col] == 0][organ].dropna()
            
            if len(mace_values) == 0 or len(non_mace_values) == 0:
                warnings.warn(f"Insufficient data for organ '{organ}', skipping")
                continue
            
            # Descriptive statistics
            mace_mean = mace_values.mean()
            mace_std = mace_values.std()
            mace_median = mace_values.median()
            
            non_mace_mean = non_mace_values.mean()
            non_mace_std = non_mace_values.std()
            non_mace_median = non_mace_values.median()
            
            mean_diff = mace_mean - non_mace_mean
            
            # Statistical tests
            # T-test (parametric)
            t_stat, t_p_value = stats.ttest_ind(mace_values, non_mace_values, equal_var=False)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, mw_p_value = stats.mannwhitneyu(mace_values, non_mace_values, 
                                                   alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(mace_values) - 1) * mace_std**2 + 
                                 (len(non_mace_values) - 1) * non_mace_std**2) / 
                                (len(mace_values) + len(non_mace_values) - 2))
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for mean difference
            se_diff = np.sqrt(mace_std**2/len(mace_values) + non_mace_std**2/len(non_mace_values))\n            \n            df = len(mace_values) + len(non_mace_values) - 2\n            t_critical = stats.t.ppf(0.975, df)\n            ci_lower = mean_diff - t_critical * se_diff\n            ci_upper = mean_diff + t_critical * se_diff\n            \n            results.append({\n                'Organ': organ,\n                'MACE_N': len(mace_values),\n                'MACE_Mean': mace_mean,\n                'MACE_Std': mace_std,\n                'MACE_Median': mace_median,\n                'NonMACE_N': len(non_mace_values),\n                'NonMACE_Mean': non_mace_mean,\n                'NonMACE_Std': non_mace_std,\n                'NonMACE_Median': non_mace_median,\n                'Mean_Difference': mean_diff,\n                'Cohens_D': cohens_d,\n                'CI_Lower': ci_lower,\n                'CI_Upper': ci_upper,\n                'T_Statistic': t_stat,\n                'T_P_Value': t_p_value,\n                'MW_U_Statistic': u_stat,\n                'MW_P_Value': mw_p_value\n            })\n        \n        # Convert to DataFrame\n        results_df = pd.DataFrame(results)\n        \n        if len(results_df) > 0:\n            # Multiple testing correction\n            _, t_q_values, _, _ = multipletests(results_df['T_P_Value'], method='fdr_bh')\n            _, mw_q_values, _, _ = multipletests(results_df['MW_P_Value'], method='fdr_bh')\n            \n            results_df['T_Q_Value'] = t_q_values\n            results_df['MW_Q_Value'] = mw_q_values\n            results_df['T_Significant'] = results_df['T_Q_Value'] < 0.05\n            results_df['MW_Significant'] = results_df['MW_Q_Value'] < 0.05\n            \n            # Sort by effect size\n            results_df = results_df.sort_values('Cohens_D', key=abs, ascending=False)\n        \n        return results_df\n    \n    def create_comparison_boxplot(self, data: pd.DataFrame, results_df: pd.DataFrame,\n                                target_col: str = 'mace', \n                                save_path: Optional[str] = None) -> plt.Figure:\n        \"\"\"\n        Create boxplot comparing organ ages between MACE and non-MACE groups.\n        \n        Args:\n            data: DataFrame with patient data\n            results_df: Statistical comparison results\n            target_col: Column name for MACE status\n            save_path: Path to save the figure\n            \n        Returns:\n            Matplotlib figure object\n        \"\"\"\n        # Prepare data for plotting\n        plot_organs = [organ for organ in self.organ_list if organ in data.columns]\n        \n        # Melt data for seaborn\n        melted_data = pd.melt(\n            data, \n            id_vars=[target_col], \n            value_vars=plot_organs,\n            var_name='organ', \n            value_name='age_gap'\n        )\n        \n        # Create figure\n        plt.figure(figsize=(10, 8))\n        \n        # Create boxplot\n        ax = sns.boxplot(\n            x='age_gap', y='organ', hue=target_col, \n            data=melted_data,\n            palette=['lightblue', 'lightcoral'],\n            orient='h'\n        )\n        \n        # Add statistical significance annotations\n        self._add_significance_annotations(ax, results_df, melted_data)\n        \n        # Customize plot\n        ax.set_xlabel('Organ Age Gap (Z-score)')\n        ax.set_ylabel('Organ')\n        ax.set_title('Organ Age Gaps: MACE vs Non-MACE Comparison')\n        \n        # Update legend\n        handles, labels = ax.get_legend_handles_labels()\n        ax.legend(handles, ['Non-MACE', 'MACE'], title='Group')\n        \n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            \n        return plt.gcf()\n    \n    def _add_significance_annotations(self, ax: plt.Axes, results_df: pd.DataFrame, \n                                    melted_data: pd.DataFrame):\n        \"\"\"\n        Add significance annotations to boxplot.\n        \n        Args:\n            ax: Matplotlib axis\n            results_df: Statistical results\n            melted_data: Melted data for plotting\n        \"\"\"\n        # Get maximum x values for each organ for annotation placement\n        max_x_values = {}\n        for organ in results_df['Organ']:\n            organ_data = melted_data[melted_data['organ'] == organ]['age_gap']\n            \n            # Calculate outlier threshold\n            q1, q3 = np.percentile(organ_data.dropna(), [25, 75])\n            iqr = q3 - q1\n            upper_bound = q3 + 1.5 * iqr\n            \n            # Find maximum value (including outliers)\n            outliers = organ_data[organ_data > upper_bound]\n            if len(outliers) > 0:\n                max_x_values[organ] = outliers.max()\n            else:\n                max_x_values[organ] = organ_data.max()\n        \n        # Add significance annotations\n        y_positions = {organ: i for i, organ in enumerate(results_df['Organ'])}\n        \n        for _, row in results_df.iterrows():\n            organ = row['Organ']\n            q_value = row['T_Q_Value']\n            \n            # Determine significance level\n            if q_value < 0.001:\n                sig_text = '***'\n            elif q_value < 0.01:\n                sig_text = '**'\n            elif q_value < 0.05:\n                sig_text = '*'\n            else:\n                sig_text = 'ns'\n            \n            # Add annotation\n            if organ in max_x_values and organ in y_positions:\n                x_pos = max_x_values[organ] + 0.1\n                y_pos = y_positions[organ]\n                \n                ax.text(x_pos, y_pos, f'q={q_value:.3f}\\n{sig_text}', \n                       ha='left', va='center', fontsize=8,\n                       bbox=dict(boxstyle=\"round,pad=0.3\", facecolor='white', alpha=0.8))\n    \n    def create_violin_plot(self, data: pd.DataFrame, target_col: str = 'mace',\n                          save_path: Optional[str] = None) -> plt.Figure:\n        \"\"\"\n        Create violin plot for more detailed distribution comparison.\n        \n        Args:\n            data: DataFrame with patient data\n            target_col: Column name for MACE status\n            save_path: Path to save the figure\n            \n        Returns:\n            Matplotlib figure object\n        \"\"\"\n        # Prepare data\n        plot_organs = [organ for organ in self.organ_list if organ in data.columns]\n        \n        melted_data = pd.melt(\n            data, \n            id_vars=[target_col], \n            value_vars=plot_organs,\n            var_name='organ', \n            value_name='age_gap'\n        )\n        \n        # Create figure\n        plt.figure(figsize=(12, 8))\n        \n        # Create violin plot\n        ax = sns.violinplot(\n            x='organ', y='age_gap', hue=target_col,\n            data=melted_data,\n            palette=['lightblue', 'lightcoral'],\n            split=True, inner='quartile'\n        )\n        \n        # Customize plot\n        ax.set_xlabel('Organ')\n        ax.set_ylabel('Organ Age Gap (Z-score)')\n        ax.set_title('Distribution of Organ Age Gaps by MACE Status')\n        plt.xticks(rotation=45, ha='right')\n        \n        # Update legend\n        handles, labels = ax.get_legend_handles_labels()\n        ax.legend(handles, ['Non-MACE', 'MACE'], title='Group')\n        \n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            \n        return plt.gcf()\n    \n    def create_effect_size_plot(self, results_df: pd.DataFrame, \n                               save_path: Optional[str] = None) -> plt.Figure:\n        \"\"\"\n        Create plot showing effect sizes (Cohen's d) for each organ.\n        \n        Args:\n            results_df: Statistical comparison results\n            save_path: Path to save the figure\n            \n        Returns:\n            Matplotlib figure object\n        \"\"\"\n        if len(results_df) == 0:\n            raise ValueError(\"No results to plot\")\n        \n        # Sort by effect size\n        plot_data = results_df.sort_values('Cohens_D', key=abs, ascending=True)\n        \n        # Create figure\n        plt.figure(figsize=(8, 6))\n        \n        # Create horizontal bar plot\n        colors = ['red' if d > 0 else 'blue' for d in plot_data['Cohens_D']]\n        bars = plt.barh(plot_data['Organ'], plot_data['Cohens_D'], color=colors, alpha=0.7)\n        \n        # Add confidence intervals\n        plt.errorbar(plot_data['Mean_Difference'], plot_data['Organ'], \n                    xerr=[plot_data['Mean_Difference'] - plot_data['CI_Lower'],\n                          plot_data['CI_Upper'] - plot_data['Mean_Difference']],\n                    fmt='none', color='black', alpha=0.5, capsize=3)\n        \n        # Add significance markers\n        for i, (_, row) in enumerate(plot_data.iterrows()):\n            if row['T_Significant']:\n                plt.text(row['Cohens_D'] + 0.05 if row['Cohens_D'] > 0 else row['Cohens_D'] - 0.05, \n                        i, '*', ha='center', va='center', fontsize=16, fontweight='bold')\n        \n        # Customize plot\n        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)\n        plt.xlabel(\"Cohen's d (Effect Size)\")\n        plt.ylabel('Organ')\n        plt.title('Effect Sizes: MACE vs Non-MACE Organ Age Differences')\n        plt.grid(axis='x', alpha=0.3)\n        \n        # Add effect size interpretation\n        plt.text(0.02, 0.98, 'Effect Size:\\nSmall: 0.2\\nMedium: 0.5\\nLarge: 0.8', \n                transform=plt.gca().transAxes, verticalalignment='top',\n                bbox=dict(boxstyle=\"round,pad=0.3\", facecolor='white', alpha=0.8))\n        \n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=300, bbox_inches='tight')\n            \n        return plt.gcf()\n    \n    def generate_summary_report(self, results_df: pd.DataFrame) -> str:\n        \"\"\"\n        Generate a text summary of the comparison analysis.\n        \n        Args:\n            results_df: Statistical comparison results\n            \n        Returns:\n            Summary report as string\n        \"\"\"\n        if len(results_df) == 0:\n            return \"No comparison results available.\"\n        \n        report = [\"MACE vs Non-MACE Organ Age Comparison Summary\"]\n        report.append(\"=\" * 50)\n        report.append(\"\")\n        \n        # Overall statistics\n        n_organs = len(results_df)\n        n_significant = results_df['T_Significant'].sum()\n        \n        report.append(f\"Total organs analyzed: {n_organs}\")\n        report.append(f\"Significant differences (q < 0.05): {n_significant} ({n_significant/n_organs*100:.1f}%)\")\n        report.append(\"\")\n        \n        # Significant results\n        if n_significant > 0:\n            report.append(\"Significant Organ Differences:\")\n            report.append(\"-\" * 30)\n            \n            sig_results = results_df[results_df['T_Significant']].sort_values('Cohens_D', key=abs, ascending=False)\n            \n            for _, row in sig_results.iterrows():\n                effect = \"Large\" if abs(row['Cohens_D']) >= 0.8 else \"Medium\" if abs(row['Cohens_D']) >= 0.5 else \"Small\"\n                direction = \"higher\" if row['Mean_Difference'] > 0 else \"lower\"\n                \n                report.append(f\"{row['Organ']}:\")\n                report.append(f\"  Mean difference: {row['Mean_Difference']:.3f} ({direction} in MACE)\")\n                report.append(f\"  Effect size (Cohen's d): {row['Cohens_D']:.3f} ({effect})\")\n                report.append(f\"  q-value: {row['T_Q_Value']:.3e}\")\n                report.append(\"\")\n        \n        # Largest effects (regardless of significance)\n        report.append(\"Largest Effect Sizes:\")\n        report.append(\"-\" * 20)\n        \n        top_effects = results_df.nlargest(3, 'Cohens_D', keep='all')\n        for _, row in top_effects.iterrows():\n            direction = \"higher\" if row['Mean_Difference'] > 0 else \"lower\"\n            sig_status = \"significant\" if row['T_Significant'] else \"not significant\"\n            \n            report.append(f\"{row['Organ']}: d={row['Cohens_D']:.3f} ({direction} in MACE, {sig_status})\")\n        \n        return \"\\n\".join(report)\n    \n    def run_comprehensive_comparison(self, data_path: str, target_col: str = 'mace') -> Dict[str, Any]:\n        \"\"\"\n        Run comprehensive MACE vs non-MACE comparison analysis.\n        \n        Args:\n            data_path: Path to dataset\n            target_col: Column name for MACE status\n            \n        Returns:\n            Dictionary with all analysis results\n        \"\"\"\n        print(\"Starting comprehensive MACE comparison analysis...\")\n        \n        # Load data\n        if data_path.endswith('.xlsx'):\n            data = pd.read_excel(data_path)\n        else:\n            data = pd.read_csv(data_path)\n        \n        # Statistical comparison\n        print(\"Performing statistical comparisons...\")\n        results_df = self.compare_organ_ages(data, target_col)\n        \n        # Save results\n        results_path = f\"{self.output_dir}/organ_age_comparison_statistics.csv\"\n        results_df.to_csv(results_path, index=False)\n        \n        # Create visualizations\n        print(\"Creating visualizations...\")\n        \n        # Boxplot\n        boxplot_fig = self.create_comparison_boxplot(\n            data, results_df, target_col,\n            save_path=f\"{self.output_dir}/organ_age_boxplot_comparison.png\"\n        )\n        \n        # Violin plot\n        violin_fig = self.create_violin_plot(\n            data, target_col,\n            save_path=f\"{self.output_dir}/organ_age_violin_comparison.png\"\n        )\n        \n        # Effect size plot\n        effect_fig = self.create_effect_size_plot(\n            results_df,\n            save_path=f\"{self.output_dir}/organ_age_effect_sizes.png\"\n        )\n        \n        # Generate summary report\n        summary = self.generate_summary_report(results_df)\n        \n        # Save summary\n        summary_path = f\"{self.output_dir}/comparison_summary.txt\"\n        with open(summary_path, 'w') as f:\n            f.write(summary)\n        \n        print(f\"Analysis completed. Results saved to: {self.output_dir}\")\n        \n        return {\n            'statistical_results': results_df,\n            'summary': summary,\n            'figures': {\n                'boxplot': boxplot_fig,\n                'violin': violin_fig,\n                'effect_sizes': effect_fig\n            }\n        }