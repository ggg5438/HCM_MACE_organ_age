"""
Results Visualization Module

This module provides comprehensive visualization tools for model performance
results, cross-validation metrics, and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
import warnings


class ResultsVisualizer:
    """
    Comprehensive visualization tools for MACE prediction results.
    
    This class provides methods to create publication-ready plots for:
    - Cross-validation performance metrics
    - Bootstrap confidence intervals
    - Model comparison charts
    - Feature importance rankings
    """
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'husl',
                 figsize: Tuple[int, int] = (10, 6), dpi: int = 300):
        """
        Initialize the results visualizer.
        
        Args:
            style: Seaborn style theme
            palette: Color palette for plots
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.style = style
        self.palette = palette
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        sns.set_style(self.style)
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
    
    def plot_cv_performance(self, results_df: pd.DataFrame,
                           metrics: List[str] = ['f1', 'auc', 'precision', 'recall'],
                           error_type: str = 'ci',
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cross-validation performance with error bars.
        
        Args:
            results_df: DataFrame with CV results
            metrics: List of metrics to plot
            error_type: Type of error bars ('ci' for confidence interval, 'std' for standard deviation)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Prepare data
        plot_data = self._prepare_cv_data(results_df, metrics, error_type)
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            self._plot_single_metric_cv(ax, plot_data, metric, error_type)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _prepare_cv_data(self, results_df: pd.DataFrame, metrics: List[str],
                        error_type: str) -> pd.DataFrame:
        """
        Prepare cross-validation data for plotting.
        
        Args:
            results_df: Raw CV results
            metrics: Metrics to include
            error_type: Error bar type
            
        Returns:
            Prepared DataFrame for plotting
        """
        # Get best results for each model-feature combination
        if 'rank_test_f1' in results_df.columns:
            group_cols = ['FeatureSet', 'Model']
            best_idx = results_df.groupby(group_cols)['rank_test_f1'].idxmin()
            plot_data = results_df.loc[best_idx].copy()
        else:
            plot_data = results_df.copy()
        
        return plot_data
    
    def _plot_single_metric_cv(self, ax: plt.Axes, data: pd.DataFrame,
                              metric: str, error_type: str):
        """
        Plot a single metric with error bars.
        
        Args:
            ax: Matplotlib axis
            data: Plotting data
            metric: Metric name
            error_type: Error bar type
        """
        mean_col = f'mean_test_{metric}'
        
        if error_type == 'ci':
            lower_col = f'test_{metric}_lowerCI'
            upper_col = f'test_{metric}_upperCI'
            error_label = '95% CI'
        else:
            std_col = f'std_test_{metric}'
            error_label = 'Standard Deviation'
        
        # Check if required columns exist
        if mean_col not in data.columns:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric.upper()} (No Data)')
            return
        
        # Group data
        feature_sets = data['FeatureSet'].unique() if 'FeatureSet' in data.columns else ['Default']
        models = data['Model'].unique() if 'Model' in data.columns else ['Default']
        
        x_pos = np.arange(len(feature_sets))
        bar_width = 0.8 / len(models)
        
        # Plot bars for each model
        for i, model in enumerate(models):
            if 'Model' in data.columns:
                model_data = data[data['Model'] == model]
            else:
                model_data = data
            
            # Calculate values and errors
            y_values = []
            y_errors = []
            
            for fs in feature_sets:
                if 'FeatureSet' in data.columns:
                    fs_data = model_data[model_data['FeatureSet'] == fs]
                else:
                    fs_data = model_data
                
                if len(fs_data) > 0:
                    y_val = fs_data[mean_col].iloc[0]
                    
                    if error_type == 'ci' and lower_col in data.columns and upper_col in data.columns:
                        lower = fs_data[lower_col].iloc[0]
                        upper = fs_data[upper_col].iloc[0]
                        y_err = [[y_val - lower], [upper - y_val]]
                    elif error_type == 'std' and std_col in data.columns:
                        std_val = fs_data[std_col].iloc[0]
                        y_err = std_val
                    else:
                        y_err = 0
                else:
                    y_val = 0
                    y_err = 0
                
                y_values.append(y_val)
                y_errors.append(y_err)
            
            # Plot bars
            x_positions = x_pos + (i - len(models)/2 + 0.5) * bar_width
            bars = ax.bar(x_positions, y_values, bar_width, 
                         label=model, alpha=0.8, capsize=4)
            
            # Add error bars
            if error_type == 'ci' and all(isinstance(err, list) for err in y_errors):
                error_array = np.array(y_errors).T
                ax.errorbar(x_positions, y_values, yerr=error_array, 
                           fmt='none', color='black', capsize=4)
            elif error_type == 'std':
                ax.errorbar(x_positions, y_values, yerr=y_errors, 
                           fmt='none', color='black', capsize=4)
        
        # Customize plot
        ax.set_xlabel('Feature Set')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'{metric.upper()} Performance ({error_label})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_sets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    def plot_bootstrap_results(self, results_df: pd.DataFrame,
                              metrics: List[str] = ['F1', 'AUC', 'Precision', 'Recall'],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot bootstrap test results with confidence intervals.
        
        Args:
            results_df: DataFrame with bootstrap results
            metrics: List of metrics to plot
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            self._plot_bootstrap_metric(ax, results_df, metric)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_bootstrap_metric(self, ax: plt.Axes, data: pd.DataFrame, metric: str):
        """
        Plot bootstrap results for a single metric.
        
        Args:
            ax: Matplotlib axis
            data: Bootstrap results data
            metric: Metric name
        """
        mean_col = f'Test{metric}_BootMean'
        lower_col = f'Test{metric}_Boot95CI_Lower'
        upper_col = f'Test{metric}_Boot95CI_Upper'
        
        # Check if columns exist
        if mean_col not in data.columns:
            ax.text(0.5, 0.5, f'No bootstrap data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Test {metric} (No Data)')
            return
        
        # Prepare data
        feature_sets = data['FeatureSet'].unique() if 'FeatureSet' in data.columns else ['Default']
        models = data['Model'].unique() if 'Model' in data.columns else ['Default']
        
        x_pos = np.arange(len(feature_sets))
        bar_width = 0.8 / len(models)
        
        # Plot bars for each model
        for i, model in enumerate(models):
            if 'Model' in data.columns:
                model_data = data[data['Model'] == model]
            else:
                model_data = data
            
            y_values = []
            y_errors = []
            
            for fs in feature_sets:
                if 'FeatureSet' in data.columns:
                    fs_data = model_data[model_data['FeatureSet'] == fs]
                else:
                    fs_data = model_data
                
                if len(fs_data) > 0:
                    mean_val = fs_data[mean_col].iloc[0]
                    
                    if lower_col in data.columns and upper_col in data.columns:
                        lower_val = fs_data[lower_col].iloc[0]
                        upper_val = fs_data[upper_col].iloc[0]
                        y_err = [[mean_val - lower_val], [upper_val - mean_val]]
                    else:
                        y_err = 0
                else:
                    mean_val = 0
                    y_err = 0
                
                y_values.append(mean_val)
                y_errors.append(y_err)
            
            # Plot bars
            x_positions = x_pos + (i - len(models)/2 + 0.5) * bar_width
            ax.bar(x_positions, y_values, bar_width, 
                  label=model, alpha=0.8)
            
            # Add error bars
            if all(isinstance(err, list) for err in y_errors):
                error_array = np.array(y_errors).T
                ax.errorbar(x_positions, y_values, yerr=error_array, 
                           fmt='none', color='black', capsize=4)
        
        # Customize plot
        ax.set_xlabel('Feature Set')
        ax.set_ylabel(f'{metric} Score')
        ax.set_title(f'Test {metric} (Bootstrap 95% CI)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_sets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    def create_performance_table(self, cv_results: pd.DataFrame,
                               bootstrap_results: Optional[pd.DataFrame] = None,
                               metrics: List[str] = ['f1', 'auc'],
                               save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a comprehensive performance table.
        
        Args:
            cv_results: Cross-validation results
            bootstrap_results: Bootstrap test results (optional)
            metrics: Metrics to include
            save_path: Path to save the table
            
        Returns:
            Performance summary table
        """
        # Prepare CV results
        cv_table = self._create_cv_table(cv_results, metrics)
        
        # Add bootstrap results if available
        if bootstrap_results is not None:
            bootstrap_table = self._create_bootstrap_table(bootstrap_results, metrics)
            # Merge tables (implementation depends on structure)
            table = cv_table  # Simplified for now
        else:
            table = cv_table
        
        if save_path:
            if save_path.endswith('.xlsx'):
                table.to_excel(save_path, index=False)
            else:
                table.to_csv(save_path, index=False)
        
        return table
    
    def _create_cv_table(self, results_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Create cross-validation results table.
        
        Args:
            results_df: CV results
            metrics: Metrics to include
            
        Returns:
            Formatted CV table
        """
        # Get best results for each combination
        if 'rank_test_f1' in results_df.columns:
            group_cols = ['FeatureSet', 'Model']
            best_idx = results_df.groupby(group_cols)['rank_test_f1'].idxmin()
            best_results = results_df.loc[best_idx].copy()
        else:
            best_results = results_df.copy()
        
        # Create table
        table_rows = []
        
        for _, row in best_results.iterrows():
            table_row = {
                'FeatureSet': row.get('FeatureSet', 'Unknown'),
                'Model': row.get('Model', 'Unknown')
            }
            
            for metric in metrics:
                mean_col = f'mean_test_{metric}'
                std_col = f'std_test_{metric}'
                
                if mean_col in row and std_col in row:
                    mean_val = row[mean_col]
                    std_val = row[std_col]
                    table_row[f'{metric.upper()}_CV'] = f"{mean_val:.3f} ± {std_val:.3f}"
                else:
                    table_row[f'{metric.upper()}_CV'] = "N/A"
            
            table_rows.append(table_row)
        
        return pd.DataFrame(table_rows)
    
    def _create_bootstrap_table(self, results_df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Create bootstrap results table.
        
        Args:
            results_df: Bootstrap results
            metrics: Metrics to include
            
        Returns:
            Formatted bootstrap table
        """
        table_rows = []
        
        for _, row in results_df.iterrows():
            table_row = {
                'FeatureSet': row.get('FeatureSet', 'Unknown'),
                'Model': row.get('Model', 'Unknown')
            }
            
            for metric in metrics.copy():
                # Map metric names
                metric_map = {'f1': 'F1', 'auc': 'AUC', 'precision': 'Precision', 'recall': 'Recall'}
                bootstrap_metric = metric_map.get(metric, metric.upper())
                
                mean_col = f'Test{bootstrap_metric}_BootMean'
                lower_col = f'Test{bootstrap_metric}_Boot95CI_Lower'
                upper_col = f'Test{bootstrap_metric}_Boot95CI_Upper'
                
                if all(col in row for col in [mean_col, lower_col, upper_col]):
                    mean_val = row[mean_col]
                    lower_val = row[lower_col]
                    upper_val = row[upper_col]
                    table_row[f'{metric.upper()}_Bootstrap'] = f"{mean_val:.3f} [{lower_val:.3f}, {upper_val:.3f}]"
                else:
                    table_row[f'{metric.upper()}_Bootstrap'] = "N/A"
            
            table_rows.append(table_row)
        
        return pd.DataFrame(table_rows)