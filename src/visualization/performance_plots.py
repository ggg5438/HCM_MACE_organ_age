"""
Performance Visualization Module

This module provides specialized visualization tools for model performance
evaluation including ROC curves, precision-recall curves, and calibration plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional, Tuple, Any
import pickle
import warnings


class PerformanceVisualizer:
    """
    Specialized visualizer for model performance evaluation.
    
    This class provides methods to create:
    - ROC curves with confidence intervals
    - Precision-recall curves
    - Calibration plots
    - Performance comparison charts
    """
    
    def __init__(self, colorblind_friendly: bool = True, dpi: int = 300):
        """
        Initialize the performance visualizer.
        
        Args:
            colorblind_friendly: Whether to use colorblind-friendly colors
            dpi: Resolution for saved figures
        """
        self.dpi = dpi
        
        # Set color palette
        if colorblind_friendly:
            self.colors = [
                '#0173B2', '#DE8F05', '#029E73', '#D55E00', 
                '#CC78BC', '#CA9161', '#FBAFE4', '#949494', 
                '#ECE133', '#56B4E9'
            ]
        else:
            self.colors = plt.cm.tab10.colors
        
        # Line styles for different feature sets
        self.line_styles = ['-', '--', '-.', ':']
    
    def load_prediction_results(self, file_path: str) -> Dict[str, Any]:
        """
        Load prediction results from pickle file.
        
        Args:
            file_path: Path to pickle file with prediction results
            
        Returns:
            Dictionary with prediction results
        """
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            return results
        except FileNotFoundError:
            raise FileNotFoundError(f"Prediction results file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error loading prediction results: {e}")
    
    def plot_roc_curves(self, prediction_results: Dict[str, Any], 
                       selected_models: Optional[List[str]] = None,
                       selected_features: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None,
                       custom_colors: Optional[Dict[str, str]] = None,
                       custom_styles: Optional[Dict[str, str]] = None,
                       custom_labels: Optional[Dict[str, str]] = None) -> plt.Figure:
        """
        Plot ROC curves for model comparison.
        
        Args:
            prediction_results: Dictionary with prediction results
            selected_models: List of model names to plot (None for all)
            selected_features: List of feature sets to plot (None for all)
            figsize: Figure size
            save_path: Path to save the figure
            custom_colors: Custom colors for models
            custom_styles: Custom line styles for feature sets
            custom_labels: Custom labels for combinations
            
        Returns:
            Matplotlib figure object
        """
        # Create figure with extra space for legend
        fig_width = figsize[0] + 4
        fig = plt.figure(figsize=(fig_width, figsize[1]))
        
        # Main plot area
        plot_width = figsize[0] / fig_width
        ax = fig.add_axes([0.1, 0.1, plot_width - 0.15, 0.8])
        
        # Select models and features
        if selected_models is None:
            selected_models = list(prediction_results.keys())
        
        count = 0
        for i, model_name in enumerate(selected_models):
            if model_name not in prediction_results:
                warnings.warn(f"Model {model_name} not found in results")
                continue
            
            model_results = prediction_results[model_name]
            
            # Select feature sets
            if selected_features is None:
                current_features = list(model_results.keys())
            else:
                current_features = selected_features
            
            for j, fs_name in enumerate(current_features):
                if fs_name not in model_results:
                    warnings.warn(f"Feature set {fs_name} not found for model {model_name}")
                    continue
                
                y_true = model_results[fs_name]['y_true']
                y_pred_proba = model_results[fs_name]['y_pred_proba']
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Select color and style
                color = custom_colors.get(model_name, self.colors[i % len(self.colors)]) if custom_colors else self.colors[i % len(self.colors)]
                linestyle = custom_styles.get(fs_name, self.line_styles[j % len(self.line_styles)]) if custom_styles else self.line_styles[j % len(self.line_styles)]
                
                # Create label
                default_label = f'{model_name} + {fs_name} (AUC = {roc_auc:.3f})'\n                label_key = f'{model_name} + {fs_name}'\n                \n                if custom_labels and label_key in custom_labels:\n                    custom_label = custom_labels[label_key]\n                    if '(AUC =' not in custom_label:\n                        label = f'{custom_label} (AUC = {roc_auc:.3f})'\n                    else:\n                        label = custom_label\n                else:\n                    label = default_label\n                \n                # Plot ROC curve\n                ax.plot(fpr, tpr, color=color, linestyle=linestyle, \n                       linewidth=2, label=label)\n                count += 1\n        \n        # Plot diagonal (random classifier)\n        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random (AUC = 0.500)')\n        \n        # Customize plot\n        ax.set_xlim([0.0, 1.0])\n        ax.set_ylim([0.0, 1.05])\n        ax.set_xlabel('False Positive Rate', fontsize=12)\n        ax.set_ylabel('True Positive Rate', fontsize=12)\n        ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)\n        ax.grid(True, alpha=0.3)\n        \n        # Add legend outside plot area\n        legend_x = plot_width + 0.02\n        ax.legend(bbox_to_anchor=(legend_x, 1), loc='upper left', fontsize=10)\n        \n        if save_path:\n            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')\n        \n        return fig\n    \n    def plot_pr_curves(self, prediction_results: Dict[str, Any],\n                      selected_models: Optional[List[str]] = None,\n                      selected_features: Optional[List[str]] = None,\n                      figsize: Tuple[int, int] = (10, 8),\n                      save_path: Optional[str] = None,\n                      custom_colors: Optional[Dict[str, str]] = None,\n                      custom_styles: Optional[Dict[str, str]] = None,\n                      custom_labels: Optional[Dict[str, str]] = None) -> plt.Figure:\n        \"\"\"\n        Plot Precision-Recall curves for model comparison.\n        \n        Args:\n            prediction_results: Dictionary with prediction results\n            selected_models: List of model names to plot (None for all)\n            selected_features: List of feature sets to plot (None for all)\n            figsize: Figure size\n            save_path: Path to save the figure\n            custom_colors: Custom colors for models\n            custom_styles: Custom line styles for feature sets\n            custom_labels: Custom labels for combinations\n            \n        Returns:\n            Matplotlib figure object\n        \"\"\"\n        # Create figure with extra space for legend\n        fig_width = figsize[0] + 4\n        fig = plt.figure(figsize=(fig_width, figsize[1]))\n        \n        # Main plot area\n        plot_width = figsize[0] / fig_width\n        ax = fig.add_axes([0.1, 0.1, plot_width - 0.15, 0.8])\n        \n        # Select models\n        if selected_models is None:\n            selected_models = list(prediction_results.keys())\n        \n        for i, model_name in enumerate(selected_models):\n            if model_name not in prediction_results:\n                warnings.warn(f\"Model {model_name} not found in results\")\n                continue\n            \n            model_results = prediction_results[model_name]\n            \n            # Select feature sets\n            if selected_features is None:\n                current_features = list(model_results.keys())\n            else:\n                current_features = selected_features\n            \n            for j, fs_name in enumerate(current_features):\n                if fs_name not in model_results:\n                    warnings.warn(f\"Feature set {fs_name} not found for model {model_name}\")\n                    continue\n                \n                y_true = model_results[fs_name]['y_true']\n                y_pred_proba = model_results[fs_name]['y_pred_proba']\n                \n                # Calculate PR curve\n                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)\n                avg_precision = average_precision_score(y_true, y_pred_proba)\n                \n                # Select color and style\n                color = custom_colors.get(model_name, self.colors[i % len(self.colors)]) if custom_colors else self.colors[i % len(self.colors)]\n                linestyle = custom_styles.get(fs_name, self.line_styles[j % len(self.line_styles)]) if custom_styles else self.line_styles[j % len(self.line_styles)]\n                \n                # Create label\n                default_label = f'{model_name} + {fs_name} (AP = {avg_precision:.3f})'\n                label_key = f'{model_name} + {fs_name}'\n                \n                if custom_labels and label_key in custom_labels:\n                    custom_label = custom_labels[label_key]\n                    if '(AP =' not in custom_label:\n                        label = f'{custom_label} (AP = {avg_precision:.3f})'\n                    else:\n                        label = custom_label\n                else:\n                    label = default_label\n                \n                # Plot PR curve\n                ax.plot(recall, precision, color=color, linestyle=linestyle,\n                       linewidth=2, label=label)\n        \n        # Calculate baseline (random classifier)\n        # For binary classification, baseline is the positive class ratio\n        if selected_models and selected_models[0] in prediction_results:\n            first_model = prediction_results[selected_models[0]]\n            first_fs = list(first_model.keys())[0]\n            y_true_sample = first_model[first_fs]['y_true']\n            baseline = np.mean(y_true_sample)\n            ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, \n                      label=f'Random (AP = {baseline:.3f})')\n        \n        # Customize plot\n        ax.set_xlim([0.0, 1.0])\n        ax.set_ylim([0.0, 1.05])\n        ax.set_xlabel('Recall', fontsize=12)\n        ax.set_ylabel('Precision', fontsize=12)\n        ax.set_title('Precision-Recall Curves', fontsize=14)\n        ax.grid(True, alpha=0.3)\n        \n        # Add legend outside plot area\n        legend_x = plot_width + 0.02\n        ax.legend(bbox_to_anchor=(legend_x, 1), loc='upper left', fontsize=10)\n        \n        if save_path:\n            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')\n        \n        return fig\n    \n    def plot_calibration_curve(self, prediction_results: Dict[str, Any],\n                              selected_models: Optional[List[str]] = None,\n                              selected_features: Optional[List[str]] = None,\n                              n_bins: int = 10,\n                              figsize: Tuple[int, int] = (10, 8),\n                              save_path: Optional[str] = None) -> plt.Figure:\n        \"\"\"\n        Plot calibration curves to assess prediction reliability.\n        \n        Args:\n            prediction_results: Dictionary with prediction results\n            selected_models: List of model names to plot (None for all)\n            selected_features: List of feature sets to plot (None for all)\n            n_bins: Number of bins for calibration\n            figsize: Figure size\n            save_path: Path to save the figure\n            \n        Returns:\n            Matplotlib figure object\n        \"\"\"\n        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)\n        \n        # Select models\n        if selected_models is None:\n            selected_models = list(prediction_results.keys())\n        \n        for i, model_name in enumerate(selected_models):\n            if model_name not in prediction_results:\n                continue\n            \n            model_results = prediction_results[model_name]\n            \n            # Select feature sets\n            if selected_features is None:\n                current_features = list(model_results.keys())\n            else:\n                current_features = selected_features\n            \n            for j, fs_name in enumerate(current_features):\n                if fs_name not in model_results:\n                    continue\n                \n                y_true = model_results[fs_name]['y_true']\n                y_pred_proba = model_results[fs_name]['y_pred_proba']\n                \n                # Calculate calibration curve\n                fraction_of_positives, mean_predicted_value = calibration_curve(\n                    y_true, y_pred_proba, n_bins=n_bins\n                )\n                \n                color = self.colors[i % len(self.colors)]\n                linestyle = self.line_styles[j % len(self.line_styles)]\n                label = f'{model_name} + {fs_name}'\n                \n                # Plot calibration curve\n                ax1.plot(mean_predicted_value, fraction_of_positives, 'o-',\n                        color=color, linestyle=linestyle, label=label)\n                \n                # Plot histogram of predictions\n                ax2.hist(y_pred_proba, bins=20, alpha=0.5, color=color, \n                        label=label, density=True)\n        \n        # Perfect calibration line\n        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')\n        \n        # Customize calibration plot\n        ax1.set_xlabel('Mean Predicted Probability')\n        ax1.set_ylabel('Fraction of Positives')\n        ax1.set_title('Calibration Curve')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        \n        # Customize histogram\n        ax2.set_xlabel('Predicted Probability')\n        ax2.set_ylabel('Density')\n        ax2.set_title('Distribution of Predictions')\n        ax2.legend()\n        ax2.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        \n        if save_path:\n            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')\n        \n        return fig\n    \n    def create_performance_comparison_table(self, prediction_results: Dict[str, Any]) -> str:\n        \"\"\"\n        Create a text table comparing model performance.\n        \n        Args:\n            prediction_results: Dictionary with prediction results\n            \n        Returns:\n            Formatted table as string\n        \"\"\"\n        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n        \n        results = []\n        \n        for model_name, model_results in prediction_results.items():\n            for fs_name, fs_results in model_results.items():\n                y_true = fs_results['y_true']\n                y_pred_proba = fs_results['y_pred_proba']\n                y_pred = (y_pred_proba > 0.5).astype(int)\n                \n                # Calculate metrics\n                accuracy = accuracy_score(y_true, y_pred)\n                precision = precision_score(y_true, y_pred, zero_division=0)\n                recall = recall_score(y_true, y_pred, zero_division=0)\n                f1 = f1_score(y_true, y_pred, zero_division=0)\n                \n                # ROC AUC\n                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)\n                roc_auc = auc(fpr, tpr)\n                \n                # Average Precision\n                avg_precision = average_precision_score(y_true, y_pred_proba)\n                \n                results.append({\n                    'Model': model_name,\n                    'Features': fs_name,\n                    'Accuracy': f'{accuracy:.3f}',\n                    'Precision': f'{precision:.3f}',\n                    'Recall': f'{recall:.3f}',\n                    'F1': f'{f1:.3f}',\n                    'ROC AUC': f'{roc_auc:.3f}',\n                    'Avg Precision': f'{avg_precision:.3f}'\n                })\n        \n        if not results:\n            return \"No results to display\"\n        \n        # Create formatted table\n        import pandas as pd\n        df = pd.DataFrame(results)\n        return df.to_string(index=False)\n    \n    def plot_all_curves(self, prediction_results: Dict[str, Any],\n                       selected_models: Optional[List[str]] = None,\n                       selected_features: Optional[List[str]] = None,\n                       save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:\n        \"\"\"\n        Plot all performance curves (ROC, PR, calibration) at once.\n        \n        Args:\n            prediction_results: Dictionary with prediction results\n            selected_models: List of model names to plot (None for all)\n            selected_features: List of feature sets to plot (None for all)\n            save_dir: Directory to save figures\n            \n        Returns:\n            Dictionary mapping plot types to figure objects\n        \"\"\"\n        figures = {}\n        \n        # ROC curves\n        roc_save_path = f\"{save_dir}/roc_curves.png\" if save_dir else None\n        figures['roc'] = self.plot_roc_curves(\n            prediction_results, selected_models, selected_features, save_path=roc_save_path\n        )\n        \n        # PR curves\n        pr_save_path = f\"{save_dir}/pr_curves.png\" if save_dir else None\n        figures['pr'] = self.plot_pr_curves(\n            prediction_results, selected_models, selected_features, save_path=pr_save_path\n        )\n        \n        # Calibration curves\n        cal_save_path = f\"{save_dir}/calibration_curves.png\" if save_dir else None\n        figures['calibration'] = self.plot_calibration_curve(\n            prediction_results, selected_models, selected_features, save_path=cal_save_path\n        )\n        \n        return figures