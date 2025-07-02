"""
Feature Importance Analysis Module

This module provides comprehensive feature importance analysis using 
permutation importance and statistical significance testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp
from functools import partial
import time
from datetime import datetime
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis using permutation importance.
    
    This class provides methods to calculate feature importance by measuring
    the decrease in model performance when individual features are permuted.
    Supports parallel processing for efficient computation on large datasets.
    """
    
    def __init__(self, n_permutations: int = 100, n_jobs: int = -1, 
                 random_state: int = 42, batch_size: int = 50):
        """
        Initialize the feature importance analyzer.
        
        Args:
            n_permutations: Number of permutations per feature
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random state for reproducibility
            batch_size: Batch size for processing features
        """
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.random_state = random_state
        self.batch_size = batch_size
        
    def calculate_importance(self, model: Any, X: pd.DataFrame, y: np.ndarray,
                           feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate feature importance using permutation importance.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X: Feature matrix
            y: Target vector
            feature_list: List of features to analyze (default: all features)
            
        Returns:
            DataFrame with importance scores for each feature
        """
        if feature_list is None:
            feature_list = X.columns.tolist()
        
        # Calculate baseline performance
        baseline_metrics = self._calculate_baseline_metrics(model, X, y)
        
        print(f"Calculating importance for {len(feature_list)} features...")
        print(f"Baseline metrics: {baseline_metrics}")
        
        # Parallel computation
        if self.n_jobs > 1:
            results = self._parallel_importance_calculation(
                model, X, y, feature_list, baseline_metrics
            )
        else:
            results = self._sequential_importance_calculation(
                model, X, y, feature_list, baseline_metrics
            )
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add feature metadata if available
        results_df = self._add_feature_metadata(results_df, X)
        
        return results_df
    
    def _calculate_baseline_metrics(self, model: Any, X: pd.DataFrame, 
                                  y: np.ndarray) -> Dict[str, float]:
        """
        Calculate baseline performance metrics.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of baseline metrics
        """
        y_pred = model.predict(X)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }
        
        if y_prob is not None and len(np.unique(y)) == 2:
            metrics['auc'] = roc_auc_score(y, y_prob)
        else:
            metrics['auc'] = np.nan
        
        return metrics
    
    def _parallel_importance_calculation(self, model: Any, X: pd.DataFrame, 
                                       y: np.ndarray, feature_list: List[str],
                                       baseline_metrics: Dict[str, float]) -> List[Dict]:
        """
        Calculate feature importance using parallel processing.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_list: List of features to analyze
            baseline_metrics: Baseline performance metrics
            
        Returns:
            List of feature importance results
        """
        # Split features into chunks
        feature_chunks = [
            feature_list[i:i + self.batch_size] 
            for i in range(0, len(feature_list), self.batch_size)
        ]
        
        # Create worker function
        worker_func = partial(
            self._process_feature_chunk,
            X=X, y=y, baseline_metrics=baseline_metrics, model=model
        )
        
        # Process chunks in parallel
        all_results = []
        
        try:
            with mp.Pool(processes=self.n_jobs) as pool:
                chunk_results = pool.map(worker_func, feature_chunks)
                
            # Flatten results
            for chunk_result in chunk_results:
                all_results.extend(chunk_result)
                
        except Exception as e:
            warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential.")
            all_results = self._sequential_importance_calculation(
                model, X, y, feature_list, baseline_metrics
            )
        
        return all_results
    
    def _sequential_importance_calculation(self, model: Any, X: pd.DataFrame,
                                        y: np.ndarray, feature_list: List[str],
                                        baseline_metrics: Dict[str, float]) -> List[Dict]:
        """
        Calculate feature importance sequentially.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            feature_list: List of features to analyze
            baseline_metrics: Baseline performance metrics
            
        Returns:
            List of feature importance results
        """
        results = []
        
        for i, feature in enumerate(feature_list):
            if i % 20 == 0:
                print(f"Processing feature {i+1}/{len(feature_list)}: {feature}")
            
            try:
                importance = self._calculate_single_feature_importance(
                    feature, X, y, baseline_metrics, model
                )
                results.append(importance)
                
            except Exception as e:
                warnings.warn(f"Error processing feature {feature}: {e}")
                # Add NaN result for failed features
                results.append(self._create_nan_result(feature))
        
        return results
    
    def _process_feature_chunk(self, feature_chunk: List[str], X: pd.DataFrame,
                             y: np.ndarray, baseline_metrics: Dict[str, float],
                             model: Any) -> List[Dict]:
        """
        Process a chunk of features (for parallel processing).
        
        Args:
            feature_chunk: List of features in this chunk
            X: Feature matrix
            y: Target vector
            baseline_metrics: Baseline performance metrics
            model: Trained model
            
        Returns:
            List of importance results for the chunk
        """
        chunk_results = []
        
        for feature in feature_chunk:
            try:
                importance = self._calculate_single_feature_importance(
                    feature, X, y, baseline_metrics, model
                )
                chunk_results.append(importance)
                
            except Exception as e:
                warnings.warn(f"Error processing feature {feature}: {e}")
                chunk_results.append(self._create_nan_result(feature))
        
        return chunk_results
    
    def _calculate_single_feature_importance(self, feature: str, X: pd.DataFrame,
                                           y: np.ndarray, baseline_metrics: Dict[str, float],
                                           model: Any) -> Dict[str, float]:
        """
        Calculate importance for a single feature using permutation.
        
        Args:
            feature: Feature name
            X: Feature matrix
            y: Target vector
            baseline_metrics: Baseline performance metrics
            model: Trained model
            
        Returns:
            Dictionary with importance statistics
        """
        importance_scores = {metric: [] for metric in baseline_metrics.keys()}
        
        for perm_idx in range(self.n_permutations):
            # Set random seed for this permutation
            perm_seed = self.random_state + perm_idx
            np.random.seed(perm_seed)
            
            # Create permuted data
            X_perm = X.copy()
            X_perm[feature] = np.random.permutation(X_perm[feature].values)
            
            # Calculate metrics on permuted data
            y_pred_perm = model.predict(X_perm)
            
            if hasattr(model, "predict_proba"):
                y_prob_perm = model.predict_proba(X_perm)[:, 1]
            else:
                y_prob_perm = None
            
            perm_metrics = {
                'accuracy': accuracy_score(y, y_pred_perm),
                'precision': precision_score(y, y_pred_perm, zero_division=0),
                'recall': recall_score(y, y_pred_perm, zero_division=0),
                'f1': f1_score(y, y_pred_perm, zero_division=0)
            }
            
            if y_prob_perm is not None and len(np.unique(y)) == 2:
                perm_metrics['auc'] = roc_auc_score(y, y_prob_perm)
            else:
                perm_metrics['auc'] = np.nan
            
            # Calculate importance as decrease in performance
            for metric in importance_scores:
                if metric in baseline_metrics and metric in perm_metrics:
                    importance = baseline_metrics[metric] - perm_metrics[metric]
                    importance_scores[metric].append(importance)
        
        # Calculate statistics
        result = {'feature_id': feature}
        
        for metric in importance_scores:
            scores = importance_scores[metric]
            if scores and not all(np.isnan(scores)):
                result[f'{metric}_mean'] = np.nanmean(scores)
                result[f'{metric}_std'] = np.nanstd(scores)
                
                # Calculate signal-to-noise ratio
                if result[f'{metric}_std'] > 0:
                    result[f'{metric}_snr'] = result[f'{metric}_mean'] / result[f'{metric}_std']
                else:
                    result[f'{metric}_snr'] = 0
            else:
                result[f'{metric}_mean'] = np.nan
                result[f'{metric}_std'] = np.nan
                result[f'{metric}_snr'] = np.nan
        
        return result
    
    def _create_nan_result(self, feature: str) -> Dict[str, float]:
        """
        Create a result dictionary with NaN values for failed features.
        
        Args:
            feature: Feature name
            
        Returns:
            Dictionary with NaN importance values
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        result = {'feature_id': feature}
        
        for metric in metrics:
            result[f'{metric}_mean'] = np.nan
            result[f'{metric}_std'] = np.nan
            result[f'{metric}_snr'] = np.nan
        
        return result
    
    def _add_feature_metadata(self, results_df: pd.DataFrame, 
                            X: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature metadata to results.
        
        Args:
            results_df: Results DataFrame
            X: Original feature matrix
            
        Returns:
            Results DataFrame with added metadata
        """
        # Add basic statistics
        results_df['feature_mean'] = results_df['feature_id'].map(
            lambda f: X[f].mean() if f in X.columns else np.nan
        )
        results_df['feature_std'] = results_df['feature_id'].map(
            lambda f: X[f].std() if f in X.columns else np.nan
        )
        results_df['feature_missing_rate'] = results_df['feature_id'].map(
            lambda f: X[f].isna().mean() if f in X.columns else np.nan
        )
        
        return results_df
    
    def filter_significant_features(self, results_df: pd.DataFrame,
                                  min_importance: float = 0.0,
                                  min_snr: float = 1.0,
                                  metric: str = 'f1') -> pd.DataFrame:
        """
        Filter features based on significance criteria.
        
        Args:
            results_df: Feature importance results
            min_importance: Minimum importance threshold
            min_snr: Minimum signal-to-noise ratio
            metric: Metric to use for filtering
            
        Returns:
            Filtered DataFrame with significant features
        """
        mean_col = f'{metric}_mean'
        snr_col = f'{metric}_snr'
        
        # Apply filters
        filtered_df = results_df[
            (results_df[mean_col] > min_importance) & 
            (results_df[snr_col] > min_snr)
        ].copy()
        
        # Sort by importance
        filtered_df = filtered_df.sort_values(mean_col, ascending=False)
        
        return filtered_df
    
    def get_top_features(self, results_df: pd.DataFrame, n_features: int = 20,
                        metric: str = 'f1') -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            results_df: Feature importance results
            n_features: Number of top features to return
            metric: Metric to use for ranking
            
        Returns:
            DataFrame with top features
        """
        mean_col = f'{metric}_mean'
        
        # Filter out NaN values and sort
        valid_results = results_df.dropna(subset=[mean_col])
        top_features = valid_results.sort_values(mean_col, ascending=False).head(n_features)
        
        return top_features
    
    def save_results(self, results_df: pd.DataFrame, output_path: str,
                    include_metadata: bool = True):
        """
        Save feature importance results to file.
        
        Args:
            results_df: Feature importance results
            output_path: Output file path
            include_metadata: Whether to include analysis metadata
        """
        if include_metadata:
            # Add metadata sheet if Excel format
            if output_path.endswith('.xlsx'):
                with pd.ExcelWriter(output_path) as writer:
                    results_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
                    
                    # Create metadata
                    metadata = pd.DataFrame({
                        'Parameter': ['n_permutations', 'random_state', 'analysis_date'],
                        'Value': [self.n_permutations, self.random_state, datetime.now().isoformat()]
                    })
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
            else:
                results_df.to_csv(output_path, index=False)
        else:
            if output_path.endswith('.xlsx'):
                results_df.to_excel(output_path, index=False)
            else:
                results_df.to_csv(output_path, index=False)
        
        print(f"Feature importance results saved to: {output_path}")