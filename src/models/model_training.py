"""
Model Training Module

This module provides comprehensive model training functionality with 
Bayesian optimization, cross-validation, and performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, accuracy_score, precision_recall_curve, auc
)
from sklearn.utils import resample
from sklearn.base import clone
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import warnings

from .splsda import SPLSDAClassifier


class ModelTrainer:
    """
    Comprehensive model training with Bayesian optimization and evaluation.
    
    This class provides a complete pipeline for training machine learning models
    on MACE prediction data, including hyperparameter optimization, cross-validation,
    and bootstrap evaluation.
    """
    
    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            n_jobs: Number of parallel jobs for training
            random_state: Random state for reproducibility
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = self._get_model_configurations()
        
    def _get_model_configurations(self) -> Dict[str, Tuple[Any, Dict]]:
        """
        Get model configurations with search spaces for Bayesian optimization.
        
        Returns:
            Dictionary mapping model names to (estimator, search_space) tuples
        """
        # SPLSDA search space
        splsda_space = {
            'n_components': Integer(2, 300),
            'lambda_val': Real(0.0001, 1.0, prior='log-uniform'),
            'max_iter': Integer(100, 3000),
            'tol': Real(1e-8, 1e-4, prior='log-uniform')
        }
        
        # Random Forest search space
        rf_space = {
            'n_estimators': Integer(100, 2000),
            'max_depth': Integer(2, 150),
            'min_samples_split': Integer(2, 50),
            'min_samples_leaf': Integer(1, 20),
            'max_features': Categorical(['sqrt', 'log2', None, 0.3, 0.5, 0.7]),
            'class_weight': Categorical(['balanced', 'balanced_subsample', None]),
            'criterion': Categorical(['gini', 'entropy']),
            'ccp_alpha': Real(0.0, 0.05),
            'max_samples': Real(0.5, 1.0)
        }
        
        # Logistic Regression search space
        lr_space = [
            {
                'solver': ['liblinear'],
                'penalty': ['l1', 'l2'],
                'C': Real(0.0001, 10000, prior='log-uniform'),
                'max_iter': Integer(100, 5000),
                'class_weight': Categorical(['balanced', None])
            },
            {
                'solver': ['lbfgs', 'newton-cg'],
                'penalty': ['l2', 'none'],
                'C': Real(0.0001, 10000, prior='log-uniform'),
                'max_iter': Integer(100, 5000),
                'class_weight': Categorical(['balanced', None])
            }
        ]
        
        return {
            'SPLSDA': (SPLSDAClassifier(), splsda_space),
            'RandomForest': (RandomForestClassifier(random_state=self.random_state), rf_space),
            'LogisticRegression': (LogisticRegression(random_state=self.random_state), lr_space)
        }
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   model_name: str, n_iter: int = 100, 
                   cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a model with Bayesian optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model to train
            n_iter: Number of Bayesian optimization iterations
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing trained model, best parameters, and CV results
            
        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        
        estimator, search_space = self.models[model_name]
        
        # Define scoring metrics
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "auc": "roc_auc",
            "average_precision": "average_precision"
        }
        
        # Bayesian optimization
        opt = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_space,
            scoring=scoring,
            refit="f1",  # Optimize for F1 score
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            n_iter=n_iter,
            verbose=1,
            return_train_score=True,
            random_state=self.random_state
        )
        
        # Fit the optimizer
        opt.fit(X_train, y_train)
        
        # Train final model with best parameters
        best_model = clone(estimator).set_params(**opt.best_params_)
        best_model.fit(X_train, y_train)
        
        return {
            'model': best_model,
            'best_params': opt.best_params_,
            'best_score': opt.best_score_,
            'cv_results': pd.DataFrame(opt.cv_results_),
            'optimizer': opt
        }
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of performance metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Add probability-based metrics if available
        if y_prob is not None and len(np.unique(y_test)) == 2:
            metrics['auc'] = roc_auc_score(y_test, y_prob)
            
            # Calculate AUPR
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            metrics['aupr'] = auc(recall_curve, precision_curve)
        else:
            metrics['auc'] = np.nan
            metrics['aupr'] = np.nan
        
        return metrics
    
    def bootstrap_evaluation(self, model: Any, X_test: np.ndarray, 
                           y_test: np.ndarray, n_bootstraps: int = 1000) -> Dict[str, float]:
        """
        Perform bootstrap evaluation for confidence intervals.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            n_bootstraps: Number of bootstrap samples
            
        Returns:
            Dictionary with mean values and confidence intervals
        """
        rng = np.random.RandomState(self.random_state)
        
        bootstrap_metrics = []
        
        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = rng.choice(len(X_test), size=len(X_test), replace=True)
            X_boot = X_test[indices]
            y_boot = y_test[indices]
            
            # Skip if only one class present
            if len(np.unique(y_boot)) < 2:
                continue
            
            # Calculate metrics for this bootstrap sample
            metrics = self.evaluate_model(model, X_boot, y_boot)
            bootstrap_metrics.append(metrics)
        
        if not bootstrap_metrics:
            # Return NaN if no valid bootstrap samples
            return {f"{metric}_{stat}": np.nan 
                   for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr']
                   for stat in ['mean', '95ci_lower', '95ci_upper']}
        
        # Calculate statistics
        metrics_df = pd.DataFrame(bootstrap_metrics)
        results = {}
        
        for metric in metrics_df.columns:
            values = metrics_df[metric].dropna()
            if len(values) > 0:
                results[f"{metric}_mean"] = values.mean()
                results[f"{metric}_95ci_lower"] = np.percentile(values, 2.5)
                results[f"{metric}_95ci_upper"] = np.percentile(values, 97.5)
            else:
                results[f"{metric}_mean"] = np.nan
                results[f"{metric}_95ci_lower"] = np.nan
                results[f"{metric}_95ci_upper"] = np.nan
        
        return results
    
    def save_model(self, model: Any, model_info: Dict[str, Any], 
                   output_dir: str, model_name: str, feature_set_name: str) -> str:
        """
        Save trained model and metadata.
        
        Args:
            model: Trained model object
            model_info: Model metadata and configuration
            output_dir: Directory to save model files
            model_name: Name of the model
            feature_set_name: Name of the feature set used
            
        Returns:
            Path to saved model info file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, f"{model_name}_{feature_set_name}_model.joblib")
        joblib.dump(model, model_path)
        
        # Save model info
        info_path = os.path.join(output_dir, f"{model_name}_{feature_set_name}_info.joblib")
        joblib.dump(model_info, info_path)
        
        # Save feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': model_info.get('feature_names', []),
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            importance_path = os.path.join(output_dir, f"{model_name}_{feature_set_name}_importance.csv")
            importance_df.to_csv(importance_path, index=False)
        
        return info_path
    
    def get_confidence_interval(self, mean: float, std: float, 
                               n_folds: int, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval using normal approximation.
        
        Args:
            mean: Sample mean
            std: Sample standard deviation
            n_folds: Number of CV folds
            alpha: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        z_score = 1.96  # For 95% CI
        sem = std / np.sqrt(n_folds)
        margin = z_score * sem
        return (mean - margin, mean + margin)