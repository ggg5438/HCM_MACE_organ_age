"""
Sparse Partial Least Squares Discriminant Analysis (SPLSDA) Implementation

This module provides a custom implementation of SPLSDA for high-dimensional
biological data classification.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from typing import Optional


class SPLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    Sparse Partial Least Squares Discriminant Analysis classifier.
    
    This implementation provides sparse feature selection through L1 regularization
    combined with dimensionality reduction via PLS components. It's particularly
    suitable for high-dimensional biological data with many irrelevant features.
    
    Parameters:
        n_components: Number of PLS components to compute
        lambda_val: L1 regularization parameter for sparsity
        max_iter: Maximum number of iterations for convergence
        tol: Tolerance for convergence criteria
        
    Attributes:
        classes_: Unique class labels
        W_: Loading vectors (features x components)
        T_train_: Transformed training data
        centroids_: Class centroids in the transformed space
        X_mean_: Feature means from training data
    """
    
    def __init__(self, n_components: int = 2, lambda_val: float = 0.1, 
                 max_iter: int = 500, tol: float = 1e-6):
        """
        Initialize SPLSDA classifier.
        
        Args:
            n_components: Number of latent components (default: 2)
            lambda_val: L1 regularization strength (default: 0.1)
            max_iter: Maximum iterations for algorithm convergence (default: 500)
            tol: Convergence tolerance (default: 1e-6)
        """
        self.n_components = n_components
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.tol = tol
    
    def _soft_thresholding(self, u: np.ndarray, lamb: float) -> np.ndarray:
        """
        Apply soft thresholding for L1 regularization.
        
        Args:
            u: Input vector
            lamb: Thresholding parameter
            
        Returns:
            Soft-thresholded vector
        """
        return np.sign(u) * np.maximum(np.abs(u) - lamb, 0)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SPLSDAClassifier':
        """
        Fit the SPLSDA model to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            Self (fitted estimator)
            
        Raises:
            ValueError: If input data is invalid
        """
        # Validate input
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=np.float64)
        
        # Center the data
        self.X_mean_ = X.mean(axis=0)
        X_centered = X - self.X_mean_
        
        # Set up class information
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X_centered.shape
        
        # Create binary indicator matrix for classes
        Y = np.zeros((n_samples, n_classes))
        for i, class_label in enumerate(self.classes_):
            Y[y == class_label, i] = 1
        
        # Initialize matrices
        W = np.zeros((n_features, self.n_components))
        T_matrix = np.zeros((n_samples, self.n_components))
        X_deflated = X_centered.copy()
        Y_deflated = Y.copy()
        
        # Iteratively compute PLS components
        for component in range(self.n_components):
            if (np.linalg.norm(X_deflated) < 1e-8 or 
                np.linalg.norm(Y_deflated) < 1e-8):
                break
            
            # Initialize loading vector using SVD
            try:
                U, s, Vt = np.linalg.svd(X_deflated.T @ Y_deflated, full_matrices=False)
                w = U[:, 0]
            except np.linalg.LinAlgError:
                break
            
            # NIPALS algorithm with sparse regularization
            for iteration in range(self.max_iter):
                w_old = w.copy()
                
                # Score vector
                t = X_deflated @ w
                norm_t_squared = t.T @ t
                
                if norm_t_squared < 1e-8:
                    break
                
                # Y-loadings
                c = Y_deflated.T @ t / norm_t_squared
                c_norm = np.linalg.norm(c)
                if c_norm != 0:
                    c = c / c_norm
                
                # X-loadings with sparsity
                u = X_deflated.T @ (Y_deflated @ c)
                w = self._soft_thresholding(u, self.lambda_val)
                
                # Normalize loading vector
                w_norm = np.linalg.norm(w)
                if w_norm != 0:
                    w = w / w_norm
                else:
                    break
                
                # Check convergence
                if np.linalg.norm(w - w_old) < self.tol:
                    break
            
            # Store component
            t = X_deflated @ w
            W[:, component] = w
            T_matrix[:, component] = t
            
            # Deflate matrices
            norm_t_squared = t.T @ t
            if norm_t_squared < 1e-8:
                break
                
            p_vector = X_deflated.T @ t / norm_t_squared
            q_vector = Y_deflated.T @ t / norm_t_squared
            
            X_deflated = X_deflated - np.outer(t, p_vector)
            Y_deflated = Y_deflated - np.outer(t, q_vector)
        
        # Store model parameters
        self.W_ = W
        self.T_train_ = T_matrix
        
        # Compute class centroids in transformed space
        self.centroids_ = {}
        for class_label in self.classes_:
            class_mask = (y == class_label)
            if np.sum(class_mask) > 0:
                self.centroids_[class_label] = T_matrix[class_mask, :].mean(axis=0)
            else:
                self.centroids_[class_label] = np.zeros(self.n_components)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to the learned latent space.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        X = check_array(X)
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.X_mean_
        return X_centered @ self.W_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for new data.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        T_new = self.transform(X)
        predictions = []
        
        for sample in T_new:
            # Find closest centroid
            distances = {
                class_label: np.linalg.norm(sample - centroid)
                for class_label, centroid in self.centroids_.items()
            }
            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new data.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        T_new = self.transform(X)
        probabilities = []
        
        for sample in T_new:
            # Calculate distances to centroids
            distances = np.array([
                np.linalg.norm(sample - self.centroids_[class_label])
                for class_label in self.classes_
            ])
            
            # Convert distances to probabilities using softmax
            exp_neg_distances = np.exp(-distances)
            total = exp_neg_distances.sum()
            
            if total == 0 or np.isnan(total):
                # Uniform probability if all distances are very large
                probs = np.ones(len(self.classes_)) / len(self.classes_)
            else:
                probs = exp_neg_distances / total
            
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on loading magnitudes.
        
        Returns:
            Feature importance scores of shape (n_features,)
        """
        if not hasattr(self, 'W_'):
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Sum of absolute loadings across components
        return np.sum(np.abs(self.W_), axis=1)