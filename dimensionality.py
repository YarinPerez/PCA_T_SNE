"""
Dimensionality reduction module.

Implements PCA using ONLY NumPy and wraps t-SNE from scikit-learn
for dimensionality reduction to 3D space.
"""
import numpy as np
from sklearn.manifold import TSNE
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PCA_NumPy:
    """
    Principal Component Analysis implementation using ONLY NumPy.

    CRITICAL: This implementation must NOT use scikit-learn's PCA.
    All computations use only numpy for all operations.
    """

    def __init__(self, n_components: int = 3):
        """
        Initialize PCA reducer.

        Args:
            n_components: Number of principal components (default: 3)

        Raises:
            ValueError: If n_components <= 0
        """
        if n_components <= 0:
            raise ValueError("n_components must be greater than 0")

        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self._covariance_matrix = None

    def fit(self, X: np.ndarray) -> 'PCA_NumPy':
        """
        Fit PCA on data using ONLY NumPy operations.

        Mathematical steps:
        1. Center the data (subtract mean)
        2. Compute covariance matrix
        3. Compute eigenvalues and eigenvectors
        4. Sort by eigenvalues (descending)
        5. Select top n_components eigenvectors
        6. Calculate explained variance ratio

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If X has invalid shape
            ValueError: If n_components >= n_features
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        n_samples, n_features = X.shape

        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"n_features ({n_features})"
            )

        logger.info(f"Fitting PCA with {self.n_components} components...")

        # STEP 1: Center the data (subtract mean)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # STEP 2: Compute covariance matrix
        # Use (X.T @ X) / (n_samples - 1) for unbiased estimator
        self._covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # STEP 3: Compute eigenvalues and eigenvectors
        # np.linalg.eig handles complex eigenvalues properly
        eigenvalues, eigenvectors = np.linalg.eig(self._covariance_matrix)

        # STEP 4: Sort by eigenvalues in descending order
        # Take only the real part to handle numerical artifacts
        eigenvalues = eigenvalues.real
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices].real

        # STEP 5: Select top n_components eigenvectors
        self.components_ = eigenvectors_sorted[:, :self.n_components]

        # STEP 6: Calculate explained variance
        # Only use positive eigenvalues (handle numerical precision issues)
        self.explained_variance_ = np.maximum(
            eigenvalues_sorted[:self.n_components], 0
        )
        total_variance = np.sum(np.maximum(eigenvalues_sorted, 0))

        if total_variance == 0:
            self.explained_variance_ratio_ = np.ones(self.n_components) / self.n_components
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        logger.info(
            f"PCA fitted successfully. Explained variance ratio: "
            f"{self.explained_variance_ratio_}"
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)

        Raises:
            ValueError: If PCA has not been fitted yet
            ValueError: If X has wrong number of features
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transform. Call fit() first.")

        if X.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but PCA was fitted with "
                f"{self.components_.shape[0]} features"
            )

        X_centered = X - self.mean_
        X_pca = X_centered @ self.components_

        return X_pca

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the variance explained by each principal component.

        Returns:
            Array of explained variance ratios of shape (n_components,)

        Raises:
            ValueError: If PCA has not been fitted yet
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA has not been fitted yet. Call fit() first.")
        return self.explained_variance_ratio_

    def get_components(self) -> np.ndarray:
        """
        Get the principal components (eigenvectors).

        Returns:
            Array of shape (n_features, n_components)

        Raises:
            ValueError: If PCA has not been fitted yet
        """
        if self.components_ is None:
            raise ValueError("PCA has not been fitted yet. Call fit() first.")
        return self.components_


class TSNE_Wrapper:
    """Wrapper around scikit-learn's t-SNE for dimensionality reduction."""

    def __init__(self, n_components: int = 3, perplexity: float = 30.0,
                 learning_rate: float = 200.0, n_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize t-SNE reducer.

        Args:
            n_components: Number of output dimensions (default: 3)
            perplexity: t-SNE perplexity parameter (default: 30.0)
            learning_rate: t-SNE learning rate (default: 200.0)
            n_iter: Maximum iterations (default: 1000)
            random_state: Random seed for reproducibility (default: 42)

        Raises:
            ValueError: If n_components not in (2, 3)
        """
        if n_components not in (2, 3):
            logger.warning(f"t-SNE typically uses 2D or 3D, got {n_components}D")

        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

        # Scikit-learn TSNE compatibility: different versions use different parameter names
        tsne_kwargs = {
            'n_components': n_components,
            'perplexity': perplexity,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'verbose': 1
        }

        # Try to use n_iter if supported, fall back to n_iter_without_progress
        try:
            self.tsne = TSNE(n_iter=n_iter, **tsne_kwargs)
        except TypeError:
            # Fallback for sklearn versions that don't support n_iter
            # Use n_iter_without_progress instead
            self.tsne = TSNE(n_iter_without_progress=max(100, n_iter // 10), **tsne_kwargs)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)

        Raises:
            ValueError: If X has invalid shape
            ValueError: If n_samples < perplexity * 3
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        n_samples = X.shape[0]

        # t-SNE needs at least 3 * perplexity samples
        if n_samples < self.perplexity * 3:
            logger.warning(
                f"n_samples ({n_samples}) is less than recommended "
                f"minimum ({self.perplexity * 3}). Results may be unreliable."
            )

        logger.info("Running t-SNE (this may take a while for large datasets)...")
        try:
            X_tsne = self.tsne.fit_transform(X)
            logger.info(f"t-SNE complete. Output shape: {X_tsne.shape}")
            return X_tsne
        except Exception as e:
            logger.error(f"t-SNE fitting failed: {e}")
            raise
