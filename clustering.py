"""
K-Means clustering module for sentence vectors.

Wraps scikit-learn's K-Means implementation and provides clustering
metrics and analysis.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KMeansClustering:
    """Wrapper for K-Means clustering with metrics calculation."""

    def __init__(self, n_clusters: int, random_state: int = 42,
                 max_iter: int = 300, n_init: int = 10):
        """
        Initialize K-Means clusterer.

        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility (default: 42)
            max_iter: Maximum iterations (default: 300)
            n_init: Number of initializations (default: 10)

        Raises:
            ValueError: If n_clusters <= 0
        """
        if n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            verbose=0
        )
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Means clustering to the data.

        Args:
            X: Input vectors of shape (n_samples, n_features)

        Returns:
            Cluster labels of shape (n_samples,)

        Raises:
            ValueError: If X has invalid shape
            ValueError: If n_clusters >= n_samples
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")

        n_samples = X.shape[0]
        if self.n_clusters >= n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) must be less than "
                f"n_samples ({n_samples})"
            )

        logger.info(f"Fitting K-Means with k={self.n_clusters}...")
        self.kmeans.fit(X)
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_

        logger.info(f"K-Means fitting complete. Inertia: {self.kmeans.inertia_:.2f}")
        return self.labels_

    def get_cluster_labels(self) -> np.ndarray:
        """
        Get cluster labels for each sample.

        Returns:
            Cluster labels of shape (n_samples,)

        Raises:
            ValueError: If clustering has not been fit yet
        """
        if self.labels_ is None:
            raise ValueError("K-Means has not been fitted yet. Call fit() first.")
        return self.labels_

    def get_centroids(self) -> np.ndarray:
        """
        Get cluster centroids.

        Returns:
            Centroid coordinates of shape (n_clusters, n_features)

        Raises:
            ValueError: If clustering has not been fit yet
        """
        if self.cluster_centers_ is None:
            raise ValueError("K-Means has not been fitted yet. Call fit() first.")
        return self.cluster_centers_

    def calculate_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.

        Args:
            X: Input vectors of shape (n_samples, n_features)

        Returns:
            Dictionary with keys:
            - 'inertia': Within-cluster sum of squares
            - 'silhouette_score': Silhouette coefficient

        Raises:
            ValueError: If clustering has not been fit yet
        """
        if self.labels_ is None:
            raise ValueError("K-Means has not been fitted yet. Call fit() first.")

        inertia = self.kmeans.inertia_

        # Calculate silhouette score
        try:
            silhouette = silhouette_score(X, self.labels_)
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
            silhouette = None

        silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
        logger.info(
            f"Clustering metrics - Inertia: {inertia:.2f}, "
            f"Silhouette: {silhouette_str}"
        )

        return {
            'inertia': float(inertia),
            'silhouette_score': float(silhouette) if silhouette is not None else None
        }

    def get_cluster_summary(self, labels: Optional[np.ndarray] = None) -> Dict[int, int]:
        """
        Get summary statistics for each cluster.

        Args:
            labels: Cluster labels (uses internal labels if not provided)

        Returns:
            Dictionary mapping cluster ID to number of samples

        Raises:
            ValueError: If clustering has not been fit yet and no labels provided
        """
        if labels is None:
            if self.labels_ is None:
                raise ValueError("K-Means has not been fitted yet.")
            labels = self.labels_

        unique_labels, counts = np.unique(labels, return_counts=True)
        summary = dict(zip(unique_labels, counts))

        logger.info("Cluster summary:")
        for cluster_id, count in sorted(summary.items()):
            logger.info(f"  Cluster {cluster_id}: {count} samples")

        return summary
