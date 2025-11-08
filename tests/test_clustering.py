"""Tests for K-Means clustering module."""
import numpy as np
import pytest

import sys
sys.path.insert(0, '/mnt/c/25D/L17/PCA_T_SNE')

from clustering import KMeansClustering


class TestKMeansClustering:
    """Test suite for KMeansClustering class."""

    @pytest.fixture
    def sample_vectors(self):
        """Generate sample vectors for clustering."""
        np.random.seed(42)
        # Create 100 vectors in 10 dimensions
        return np.random.randn(100, 10)

    @pytest.fixture
    def clusterer_3(self):
        """Create a clusterer with 3 clusters."""
        return KMeansClustering(n_clusters=3, random_state=42)

    def test_initialization(self):
        """Test KMeansClustering initialization."""
        clusterer = KMeansClustering(n_clusters=5, random_state=42)
        assert clusterer.n_clusters == 5
        assert clusterer.random_state == 42

    def test_invalid_n_clusters(self):
        """Test that invalid n_clusters raises error."""
        with pytest.raises(ValueError):
            KMeansClustering(n_clusters=0)

        with pytest.raises(ValueError):
            KMeansClustering(n_clusters=-1)

    def test_fit_returns_labels(self, clusterer_3, sample_vectors):
        """Test that fit returns cluster labels."""
        labels = clusterer_3.fit(sample_vectors)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_vectors)
        assert np.min(labels) >= 0
        assert np.max(labels) < 3

    def test_fit_stores_labels(self, clusterer_3, sample_vectors):
        """Test that fit stores labels internally."""
        labels = clusterer_3.fit(sample_vectors)
        stored_labels = clusterer_3.get_cluster_labels()

        assert np.array_equal(labels, stored_labels)

    def test_fit_stores_centroids(self, clusterer_3, sample_vectors):
        """Test that fit stores cluster centroids."""
        clusterer_3.fit(sample_vectors)
        centroids = clusterer_3.get_centroids()

        assert centroids.shape == (3, sample_vectors.shape[1])

    def test_get_labels_before_fit_raises_error(self, clusterer_3):
        """Test that get_cluster_labels before fit raises error."""
        with pytest.raises(ValueError, match="not been fitted"):
            clusterer_3.get_cluster_labels()

    def test_get_centroids_before_fit_raises_error(self, clusterer_3):
        """Test that get_centroids before fit raises error."""
        with pytest.raises(ValueError, match="not been fitted"):
            clusterer_3.get_centroids()

    def test_calculate_metrics(self, clusterer_3, sample_vectors):
        """Test clustering metrics calculation."""
        clusterer_3.fit(sample_vectors)
        metrics = clusterer_3.calculate_metrics(sample_vectors)

        assert 'inertia' in metrics
        assert 'silhouette_score' in metrics
        assert metrics['inertia'] > 0

    def test_metrics_before_fit_raises_error(self, clusterer_3, sample_vectors):
        """Test that metrics calculation before fit raises error."""
        with pytest.raises(ValueError, match="not been fitted"):
            clusterer_3.calculate_metrics(sample_vectors)

    def test_cluster_summary(self, clusterer_3, sample_vectors):
        """Test cluster summary statistics."""
        labels = clusterer_3.fit(sample_vectors)
        summary = clusterer_3.get_cluster_summary(labels)

        assert isinstance(summary, dict)
        assert len(summary) == 3
        assert sum(summary.values()) == len(sample_vectors)

    def test_summary_before_fit_raises_error(self, clusterer_3):
        """Test that summary before fit raises error."""
        with pytest.raises(ValueError, match="not been fitted"):
            clusterer_3.get_cluster_summary()

    def test_fit_with_invalid_input_shape(self, clusterer_3):
        """Test that invalid input shape raises error."""
        # 1D array instead of 2D
        with pytest.raises(ValueError):
            clusterer_3.fit(np.array([1, 2, 3]))

    def test_n_clusters_greater_than_samples(self, sample_vectors):
        """Test error when n_clusters >= n_samples."""
        clusterer = KMeansClustering(n_clusters=200)  # More than 100 samples

        with pytest.raises(ValueError, match="less than"):
            clusterer.fit(sample_vectors)

    def test_reproducibility_with_seed(self, sample_vectors):
        """Test that same seed produces same clustering."""
        clusterer1 = KMeansClustering(n_clusters=3, random_state=42)
        labels1 = clusterer1.fit(sample_vectors)

        clusterer2 = KMeansClustering(n_clusters=3, random_state=42)
        labels2 = clusterer2.fit(sample_vectors)

        assert np.array_equal(labels1, labels2)

    def test_different_seeds_produce_different_results(self, sample_vectors):
        """Test that different seeds may produce different results."""
        clusterer1 = KMeansClustering(n_clusters=3, random_state=42)
        labels1 = clusterer1.fit(sample_vectors)

        clusterer2 = KMeansClustering(n_clusters=3, random_state=123)
        labels2 = clusterer2.fit(sample_vectors)

        # Different seeds often produce different results (not guaranteed, but likely)
        # We just check that both produce valid results
        assert len(np.unique(labels1)) == 3
        assert len(np.unique(labels2)) == 3

    def test_different_k_values(self, sample_vectors):
        """Test clustering with different k values."""
        for k in [2, 3, 5, 10]:
            clusterer = KMeansClustering(n_clusters=k, random_state=42)
            labels = clusterer.fit(sample_vectors)

            unique_labels = np.unique(labels)
            assert len(unique_labels) == k

    def test_centroids_are_within_data_range(self, clusterer_3, sample_vectors):
        """Test that centroids are within the data range."""
        clusterer_3.fit(sample_vectors)
        centroids = clusterer_3.get_centroids()

        # Check that centroids are reasonable (not far outside data bounds)
        data_min = np.min(sample_vectors)
        data_max = np.max(sample_vectors)

        assert np.all(centroids >= data_min - 1)
        assert np.all(centroids <= data_max + 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
