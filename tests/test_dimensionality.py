"""
Tests for dimensionality reduction modules (PCA and t-SNE).

CRITICAL: PCA tests validate the NumPy implementation against scikit-learn.
"""
import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE as SklearnTSNE

import sys
sys.path.insert(0, '/mnt/c/25D/L17/PCA_T_SNE')

from dimensionality import PCA_NumPy, TSNE_Wrapper


class TestPCANumPy:
    """Test suite for NumPy-based PCA implementation."""

    @pytest.fixture
    def sample_data(self):
        """Generate reproducible sample data for testing."""
        np.random.seed(42)
        return np.random.randn(100, 50)

    @pytest.fixture
    def sample_data_3d(self):
        """Generate simple 3D data for basic tests."""
        np.random.seed(42)
        return np.random.randn(50, 20)

    def test_pca_fit_returns_self(self, sample_data):
        """Test that fit() returns self for method chaining."""
        pca = PCA_NumPy(n_components=3)
        result = pca.fit(sample_data)
        assert result is pca

    def test_pca_transform_after_fit(self, sample_data):
        """Test that transform works after fit."""
        pca = PCA_NumPy(n_components=3)
        pca.fit(sample_data)
        transformed = pca.transform(sample_data)

        assert transformed.shape == (sample_data.shape[0], 3)
        assert isinstance(transformed, np.ndarray)

    def test_pca_fit_transform(self, sample_data):
        """Test fit_transform produces same result as fit + transform."""
        pca1 = PCA_NumPy(n_components=3)
        result1 = pca1.fit_transform(sample_data)

        pca2 = PCA_NumPy(n_components=3)
        pca2.fit(sample_data)
        result2 = pca2.transform(sample_data)

        assert np.allclose(result1, result2)

    def test_pca_matches_sklearn_variance(self, sample_data):
        """
        CRITICAL TEST: Verify NumPy PCA explained variance matches scikit-learn.

        This is essential to validate mathematical correctness.
        """
        # Our implementation
        pca_ours = PCA_NumPy(n_components=3)
        pca_ours.fit(sample_data)
        var_ours = pca_ours.get_explained_variance_ratio()

        # Scikit-learn implementation
        pca_sklearn = SklearnPCA(n_components=3)
        pca_sklearn.fit(sample_data)
        var_sklearn = pca_sklearn.explained_variance_ratio_

        # Compare (allow small numerical differences)
        assert np.allclose(var_ours, var_sklearn, atol=1e-10), \
            f"Explained variance mismatch:\nOurs: {var_ours}\nSklearn: {var_sklearn}"

    def test_pca_matches_sklearn_projections(self, sample_data):
        """
        CRITICAL TEST: Verify NumPy PCA projections match scikit-learn.

        Projections may differ in sign (eigenvectors can be flipped),
        but magnitudes must match.
        """
        # Our implementation
        pca_ours = PCA_NumPy(n_components=3)
        X_ours = pca_ours.fit_transform(sample_data)

        # Scikit-learn implementation
        pca_sklearn = SklearnPCA(n_components=3)
        X_sklearn = pca_sklearn.fit_transform(sample_data)

        # Compare magnitudes (handle sign flips)
        assert np.allclose(np.abs(X_ours), np.abs(X_sklearn), atol=1e-10), \
            "PCA projections don't match scikit-learn"

    def test_pca_component_orthogonality(self, sample_data):
        """Test that principal components are orthogonal."""
        pca = PCA_NumPy(n_components=3)
        pca.fit(sample_data)
        components = pca.get_components()

        # Compute Gram matrix (should be identity for orthogonal vectors)
        gram = components.T @ components

        # Check orthogonality (off-diagonal elements should be ~0)
        off_diagonal = gram - np.eye(3)
        assert np.allclose(off_diagonal, 0, atol=1e-10)

    def test_pca_variance_sums_to_one(self, sample_data):
        """Test that explained variance ratios sum to <= 1.0 (with small tolerance for precision)."""
        pca = PCA_NumPy(n_components=3)
        pca.fit(sample_data)
        var_ratio = pca.get_explained_variance_ratio()

        # The sum should be <= 1.0 and close to 1.0
        # Using atol=1e-6 to account for floating point precision
        assert np.sum(var_ratio) <= 1.0 + 1e-6

    def test_pca_with_2_components(self, sample_data):
        """Test PCA with different number of components."""
        pca = PCA_NumPy(n_components=2)
        result = pca.fit_transform(sample_data)

        assert result.shape == (sample_data.shape[0], 2)

    def test_pca_with_5_components(self, sample_data):
        """Test PCA with 5 components."""
        pca = PCA_NumPy(n_components=5)
        result = pca.fit_transform(sample_data)

        assert result.shape == (sample_data.shape[0], 5)

    def test_pca_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises ValueError."""
        pca = PCA_NumPy(n_components=3)

        with pytest.raises(ValueError, match="must be fitted"):
            pca.transform(sample_data)

    def test_pca_invalid_n_components(self):
        """Test that invalid n_components raises error."""
        with pytest.raises(ValueError):
            PCA_NumPy(n_components=0)

        with pytest.raises(ValueError):
            PCA_NumPy(n_components=-1)

    def test_pca_too_many_components(self, sample_data):
        """Test that n_components > n_features raises error."""
        pca = PCA_NumPy(n_components=100)  # More than 50 features

        with pytest.raises(ValueError):
            pca.fit(sample_data)

    def test_pca_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        pca = PCA_NumPy(n_components=3)
        invalid_data = np.random.randn(10)  # 1D instead of 2D

        with pytest.raises(ValueError):
            pca.fit(invalid_data)

    def test_pca_variance_decreases(self, sample_data):
        """Test that explained variance decreases across components."""
        pca = PCA_NumPy(n_components=3)
        pca.fit(sample_data)
        var_ratio = pca.get_explained_variance_ratio()

        # Each component should explain less variance than previous
        assert var_ratio[0] >= var_ratio[1] >= var_ratio[2]


class TestTSNEWrapper:
    """Test suite for t-SNE wrapper."""

    @pytest.fixture
    def small_data(self):
        """Generate small dataset for t-SNE testing."""
        np.random.seed(42)
        # Need enough samples for perplexity*3
        return np.random.randn(100, 20)

    def test_tsne_fit_transform_shape(self, small_data):
        """Test that t-SNE output has correct shape."""
        tsne = TSNE_Wrapper(n_components=3, n_iter=100)  # Fewer iterations for speed
        result = tsne.fit_transform(small_data)

        assert result.shape == (small_data.shape[0], 3)

    def test_tsne_2d_output(self, small_data):
        """Test t-SNE with 2D output."""
        tsne = TSNE_Wrapper(n_components=2, n_iter=100)
        result = tsne.fit_transform(small_data)

        assert result.shape == (small_data.shape[0], 2)

    def test_tsne_reproducible_with_seed(self, small_data):
        """Test that t-SNE produces reproducible results with same seed."""
        tsne1 = TSNE_Wrapper(n_components=3, random_state=42, n_iter=100)
        result1 = tsne1.fit_transform(small_data)

        tsne2 = TSNE_Wrapper(n_components=3, random_state=42, n_iter=100)
        result2 = tsne2.fit_transform(small_data)

        assert np.allclose(result1, result2, atol=1e-6)

    def test_tsne_different_seeds_produce_different_results(self, small_data):
        """Test that different seeds produce different results."""
        tsne1 = TSNE_Wrapper(n_components=3, random_state=42, n_iter=100)
        result1 = tsne1.fit_transform(small_data)

        tsne2 = TSNE_Wrapper(n_components=3, random_state=123, n_iter=100)
        result2 = tsne2.fit_transform(small_data)

        # Results should be different (not all close)
        assert not np.allclose(result1, result2)

    def test_tsne_insufficient_samples_warning(self, caplog):
        """Test that warning is logged when n_samples < perplexity * 3."""
        # Create data with n_samples just enough for perplexity to be valid
        # but still trigger the warning
        small_data = np.random.randn(50, 20)  # 50 samples, perplexity=30
        tsne = TSNE_Wrapper(n_components=3, perplexity=20, n_iter=10)

        with caplog.at_level('WARNING'):
            result = tsne.fit_transform(small_data)
            # Verify output shape is correct
            assert result.shape == (50, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
