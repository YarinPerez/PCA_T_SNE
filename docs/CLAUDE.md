# CLAUDE.md - AI Assistant Implementation Guide

This document provides specific instructions for Claude (or any AI assistant) to implement the Sentence Clustering & Visualization Tool efficiently and correctly.

---

## 🎯 Implementation Overview

You are implementing a Python program that:
1. Loads sentences from CSV → converts to vectors
2. Clusters vectors using K-Means
3. Visualizes in 3D using PCA (NumPy-only implementation)
4. Visualizes in 3D using t-SNE (pre-built library)

**Environment**: Python 3.10+ with uv virtual environment

---

## 🚨 CRITICAL REQUIREMENTS

### Must-Follow Rules
1. **PCA MUST be implemented using ONLY NumPy** - Do NOT use scikit-learn's PCA
2. **t-SNE MUST use pre-built library** - scikit-learn is recommended
3. **All visualizations must be 3D** - not 2D
4. **Must work in uv virtual environment**
5. **Validate PCA against scikit-learn** - results must match within numerical precision

### Quality Standards
- Follow PEP 8 style guidelines
- Include docstrings for all functions/classes
- Add type hints where beneficial
- Write clear, descriptive variable names
- Handle errors gracefully with informative messages
- Log important steps and timing information

---

## 📁 Project Structure to Create

```
sentence-clustering/
├── main.py                 # Entry point and pipeline orchestrator
├── data_loader.py          # CSV loading and validation
├── vectorizer.py           # Sentence to vector conversion
├── clustering.py           # K-Means clustering wrapper
├── dimensionality.py       # PCA (NumPy) and t-SNE implementations
├── visualizer.py           # 3D plotting functions
├── config.py              # Configuration management
├── utils.py               # Helper functions (logging, timing)
├── requirements.txt       # Python dependencies
├── README.md             # User documentation
├── tests/                # Test directory
│   ├── test_data_loader.py
│   ├── test_vectorizer.py
│   ├── test_clustering.py
│   ├── test_dimensionality.py
│   └── data/
│       └── sample.csv    # Test data
└── output/               # Output directory for plots
```

---

## 🔧 Implementation Order

### Phase 1: Setup (Start Here)
1. Create all module files with basic structure
2. Create requirements.txt
3. Implement utils.py (logging, timing decorators)
4. Create sample CSV for testing

### Phase 2: Data Pipeline
1. Implement data_loader.py
2. Implement vectorizer.py
3. Test data loading → vectorization flow

### Phase 3: Core Processing
1. Implement clustering.py (easiest)
2. **Implement dimensionality.py PCA** (most critical/complex)
3. Implement dimensionality.py t-SNE wrapper (easy)
4. Validate PCA thoroughly

### Phase 4: Visualization
1. Implement visualizer.py base functionality
2. Add PCA-specific visualization
3. Add t-SNE-specific visualization

### Phase 5: Integration
1. Implement main.py pipeline
2. Add CLI interface
3. Implement config.py
4. End-to-end testing

---

## 💻 Code Templates & Guidance

### Template 1: data_loader.py

```python
"""
Module for loading and validating CSV data containing sentences.
"""
import pandas as pd
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading sentences from CSV files."""
    
    def __init__(self, csv_path: str, text_column: str = 'text', 
                 encoding: str = 'utf-8'):
        """
        Initialize DataLoader.
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of column containing sentences
            encoding: File encoding (default: 'utf-8')
        """
        self.csv_path = csv_path
        self.text_column = text_column
        self.encoding = encoding
    
    def load(self) -> Tuple[List[str], pd.DataFrame]:
        """
        Load sentences from CSV.
        
        Returns:
            Tuple of (sentences list, original dataframe)
        
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If text column not found
        """
        # TODO: Implement
        # 1. Check file exists
        # 2. Load CSV with pandas
        # 3. Validate text_column exists
        # 4. Extract sentences, skip empty/null
        # 5. Log statistics (loaded, skipped)
        # 6. Return sentences and df
        pass
    
    def validate_data(self, df: pd.DataFrame) -> List[str]:
        """Extract and validate sentences from dataframe."""
        # TODO: Implement validation logic
        pass
```

**Key Points**:
- Use pandas for CSV loading (it's robust)
- Skip empty/null sentences
- Log how many sentences loaded vs skipped
- Return both sentences and original dataframe (useful for metadata)

---

### Template 2: vectorizer.py

```python
"""
Module for converting sentences into vector embeddings.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class SentenceVectorizer:
    """Converts sentences to dense vector embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vectorizer with specified model.
        
        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def vectorize(self, sentences: List[str], 
                  batch_size: int = 32,
                  show_progress: bool = True) -> np.ndarray:
        """
        Convert sentences to vectors.
        
        Args:
            sentences: List of sentences to vectorize
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            NumPy array of shape (n_sentences, embedding_dim)
        """
        # TODO: Implement
        # Use model.encode() with show_progress_bar parameter
        # Return numpy array
        pass
```

**Key Points**:
- Default model: 'all-MiniLM-L6-v2' (good balance of speed/quality)
- Use batch processing for efficiency
- Show progress bar for large datasets
- Return numpy array directly

---

### Template 3: clustering.py

```python
"""
K-Means clustering for sentence vectors.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class KMeansClustering:
    """Wrapper for K-Means clustering with metrics."""
    
    def __init__(self, n_clusters: int, random_state: int = 42,
                 max_iter: int = 300, n_init: int = 10):
        """
        Initialize K-Means clusterer.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations
            n_init: Number of initializations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init
        )
        self.labels_ = None
        self.cluster_centers_ = None
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Means and return cluster labels.
        
        Args:
            X: Input vectors of shape (n_samples, n_features)
        
        Returns:
            Cluster labels of shape (n_samples,)
        """
        # TODO: Implement
        # 1. Fit K-Means
        # 2. Store labels and centers
        # 3. Log fitting info
        # 4. Return labels
        pass
    
    def calculate_metrics(self, X: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        
        Returns:
            Dictionary with 'inertia' and 'silhouette_score'
        """
        # TODO: Implement
        # Calculate inertia (within-cluster sum of squares)
        # Calculate silhouette score
        pass
```

**Key Points**:
- Use scikit-learn's KMeans (it's optimized)
- Always set random_state for reproducibility
- Calculate and return both inertia and silhouette score
- Store labels and centroids for later use

---

### Template 4: dimensionality.py (MOST CRITICAL)

```python
"""
Dimensionality reduction: PCA (NumPy-only) and t-SNE (scikit-learn).
"""
import numpy as np
from sklearn.manifold import TSNE
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PCA_NumPy:
    """
    Principal Component Analysis using ONLY NumPy.
    
    CRITICAL: This implementation must NOT use scikit-learn's PCA.
    Must use only numpy for all operations.
    """
    
    def __init__(self, n_components: int = 3):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of principal components (must be 3 for this project)
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA_NumPy':
        """
        Fit PCA on data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            self
        """
        # STEP 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # STEP 2: Compute covariance matrix
        n_samples = X_centered.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # STEP 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # STEP 4: Sort by eigenvalues (descending order)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_indices]
        eigenvectors_sorted = eigenvectors[:, sorted_indices]
        
        # STEP 5: Select top n_components
        self.components_ = eigenvectors_sorted[:, :self.n_components]
        
        # STEP 6: Calculate explained variance
        self.explained_variance_ = eigenvalues_sorted[:self.n_components].real
        total_variance = np.sum(eigenvalues_sorted.real)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        logger.info(f"PCA fitted. Explained variance ratio: {self.explained_variance_ratio_}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transform")
        
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and transform data in one step."""
        self.fit(X)
        return self.transform(X)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Return explained variance ratio for each component."""
        return self.explained_variance_ratio_


class TSNE_Wrapper:
    """Wrapper for scikit-learn's t-SNE."""
    
    def __init__(self, n_components: int = 3, perplexity: float = 30.0,
                 learning_rate: float = 200.0, n_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize t-SNE.
        
        Args:
            n_components: Number of dimensions (must be 3)
            perplexity: t-SNE perplexity parameter
            learning_rate: t-SNE learning rate
            n_iter: Maximum iterations
            random_state: Random seed
        """
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
            verbose=1  # Show progress
        )
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply t-SNE dimensionality reduction.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Transformed data of shape (n_samples, 3)
        """
        logger.info("Running t-SNE (this may take a while)...")
        X_tsne = self.tsne.fit_transform(X)
        logger.info("t-SNE complete")
        return X_tsne
```

**CRITICAL PCA Notes**:
- Use `np.linalg.eig()` NOT `np.linalg.eigh()` initially (eig handles complex eigenvalues)
- Take `.real` part of eigenvalues to handle numerical artifacts
- Sort eigenvalues in DESCENDING order
- Remember to center data before computing covariance
- Store mean for transforming new data
- Validate results against scikit-learn (in tests, not in production code)

---

### Template 5: visualizer.py

```python
"""
3D visualization of clusters using matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class Visualizer3D:
    """Creates 3D visualizations of clustered data."""
    
    def __init__(self, output_dir: str = './output', figsize: tuple = (12, 10)):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size in inches
        """
        self.output_dir = output_dir
        self.figsize = figsize
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_3d_clusters(self, coordinates: np.ndarray, labels: np.ndarray,
                         title: str, axis_labels: List[str],
                         filename: Optional[str] = None,
                         show: bool = True) -> None:
        """
        Create 3D scatter plot of clusters.
        
        Args:
            coordinates: 3D coordinates of shape (n_samples, 3)
            labels: Cluster labels of shape (n_samples,)
            title: Plot title
            axis_labels: List of 3 axis labels [x, y, z]
            filename: Filename to save (without extension)
            show: Whether to display plot
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Use colormap for cluster colors
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                coordinates[mask, 2],
                c=[colors[i]],
                label=f'Cluster {label}',
                s=50,
                alpha=0.6,
                edgecolors='k',
                linewidth=0.5
            )
        
        # Set labels and title
        ax.set_xlabel(axis_labels[0], fontsize=12)
        ax.set_ylabel(axis_labels[1], fontsize=12)
        ax.set_zlabel(axis_labels[2], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        
        # Improve viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            filepath = f"{self.output_dir}/{filename}.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_pca(self, coordinates: np.ndarray, labels: np.ndarray,
                 explained_variance: np.ndarray, 
                 filename: str = 'pca_clusters_3d',
                 show: bool = True) -> None:
        """
        Plot PCA projection with explained variance.
        
        Args:
            coordinates: PCA coordinates (n_samples, 3)
            labels: Cluster labels
            explained_variance: Explained variance ratio for 3 components
            filename: Output filename
            show: Whether to display
        """
        title = (f"PCA Projection (Variance Explained: "
                f"{explained_variance[0]:.1%}, {explained_variance[1]:.1%}, "
                f"{explained_variance[2]:.1%})")
        
        axis_labels = [
            f'PC1 ({explained_variance[0]:.1%})',
            f'PC2 ({explained_variance[1]:.1%})',
            f'PC3 ({explained_variance[2]:.1%})'
        ]
        
        self.plot_3d_clusters(coordinates, labels, title, axis_labels, 
                            filename, show)
    
    def plot_tsne(self, coordinates: np.ndarray, labels: np.ndarray,
                  filename: str = 'tsne_clusters_3d',
                  show: bool = True) -> None:
        """
        Plot t-SNE projection.
        
        Args:
            coordinates: t-SNE coordinates (n_samples, 3)
            labels: Cluster labels
            filename: Output filename
            show: Whether to display
        """
        title = "t-SNE Projection"
        axis_labels = ['t-SNE 1', 't-SNE 2', 't-SNE 3']
        
        self.plot_3d_clusters(coordinates, labels, title, axis_labels,
                            filename, show)
```

**Key Points**:
- Use matplotlib for 3D plotting (reliable, well-documented)
- Use consistent colors across both PCA and t-SNE plots
- Save at high DPI (300) for quality
- Include explained variance in PCA plot title
- Make plots interactive (rotatable) by default

---

### Template 6: main.py

```python
"""
Main pipeline for sentence clustering and visualization.
"""
import argparse
import logging
import time
from pathlib import Path

from data_loader import DataLoader
from vectorizer import SentenceVectorizer
from clustering import KMeansClustering
from dimensionality import PCA_NumPy, TSNE_Wrapper
from visualizer import Visualizer3D
from utils import setup_logging, Timer

logger = logging.getLogger(__name__)

class SentenceClusteringPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.loader = DataLoader(
            csv_path=config['csv_path'],
            text_column=config.get('text_column', 'text')
        )
        self.vectorizer = SentenceVectorizer(
            model_name=config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.clusterer = KMeansClustering(
            n_clusters=config['n_clusters'],
            random_state=config.get('random_seed', 42)
        )
        self.pca = PCA_NumPy(n_components=3)
        self.tsne = TSNE_Wrapper(
            n_components=3,
            perplexity=config.get('tsne_perplexity', 30),
            n_iter=config.get('tsne_iterations', 1000),
            random_state=config.get('random_seed', 42)
        )
        self.visualizer = Visualizer3D(
            output_dir=config.get('output_dir', './output')
        )
    
    def run(self):
        """Execute the full pipeline."""
        logger.info("="*60)
        logger.info("Starting Sentence Clustering & Visualization Pipeline")
        logger.info("="*60)
        
        # Step 1: Load data
        with Timer("Data Loading"):
            sentences, df = self.loader.load()
            logger.info(f"Loaded {len(sentences)} sentences")
        
        # Step 2: Vectorize
        with Timer("Vectorization"):
            vectors = self.vectorizer.vectorize(sentences)
            logger.info(f"Generated vectors of shape {vectors.shape}")
        
        # Step 3: Cluster
        with Timer("Clustering"):
            labels = self.clusterer.fit(vectors)
            metrics = self.clusterer.calculate_metrics(vectors)
            logger.info(f"Clustering complete. Inertia: {metrics['inertia']:.2f}, "
                       f"Silhouette: {metrics['silhouette_score']:.3f}")
        
        # Step 4: PCA
        with Timer("PCA Dimensionality Reduction"):
            coords_pca = self.pca.fit_transform(vectors)
            explained_var = self.pca.get_explained_variance_ratio()
            logger.info(f"PCA complete. Explained variance: {explained_var}")
        
        # Step 5: t-SNE
        with Timer("t-SNE Dimensionality Reduction"):
            coords_tsne = self.tsne.fit_transform(vectors)
            logger.info("t-SNE complete")
        
        # Step 6: Visualize
        with Timer("Visualization"):
            # PCA plot
            self.visualizer.plot_pca(
                coords_pca, labels, explained_var,
                filename='pca_clusters_3d',
                show=self.config.get('show_plots', True)
            )
            
            # t-SNE plot
            self.visualizer.plot_tsne(
                coords_tsne, labels,
                filename='tsne_clusters_3d',
                show=self.config.get('show_plots', True)
            )
        
        logger.info("="*60)
        logger.info("Pipeline Complete!")
        logger.info("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sentence Clustering & Visualization Tool'
    )
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file containing sentences')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of column containing sentences')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters for K-Means')
    parser.add_argument('--embedding_model', type=str, 
                       default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory for output files')
    parser.add_argument('--tsne_perplexity', type=float, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--tsne_iterations', type=int, default=1000,
                       help='t-SNE maximum iterations')
    parser.add_argument('--no_show_plots', action='store_true',
                       help='Do not display plots (only save)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Create config from args
    config = {
        'csv_path': args.csv_path,
        'text_column': args.text_column,
        'n_clusters': args.n_clusters,
        'embedding_model': args.embedding_model,
        'random_seed': args.random_seed,
        'output_dir': args.output_dir,
        'tsne_perplexity': args.tsne_perplexity,
        'tsne_iterations': args.tsne_iterations,
        'show_plots': not args.no_show_plots
    }
    
    # Run pipeline
    pipeline = SentenceClusteringPipeline(config)
    pipeline.run()


if __name__ == '__main__':
    main()
```

---

### Template 7: utils.py

```python
"""
Utility functions for the project.
"""
import logging
import time
from contextlib import contextmanager

def setup_logging(level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

@contextmanager
def Timer(name: str):
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer("My Operation"):
            # code here
            pass
    """
    logger = logging.getLogger(__name__)
    start = time.time()
    logger.info(f"Starting: {name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {name} (took {elapsed:.2f}s)")
```

---

## 🧪 Testing Strategy

### Critical Test: PCA Validation

```python
# tests/test_dimensionality.py

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA
from dimensionality import PCA_NumPy

def test_pca_matches_sklearn():
    """CRITICAL: Verify our PCA matches scikit-learn."""
    # Generate random data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    
    # Our implementation
    pca_ours = PCA_NumPy(n_components=3)
    X_ours = pca_ours.fit_transform(X)
    
    # Scikit-learn
    pca_sklearn = SklearnPCA(n_components=3)
    X_sklearn = pca_sklearn.fit_transform(X)
    
    # Compare explained variance ratios
    assert np.allclose(
        pca_ours.explained_variance_ratio_,
        pca_sklearn.explained_variance_ratio_,
        atol=1e-10
    ), "Explained variance ratios don't match"
    
    # Compare projections (magnitude only, signs may differ)
    assert np.allclose(
        np.abs(X_ours),
        np.abs(X_sklearn),
        atol=1e-10
    ), "PCA projections don't match scikit-learn"
    
    print("✅ PCA validation passed!")
```

**Run this test immediately after implementing PCA!**

---

## 📝 requirements.txt

```txt
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
matplotlib>=3.7.0

# Required by sentence-transformers
torch>=2.0.0
transformers>=4.30.0

# Development (optional)
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.1.0
```

---

## 🎨 Sample CSV for Testing

Create `tests/data/sample.csv`:

```csv
id,text,category
1,Machine learning is transforming industries,tech
2,Artificial intelligence will change the world,tech
3,I love pizza and pasta,food
4,The weather is beautiful today,casual
5,Neural networks are fascinating,tech
6,Cooking is my favorite hobby,food
7,I enjoy hiking in the mountains,casual
8,Deep learning requires lots of data,tech
9,Italian cuisine is delicious,food
10,Summer is the best season,casual
11,Computer vision has many applications,tech
12,I prefer tea over coffee,food
13,Reading books is relaxing,casual
14,Natural language processing is complex,tech
15,Baking bread is an art,food
```

---

## ✅ Implementation Checklist

Use this checklist as you implement:

### Setup
- [ ] Create project structure
- [ ] Create all module files
- [ ] Create requirements.txt
- [ ] Create sample CSV

### Core Modules
- [ ] Implement utils.py (logging, Timer)
- [ ] Implement data_loader.py
- [ ] Implement vectorizer.py
- [ ] Implement clustering.py
- [ ] **Implement dimensionality.py PCA (CRITICAL)**
- [ ] Implement dimensionality.py t-SNE
- [ ] Implement visualizer.py
- [ ] Implement main.py

### Validation
- [ ] **Test PCA against scikit-learn** (MUST PASS)
- [ ] Test data loading with sample CSV
- [ ] Test end-to-end pipeline
- [ ] Verify both plots are generated
- [ ] Check plot quality

### Polish
- [ ] Add docstrings to all functions
- [ ] Add error handling
- [ ] Test CLI interface
- [ ] Create README.md

---

## 🚀 Quick Start Commands

```bash
# Setup
mkdir sentence-clustering && cd sentence-clustering
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install numpy pandas scikit-learn sentence-transformers matplotlib torch

# Create structure
mkdir tests tests/data output

# Run (after implementation)
python main.py --csv_path tests/data/sample.csv --n_clusters 3

# Test (after creating tests)
pytest tests/ -v
```

---

## ⚠️ Common Pitfalls to Avoid

1. **PCA Implementation**
   - ❌ Don't use `sklearn.decomposition.PCA` in production code
   - ❌ Don't forget to center data before covariance
   - ❌ Don't forget to sort eigenvalues in descending order
   - ✅ Do validate against scikit-learn in tests

2. **Memory Management**
   - ❌ Don't load entire CSV into memory if very large
   - ✅ Do use batch processing for vectorization if needed

3. **Error Handling**
   - ❌ Don't let program crash on invalid input
   - ✅ Do provide helpful error messages

4. **Reproducibility**
   - ❌ Don't forget to set random seeds
   - ✅ Do use consistent random_state everywhere

5. **Visualization**
   - ❌ Don't use 2D projections (must be 3D)
   - ✅ Do use consistent colors between PCA and t-SNE plots

---

## 📊 Expected Output

When pipeline runs successfully, you should see:

```
2025-11-08 10:00:00 - __main__ - INFO - ============================================================
2025-11-08 10:00:00 - __main__ - INFO - Starting Sentence Clustering & Visualization Pipeline
2025-11-08 10:00:00 - __main__ - INFO - ============================================================
2025-11-08 10:00:00 - __main__ - INFO - Starting: Data Loading
2025-11-08 10:00:01 - data_loader - INFO - Loaded 15 sentences (0 skipped)
2025-11-08 10:00:01 - __main__ - INFO - Loaded 15 sentences
2025-11-08 10:00:01 - __main__ - INFO - Completed: Data Loading (took 1.23s)
2025-11-08 10:00:01 - __main__ - INFO - Starting: Vectorization
2025-11-08 10:00:01 - vectorizer - INFO - Loading sentence transformer model: all-MiniLM-L6-v2
2025-11-08 10:00:05 - vectorizer - INFO - Model loaded. Embedding dimension: 384
2025-11-08 10:00:06 - __main__ - INFO - Generated vectors of shape (15, 384)
2025-11-08 10:00:06 - __main__ - INFO - Completed: Vectorization (took 5.12s)
2025-11-08 10:00:06 - __main__ - INFO - Starting: Clustering
2025-11-08 10:00:06 - __main__ - INFO - Clustering complete. Inertia: 45.32, Silhouette: 0.421
2025-11-08 10:00:06 - __main__ - INFO - Completed: Clustering (took 0.23s)
2025-11-08 10:00:06 - __main__ - INFO - Starting: PCA Dimensionality Reduction
2025-11-08 10:00:06 - dimensionality - INFO - PCA fitted. Explained variance ratio: [0.234 0.189 0.156]
2025-11-08 10:00:06 - __main__ - INFO - Completed: PCA Dimensionality Reduction (took 0.15s)
2025-11-08 10:00:06 - __main__ - INFO - Starting: t-SNE Dimensionality Reduction
2025-11-08 10:00:06 - dimensionality - INFO - Running t-SNE (this may take a while)...
2025-11-08 10:00:12 - dimensionality - INFO - t-SNE complete
2025-11-08 10:00:12 - __main__ - INFO - Completed: t-SNE Dimensionality Reduction (took 6.45s)
2025-11-08 10:00:12 - __main__ - INFO - Starting: Visualization
2025-11-08 10:00:14 - visualizer - INFO - Plot saved to ./output/pca_clusters_3d.png
2025-11-08 10:00:16 - visualizer - INFO - Plot saved to ./output/tsne_clusters_3d.png
2025-11-08 10:00:16 - __main__ - INFO - Completed: Visualization (took 4.21s)
2025-11-08 10:00:16 - __main__ - INFO - ============================================================
2025-11-08 10:00:16 - __main__ - INFO - Pipeline Complete!
2025-11-08 10:00:16 - __main__ - INFO - ============================================================
```

---

## 🎯 Priority Order

**Implement in this order for maximum efficiency:**

1. **High Priority** (Do First)
   - utils.py (needed by everyone)
   - data_loader.py (simple, quick win)
   - vectorizer.py (straightforward)
   - Sample CSV creation

2. **Medium Priority** (Core Logic)
   - clustering.py (easy, uses scikit-learn)
   - **dimensionality.py PCA** (complex, critical)
   - dimensionality.py t-SNE (easy wrapper)

3. **Lower Priority** (Integration)
   - visualizer.py
   - main.py
   - Testing

---

## 💡 Pro Tips for Claude

1. **Start with the template code provided** - it has the correct structure
2. **Implement PCA carefully** - this is the most critical and complex part
3. **Test PCA immediately** after implementation against scikit-learn
4. **Use descriptive logging** throughout to help users understand progress
5. **Handle errors gracefully** with actionable error messages
6. **Add docstrings** as you write code, not after
7. **Keep it simple** - don't over-engineer
8. **Test incrementally** - don't wait until everything is done

---

## 🏁 Success Criteria

Your implementation is complete when:

1. ✅ PCA implementation uses ONLY NumPy
2. ✅ PCA results match scikit-learn (validated in tests)
3. ✅ Sample CSV processes successfully end-to-end
4. ✅ Both 3D visualizations are generated
5. ✅ Plots clearly show cluster separation
6. ✅ All code has docstrings
7. ✅ Error handling is in place
8. ✅ CLI interface works
9. ✅ README is complete

---

**Good luck with the implementation! Focus on getting PCA right - that's the most critical component.**

---

**Version**: 1.0
**Last Updated**: 2025-11-08
