# Product Requirements Document: Sentence Clustering & Visualization Tool

## 1. Overview

### 1.1 Product Description
A Python-based analytical tool that loads sentences from CSV files, converts them into vector embeddings, clusters them using K-Means algorithm, and visualizes the clusters in 3D space using both PCA (Principal Component Analysis) and t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction techniques.

### 1.2 Purpose
Enable users to analyze and visualize semantic relationships between sentences by clustering similar sentences and displaying them in an interpretable 3D space.

### 1.3 Target Users
- Data scientists analyzing text datasets
- NLP researchers exploring sentence embeddings
- Machine learning engineers working on text clustering tasks
- Anyone needing to visualize high-dimensional text data

---

## 2. Technical Stack

### 2.1 Runtime Environment
- **Language**: Python 3.10+
- **Environment Manager**: uv (virtual environment)
- **Package Manager**: pip (via uv)

### 2.2 Core Dependencies
- **Sentence Vectorization**: sentence-transformers, transformers, or similar NLP library
- **Clustering**: scikit-learn (K-Means)
- **Dimensionality Reduction**: 
  - PCA: Custom implementation using NumPy only
  - t-SNE: scikit-learn or other pre-built library
- **Numerical Computing**: NumPy
- **Data Handling**: pandas (for CSV loading)
- **Visualization**: matplotlib or plotly (for 3D plotting)

---

## 3. Functional Requirements

### 3.1 Step 1: Data Loading & Vectorization

#### 3.1.1 CSV Input
- **FR-1.1**: System SHALL accept CSV files as input
- **FR-1.2**: CSV file MUST contain at least one column with sentence text
- **FR-1.3**: System SHALL support configurable column selection for sentence extraction
- **FR-1.4**: System SHALL handle UTF-8 encoded text

#### 3.1.2 Sentence Vectorization
- **FR-1.5**: System SHALL convert each sentence into a numerical vector representation
- **FR-1.6**: System SHALL use pre-trained sentence embedding models (e.g., sentence-transformers)
- **FR-1.7**: System SHALL handle variable-length sentences
- **FR-1.8**: Output vectors SHALL have consistent dimensionality across all sentences

#### 3.1.3 Data Validation
- **FR-1.9**: System SHALL skip empty or null sentences
- **FR-1.10**: System SHALL log warnings for any skipped sentences
- **FR-1.11**: System SHALL validate that at least minimum required sentences exist (e.g., >= number of clusters)

---

### 3.2 Step 2: K-Means Clustering

#### 3.2.1 Clustering Algorithm
- **FR-2.1**: System SHALL apply K-Means clustering to sentence vectors
- **FR-2.2**: Number of clusters (k) SHALL be configurable by user
- **FR-2.3**: System SHALL use scikit-learn's K-Means implementation
- **FR-2.4**: System SHALL support configurable random seed for reproducibility

#### 3.2.2 Clustering Configuration
- **FR-2.5**: System SHALL support configuration of:
  - Number of clusters (k)
  - Maximum iterations
  - Number of initializations (n_init)
  - Convergence tolerance
- **FR-2.6**: System SHALL assign each sentence to exactly one cluster

#### 3.2.3 Clustering Output
- **FR-2.7**: System SHALL output cluster assignments for each sentence
- **FR-2.8**: System SHALL calculate and display clustering metrics:
  - Inertia (within-cluster sum of squares)
  - Silhouette score
- **FR-2.9**: System SHALL identify cluster centroids

---

### 3.3 Step 3: PCA Visualization (NumPy-only Implementation)

#### 3.3.1 PCA Implementation
- **FR-3.1**: System SHALL implement PCA using ONLY NumPy (no scikit-learn)
- **FR-3.2**: PCA SHALL reduce vector dimensionality to exactly 3 components
- **FR-3.3**: Implementation SHALL include:
  - Data centering (mean subtraction)
  - Covariance matrix computation
  - Eigenvalue/eigenvector calculation
  - Projection onto top 3 principal components

#### 3.3.2 PCA Visualization
- **FR-3.4**: System SHALL display sentence vectors in 3D space after PCA reduction
- **FR-3.5**: Each point SHALL be colored by its cluster assignment
- **FR-3.6**: Visualization SHALL include:
  - Axis labels (PC1, PC2, PC3)
  - Legend showing cluster colors
  - Title indicating "PCA Projection"
- **FR-3.7**: Plot SHALL be interactive (rotatable) if using appropriate backend
- **FR-3.8**: System SHALL display explained variance ratio for each principal component

---

### 3.4 Step 4: t-SNE Visualization

#### 3.4.1 t-SNE Implementation
- **FR-4.1**: System SHALL use pre-built t-SNE library (e.g., scikit-learn)
- **FR-4.2**: t-SNE SHALL reduce vector dimensionality to exactly 3 components
- **FR-4.3**: System SHALL support configurable t-SNE parameters:
  - Perplexity (default: 30)
  - Learning rate (default: 200)
  - Number of iterations (default: 1000)
  - Random seed for reproducibility

#### 3.4.2 t-SNE Visualization
- **FR-4.4**: System SHALL display sentence vectors in 3D space after t-SNE reduction
- **FR-4.5**: Each point SHALL be colored by its cluster assignment
- **FR-4.6**: Visualization SHALL include:
  - Axis labels (t-SNE 1, t-SNE 2, t-SNE 3)
  - Legend showing cluster colors
  - Title indicating "t-SNE Projection"
- **FR-4.7**: Plot SHALL be interactive (rotatable) if using appropriate backend

---

### 3.5 General Visualization Requirements

- **FR-5.1**: Both plots SHALL use consistent color mapping for clusters
- **FR-5.2**: System SHALL support saving plots as image files (PNG, PDF)
- **FR-5.3**: System SHALL display both plots simultaneously or sequentially based on configuration
- **FR-5.4**: Cluster centroids MAY be optionally displayed on plots
- **FR-5.5**: Point size SHALL be configurable
- **FR-5.6**: System SHALL support hover tooltips showing sentence text (if using interactive backend)

---

## 4. Non-Functional Requirements

### 4.1 Performance
- **NFR-1.1**: System SHALL process up to 10,000 sentences within 5 minutes on standard hardware
- **NFR-1.2**: PCA computation SHALL complete in < 30 seconds for datasets up to 10,000 vectors
- **NFR-1.3**: t-SNE computation time SHALL be acceptable (may take several minutes for large datasets)
- **NFR-1.4**: System SHALL provide progress indicators for long-running operations

### 4.2 Usability
- **NFR-2.1**: System SHALL provide clear command-line interface or configuration file
- **NFR-2.2**: Error messages SHALL be descriptive and actionable
- **NFR-2.3**: System SHALL provide help documentation via --help flag
- **NFR-2.4**: System SHALL log execution steps and timing information

### 4.3 Maintainability
- **NFR-3.1**: Code SHALL follow PEP 8 style guidelines
- **NFR-3.2**: Functions SHALL have docstrings describing parameters and return values
- **NFR-3.3**: Code SHALL be modular with separate functions for each major step
- **NFR-3.4**: Configuration parameters SHALL be centralized

### 4.4 Reliability
- **NFR-4.1**: System SHALL handle errors gracefully without crashing
- **NFR-4.2**: System SHALL validate input data before processing
- **NFR-4.3**: System SHALL produce reproducible results when using fixed random seeds

### 4.5 Portability
- **NFR-5.1**: System SHALL run on Linux, macOS, and Windows
- **NFR-5.2**: System SHALL be installed and run within uv virtual environment
- **NFR-5.3**: All dependencies SHALL be specified in requirements.txt or pyproject.toml

---

## 5. System Architecture

### 5.1 Module Structure

```
sentence-clustering/
├── main.py                 # Entry point, orchestrates workflow
├── data_loader.py          # CSV loading and preprocessing
├── vectorizer.py           # Sentence to vector conversion
├── clustering.py           # K-Means clustering logic
├── dimensionality.py       # PCA (NumPy) and t-SNE implementations
├── visualizer.py           # 3D plotting functions
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
└── README.md             # User documentation
```

### 5.2 Data Flow

```
CSV File → Load Data → Vectorize Sentences → K-Means Clustering
                                                    ↓
                                            Cluster Labels
                                                    ↓
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                            PCA (NumPy only)                t-SNE (pre-built)
                                    ↓                               ↓
                            3D Visualization                3D Visualization
```

---

## 6. Configuration Parameters

### 6.1 User-Configurable Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | string | required | Path to input CSV file |
| `text_column` | string | "text" | Column name containing sentences |
| `n_clusters` | int | 5 | Number of K-Means clusters |
| `embedding_model` | string | "all-MiniLM-L6-v2" | Sentence transformer model name |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `pca_components` | int | 3 | Number of PCA components (fixed) |
| `tsne_perplexity` | int | 30 | t-SNE perplexity parameter |
| `tsne_iterations` | int | 1000 | t-SNE max iterations |
| `output_dir` | string | "./output" | Directory for saving plots |
| `plot_format` | string | "png" | Image format for saved plots |
| `interactive` | bool | true | Enable interactive plots |

---

## 7. Input/Output Specifications

### 7.1 Input Format

**CSV File Structure:**
```csv
id,text,metadata
1,"This is the first sentence.",category_a
2,"Another example sentence here.",category_b
3,"Third sentence for clustering.",category_a
```

Requirements:
- Header row required
- At least one column with text data
- UTF-8 encoding
- Comma-separated (default), configurable delimiter

### 7.2 Output Artifacts

1. **Console Output:**
   - Number of sentences loaded
   - Vector dimensions
   - Clustering metrics (inertia, silhouette score)
   - PCA explained variance ratios
   - Execution time for each step

2. **Visualization Files:**
   - `pca_clusters_3d.png` - PCA projection plot
   - `tsne_clusters_3d.png` - t-SNE projection plot

3. **Optional Data Exports:**
   - `cluster_assignments.csv` - Sentences with cluster labels
   - `cluster_statistics.json` - Clustering metrics and summary

---

## 8. Constraints & Assumptions

### 8.1 Constraints
- PCA implementation MUST use only NumPy (no scikit-learn for PCA)
- t-SNE MAY use any pre-built library
- Must run in uv virtual environment
- 3D visualizations only (not 2D)

### 8.2 Assumptions
- Input sentences are in English (or compatible with chosen embedding model)
- Sentences are preprocessed (no need for extensive cleaning)
- User has basic Python knowledge for setup
- Sufficient memory available for loading all vectors into memory
- Number of clusters (k) is predetermined by user

---

## 9. Future Enhancements (Out of Scope for v1)

- **FE-1**: Support for multiple embedding models comparison
- **FE-2**: Automatic optimal cluster number detection (elbow method, silhouette analysis)
- **FE-3**: Hierarchical clustering alternatives
- **FE-4**: Web-based interactive visualization dashboard
- **FE-5**: Batch processing of multiple CSV files
- **FE-6**: Export to interactive HTML files
- **FE-7**: UMAP as additional dimensionality reduction option
- **FE-8**: Cluster labeling/naming based on representative sentences
- **FE-9**: GPU acceleration for embedding generation

---

## 10. Success Metrics

### 10.1 Technical Metrics
- Successfully loads and vectorizes 100% of valid sentences
- PCA implementation produces mathematically correct results (verified against scikit-learn)
- Clusters are well-separated (silhouette score > 0.3)
- Visualizations clearly show cluster separation

### 10.2 User Experience Metrics
- Setup time < 10 minutes for new users
- Total execution time acceptable for typical datasets (< 5 minutes for 5000 sentences)
- Visualizations are interpretable and actionable

---

## 11. Testing Requirements

### 11.1 Unit Tests
- Test CSV loading with various formats
- Test vectorization with edge cases (empty strings, special characters)
- Test PCA implementation against known results
- Test clustering with different k values

### 11.2 Integration Tests
- End-to-end workflow with sample dataset
- Verify output file generation
- Validate visualization rendering

### 11.3 Validation Tests
- Compare NumPy PCA results with scikit-learn PCA (mathematical correctness)
- Verify cluster assignments are consistent with given random seed
- Validate 3D plot generation for both PCA and t-SNE

---

## 12. Installation & Setup

### 12.1 Environment Setup

```bash
# Create uv virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### 12.2 Required Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
matplotlib>=3.7.0
```

---

## 13. Documentation Requirements

- **DR-1**: README.md with quick start guide
- **DR-2**: Code comments for complex logic (especially PCA implementation)
- **DR-3**: Example configuration file
- **DR-4**: Sample CSV dataset for testing
- **DR-5**: Usage examples with different parameters
- **DR-6**: Troubleshooting section for common issues

---

## 14. Acceptance Criteria

The product is considered complete when:

1. ✅ All functional requirements (FR-1.x through FR-5.x) are implemented
2. ✅ PCA is implemented using ONLY NumPy
3. ✅ t-SNE uses pre-built library
4. ✅ Both 3D visualizations render correctly
5. ✅ Program runs successfully in uv virtual environment
6. ✅ Sample dataset processes end-to-end without errors
7. ✅ Output plots clearly show cluster separation
8. ✅ Basic documentation is provided
9. ✅ Code follows Python best practices

---

## 15. Timeline & Milestones

### Phase 1: Foundation (Week 1)
- Setup project structure
- Implement data loading and vectorization
- Unit tests for data pipeline

### Phase 2: Core Processing (Week 1-2)
- Implement K-Means clustering
- Implement NumPy-only PCA
- Implement t-SNE integration
- Validation tests

### Phase 3: Visualization (Week 2)
- 3D plotting for PCA
- 3D plotting for t-SNE
- Polish visualizations

### Phase 4: Integration & Polish (Week 2-3)
- End-to-end integration
- Configuration management
- Documentation
- Testing with various datasets

---

## 16. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-08 | Initial | Initial PRD creation |

---

## 17. Appendix

### 17.1 PCA Algorithm Overview (NumPy Implementation)

```python
# Pseudocode for NumPy-only PCA
1. Center the data: X_centered = X - X.mean(axis=0)
2. Compute covariance matrix: C = (X_centered.T @ X_centered) / (n - 1)
3. Compute eigenvalues and eigenvectors: eigenvalues, eigenvectors = np.linalg.eig(C)
4. Sort by eigenvalues (descending): sorted_indices = np.argsort(eigenvalues)[::-1]
5. Select top 3 eigenvectors: top_3_eigenvectors = eigenvectors[:, sorted_indices[:3]]
6. Project data: X_pca = X_centered @ top_3_eigenvectors
7. Calculate explained variance ratio
```

### 17.2 Example Usage

```python
# Command-line usage example
python main.py --csv_path data/sentences.csv --n_clusters 5 --text_column "sentence"

# Programmatic usage example
from main import SentenceClusteringPipeline

pipeline = SentenceClusteringPipeline(
    csv_path="data/sentences.csv",
    n_clusters=5,
    random_seed=42
)
pipeline.run()
```

---

**End of PRD**
