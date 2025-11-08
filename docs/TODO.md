# TODO List: Sentence Clustering & Visualization Tool

## Project Setup

### Environment & Dependencies
- [x] Create project directory structure
- [x] Initialize uv virtual environment
- [x] Create `requirements.txt` with all dependencies
- [ ] Create `pyproject.toml` (optional, for modern Python packaging)
- [ ] Create `.gitignore` file
- [🚧] Test environment activation and package installation
- [x] Create `output/` directory for generated plots

### Project Structure
- [x] Create `main.py` (entry point)
- [x] Create `data_loader.py` (CSV loading)
- [x] Create `vectorizer.py` (sentence embeddings)
- [x] Create `clustering.py` (K-Means)
- [x] Create `dimensionality.py` (PCA and t-SNE)
- [x] Create `visualizer.py` (3D plotting)
- [ ] Create `config.py` (configuration management)
- [x] Create `utils.py` (helper functions)
- [ ] Create `README.md` (user documentation)

---

## Step 1: Data Loading & Vectorization

### CSV Loading Module (`data_loader.py`)
- [x] Implement `load_csv()` function
  - [x] Accept file path as parameter
  - [x] Support configurable text column name
  - [x] Support configurable delimiter (default: comma)
  - [x] Handle UTF-8 encoding
  - [x] Validate CSV structure
- [x] Implement data validation
  - [x] Check if file exists
  - [x] Verify specified column exists
  - [x] Count total rows
  - [x] Detect and skip empty/null sentences
  - [x] Log warnings for skipped rows
- [x] Implement `extract_sentences()` function
  - [x] Return list of valid sentences
  - [x] Return sentence metadata (optional: IDs, original indices)
- [x] Add error handling for file I/O issues
- [x] Write unit tests for data loading

### Vectorization Module (`vectorizer.py`)
- [x] Implement `SentenceVectorizer` class
  - [x] Initialize with model name (default: 'all-MiniLM-L6-v2')
  - [x] Load sentence-transformers model
  - [x] Handle model download on first run
- [x] Implement `vectorize_sentences()` method
  - [x] Accept list of sentences
  - [x] Generate embeddings using model.encode()
  - [x] Return numpy array of vectors
  - [x] Show progress bar for large datasets
- [x] Implement `get_vector_dimension()` method
- [x] Add batch processing for memory efficiency
- [x] Handle edge cases (empty strings, special characters)
- [x] Add error handling for model loading failures
- [x] Write unit tests for vectorization

---

## Step 2: K-Means Clustering

### Clustering Module (`clustering.py`)
- [x] Implement `KMeansClustering` class
  - [x] Initialize with configurable parameters:
    - [x] `n_clusters` (required)
    - [x] `max_iter` (default: 300)
    - [x] `n_init` (default: 3)
    - [x] `random_state` (for reproducibility)
- [x] Implement `fit()` method
  - [x] Use scikit-learn's KMeans
  - [x] Fit on sentence vectors
  - [x] Store cluster labels
  - [x] Store cluster centroids
- [x] Implement `get_cluster_labels()` method
- [x] Implement `get_centroids()` method
- [x] Implement `calculate_metrics()` method
  - [x] Calculate inertia (within-cluster sum of squares)
  - [x] Calculate silhouette score
  - [x] Return metrics as dictionary
- [x] Implement `get_cluster_summary()` method
  - [x] Count sentences per cluster
  - [x] Return cluster statistics
- [x] Add validation for minimum samples requirement
- [x] Write unit tests for clustering

---

## Step 3: PCA Implementation (NumPy Only)

### PCA Module (`dimensionality.py`)
- [x] Implement `PCA_NumPy` class (no scikit-learn!)
  - [x] Initialize with `n_components=3`
- [x] Implement `fit()` method
  - [x] Center the data (subtract mean)
  - [x] Compute covariance matrix using numpy
  - [x] Calculate eigenvalues and eigenvectors
  - [x] Sort eigenvalues in descending order
  - [x] Select top 3 eigenvectors
  - [x] Store principal components
  - [x] Calculate explained variance ratio
- [x] Implement `transform()` method
  - [x] Project data onto principal components
  - [x] Return 3D coordinates
- [x] Implement `fit_transform()` method
  - [x] Combine fit and transform
- [x] Implement `get_explained_variance()` method
  - [x] Return variance explained by each PC
- [x] Add numerical stability checks
- [🚧] **Validate against scikit-learn PCA** (for testing only)
- [x] Write comprehensive unit tests
  - [x] Test with known datasets
  - [x] Compare results with scikit-learn
  - [x] Test edge cases (singular matrices, etc.)

---

## Step 4: t-SNE Implementation

### t-SNE Module (`dimensionality.py`)
- [x] Implement `TSNE_Wrapper` class
  - [x] Initialize with configurable parameters:
    - [x] `n_components=3`
    - [x] `perplexity` (default: 30)
    - [x] `learning_rate` (default: 200)
    - [x] `n_iter` (default: 1000)
    - [x] `random_state` (for reproducibility)
- [x] Implement `fit_transform()` method
  - [x] Use scikit-learn's TSNE
  - [x] Return 3D coordinates
  - [x] Add progress callback if available
- [x] Add warning for computational time on large datasets
- [x] Handle edge cases (n_samples < perplexity)
- [x] Write unit tests for t-SNE wrapper

---

## Visualization

### Visualization Module (`visualizer.py`)
- [x] Implement `Visualizer3D` class
  - [x] Initialize with output directory
  - [x] Set up matplotlib/plotly backend
- [x] Implement `plot_3d_clusters()` method
  - [x] Accept 3D coordinates and cluster labels
  - [x] Create 3D scatter plot
  - [x] Color points by cluster
  - [x] Add axis labels
  - [x] Add legend
  - [x] Add title
  - [x] Support interactive rotation
- [x] Implement `plot_pca()` method
  - [x] Call plot_3d_clusters with PCA data
  - [x] Set title to "PCA Projection"
  - [x] Label axes as PC1, PC2, PC3
  - [x] Display explained variance in title/subtitle
- [x] Implement `plot_tsne()` method
  - [x] Call plot_3d_clusters with t-SNE data
  - [x] Set title to "t-SNE Projection"
  - [x] Label axes as t-SNE 1, 2, 3
- [x] Implement `save_plot()` method
  - [x] Support PNG, PDF, SVG formats
  - [x] Use configurable DPI
- [x] Implement consistent color mapping across plots
- [ ] Add optional centroid plotting
- [ ] Add optional hover tooltips (for interactive plots)
- [x] Implement `show_plots()` method for display
- [ ] Write tests for plot generation

---

## Configuration Management

### Config Module (`config.py`)
- [ ] Create `Config` class or dataclass
  - [ ] Define all configurable parameters
  - [ ] Set default values
- [ ] Implement configuration loading from:
  - [ ] Command-line arguments
  - [ ] Config file (JSON/YAML)
  - [ ] Environment variables
- [ ] Implement configuration validation
- [ ] Implement `get_config()` helper function
- [ ] Document all configuration options

---

## Main Pipeline

### Main Module (`main.py`)
- [x] Implement `SentenceClusteringPipeline` class
  - [x] Initialize with configuration
  - [x] Store all components (loader, vectorizer, etc.)
- [x] Implement `run()` method - orchestrate full pipeline:
  - [x] Load CSV data
  - [x] Vectorize sentences
  - [x] Perform K-Means clustering
  - [x] Calculate clustering metrics
  - [x] Reduce dimensions with PCA
  - [x] Reduce dimensions with t-SNE
  - [x] Generate visualizations
  - [x] Save outputs
  - [x] Print summary statistics
- [x] Implement logging throughout pipeline
  - [x] Log each step with timestamp
  - [x] Log execution time per step
  - [x] Log final summary
- [x] Implement error handling and rollback
- [x] Add progress indicators for long operations
- [x] Implement command-line interface
  - [x] Use argparse or click
  - [x] Support --help flag
  - [x] Support all config parameters as CLI args
- [x] Add `if __name__ == "__main__":` entry point

---

## Utilities

### Utils Module (`utils.py`)
- [x] Implement timing decorator
- [ ] Implement progress bar helper
- [x] Implement logging setup function
- [x] Implement output directory creation
- [ ] Implement file validation helpers
- [ ] Implement data export functions (optional):
  - [ ] Export cluster assignments to CSV
  - [ ] Export metrics to JSON
- [ ] Write utility tests

---

## Testing

### Unit Tests
- [x] Create `tests/` directory
- [x] Write tests for `data_loader.py`
  - [x] Test CSV loading with valid data
  - [x] Test with missing file
  - [x] Test with invalid column name
  - [x] Test with empty CSV
- [ ] Write tests for `vectorizer.py`
  - [ ] Test vectorization output shape
  - [ ] Test with various sentence lengths
  - [ ] Test with special characters
- [x] Write tests for `clustering.py`
  - [x] Test clustering with known data
  - [x] Test metrics calculation
  - [x] Test with invalid n_clusters
- [x] Write tests for `dimensionality.py` (PCA)
  - [🚧] **CRITICAL**: Validate NumPy PCA against scikit-learn
  - [x] Test with synthetic data
  - [x] Test explained variance calculation
  - [x] Test edge cases
- [x] Write tests for `dimensionality.py` (t-SNE)
  - [x] Test output shape
  - [x] Test reproducibility with random_state
- [ ] Write tests for `visualizer.py`
  - [ ] Test plot generation (if possible)
  - [ ] Test file saving
- [🚧] Run all tests with pytest

### Integration Tests
- [x] Create sample CSV dataset (100-500 sentences)
- [ ] Test end-to-end pipeline with sample data
- [ ] Verify all output files are created
- [ ] Verify plots render correctly
- [ ] Test with different configurations
- [ ] Test error scenarios:
  - [ ] Missing CSV file
  - [ ] Invalid parameters
  - [ ] Insufficient samples

### Validation Tests
- [🚧] Verify PCA mathematical correctness
  - [🚧] Compare eigenvalues with scikit-learn
  - [🚧] Compare projections with scikit-learn
  - [x] Check orthogonality of principal components
- [ ] Verify clustering quality
  - [ ] Check silhouette scores
  - [ ] Visual inspection of plots
- [ ] Verify reproducibility
  - [ ] Run pipeline twice with same seed
  - [ ] Compare outputs

---

## Documentation

### User Documentation
- [ ] Write comprehensive README.md
  - [ ] Project description
  - [ ] Features list
  - [ ] Installation instructions
  - [ ] Quick start guide
  - [ ] Usage examples
  - [ ] Configuration options
  - [ ] Troubleshooting section
- [ ] Create example configuration file
- [ ] Create sample CSV dataset
- [ ] Add inline code comments
  - [ ] Especially for PCA implementation
  - [ ] Complex algorithm sections
- [ ] Write docstrings for all functions/classes
  - [ ] Follow NumPy or Google docstring format
  - [ ] Include parameters, returns, raises

### Developer Documentation
- [ ] Document project architecture
- [ ] Document module dependencies
- [ ] Document testing procedures
- [ ] Create contribution guidelines (if open source)

---

## Polish & Finalization

### Code Quality
- [ ] Run code formatter (black or autopep8)
- [ ] Run linter (pylint or flake8)
- [ ] Fix all linting issues
- [ ] Ensure PEP 8 compliance
- [ ] Remove debug print statements
- [ ] Remove commented-out code
- [ ] Add type hints (optional but recommended)

### Performance
- [ ] Profile code for bottlenecks
- [ ] Optimize slow sections
- [ ] Add batch processing for large datasets
- [ ] Verify memory usage is acceptable

### User Experience
- [ ] Test CLI usability
- [ ] Improve error messages
- [ ] Add helpful warnings
- [ ] Test on different operating systems
  - [ ] Linux
  - [ ] macOS
  - [ ] Windows

### Final Testing
- [ ] Run full test suite
- [ ] Test with multiple datasets
- [ ] Test with edge cases:
  - [ ] Very small dataset (10 sentences)
  - [ ] Large dataset (10,000 sentences)
  - [ ] Different numbers of clusters
- [ ] Get peer review (if applicable)

---

## Deployment & Delivery

- [ ] Create requirements.txt with pinned versions
- [ ] Verify installation in fresh uv environment
- [ ] Create distribution package (optional)
- [ ] Tag release version (v1.0.0)
- [ ] Create release notes
- [ ] Archive final deliverables

---

## Optional Enhancements (Future)

- [ ] Add automatic optimal k detection (elbow method)
- [ ] Add UMAP as alternative to t-SNE
- [ ] Add web-based dashboard
- [ ] Add GPU acceleration for embeddings
- [ ] Add cluster naming/labeling
- [ ] Add export to interactive HTML
- [ ] Add support for multiple embedding models
- [ ] Add hierarchical clustering option

---

## Notes & Reminders

### Critical Requirements
⚠️ **PCA MUST use ONLY NumPy** - no scikit-learn for PCA implementation!
⚠️ **t-SNE CAN use pre-built libraries** - scikit-learn is fine
⚠️ **Must run in uv virtual environment**
⚠️ **All visualizations must be 3D** (not 2D)

### Testing Priority
1. **Highest**: PCA mathematical correctness
2. **High**: End-to-end pipeline functionality
3. **Medium**: Individual component unit tests
4. **Low**: Edge case handling

### Time Estimates
- Project setup: 2-4 hours
- Data loading & vectorization: 4-6 hours
- K-Means clustering: 2-3 hours
- **PCA implementation (NumPy): 6-8 hours** (most complex)
- t-SNE integration: 1-2 hours
- Visualization: 4-6 hours
- Testing: 6-8 hours
- Documentation: 4-6 hours
- Polish & finalization: 4-6 hours

**Total Estimated Time: 35-50 hours**

---

## Progress Tracking

### Legend
- [ ] Not started
- [🚧] In progress
- [✅] Completed
- [⏸️] Blocked/On hold
- [❌] Cancelled/Removed

### Current Sprint Focus
<!-- Update this section as you work -->
- Currently working on: PCA validation testing and end-to-end tests
- Blockers: Installing dependencies in uv venv (in progress)
- Next up: Run full test suite, create README.md, end-to-end validation

---

**Last Updated:** 2025-11-08
