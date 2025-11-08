# Project Planning: Sentence Clustering & Visualization Tool

## 1. Project Overview

### 1.1 Vision
Build a robust, maintainable Python tool that transforms sentences into vectors, clusters them, and visualizes the results in 3D space using two different dimensionality reduction techniques.

### 1.2 Success Criteria
- ✅ Successfully processes CSV files with sentences
- ✅ Generates accurate vector embeddings
- ✅ Produces meaningful clusters using K-Means
- ✅ Implements PCA using ONLY NumPy (mathematically correct)
- ✅ Integrates t-SNE from pre-built library
- ✅ Creates clear, interpretable 3D visualizations
- ✅ Runs smoothly in uv virtual environment
- ✅ Code is maintainable and well-documented

---

## 2. Architecture & Design

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                          │
│                  (CSV file + Configuration)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      main.py                                │
│              (Pipeline Orchestrator)                        │
│  - Coordinates all components                               │
│  - Handles CLI arguments                                    │
│  - Manages execution flow                                   │
└────┬──────────┬──────────┬──────────┬──────────┬───────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│  data   │ │vectori- │ │clusteri-│ │dimensi- │ │visuali- │
│ loader  │ │  zer    │ │   ng    │ │ onality │ │  zer    │
│   .py   │ │   .py   │ │   .py   │ │   .py   │ │   .py   │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                        Outputs                              │
│  - Clustering metrics (console)                             │
│  - PCA 3D visualization (PNG/interactive)                   │
│  - t-SNE 3D visualization (PNG/interactive)                 │
│  - Optional: CSV with cluster assignments                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Module Responsibilities

#### `main.py` - Pipeline Orchestrator
- **Purpose**: Entry point and workflow coordination
- **Responsibilities**:
  - Parse CLI arguments
  - Initialize all components
  - Execute pipeline steps in sequence
  - Handle errors and logging
  - Display summary statistics
- **Dependencies**: All other modules

#### `data_loader.py` - Data Ingestion
- **Purpose**: Load and validate CSV data
- **Responsibilities**:
  - Read CSV files with pandas
  - Extract sentence column
  - Validate data quality
  - Skip empty/null entries
  - Return clean sentence list
- **Dependencies**: pandas, logging
- **Input**: CSV file path, column name
- **Output**: List of sentences, metadata

#### `vectorizer.py` - Sentence Embeddings
- **Purpose**: Convert sentences to numerical vectors
- **Responsibilities**:
  - Load sentence-transformer model
  - Generate embeddings for all sentences
  - Handle batch processing
  - Manage memory efficiently
- **Dependencies**: sentence-transformers, numpy
- **Input**: List of sentences
- **Output**: NumPy array of shape (n_sentences, embedding_dim)

#### `clustering.py` - K-Means Clustering
- **Purpose**: Group similar sentence vectors
- **Responsibilities**:
  - Apply K-Means algorithm
  - Assign cluster labels
  - Calculate clustering metrics
  - Provide cluster statistics
- **Dependencies**: scikit-learn, numpy
- **Input**: Sentence vectors, n_clusters
- **Output**: Cluster labels, centroids, metrics

#### `dimensionality.py` - Dimensionality Reduction
- **Purpose**: Reduce vectors to 3D for visualization
- **Responsibilities**:
  - Implement PCA using NumPy only
  - Wrap t-SNE from scikit-learn
  - Calculate explained variance (PCA)
  - Handle numerical stability
- **Dependencies**: numpy (PCA), scikit-learn (t-SNE)
- **Input**: High-dimensional vectors
- **Output**: 3D coordinates

#### `visualizer.py` - 3D Plotting
- **Purpose**: Create visual representations
- **Responsibilities**:
  - Generate 3D scatter plots
  - Color by cluster
  - Add labels and legends
  - Save to file
  - Support interactive viewing
- **Dependencies**: matplotlib (or plotly)
- **Input**: 3D coordinates, cluster labels
- **Output**: Plot files, interactive displays

#### `config.py` - Configuration Management
- **Purpose**: Centralize all settings
- **Responsibilities**:
  - Define configuration schema
  - Load from CLI/file/env
  - Validate parameters
  - Provide defaults
- **Dependencies**: argparse, json/yaml
- **Input**: CLI args, config file
- **Output**: Config object

#### `utils.py` - Helper Functions
- **Purpose**: Shared utilities
- **Responsibilities**:
  - Timing decorators
  - Logging setup
  - Progress bars
  - File I/O helpers
- **Dependencies**: Various

### 2.3 Data Flow

```
1. CSV File → [data_loader] → List of Sentences
                                      ↓
2. Sentences → [vectorizer] → Vectors (n × 768)
                                      ↓
3. Vectors → [clustering] → Cluster Labels (n × 1)
                                      ↓
4. Vectors + Labels → [dimensionality] → 3D Coords (PCA & t-SNE)
                                      ↓
5. 3D Coords + Labels → [visualizer] → Plots (PNG/Interactive)
```

---

## 3. Implementation Strategy

### 3.1 Development Phases

#### Phase 1: Foundation (Week 1, Days 1-2)
**Goal**: Set up project structure and basic data pipeline

**Tasks**:
1. Project setup
   - Create directory structure
   - Initialize uv environment
   - Install core dependencies
   - Set up version control
2. Data loading implementation
   - Implement CSV loader
   - Add validation
   - Write unit tests
3. Vectorization implementation
   - Integrate sentence-transformers
   - Test embedding generation
   - Handle edge cases

**Deliverables**:
- Working data loader
- Working vectorizer
- Unit tests passing
- Can load CSV and generate embeddings

**Success Metrics**:
- Load 1000 sentences in < 5 seconds
- Generate embeddings without errors
- 100% test coverage for data_loader

---

#### Phase 2: Core Processing (Week 1-2, Days 3-5)
**Goal**: Implement clustering and dimensionality reduction

**Tasks**:
1. K-Means clustering
   - Implement clustering wrapper
   - Add metrics calculation
   - Test with synthetic data
2. **PCA implementation (CRITICAL)**
   - Implement from scratch using NumPy
   - Validate against scikit-learn
   - Ensure mathematical correctness
   - Handle edge cases (singular matrices)
3. t-SNE integration
   - Wrap scikit-learn t-SNE
   - Configure parameters
   - Test reproducibility

**Deliverables**:
- Working K-Means clustering
- **NumPy-only PCA (validated)**
- t-SNE integration
- Comprehensive tests

**Success Metrics**:
- PCA results match scikit-learn (within numerical precision)
- Silhouette score > 0.3 for test data
- t-SNE produces separable clusters
- All unit tests pass

---

#### Phase 3: Visualization (Week 2, Days 6-7)
**Goal**: Create clear, informative 3D visualizations

**Tasks**:
1. 3D plotting infrastructure
   - Choose backend (matplotlib vs plotly)
   - Implement base plotting function
   - Add color mapping
2. PCA visualization
   - Create PCA-specific plot
   - Add explained variance
   - Label axes appropriately
3. t-SNE visualization
   - Create t-SNE-specific plot
   - Ensure consistent styling
   - Add appropriate labels
4. Polish visuals
   - Adjust colors for clarity
   - Add legends
   - Test interactivity

**Deliverables**:
- Two 3D visualization functions
- Consistent styling across plots
- Save functionality
- Interactive display option

**Success Metrics**:
- Plots are visually clear
- Clusters are distinguishable
- Files saved successfully
- Interactive rotation works

---

#### Phase 4: Integration & Pipeline (Week 2-3, Days 8-10)
**Goal**: Connect all components into cohesive pipeline

**Tasks**:
1. Main pipeline implementation
   - Create pipeline orchestrator
   - Wire all components
   - Add error handling
   - Implement logging
2. CLI interface
   - Add argparse/click
   - Support all parameters
   - Add --help documentation
3. Configuration management
   - Support config files
   - Validate parameters
   - Set sensible defaults
4. End-to-end testing
   - Test full pipeline
   - Test various configurations
   - Test error scenarios

**Deliverables**:
- Working end-to-end pipeline
- CLI interface
- Configuration system
- Integration tests

**Success Metrics**:
- Process sample dataset successfully
- Generate both visualizations
- Handle errors gracefully
- CLI is user-friendly

---

#### Phase 5: Documentation & Polish (Week 3, Days 11-14)
**Goal**: Finalize and document the project

**Tasks**:
1. Documentation
   - Write comprehensive README
   - Add inline comments
   - Write docstrings
   - Create usage examples
2. Code quality
   - Run formatter (black)
   - Run linter (flake8/pylint)
   - Fix all issues
   - Remove debug code
3. Testing
   - Achieve high test coverage
   - Add edge case tests
   - Test on different OS
4. Final validation
   - Test with real datasets
   - Performance benchmarking
   - Get peer review

**Deliverables**:
- Complete documentation
- Clean, formatted code
- High test coverage
- Performance benchmarks

**Success Metrics**:
- README covers all features
- Code passes linting
- Test coverage > 80%
- Performance meets requirements

---

### 3.2 Critical Implementation Details

#### 3.2.1 NumPy-Only PCA Implementation

**Mathematical Steps**:
```python
# STEP 1: Center the data
X_centered = X - np.mean(X, axis=0)

# STEP 2: Compute covariance matrix
n_samples = X_centered.shape[0]
covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

# STEP 3: Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# STEP 4: Sort by eigenvalues (descending)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# STEP 5: Select top 3 components
top_3_eigenvectors = eigenvectors_sorted[:, :3]

# STEP 6: Project data
X_pca = X_centered @ top_3_eigenvectors

# STEP 7: Calculate explained variance ratio
total_variance = np.sum(eigenvalues)
explained_variance_ratio = eigenvalues_sorted[:3] / total_variance
```

**Validation Strategy**:
1. Implement PCA in NumPy
2. Run same data through scikit-learn PCA
3. Compare:
   - Eigenvalues (should match closely)
   - Projections (may differ in sign, but magnitudes should match)
   - Explained variance ratios
4. Accept if differences are < 1e-10 (numerical precision)

**Edge Cases to Handle**:
- Singular covariance matrices
- All features have zero variance
- More features than samples
- Very small eigenvalues (numerical stability)

---

#### 3.2.2 Memory Management

**Expected Memory Usage**:
- 10,000 sentences × 768 dims × 8 bytes = ~61 MB (vectors)
- 3 × 8 bytes × 10,000 = 240 KB (3D coordinates)
- Clustering: minimal additional memory
- Visualizations: depends on backend

**Optimization Strategies**:
1. Batch processing for vectorization (if memory constrained)
2. Use float32 instead of float64 where appropriate
3. Delete intermediate results after use
4. Stream processing for very large datasets (future)

---

#### 3.2.3 Error Handling Strategy

**Categories of Errors**:
1. **User Input Errors**
   - Missing CSV file → Clear error + suggestion
   - Invalid column name → List available columns
   - Too few samples → Minimum requirement message
2. **Processing Errors**
   - Model download failure → Check internet connection
   - Clustering convergence failure → Suggest parameter changes
   - Numerical instability → Fallback or warning
3. **System Errors**
   - Out of memory → Suggest smaller dataset
   - Disk space → Check output directory
   - Permission denied → Check file permissions

**Error Handling Pattern**:
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Provide actionable guidance
    print("Suggestion: Try XYZ")
    raise  # or handle gracefully
```

---

## 4. Technical Decisions

### 4.1 Technology Choices

#### Python Version
- **Choice**: Python 3.10+
- **Rationale**: Modern features, good library support, type hints

#### Virtual Environment
- **Choice**: uv
- **Rationale**: Required by PRD, fast, modern

#### Sentence Embeddings
- **Choice**: sentence-transformers (HuggingFace)
- **Default Model**: all-MiniLM-L6-v2
- **Rationale**:
  - Excellent quality/speed tradeoff
  - 384-768 dimensions
  - Pre-trained on large corpus
  - Easy to use
  - Active maintenance

#### Clustering
- **Choice**: scikit-learn KMeans
- **Rationale**:
  - Well-tested implementation
  - Fast and efficient
  - Good documentation
  - Standard in industry

#### Visualization Backend
- **Choice**: matplotlib (primary), plotly (optional)
- **Rationale**:
  - matplotlib: Universal, reliable, good 3D support
  - plotly: Better interactivity, modern interface
  - Both widely used and documented

### 4.2 Design Patterns

#### Pattern 1: Facade Pattern (Pipeline)
```python
class SentenceClusteringPipeline:
    """Facade that simplifies complex subsystem"""
    def __init__(self, config):
        self.loader = DataLoader(config)
        self.vectorizer = SentenceVectorizer(config)
        self.clusterer = KMeansClustering(config)
        self.pca = PCA_NumPy(n_components=3)
        self.tsne = TSNE_Wrapper(config)
        self.visualizer = Visualizer3D(config)
    
    def run(self):
        """Simple interface to complex workflow"""
        sentences = self.loader.load()
        vectors = self.vectorizer.vectorize(sentences)
        labels = self.clusterer.fit(vectors)
        # ... etc
```

#### Pattern 2: Strategy Pattern (Dimensionality Reduction)
```python
class DimensionalityReducer(ABC):
    @abstractmethod
    def fit_transform(self, X):
        pass

class PCA_NumPy(DimensionalityReducer):
    def fit_transform(self, X):
        # NumPy implementation
        pass

class TSNE_Wrapper(DimensionalityReducer):
    def fit_transform(self, X):
        # Scikit-learn wrapper
        pass
```

#### Pattern 3: Builder Pattern (Configuration)
```python
class ConfigBuilder:
    def __init__(self):
        self.config = {}
    
    def with_csv_path(self, path):
        self.config['csv_path'] = path
        return self
    
    def with_n_clusters(self, n):
        self.config['n_clusters'] = n
        return self
    
    def build(self):
        return Config(**self.config)
```

---

## 5. Testing Strategy

### 5.1 Test Pyramid

```
           ▲
          /E\      E2E Tests (10%)
         /───\     - Full pipeline tests
        /     \    - Real datasets
       / Integ \   Integration Tests (30%)
      /─────────\  - Component interactions
     /           \ - Multi-step workflows
    /    Unit     \ Unit Tests (60%)
   /───────────────\ - Individual functions
  /                 \ - Edge cases
 /___________________\ - Mocks and stubs
```

### 5.2 Test Coverage Goals

**Target Coverage**: 80%+

**Priority Areas** (95%+ coverage):
1. PCA implementation (CRITICAL)
2. Data loading and validation
3. Clustering logic

**Lower Priority** (60%+ coverage):
1. Visualization (hard to test)
2. CLI interface
3. Logging utilities

### 5.3 Key Test Cases

#### PCA Validation Tests
```python
def test_pca_matches_sklearn():
    """PCA results must match scikit-learn"""
    X = np.random.randn(100, 50)
    
    # Our implementation
    pca_ours = PCA_NumPy(n_components=3)
    X_ours = pca_ours.fit_transform(X)
    
    # Scikit-learn
    pca_sklearn = sklearn.decomposition.PCA(n_components=3)
    X_sklearn = pca_sklearn.fit_transform(X)
    
    # Compare (allowing for sign flips)
    assert np.allclose(np.abs(X_ours), np.abs(X_sklearn), atol=1e-10)
```

#### Integration Test
```python
def test_full_pipeline():
    """End-to-end pipeline test"""
    config = Config(
        csv_path='tests/data/sample.csv',
        n_clusters=3,
        random_seed=42
    )
    
    pipeline = SentenceClusteringPipeline(config)
    pipeline.run()
    
    # Verify outputs exist
    assert os.path.exists('output/pca_clusters_3d.png')
    assert os.path.exists('output/tsne_clusters_3d.png')
```

---

## 6. Risk Management

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PCA implementation incorrect | Medium | High | Extensive validation, compare with scikit-learn |
| Memory overflow with large datasets | Low | Medium | Batch processing, memory monitoring |
| Model download fails | Low | Medium | Cache models, provide clear error messages |
| Numerical instability in PCA | Low | Medium | Add regularization, handle edge cases |
| Poor clustering quality | Medium | Low | Provide metrics, allow parameter tuning |
| Visualization rendering issues | Low | Low | Test on multiple platforms |

### 6.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PCA takes longer than expected | Medium | Medium | Allocate extra time, start early |
| Testing takes longer than planned | Medium | Low | Incremental testing, prioritize critical tests |
| Dependency issues in uv | Low | Medium | Document dependencies clearly |
| Platform-specific bugs | Low | Low | Test on multiple OS early |

---

## 7. Performance Targets

### 7.1 Benchmarks

| Dataset Size | Vectorization | Clustering | PCA | t-SNE | Total |
|--------------|---------------|------------|-----|-------|-------|
| 100 sentences | < 5s | < 1s | < 1s | < 5s | < 15s |
| 1,000 sentences | < 30s | < 2s | < 2s | < 20s | < 60s |
| 10,000 sentences | < 5min | < 10s | < 10s | < 3min | < 10min |

### 7.2 Resource Constraints

- **Maximum Memory**: 4 GB for 10,000 sentences
- **Disk Space**: < 100 MB for outputs
- **CPU**: Single-core capable (multi-core beneficial)

---

## 8. Dependencies & Requirements

### 8.1 Core Dependencies

```txt
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
matplotlib>=3.7.0
torch>=2.0.0  # Required by sentence-transformers
```

### 8.2 Development Dependencies

```txt
# requirements-dev.txt
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.7.0
flake8>=6.1.0
mypy>=1.5.0
```

---

## 9. Deployment Checklist

### Pre-Release
- [ ] All tests passing
- [ ] Code formatted and linted
- [ ] Documentation complete
- [ ] Examples working
- [ ] Performance benchmarks met

### Release Artifacts
- [ ] Source code
- [ ] requirements.txt
- [ ] README.md
- [ ] Sample dataset
- [ ] Example outputs

### Post-Release
- [ ] Installation verified in clean environment
- [ ] User feedback collected
- [ ] Known issues documented

---

## 10. Success Metrics

### Quantitative Metrics
- **Code Quality**: 0 linting errors, test coverage > 80%
- **Performance**: Processes 1000 sentences in < 60 seconds
- **Correctness**: PCA matches scikit-learn within 1e-10
- **Usability**: Setup time < 10 minutes

### Qualitative Metrics
- Code is readable and maintainable
- Documentation is clear and complete
- Visualizations are interpretable
- Error messages are helpful

---

## 11. Next Steps

### Immediate Actions (Start Here)
1. Set up project structure
2. Initialize uv environment
3. Create all module files
4. Implement data loader first (quick win)

### First Milestone (Week 1)
- Working data pipeline (load → vectorize)
- Basic tests passing
- Can generate embeddings

### Critical Path
**Data Loading → Vectorization → PCA Implementation → Integration**

The PCA implementation is the most complex and risky component, so validate it early and thoroughly.

---

## 12. Questions & Decisions Needed

### Open Questions
1. Which visualization backend? (matplotlib vs plotly)
   - **Recommendation**: Start with matplotlib, add plotly as enhancement
2. Should we support config files or CLI-only?
   - **Recommendation**: Both, CLI overrides config file
3. What should be default number of clusters?
   - **Recommendation**: 5 (good balance for most datasets)
4. Should we cache vectorization results?
   - **Recommendation**: Yes, as future enhancement

### Decisions Made
- ✅ Use sentence-transformers for embeddings
- ✅ Use scikit-learn for K-Means and t-SNE
- ✅ Implement PCA from scratch using NumPy
- ✅ Target Python 3.10+
- ✅ Use uv for virtual environment

---

## 13. References & Resources

### Documentation
- [sentence-transformers docs](https://www.sbert.net/)
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [NumPy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [matplotlib 3D plotting](https://matplotlib.org/stable/gallery/mplot3d/index.html)

### PCA Theory
- [PCA Tutorial](https://arxiv.org/abs/1404.1100)
- [PCA from Scratch](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html)

### Similar Projects
- Research existing clustering visualization tools
- Study best practices for dimensionality reduction

---

**Last Updated**: 2025-11-08
**Version**: 1.0
