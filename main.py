"""
Main pipeline orchestrator for Sentence Clustering & Visualization.

Coordinates all components to load data, vectorize sentences, cluster,
reduce dimensionality, and visualize results.
"""
import argparse
import logging
from pathlib import Path

from data_loader import DataLoader
from vectorizer import SentenceVectorizer
from clustering import KMeansClustering
from dimensionality import PCA_NumPy, TSNE_Wrapper
from visualizer import Visualizer3D
from utils import setup_logging, Timer, ensure_output_dir

logger = logging.getLogger(__name__)


class SentenceClusteringPipeline:
    """Main pipeline orchestrator for sentence clustering and visualization."""

    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary with keys:
                - csv_path: Path to input CSV file
                - text_column: Column name with sentences
                - n_clusters: Number of clusters
                - embedding_model: Sentence transformer model name
                - random_seed: Random seed for reproducibility
                - output_dir: Directory for output files
                - tsne_perplexity: t-SNE perplexity parameter
                - tsne_iterations: t-SNE max iterations
                - show_plots: Whether to display plots

        Raises:
            ValueError: If required config keys are missing
        """
        # Validate required config
        if 'csv_path' not in config:
            raise ValueError("Config must include 'csv_path'")
        if 'n_clusters' not in config:
            raise ValueError("Config must include 'n_clusters'")

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

        # Store config for later use (needed for adaptive perplexity)
        self.config_tsne_perplexity = config.get('tsne_perplexity', 30)
        self.config_tsne_iterations = config.get('tsne_iterations', 1000)
        self.random_seed = config.get('random_seed', 42)

        # t-SNE will be initialized in run() after we know the number of samples
        self.tsne = None

        self.visualizer = Visualizer3D(
            output_dir=config.get('output_dir', './output')
        )

        logger.info("Pipeline initialized successfully")

    def run(self):
        """Execute the full clustering and visualization pipeline."""
        logger.info("=" * 70)
        logger.info("SENTENCE CLUSTERING & VISUALIZATION PIPELINE")
        logger.info("=" * 70)

        # Step 1: Load data
        with Timer("Data Loading"):
            sentences, df = self.loader.load()
            logger.info(f"✓ Loaded {len(sentences)} sentences from CSV")

        # Initialize t-SNE with adaptive perplexity based on number of samples
        n_samples = len(sentences)
        # Perplexity should be less than n_samples / 3
        # Default to min(30, n_samples // 3)
        adaptive_perplexity = min(self.config_tsne_perplexity, max(5, n_samples // 3 - 1))

        self.tsne = TSNE_Wrapper(
            n_components=3,
            perplexity=adaptive_perplexity,
            n_iter=self.config_tsne_iterations,
            random_state=self.random_seed
        )

        logger.info(f"t-SNE configured with perplexity={adaptive_perplexity} "
                   f"(adaptive based on n_samples={n_samples})")

        # Step 2: Vectorize sentences
        with Timer("Sentence Vectorization"):
            vectors = self.vectorizer.vectorize(sentences)
            logger.info(f"✓ Generated vectors of shape {vectors.shape}")

        # Step 3: Cluster with K-Means
        with Timer("K-Means Clustering"):
            labels = self.clusterer.fit(vectors)
            metrics = self.clusterer.calculate_metrics(vectors)
            cluster_summary = self.clusterer.get_cluster_summary(labels)
            logger.info(f"✓ Clustering complete")
            logger.info(f"  - Inertia: {metrics['inertia']:.2f}")
            if metrics['silhouette_score'] is not None:
                logger.info(f"  - Silhouette Score: {metrics['silhouette_score']:.3f}")

        # Step 4: PCA Dimensionality Reduction
        with Timer("PCA Dimensionality Reduction (NumPy)"):
            coords_pca = self.pca.fit_transform(vectors)
            explained_var = self.pca.get_explained_variance_ratio()
            logger.info(f"✓ PCA complete")
            logger.info(
                f"  - Explained Variance: {explained_var[0]:.1%}, "
                f"{explained_var[1]:.1%}, {explained_var[2]:.1%}"
            )

        # Step 5: t-SNE Dimensionality Reduction
        with Timer("t-SNE Dimensionality Reduction"):
            coords_tsne = self.tsne.fit_transform(vectors)
            logger.info(f"✓ t-SNE complete. Output shape: {coords_tsne.shape}")

        # Step 6: Visualization
        with Timer("Visualization"):
            # PCA plot
            self.visualizer.plot_pca(
                coords_pca, labels, explained_var,
                filename='pca_clusters_3d',
                show=self.config.get('show_plots', False)
            )
            logger.info("✓ PCA visualization saved")

            # t-SNE plot
            self.visualizer.plot_tsne(
                coords_tsne, labels,
                filename='tsne_clusters_3d',
                show=self.config.get('show_plots', False)
            )
            logger.info("✓ t-SNE visualization saved")

        # Print summary
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total sentences processed: {len(sentences)}")
        logger.info(f"Number of clusters: {self.config['n_clusters']}")
        logger.info(f"Cluster distribution:")
        for cluster_id in sorted(cluster_summary.keys()):
            logger.info(f"  - Cluster {cluster_id}: {cluster_summary[cluster_id]} sentences")
        logger.info(f"Output directory: {self.config['output_dir']}")
        logger.info(f"  - PCA plot: {Path(self.config['output_dir']) / 'pca_clusters_3d.png'}")
        logger.info(f"  - t-SNE plot: {Path(self.config['output_dir']) / 'tsne_clusters_3d.png'}")
        logger.info("=" * 70)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Sentence Clustering & Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --csv_path data.csv --n_clusters 5
  python main.py --csv_path data.csv --n_clusters 3 --text_column "sentence"
  python main.py --csv_path data.csv --n_clusters 5 --show_plots
        """
    )

    parser.add_argument(
        '--csv_path', type=str, required=True,
        help='Path to CSV file containing sentences'
    )

    parser.add_argument(
        '--text_column', type=str, default='text',
        help='Name of column containing sentences (default: "text")'
    )

    parser.add_argument(
        '--n_clusters', type=int, default=5,
        help='Number of clusters for K-Means (default: 5)'
    )

    parser.add_argument(
        '--embedding_model', type=str, default='all-MiniLM-L6-v2',
        help='Sentence transformer model name (default: "all-MiniLM-L6-v2")'
    )

    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--output_dir', type=str, default='./output',
        help='Directory for output files (default: "./output")'
    )

    parser.add_argument(
        '--tsne_perplexity', type=float, default=30,
        help='t-SNE perplexity parameter (default: 30)'
    )

    parser.add_argument(
        '--tsne_iterations', type=int, default=1000,
        help='t-SNE maximum iterations (default: 1000)'
    )

    parser.add_argument(
        '--show_plots', action='store_true',
        help='Display plots interactively (default: False)'
    )

    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main entry point for the application."""
    # Parse arguments
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Ensure output directory exists
    ensure_output_dir(args.output_dir)

    # Create config from arguments
    config = {
        'csv_path': args.csv_path,
        'text_column': args.text_column,
        'n_clusters': args.n_clusters,
        'embedding_model': args.embedding_model,
        'random_seed': args.random_seed,
        'output_dir': args.output_dir,
        'tsne_perplexity': args.tsne_perplexity,
        'tsne_iterations': args.tsne_iterations,
        'show_plots': args.show_plots
    }

    try:
        # Create and run pipeline
        pipeline = SentenceClusteringPipeline(config)
        pipeline.run()

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.debug("", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
