"""
3D visualization module for clustered data.

Uses matplotlib to create interactive 3D scatter plots of clustered
sentence embeddings reduced to 3D space.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Visualizer3D:
    """Creates 3D visualizations of clustered and reduced data."""

    def __init__(self, output_dir: str = './output', figsize: tuple = (12, 10)):
        """
        Initialize 3D visualizer.

        Args:
            output_dir: Directory to save plot files (default: './output')
            figsize: Figure size in inches (width, height) (default: (12, 10))
        """
        self.output_dir = output_dir
        self.figsize = figsize

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    def plot_3d_clusters(self, coordinates: np.ndarray, labels: np.ndarray,
                         title: str, axis_labels: List[str],
                         filename: Optional[str] = None,
                         show: bool = False) -> None:
        """
        Create and save/display a 3D scatter plot of clusters.

        Args:
            coordinates: 3D coordinates of shape (n_samples, 3)
            labels: Cluster labels of shape (n_samples,)
            title: Plot title
            axis_labels: List of 3 axis labels [x_label, y_label, z_label]
            filename: Filename to save (without extension) (default: None)
            show: Whether to display the plot (default: False)

        Raises:
            ValueError: If coordinates don't have exactly 3 columns
            ValueError: If coordinates and labels have mismatched lengths
        """
        if coordinates.shape[1] != 3:
            raise ValueError(f"Expected 3 columns, got {coordinates.shape[1]}")

        if len(coordinates) != len(labels):
            raise ValueError(
                f"Coordinates ({len(coordinates)}) and labels ({len(labels)}) "
                "have different lengths"
            )

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Get unique clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Use colormap for consistent cluster colors
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))

        # Plot each cluster with distinct color
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                coordinates[mask, 2],
                c=[colors[i % len(colors)]],
                label=f'Cluster {int(label)}',
                s=50,
                alpha=0.6,
                edgecolors='k',
                linewidth=0.5
            )

        # Set labels and title
        ax.set_xlabel(axis_labels[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(axis_labels[1], fontsize=12, fontweight='bold')
        ax.set_zlabel(axis_labels[2], fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if filename provided
        if filename:
            filepath = Path(self.output_dir) / f"{filename}.png"
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
                 show: bool = False) -> None:
        """
        Plot PCA projection with explained variance information.

        Args:
            coordinates: PCA-reduced coordinates (n_samples, 3)
            labels: Cluster labels (n_samples,)
            explained_variance: Explained variance ratio for 3 components
            filename: Output filename without extension (default: 'pca_clusters_3d')
            show: Whether to display the plot (default: False)
        """
        # Create informative title with explained variance
        title = (
            f"PCA Projection\n"
            f"Explained Variance: {explained_variance[0]:.1%} | "
            f"{explained_variance[1]:.1%} | {explained_variance[2]:.1%}"
        )

        # Create axis labels with explained variance
        axis_labels = [
            f'PC1 ({explained_variance[0]:.1%})',
            f'PC2 ({explained_variance[1]:.1%})',
            f'PC3 ({explained_variance[2]:.1%})'
        ]

        self.plot_3d_clusters(
            coordinates, labels, title, axis_labels,
            filename=filename, show=show
        )

    def plot_tsne(self, coordinates: np.ndarray, labels: np.ndarray,
                  filename: str = 'tsne_clusters_3d',
                  show: bool = False) -> None:
        """
        Plot t-SNE projection.

        Args:
            coordinates: t-SNE-reduced coordinates (n_samples, 3)
            labels: Cluster labels (n_samples,)
            filename: Output filename without extension (default: 'tsne_clusters_3d')
            show: Whether to display the plot (default: False)
        """
        title = "t-SNE Projection"
        axis_labels = ['t-SNE 1', 't-SNE 2', 't-SNE 3']

        self.plot_3d_clusters(
            coordinates, labels, title, axis_labels,
            filename=filename, show=show
        )

    @staticmethod
    def show_plots():
        """Display all generated plots (non-blocking)."""
        plt.show()

    @staticmethod
    def close_all_plots():
        """Close all open figure windows."""
        plt.close('all')
        logger.info("Closed all plot windows")
