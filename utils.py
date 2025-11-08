"""
Utility functions for the Sentence Clustering & Visualization project.

Provides logging setup, timing decorators, and common helper functions.
"""
import logging
import time
from contextlib import contextmanager
from pathlib import Path


def setup_logging(level=logging.INFO):
    """
    Setup logging configuration for the entire application.

    Args:
        level: Logging level (default: logging.INFO)
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

    Args:
        name: Name of the operation being timed
    """
    logger = logging.getLogger(__name__)
    start = time.time()
    logger.info(f"Starting: {name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {name} (took {elapsed:.2f}s)")


def ensure_output_dir(output_dir: str = './output') -> Path:
    """
    Ensure output directory exists, creating if necessary.

    Args:
        output_dir: Path to output directory

    Returns:
        Path object pointing to output directory
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
