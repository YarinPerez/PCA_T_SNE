"""
Module for converting sentences into vector embeddings.

Uses sentence-transformers to generate dense vector representations
of input sentences.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)


class SentenceVectorizer:
    """Converts sentences to dense vector embeddings using transformer models."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vectorizer with specified sentence transformer model.

        Args:
            model_name: HuggingFace model name for sentence-transformers
                       (default: 'all-MiniLM-L6-v2')

        Raises:
            Exception: If model fails to load
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def vectorize(self, sentences: List[str],
                  batch_size: int = 32,
                  show_progress: bool = True) -> np.ndarray:
        """
        Convert sentences to vector embeddings.

        Args:
            sentences: List of sentences to vectorize
            batch_size: Batch size for encoding (default: 32)
            show_progress: Show progress bar during encoding (default: True)

        Returns:
            NumPy array of shape (n_sentences, embedding_dim)

        Raises:
            ValueError: If sentences list is empty
            Exception: If vectorization fails
        """
        if not sentences:
            raise ValueError("Cannot vectorize empty list of sentences")

        logger.info(f"Vectorizing {len(sentences)} sentences...")
        try:
            vectors = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            logger.info(f"Vectorization complete. Output shape: {vectors.shape}")
            return vectors
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings.

        Returns:
            Integer dimension of embedding vectors
        """
        return self.embedding_dim
