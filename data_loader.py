"""
Module for loading and validating CSV data containing sentences.

Handles CSV file reading, data validation, and sentence extraction.
"""
import pandas as pd
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading sentences from CSV files."""

    def __init__(self, csv_path: str, text_column: str = 'text',
                 encoding: str = 'utf-8'):
        """
        Initialize DataLoader.

        Args:
            csv_path: Path to CSV file
            text_column: Name of column containing sentences (default: 'text')
            encoding: File encoding (default: 'utf-8')
        """
        self.csv_path = csv_path
        self.text_column = text_column
        self.encoding = encoding

    def load(self) -> Tuple[List[str], pd.DataFrame]:
        """
        Load sentences from CSV file.

        Returns:
            Tuple of (sentences list, original dataframe)

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If text column not found in CSV
            ValueError: If no valid sentences found after filtering
        """
        # Check file exists
        import os
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Load CSV
        logger.info(f"Loading CSV from: {self.csv_path}")
        try:
            df = pd.read_csv(self.csv_path, encoding=self.encoding)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        # Validate column exists
        if self.text_column not in df.columns:
            available = ', '.join(df.columns)
            raise ValueError(
                f"Column '{self.text_column}' not found in CSV. "
                f"Available columns: {available}"
            )

        logger.info(f"CSV loaded. Total rows: {len(df)}")

        # Extract and validate sentences
        sentences = self._extract_sentences(df)

        if not sentences:
            raise ValueError(
                "No valid sentences found after filtering empty/null entries"
            )

        logger.info(
            f"Extracted {len(sentences)} valid sentences "
            f"({len(df) - len(sentences)} rows skipped)"
        )

        return sentences, df

    def _extract_sentences(self, df: pd.DataFrame) -> List[str]:
        """
        Extract and validate sentences from dataframe.

        Skips empty, null, or whitespace-only sentences.

        Args:
            df: Input dataframe

        Returns:
            List of valid sentences
        """
        sentences = []
        skipped = 0

        for idx, text in enumerate(df[self.text_column]):
            # Check for null/NaN
            if pd.isna(text):
                skipped += 1
                logger.debug(f"Row {idx}: Skipped null value")
                continue

            # Convert to string and strip whitespace
            text_str = str(text).strip()

            # Check for empty string
            if not text_str:
                skipped += 1
                logger.debug(f"Row {idx}: Skipped empty string")
                continue

            sentences.append(text_str)

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} rows with empty or null sentences"
            )

        return sentences
