"""Tests for data loading and validation module."""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '/mnt/c/25D/L17/PCA_T_SNE')

from data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        csv_path = tmp_path / "test_data.csv"
        data = {
            'id': [1, 2, 3, 4, 5],
            'text': ['Sentence one', 'Sentence two', '', None, 'Sentence five'],
            'category': ['A', 'B', 'A', 'B', 'A']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_csv_alt_column(self, tmp_path):
        """Create CSV with different column name."""
        csv_path = tmp_path / "test_data_alt.csv"
        data = {
            'id': [1, 2, 3],
            'sentence': ['Text one', 'Text two', 'Text three'],
            'label': ['X', 'Y', 'X']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_data_loader_initialization(self, sample_csv):
        """Test DataLoader initialization."""
        loader = DataLoader(str(sample_csv), text_column='text')
        assert loader.csv_path == str(sample_csv)
        assert loader.text_column == 'text'

    def test_load_csv_success(self, sample_csv):
        """Test successful CSV loading."""
        loader = DataLoader(str(sample_csv), text_column='text')
        sentences, df = loader.load()

        # Should have 3 valid sentences (1 empty, 1 null skipped)
        assert len(sentences) == 3
        assert 'Sentence one' in sentences
        assert 'Sentence two' in sentences
        assert 'Sentence five' in sentences
        assert isinstance(df, pd.DataFrame)

    def test_load_csv_file_not_found(self, tmp_path):
        """Test error when CSV file doesn't exist."""
        loader = DataLoader(str(tmp_path / "nonexistent.csv"), text_column='text')

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_csv_column_not_found(self, sample_csv):
        """Test error when specified column doesn't exist."""
        loader = DataLoader(str(sample_csv), text_column='nonexistent')

        with pytest.raises(ValueError, match="Column.*not found"):
            loader.load()

    def test_load_csv_alternative_column(self, sample_csv_alt_column):
        """Test loading with alternative column name."""
        loader = DataLoader(str(sample_csv_alt_column), text_column='sentence')
        sentences, df = loader.load()

        assert len(sentences) == 3
        assert all(len(s) > 0 for s in sentences)

    def test_load_csv_skips_empty_strings(self, sample_csv):
        """Test that empty strings are skipped."""
        loader = DataLoader(str(sample_csv), text_column='text')
        sentences, df = loader.load()

        # Empty string should be skipped
        assert '' not in sentences
        assert all(s.strip() for s in sentences)

    def test_load_csv_skips_null_values(self, sample_csv):
        """Test that null values are skipped."""
        loader = DataLoader(str(sample_csv), text_column='text')
        sentences, df = loader.load()

        # None should not create a sentence
        assert None not in sentences

    def test_load_csv_strips_whitespace(self, tmp_path):
        """Test that whitespace is stripped from sentences."""
        csv_path = tmp_path / "whitespace_test.csv"
        data = {
            'text': ['  Sentence one  ', '\tSentence two\n', '  Sentence three  ']
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        loader = DataLoader(str(csv_path), text_column='text')
        sentences, df = loader.load()

        assert sentences == ['Sentence one', 'Sentence two', 'Sentence three']

    def test_load_csv_no_valid_sentences_error(self, tmp_path):
        """Test error when no valid sentences found."""
        csv_path = tmp_path / "empty_test.csv"
        data = {
            'text': ['', None, '   ']
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        loader = DataLoader(str(csv_path), text_column='text')

        with pytest.raises(ValueError, match="No valid sentences"):
            loader.load()

    def test_load_csv_returns_original_dataframe(self, sample_csv):
        """Test that original dataframe is returned alongside sentences."""
        loader = DataLoader(str(sample_csv), text_column='text')
        sentences, df = loader.load()

        assert isinstance(df, pd.DataFrame)
        assert 'id' in df.columns
        assert 'text' in df.columns
        assert 'category' in df.columns
        assert len(df) == 5  # Original length, not filtered

    def test_load_csv_custom_encoding(self, tmp_path):
        """Test loading with UTF-8 encoding."""
        csv_path = tmp_path / "encoding_test.csv"
        data = {
            'text': ['Hello', 'World', 'Café']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8')

        loader = DataLoader(str(csv_path), text_column='text', encoding='utf-8')
        sentences, _ = loader.load()

        assert len(sentences) == 3
        assert 'Café' in sentences


class TestDataExtraction:
    """Test suite for sentence extraction logic."""

    @pytest.fixture
    def mixed_data_csv(self, tmp_path):
        """Create CSV with various edge cases."""
        csv_path = tmp_path / "mixed_data.csv"
        data = {
            'text': [
                'Normal sentence',
                '   Sentence with leading/trailing spaces   ',
                '',
                None,
                'Sentence\nwith\nnewlines',
                '123',
                'Special chars: @#$%'
            ]
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)
        return csv_path

    def test_extract_handles_mixed_data(self, mixed_data_csv):
        """Test extraction with various edge cases."""
        loader = DataLoader(str(mixed_data_csv), text_column='text')
        sentences, _ = loader.load()

        # Should extract valid sentences
        assert len(sentences) > 0
        assert all(isinstance(s, str) for s in sentences)
        assert all(len(s) > 0 for s in sentences)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
