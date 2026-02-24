import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import numpy as np
import torch
import pandas as pd
import tempfile
import shutil
from src.data.loader import OFIDataLoader


def create_test_parquet(filepath: str, n_sequences: int = 2, seq_length: int = 200):
    """Create a test parquet file for testing."""
    data = []
    
    for seq_id in range(n_sequences):
        for step in range(seq_length):
            row = {
                'seq_ix': seq_id,
                'step_in_seq': step,
                'need_prediction': 1 if step >= seq_length - 10 else 0,
            }
            
            # Create realistic price data (bid prices p0-p5, ask prices p6-p11)
            base_price = 1000.0 + seq_id * 100
            spread = 0.1
            
            # Bid prices (decreasing from best bid)
            for i in range(6):
                row[f'p{i}'] = base_price - i * spread - np.random.rand() * 0.01
            
            # Ask prices (increasing from best ask)
            for i in range(6, 12):
                row[f'p{i}'] = base_price + (i-5) * spread + np.random.rand() * 0.01
            
            # Volumes (bid volumes v0-v5, ask volumes v6-v11)
            for i in range(12):
                row[f'v{i}'] = np.random.rand() * 100
            
            # Delta prices and volumes
            for i in range(4):
                row[f'dp{i}'] = np.random.randn() * 0.1
                row[f'dv{i}'] = np.random.rand() * 10
            
            # Targets
            row['t0'] = np.random.randn() * 0.1
            row['t1'] = np.random.randn() * 0.1
            
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_parquet(filepath)
    return df


@pytest.fixture
def test_parquet_file():
    """Create a temporary parquet file for testing."""
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, 'test.parquet')
    
    # Create test data
    create_test_parquet(filepath, n_sequences=3, seq_length=150)
    
    yield filepath
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_data_loader_initialization(test_parquet_file):
    """Test that data loader initializes correctly."""
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=32,
        shuffle=True,
        normalize_prices=True,
        device='cpu',
        max_sequences=10
    )
    
    assert loader is not None
    assert loader.sequence_length == 50
    assert loader.batch_size == 32
    assert len(loader) > 0


def test_output_shape(test_parquet_file):
    """Test that data loader outputs correct shape (batch, sequence_length, 32)."""
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=16,
        shuffle=True,
        normalize_prices=True,
        device='cpu',
        max_sequences=20  # Increase to ensure we have enough sequences
    )
    
    # Test single sequence
    features, targets = loader[0]
    assert features.shape == (50, 32), f"Expected (50, 32), got {features.shape}"
    assert targets.shape == (50, 2), f"Expected (50, 2), got {targets.shape}"
    
    # Test batch from DataLoader
    dataloader = loader.get_dataloader()
    for batch_features, batch_targets in dataloader:
        # Batch size might be smaller than requested if not enough sequences
        expected_batch_size = min(16, len(loader))
        assert batch_features.shape == (expected_batch_size, 50, 32), \
            f"Expected ({expected_batch_size}, 50, 32), got {batch_features.shape}"
        assert batch_targets.shape == (expected_batch_size, 50, 2), \
            f"Expected ({expected_batch_size}, 50, 2), got {batch_targets.shape}"
        break  # Just test first batch


def test_ofi_transformation(test_parquet_file):
    """Test that volume is transformed to OFI correctly."""
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10
    )
    
    # Get a sequence
    features, _ = loader[0]
    
    # Check that features contain OFI values
    # OFI features should be at positions 12-17 (after 12 normalized prices)
    ofi_features = features[:, 12:18].numpy()
    
    # OFI should be difference between bid and ask volumes
    # Check that OFI values are finite
    assert np.isfinite(ofi_features).all(), "OFI features should be finite"
    
    # Check shape
    assert ofi_features.shape == (50, 6), f"Expected shape (50, 6), got {ofi_features.shape}"


def test_price_normalization(test_parquet_file):
    """Test that prices are normalized relative to mid-price."""
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10
    )
    
    # Get a sequence and its info
    features, _ = loader[0]
    seq_info = loader.get_sequence_info(0)
    mid_prices = seq_info['mid_prices']
    
    # Normalized prices should be at positions 0-11
    normalized_prices = features[:, :12].numpy()
    
    # Check that normalized prices are finite
    assert np.isfinite(normalized_prices).all(), "Normalized prices should be finite"
    
    # Check that mid prices are finite
    assert np.isfinite(mid_prices).all(), "Mid prices should be finite"
    
    # Check shapes
    assert normalized_prices.shape == (50, 12), f"Expected shape (50, 12), got {normalized_prices.shape}"
    assert mid_prices.shape == (50,), f"Expected shape (50,), got {mid_prices.shape}"


def test_feature_count(test_parquet_file):
    """Test that we have exactly 32 features."""
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10
    )
    
    feature_names = loader.get_feature_names()
    assert len(feature_names) == 32, f"Expected 32 features, got {len(feature_names)}"
    
    # Check feature names structure
    assert feature_names[0] == 'norm_p0', "First feature should be norm_p0"
    assert feature_names[11] == 'norm_p11', "12th feature should be norm_p11"
    assert feature_names[12] == 'ofi_0', "13th feature should be ofi_0"
    assert feature_names[17] == 'ofi_5', "18th feature should be ofi_5"
    assert feature_names[18] == 'dp0', "19th feature should be dp0"
    assert feature_names[21] == 'dp3', "22nd feature should be dp3"
    assert feature_names[22] == 'dv0', "23rd feature should be dv0"
    assert feature_names[25] == 'dv3', "26th feature should be dv3"
    assert feature_names[26] == 'orig_p0', "27th feature should be orig_p0"
    assert feature_names[31] == 'orig_p5', "32nd feature should be orig_p5"


def test_data_types(test_parquet_file):
    """Test that data has correct types."""
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10
    )
    
    features, targets = loader[0]
    
    # Check tensor types
    assert features.dtype == torch.float32, f"Features should be float32, got {features.dtype}"
    assert targets.dtype == torch.float32, f"Targets should be float32, got {targets.dtype}"
    
    # Check numpy types in sequence info
    seq_info = loader.get_sequence_info(0)
    assert seq_info['mid_prices'].dtype == np.float32, \
        f"Mid prices should be float32, got {seq_info['mid_prices'].dtype}"


def test_sequence_integrity(test_parquet_file):
    """Test that sequences maintain temporal integrity."""
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=8,
        shuffle=False,  # Important for this test
        normalize_prices=True,
        device='cpu',
        max_sequences=10
    )
    
    # Skip if not enough sequences
    if len(loader) < 2:
        pytest.skip("Not enough sequences found in data")
    
    # Get first sequence and its info
    features1, targets1 = loader[0]
    info1 = loader.get_sequence_info(0)
    
    # Get second sequence
    features2, targets2 = loader[1]
    info2 = loader.get_sequence_info(1)
    
    # Check shapes
    assert features1.shape == (50, 32), f"Expected shape (50, 32), got {features1.shape}"
    assert features2.shape == (50, 32), f"Expected shape (50, 32), got {features2.shape}"
    
    # Check metadata
    assert 'seq_id' in info1
    assert 'window_start' in info1
    assert 'metadata' in info1
    assert 'mid_prices' in info1
    
    # Metadata should have correct shape
    assert info1['metadata'].shape == (50, 3), \
        f"Metadata shape should be (50, 3), got {info1['metadata'].shape}"


def test_edge_cases(test_parquet_file):
    """Test edge cases and error handling."""
    # Test with different sequence lengths
    for seq_len in [20, 50, 80]:
        loader = OFIDataLoader(
            parquet_path=test_parquet_file,
            sequence_length=seq_len,
            batch_size=4,
            shuffle=False,
            normalize_prices=True,
            device='cpu',
            max_sequences=5
        )
        
        if len(loader) > 0:
            features, _ = loader[0]
            assert features.shape == (seq_len, 32), \
                f"Expected ({seq_len}, 32), got {features.shape}"
    
    # Test without price normalization
    loader = OFIDataLoader(
        parquet_path=test_parquet_file,
        sequence_length=50,
        batch_size=4,
        shuffle=False,
        normalize_prices=False,
        device='cpu',
        max_sequences=5
    )
    
    if len(loader) > 0:
        features, _ = loader[0]
        assert features.shape == (50, 32), \
            f"Expected (50, 32), got {features.shape}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])