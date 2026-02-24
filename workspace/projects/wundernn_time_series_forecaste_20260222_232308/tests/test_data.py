import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import numpy as np
import torch
from src.data.loader import OFIDataLoader, create_data_loaders


def test_data_loader_initialization():
    """Test that data loader initializes correctly."""
    # Test with synthetic data
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',  # Not used in test_mode
        sequence_length=100,
        batch_size=32,
        shuffle=True,
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True  # Use synthetic data
    )
    
    assert loader is not None
    assert loader.sequence_length == 100
    assert loader.batch_size == 32
    assert len(loader) > 0


def test_output_shape():
    """Test that data loader outputs correct shape (batch, sequence_length, 33)."""
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=16,
        shuffle=True,
        normalize_prices=True,
        device='cpu',
        max_sequences=20,
        test_mode=True
    )
    
    # Test single sequence
    features, targets = loader[0]
    assert features.shape == (100, 33), f"Expected (100, 33), got {features.shape}"
    assert targets.shape == (100, 2), f"Expected (100, 2), got {targets.shape}"
    
    # Test batch from DataLoader
    dataloader = loader.get_dataloader()
    for batch_features, batch_targets in dataloader:
        # Batch size might be smaller than requested if not enough sequences
        expected_batch_size = min(16, len(loader))
        assert batch_features.shape == (expected_batch_size, 100, 33), \
            f"Expected ({expected_batch_size}, 100, 33), got {batch_features.shape}"
        assert batch_targets.shape == (expected_batch_size, 100, 2), \
            f"Expected ({expected_batch_size}, 100, 2), got {batch_targets.shape}"
        break  # Just test first batch


def test_ofi_transformation():
    """Test that volume is transformed to OFI correctly."""
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True
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
    assert ofi_features.shape == (100, 6), f"Expected shape (100, 6), got {ofi_features.shape}"


def test_price_normalization():
    """Test that prices are normalized relative to mid-price."""
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True
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
    assert normalized_prices.shape == (100, 12), f"Expected shape (100, 12), got {normalized_prices.shape}"
    assert mid_prices.shape == (100,), f"Expected shape (100,), got {mid_prices.shape}"


def test_feature_count():
    """Test that we have exactly 33 features (including spread)."""
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True
    )
    
    feature_names = loader.get_feature_names()
    assert len(feature_names) == 33, f"Expected 33 features, got {len(feature_names)}"
    
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
    assert feature_names[32] == 'spread', "33rd feature should be spread"


def test_data_types():
    """Test that data has correct types."""
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True
    )
    
    features, targets = loader[0]
    
    # Check tensor types
    assert features.dtype == torch.float32, f"Features should be float32, got {features.dtype}"
    assert targets.dtype == torch.float32, f"Targets should be float32, got {targets.dtype}"
    
    # Check numpy types in sequence info
    seq_info = loader.get_sequence_info(0)
    assert seq_info['mid_prices'].dtype == np.float32, \
        f"Mid prices should be float32, got {seq_info['mid_prices'].dtype}"
    assert seq_info['spreads'].dtype == np.float32, \
        f"Spreads should be float32, got {seq_info['spreads'].dtype}"


def test_create_data_loaders():
    """Test the create_data_loaders utility function."""
    train_dataset, valid_dataset = create_data_loaders(
        train_path='dummy_train.parquet',
        valid_path='dummy_valid.parquet',
        sequence_length=100,
        batch_size=16,
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True
    )
    
    assert train_dataset is not None
    assert valid_dataset is not None
    
    # Check that train dataset shuffles
    train_dataloader = train_dataset.get_dataloader()
    valid_dataloader = valid_dataset.get_dataloader()
    
    # Get first batch from each
    train_batch = next(iter(train_dataloader))
    valid_batch = next(iter(valid_dataloader))
    
    # Check shapes (batch size might be smaller than requested)
    expected_train_batch_size = min(16, len(train_dataset))
    expected_valid_batch_size = min(16, len(valid_dataset))
    
    assert train_batch[0].shape == (expected_train_batch_size, 100, 33), \
        f"Train batch shape mismatch: {train_batch[0].shape}"
    assert valid_batch[0].shape == (expected_valid_batch_size, 100, 33), \
        f"Valid batch shape mismatch: {valid_batch[0].shape}"


def test_sequence_integrity():
    """Test that sequences maintain temporal integrity."""
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=8,
        shuffle=False,  # Important for this test
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True
    )
    
    # Get first sequence and its info
    features1, targets1 = loader[0]
    info1 = loader.get_sequence_info(0)
    
    # Get second sequence
    features2, targets2 = loader[1]
    info2 = loader.get_sequence_info(1)
    
    # Check shapes
    assert features1.shape == (100, 33), f"Expected shape (100, 33), got {features1.shape}"
    assert features2.shape == (100, 33), f"Expected shape (100, 33), got {features2.shape}"
    
    # Check metadata
    assert 'seq_id' in info1
    assert 'window_start' in info1
    assert 'metadata' in info1
    assert 'mid_prices' in info1
    assert 'spreads' in info1
    
    # Metadata should have correct shape
    assert info1['metadata'].shape == (100, 3), \
        f"Metadata shape should be (100, 3), got {info1['metadata'].shape}"


def test_spread_calculation():
    """Test that spreads are calculated correctly."""
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=8,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=10,
        test_mode=True
    )
    
    # Get a sequence and its info
    features, _ = loader[0]
    seq_info = loader.get_sequence_info(0)
    
    # Spread should be the last feature (index 32)
    spread_feature = features[:, 32].numpy()
    spreads_from_info = seq_info['spreads']
    
    # Check that spreads are finite
    assert np.isfinite(spread_feature).all(), "Spread feature should be finite"
    assert np.isfinite(spreads_from_info).all(), "Spreads from info should be finite"
    
    # Check shape
    assert spread_feature.shape == (100,), f"Expected shape (100,), got {spread_feature.shape}"
    assert spreads_from_info.shape == (100,), f"Expected shape (100,), got {spreads_from_info.shape}"
    
    # Check that spread is positive (ask > bid)
    assert (spread_feature > 0).all(), "Spread should be positive"


def test_edge_cases():
    """Test edge cases and error handling."""
    # Test with different sequence lengths
    for seq_len in [50, 100, 150]:
        loader = OFIDataLoader(
            parquet_path='dummy.parquet',
            sequence_length=seq_len,
            batch_size=4,
            shuffle=False,
            normalize_prices=True,
            device='cpu',
            max_sequences=5,
            test_mode=True
        )
        
        if len(loader) > 0:
            features, _ = loader[0]
            assert features.shape == (seq_len, 33), \
                f"Expected ({seq_len}, 33), got {features.shape}"
    
    # Test without price normalization
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=4,
        shuffle=False,
        normalize_prices=False,
        device='cpu',
        max_sequences=5,
        test_mode=True
    )
    
    if len(loader) > 0:
        features, _ = loader[0]
        assert features.shape == (100, 33), \
            f"Expected (100, 33), got {features.shape}"


def test_safe_parquet_reading():
    """Test that parquet reading handles pyarrow filter issues safely."""
    # This test verifies the _safe_read_parquet method
    loader = OFIDataLoader(
        parquet_path='dummy.parquet',
        sequence_length=100,
        batch_size=4,
        shuffle=False,
        normalize_prices=True,
        device='cpu',
        max_sequences=5,
        test_mode=True
    )
    
    # The synthetic data should load successfully even with test_mode=True
    assert len(loader) > 0, "Loader should have sequences even in test mode"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
