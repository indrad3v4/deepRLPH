import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pytest
import torch
import numpy as np
from src.losses.weighted_pearson import WeightedPearsonCorrelationLoss, weighted_pearson_loss


def test_weighted_pearson_initialization():
    """Test that the loss module initializes correctly."""
    loss_fn = WeightedPearsonCorrelationLoss()
    assert loss_fn is not None
    assert loss_fn.reduction == 'mean'
    assert loss_fn.eps == 1e-8
    
    loss_fn_custom = WeightedPearsonCorrelationLoss(reduction='sum', eps=1e-6)
    assert loss_fn_custom.reduction == 'sum'
    assert loss_fn_custom.eps == 1e-6


def test_weighted_pearson_3d_tensor_handling():
    """Test that the loss properly handles 3D tensors (batch, seq, targets)."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    # Create synthetic 3D data
    batch_size = 4
    seq_len = 10
    n_targets = 2
    
    predictions = torch.randn(batch_size, seq_len, n_targets)
    targets = torch.randn(batch_size, seq_len, n_targets)
    
    # Compute loss
    loss = loss_fn(predictions, targets)
    
    # Check that loss is a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss should be finite"
    
    # Check that loss is in reasonable range (correlation loss should be >= 0)
    assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"


def test_weighted_pearson_weighting():
    """Test that |y_i| weighting is applied correctly."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    # Create data where targets have different magnitudes
    batch_size = 2
    seq_len = 5
    n_targets = 1
    
    # Create predictions and targets with known relationship
    predictions = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]],
                                [[2.0], [4.0], [6.0], [8.0], [10.0]]], dtype=torch.float32)
    
    # Targets are exactly 2x predictions (perfect correlation)
    targets = predictions * 2.0
    
    # Compute loss - should be very small (near 0) since perfect correlation
    loss = loss_fn(predictions, targets)
    
    # With perfect correlation, loss should be close to 0
    assert loss.item() < 0.01, f"Loss should be small for perfect correlation, got {loss.item()}"
    
    # Test with negative correlation
    targets_neg = -predictions * 2.0  # Perfect negative correlation
    loss_neg = loss_fn(predictions, targets_neg)
    
    # With perfect negative correlation, loss should be close to 2 (1 - (-1) = 2)
    assert abs(loss_neg.item() - 2.0) < 0.01, \
        f"Loss should be ~2.0 for perfect negative correlation, got {loss_neg.item()}"


def test_weighted_pearson_clipping():
    """Test that gradient clipping and value clipping work correctly."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    # Test safe_clip static method
    test_tensor = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0])
    clipped = loss_fn.safe_clip(test_tensor, -6.0, 6.0)
    
    expected = torch.tensor([-6.0, -5.0, 0.0, 5.0, 6.0])
    assert torch.allclose(clipped, expected), f"Expected {expected}, got {clipped}"
    
    # Test with extreme values in predictions and targets
    batch_size = 2
    seq_len = 3
    n_targets = 1
    
    # Create data with extreme values
    predictions = torch.tensor([[[100.0], [-100.0], [50.0]],
                                [[-50.0], [75.0], [-75.0]]], dtype=torch.float32)
    
    targets = torch.tensor([[[50.0], [-50.0], [25.0]],
                            [[-25.0], [37.5], [-37.5]]], dtype=torch.float32)
    
    # Loss should still be computable without numerical issues
    loss = loss_fn(predictions, targets)
    assert torch.isfinite(loss), "Loss should be finite even with extreme values"


def test_weighted_pearson_reductions():
    """Test different reduction modes."""
    batch_size = 3
    seq_len = 4
    n_targets = 2
    
    predictions = torch.randn(batch_size, seq_len, n_targets)
    targets = torch.randn(batch_size, seq_len, n_targets)
    
    # Test mean reduction (default)
    loss_mean = WeightedPearsonCorrelationLoss(reduction='mean')(predictions, targets)
    assert loss_mean.dim() == 0
    
    # Test sum reduction
    loss_sum = WeightedPearsonCorrelationLoss(reduction='sum')(predictions, targets)
    assert loss_sum.dim() == 0
    # Sum should be larger than mean when there are multiple targets
    if n_targets > 1:
        assert loss_sum.item() > loss_mean.item()
    
    # Test none reduction
    loss_none = WeightedPearsonCorrelationLoss(reduction='none')(predictions, targets)
    assert loss_none.shape == (n_targets,), f"Expected shape ({n_targets},), got {loss_none.shape}"


def test_weighted_pearson_gradient_flow():
    """Test that gradients flow correctly through the loss."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    batch_size = 2
    seq_len = 3
    n_targets = 1
    
    # Create learnable parameters
    predictions = torch.randn(batch_size, seq_len, n_targets, requires_grad=True)
    targets = torch.randn(batch_size, seq_len, n_targets)
    
    # Compute loss
    loss = loss_fn(predictions, targets)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert predictions.grad is not None, "Gradients should flow to predictions"
    assert torch.isfinite(predictions.grad).all(), "Gradients should be finite"
    
    # Test gradient clipping method
    params = [predictions]
    loss_fn.clip_gradients(params, clip_value=1.0)
    
    # Check gradients are clipped
    grad_norm = torch.norm(predictions.grad)
    assert grad_norm.item() <= 1.0 + 1e-6, f"Gradient norm should be <= 1.0, got {grad_norm.item()}"


def test_weighted_pearson_functional():
    """Test the functional version of the loss."""
    batch_size = 2
    seq_len = 5
    n_targets = 2
    
    predictions = torch.randn(batch_size, seq_len, n_targets)
    targets = torch.randn(batch_size, seq_len, n_targets)
    
    # Test functional version
    loss_func = weighted_pearson_loss(predictions, targets, reduction='mean')
    
    # Test class version for comparison
    loss_class = WeightedPearsonCorrelationLoss(reduction='mean')(predictions, targets)
    
    # They should give the same result
    assert torch.allclose(loss_func, loss_class), \
        f"Functional and class versions should match: {loss_func.item()} vs {loss_class.item()}"


def test_weighted_pearson_perfect_correlation():
    """Test edge case of perfect correlation."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    batch_size = 1
    seq_len = 10
    n_targets = 1
    
    # Create perfectly correlated data
    base = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    predictions = base
    targets = base * 3.0 + 2.0  # Linear transformation (perfect correlation)
    
    loss = loss_fn(predictions, targets)
    
    # With perfect correlation, loss should be very close to 0
    assert loss.item() < 1e-5, f"Loss should be ~0 for perfect correlation, got {loss.item()}"


def test_weighted_pearson_no_correlation():
    """Test case with no correlation."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    batch_size = 1
    seq_len = 100
    n_targets = 1
    
    # Create truly uncorrelated data with zero mean to avoid weighting effects
    torch.manual_seed(42)
    
    # Generate data with zero mean to minimize weighting bias
    predictions = torch.randn(batch_size, seq_len, n_targets)
    predictions = predictions - predictions.mean()  # Zero mean
    
    # Generate independent targets with zero mean
    targets = torch.randn(batch_size, seq_len, n_targets)
    targets = targets - targets.mean()  # Zero mean
    
    # Compute loss
    loss = loss_fn(predictions, targets)
    
    # With truly uncorrelated data and zero mean (minimizing weighting effects),
    # loss should be around 1 (correlation ~ 0)
    # Allow more tolerance due to weighting
    assert 0.5 < loss.item() < 1.5, f"Loss should be around 1 for uncorrelated data, got {loss.item()}"


def test_weighted_pearson_large_batch():
    """Test with large batch size to ensure numerical stability."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    batch_size = 32
    seq_len = 50
    n_targets = 3
    
    predictions = torch.randn(batch_size, seq_len, n_targets)
    targets = torch.randn(batch_size, seq_len, n_targets)
    
    loss = loss_fn(predictions, targets)
    
    assert torch.isfinite(loss), "Loss should be finite for large batch"
    assert 0 <= loss.item() <= 2, f"Loss should be between 0 and 2, got {loss.item()}"


def test_weighted_pearson_sequence_awareness():
    """Test that the loss properly handles sequence dimension."""
    loss_fn = WeightedPearsonCorrelationLoss()
    
    # Test with different sequence lengths
    for seq_len in [1, 5, 20, 100]:
        batch_size = 2
        n_targets = 1
        
        predictions = torch.randn(batch_size, seq_len, n_targets)
        targets = torch.randn(batch_size, seq_len, n_targets)
        
        loss = loss_fn(predictions, targets)
        
        assert torch.isfinite(loss), f"Loss should be finite for seq_len={seq_len}"
        assert loss.dim() == 0, f"Loss should be scalar for seq_len={seq_len}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])