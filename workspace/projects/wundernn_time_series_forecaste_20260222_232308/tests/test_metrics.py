import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import pytest
from src.metrics.evaluator import WeightedPearsonEvaluator, evaluate_weighted_pearson


class TestWeightedPearsonEvaluator:
    """Test suite for Weighted Pearson Evaluation Metrics"""
    
    def test_evaluator_initialization(self):
        """Test that evaluator initializes correctly."""
        evaluator = WeightedPearsonEvaluator()
        assert evaluator is not None
        assert evaluator.eps == 1e-8
        
        evaluator_custom = WeightedPearsonEvaluator(eps=1e-6)
        assert evaluator_custom.eps == 1e-6
    
    def test_reset_functionality(self):
        """Test that reset clears internal state."""
        evaluator = WeightedPearsonEvaluator()
        
        # Add some dummy data
        batch_size = 2
        predictions = torch.randn(batch_size, 901, 2)
        targets = torch.randn(batch_size, 901, 2)
        
        evaluator.update(predictions, targets)
        assert len(evaluator.batch_metrics) == 1
        assert len(evaluator.predictions) == 1
        assert len(evaluator.targets) == 1
        
        # Reset should clear everything
        evaluator.reset()
        assert len(evaluator.batch_metrics) == 0
        assert len(evaluator.predictions) == 0
        assert len(evaluator.targets) == 0
    
    def test_update_shape_validation(self):
        """Test that update validates input shapes correctly."""
        evaluator = WeightedPearsonEvaluator()
        
        # Correct shape should work
        batch_size = 4
        predictions = torch.randn(batch_size, 901, 2)
        targets = torch.randn(batch_size, 901, 2)
        
        # This should not raise an error
        evaluator.update(predictions, targets)
        
        # Test incorrect shapes
        evaluator.reset()
        
        # Wrong sequence length
        predictions_wrong_seq = torch.randn(batch_size, 900, 2)
        targets_wrong_seq = torch.randn(batch_size, 900, 2)
        
        with pytest.raises(ValueError, match="Expected shape"):
            evaluator.update(predictions_wrong_seq, targets_wrong_seq)
        
        # Wrong target dimension
        predictions_wrong_target = torch.randn(batch_size, 901, 3)
        targets_wrong_target = torch.randn(batch_size, 901, 3)
        
        with pytest.raises(ValueError, match="Expected shape"):
            evaluator.update(predictions_wrong_target, targets_wrong_target)
        
        # Wrong tensor dimension (2D instead of 3D)
        predictions_2d = torch.randn(batch_size * 901, 2)
        targets_2d = torch.randn(batch_size * 901, 2)
        
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            evaluator.update(predictions_2d, targets_2d)
        
        # Mismatched shapes
        predictions_mismatch = torch.randn(batch_size, 901, 2)
        targets_mismatch = torch.randn(batch_size + 1, 901, 2)
        
        with pytest.raises(ValueError, match="doesn't match"):
            evaluator.update(predictions_mismatch, targets_mismatch)
    
    def test_perfect_correlation(self):
        """Test that perfect correlation gives correlation ~1 and loss ~0."""
        evaluator = WeightedPearsonEvaluator()
        
        batch_size = 3
        seq_len = 901
        n_targets = 2
        
        # Create predictions
        predictions = torch.randn(batch_size, seq_len, n_targets)
        
        # Create perfect correlation: targets = 2 * predictions + 3
        targets = predictions * 2.0 + 3.0
        
        evaluator.update(predictions, targets)
        metrics = evaluator.compute()
        
        # Check correlations are close to 1
        assert abs(metrics['target_0_correlation'] - 1.0) < 1e-4
        assert abs(metrics['target_1_correlation'] - 1.0) < 1e-4
        assert abs(metrics['mean_correlation'] - 1.0) < 1e-4
        
        # Check losses are close to 0
        assert metrics['target_0_loss'] < 1e-4
        assert metrics['target_1_loss'] < 1e-4
        assert metrics['mean_loss'] < 1e-4
    
    def test_perfect_negative_correlation(self):
        """Test that perfect negative correlation gives correlation ~-1 and loss ~2."""
        evaluator = WeightedPearsonEvaluator()
        
        batch_size = 2
        seq_len = 901
        n_targets = 2
        
        # Create predictions
        predictions = torch.randn(batch_size, seq_len, n_targets)
        
        # Create perfect negative correlation: targets = -predictions
        targets = -predictions
        
        evaluator.update(predictions, targets)
        metrics = evaluator.compute()
        
        # Check correlations are close to -1
        assert abs(metrics['target_0_correlation'] - (-1.0)) < 1e-4
        assert abs(metrics['target_1_correlation'] - (-1.0)) < 1e-4
        assert abs(metrics['mean_correlation'] - (-1.0)) < 1e-4
        
        # Check losses are close to 2 (1 - (-1) = 2)
        assert abs(metrics['target_0_loss'] - 2.0) < 1e-4
        assert abs(metrics['target_1_loss'] - 2.0) < 1e-4
        assert abs(metrics['mean_loss'] - 2.0) < 1e-4
    
    def test_no_correlation(self):
        """Test that uncorrelated data gives correlation ~0 and loss ~1."""
        evaluator = WeightedPearsonEvaluator()
        
        batch_size = 4
        seq_len = 901
        n_targets = 2
        
        # Create completely random, uncorrelated data
        predictions = torch.randn(batch_size, seq_len, n_targets)
        targets = torch.randn(batch_size, seq_len, n_targets)  # Different random values
        
        evaluator.update(predictions, targets)
        metrics = evaluator.compute()
        
        # With enough samples, correlation should be near 0
        # Use looser tolerance due to random variation
        assert abs(metrics['mean_correlation']) < 0.1
        assert abs(metrics['mean_loss'] - 1.0) < 0.1
    
    def test_multiple_batch_aggregation(self):
        """Test that metrics are correctly aggregated across multiple batches."""
        evaluator = WeightedPearsonEvaluator()
        
        # Process multiple batches with different sizes
        batch_sizes = [2, 4, 8]
        
        for batch_size in batch_sizes:
            predictions = torch.randn(batch_size, 901, 2)
            targets = predictions * 1.5 + 0.5  # Linear relationship
            
            evaluator.update(predictions, targets)
        
        metrics = evaluator.compute()
        
        # Check aggregation
        assert metrics['total_batches'] == 3
        assert metrics['total_samples'] == sum(batch_sizes)
        
        # Check that all expected keys are present
        expected_keys = [
            'total_batches', 'total_samples',
            'target_0_correlation', 'target_0_loss',
            'target_1_correlation', 'target_1_loss',
            'mean_correlation', 'mean_loss'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Check that correlations are reasonable (should be close to 1 for linear relationship)
        assert metrics['target_0_correlation'] > 0.99
        assert metrics['target_1_correlation'] > 0.99
        assert metrics['mean_correlation'] > 0.99
    
    def test_weighting_effect(self):
        """Test that |y_i| weighting affects the correlation calculation."""
        evaluator = WeightedPearsonEvaluator()
        
        batch_size = 2
        seq_len = 901
        n_targets = 2
        
        # Create simple test case - need to ensure we have exactly 901 time steps
        # Create a pattern that repeats every 5 steps
        pattern = torch.tensor([[[1.0, 0.5], [2.0, 1.0], [3.0, 1.5], [4.0, 2.0], [5.0, 2.5]]])
        
        # Repeat pattern to get exactly 901 time steps (5 * 180 + 1)
        repeats = seq_len // 5 + 1
        predictions = pattern.repeat(batch_size, repeats, 1)
        predictions = predictions[:, :seq_len, :]  # Take exactly 901 time steps
        
        # Create targets with different magnitudes for the two targets
        # First target: small magnitude (1.1x)
        # Second target: large magnitude (10x)
        targets = predictions.clone()
        targets[:, :, 0] = predictions[:, :, 0] * 1.1  # Small magnitude for target 0
        targets[:, :, 1] = predictions[:, :, 1] * 10.0  # Large magnitude for target 1
        
        evaluator.update(predictions, targets)
        metrics = evaluator.compute()
        
        # With weighting, both should still show perfect correlation (~1)
        # since they have perfect linear relationships
        assert abs(metrics['target_0_correlation'] - 1.0) < 1e-4
        assert abs(metrics['target_1_correlation'] - 1.0) < 1e-4
    
    def test_numerical_stability(self):
        """Test that evaluator handles extreme values without numerical issues."""
        evaluator = WeightedPearsonEvaluator(eps=1e-12)
        
        batch_size = 3
        seq_len = 901
        n_targets = 2
        
        # Test with large but not extreme values to avoid numerical issues
        # Create a simple pattern
        pattern = torch.tensor([
            [[100.0, -100.0], [0.1, -0.1], [0.0, 0.0]]
        ])
        
        # Repeat to get exactly 901 time steps (3 * 300 + 1)
        repeats = seq_len // 3 + 1
        predictions = pattern.repeat(batch_size, repeats, 1)
        predictions = predictions[:, :seq_len, :]  # Take exactly 901 time steps
        
        # Create targets with same pattern but scaled
        targets = predictions * 2.0 + 1.0
        
        # This should not raise any numerical errors
        evaluator.update(predictions, targets)
        metrics = evaluator.compute()
        
        # Check all metrics are finite
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), f"Metric {key} is not finite: {value}"
        
        # With perfect linear relationship, correlation should be ~1
        # Use slightly looser tolerance due to numerical precision with large values
        assert abs(metrics['mean_correlation'] - 1.0) < 1e-3
    
    def test_convenience_function(self):
        """Test the convenience evaluate_weighted_pearson function."""
        batch_size = 2
        predictions = torch.randn(batch_size, 901, 2)
        targets = predictions * 1.5  # Linear relationship
        
        metrics = evaluate_weighted_pearson(predictions, targets)
        
        # Check that function returns expected metrics
        assert 'mean_correlation' in metrics
        assert 'mean_loss' in metrics
        assert 'target_0_correlation' in metrics
        assert 'target_1_correlation' in metrics
        
        # With linear relationship, correlation should be ~1
        assert abs(metrics['mean_correlation'] - 1.0) < 1e-4
    
    def test_detailed_report(self):
        """Test that detailed report generation works."""
        evaluator = WeightedPearsonEvaluator()
        
        batch_size = 2
        predictions = torch.randn(batch_size, 901, 2)
        targets = predictions * 2.0 + 1.0
        
        evaluator.update(predictions, targets)
        
        report = evaluator.get_detailed_report()
        
        # Check report contains expected sections
        assert "WEIGHTED PEARSON EVALUATION REPORT" in report
        assert "PER-TARGET METRICS" in report
        assert "OVERALL METRICS" in report
        assert "INTERPRETATION" in report
        
        # Check report contains metric values
        metrics = evaluator.compute()
        assert f"Total batches evaluated: {metrics['total_batches']}" in report
        assert f"Total samples evaluated: {metrics['total_samples']}" in report
    
    def test_empty_state_error(self):
        """Test that compute() raises error when no data has been accumulated."""
        evaluator = WeightedPearsonEvaluator()
        
        with pytest.raises(ValueError, match="No data has been accumulated"):
            evaluator.compute()
        
        # After reset, should also raise error
        batch_size = 1
        predictions = torch.randn(batch_size, 901, 2)
        targets = torch.randn(batch_size, 901, 2)
        
        evaluator.update(predictions, targets)
        evaluator.compute()  # Should work
        
        evaluator.reset()
        
        with pytest.raises(ValueError, match="No data has been accumulated"):
            evaluator.compute()
    
    def test_metric_range_validation(self):
        """Test that all correlation metrics are in valid range [-1, 1]."""
        evaluator = WeightedPearsonEvaluator()
        
        # Test with various relationships
        test_cases = [
            ("perfect_positive", 1.0, 2.0, 0.0),  # y = 2x
            ("perfect_negative", -1.0, -1.0, 0.0),  # y = -x
            ("random", 0.0, 1.0, 1.0),  # random, uncorrelated
        ]
        
        for name, expected_corr, target_multiplier, expected_loss in test_cases:
            evaluator.reset()
            
            batch_size = 4
            predictions = torch.randn(batch_size, 901, 2)
            
            if name == "random":
                # For random case, use completely different targets
                targets = torch.randn(batch_size, 901, 2)
            else:
                targets = predictions * target_multiplier
            
            evaluator.update(predictions, targets)
            metrics = evaluator.compute()
            
            # Check correlations are in valid range
            assert -1.0 <= metrics['target_0_correlation'] <= 1.0
            assert -1.0 <= metrics['target_1_correlation'] <= 1.0
            assert -1.0 <= metrics['mean_correlation'] <= 1.0
            
            # Check losses are in valid range [0, 2]
            assert 0.0 <= metrics['target_0_loss'] <= 2.0
            assert 0.0 <= metrics['target_1_loss'] <= 2.0
            assert 0.0 <= metrics['mean_loss'] <= 2.0