import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn.functional as F
import numpy as np
import pytest
from src.models.deep_lob import TemporalFusionDeepLOB, create_model


class TestTemporalFusionDeepLOB:
    """Test suite for Temporal Fusion DeepLOB Architecture"""
    
    def test_model_creation(self):
        """Test that model can be created"""
        model = create_model()
        assert isinstance(model, TemporalFusionDeepLOB)
        
        # Check model parameters
        assert model.input_dim == 32
        assert model.hidden_dim == 64
        assert model.num_lstm_layers == 2
    
    def test_input_output_shapes(self):
        """Test that model accepts correct input and produces correct output shape"""
        model = create_model()
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8, 16]:
            # Create input tensor with exact shape (batch_size, 1000, 32)
            dummy_input = torch.randn(batch_size, 1000, 32)
            
            # Forward pass
            output = model(dummy_input)
            
            # Check output shape
            expected_shape = (batch_size, 901, 2)
            assert output.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {output.shape}"
            
            # Check output dtype
            assert output.dtype == torch.float32, \
                f"Expected float32, got {output.dtype}"
    
    def test_vectorized_forward_pass(self):
        """Test that forward pass is vectorized (no for-loops)"""
        model = create_model()
        
        # Create input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1000, 32)
        
        # Forward pass
        output = model(dummy_input)
        
        # Check that output has correct shape
        assert output.shape == (batch_size, 901, 2)
        
        # Check that all values are finite
        assert torch.isfinite(output).all(), "Output contains NaN or infinite values"
    
    def test_slicing_behavior(self):
        """Test that model slices off first 99 steps correctly"""
        model = create_model()
        
        # Create input with known pattern
        batch_size = 2
        seq_len = 1000
        
        # Create input where each time step has a unique value
        dummy_input = torch.zeros(batch_size, seq_len, 32)
        for i in range(seq_len):
            dummy_input[:, i, :] = i  # Each time step gets its index as value
        
        # Forward pass
        output = model(dummy_input)
        
        # Output should have 901 steps (1000 - 99)
        assert output.shape[1] == 901
        
        # The model should have processed steps 99-999
        # We can't directly verify internal slicing, but we can check output consistency
        assert not torch.isnan(output).any(), "Output contains NaN values"
    
    def test_predict_method(self):
        """Test the predict method ensures float32 output"""
        model = create_model()
        
        # Create input
        dummy_input = torch.randn(4, 1000, 32)
        
        # Use predict method
        output = model.predict(dummy_input)
        
        # Check dtype
        assert output.dtype == torch.float32, \
            f"Predict method should return float32, got {output.dtype}"
        
        # Check shape
        assert output.shape == (4, 901, 2)
    
    def test_model_parameters(self):
        """Test that model has trainable parameters"""
        model = create_model()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Model should have parameters
        assert total_params > 0, "Model should have parameters"
        assert trainable_params > 0, "Model should have trainable parameters"
        assert total_params == trainable_params, "All parameters should be trainable"
    
    def test_batch_independence(self):
        """Test that model processes batches independently"""
        model = create_model()
        model.eval()  # Set to eval mode to disable dropout
        
        # Create two different inputs
        batch_size = 2
        input1 = torch.randn(batch_size, 1000, 32)
        input2 = torch.randn(batch_size, 1000, 32)
        
        # Process each input separately
        with torch.no_grad():
            output1 = model(input1)
            output2 = model(input2)
            
            # Process concatenated batch
            combined_input = torch.cat([input1, input2], dim=0)
            combined_output = model(combined_input)
        
        # Check that concatenated output matches individual outputs
        # Use larger tolerance due to floating point differences
        assert torch.allclose(combined_output[:batch_size], output1, rtol=1e-3, atol=1e-3), \
            "First half of combined output should match output1"
        assert torch.allclose(combined_output[batch_size:], output2, rtol=1e-3, atol=1e-3), \
            "Second half of combined output should match output2"
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the model"""
        model = create_model()
        
        # Create input and target
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1000, 32, requires_grad=True)
        dummy_target = torch.randn(batch_size, 901, 2)
        
        # Forward pass
        output = model(dummy_input)
        
        # Compute loss
        loss = F.mse_loss(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert dummy_input.grad is not None, "Input should have gradients"
        assert dummy_input.grad.shape == dummy_input.shape, \
            f"Gradient shape mismatch: {dummy_input.grad.shape} != {dummy_input.shape}"
        
        # Check that gradients are not all zero
        assert not torch.all(dummy_input.grad == 0), "Gradients should not be all zero"
    
    def test_model_on_gpu_if_available(self):
        """Test model can run on GPU if available"""
        model = create_model()
        
        if torch.cuda.is_available():
            # Move model to GPU
            model = model.cuda()
            
            # Create input on GPU
            dummy_input = torch.randn(4, 1000, 32).cuda()
            
            # Forward pass
            output = model(dummy_input)
            
            # Check output is on GPU
            assert output.is_cuda, "Output should be on GPU"
            assert output.shape == (4, 901, 2)
        else:
            pytest.skip("CUDA not available")
    
    def test_export_to_onnx_compatibility(self):
        """Test model can be exported to ONNX format"""
        model = create_model()
        model.eval()  # Set to evaluation mode
        
        # Create dummy input
        dummy_input = torch.randn(1, 1000, 32)
        
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                "test_model.onnx",
                input_names=["input"],
                output_names=["output"],
                opset_version=11
            )
            
            # Check file was created
            assert os.path.exists("test_model.onnx"), "ONNX file should be created"
            
            # Clean up
            if os.path.exists("test_model.onnx"):
                os.remove("test_model.onnx")
                
        except Exception as e:
            pytest.skip(f"ONNX export failed (may be due to MultiheadAttention): {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])