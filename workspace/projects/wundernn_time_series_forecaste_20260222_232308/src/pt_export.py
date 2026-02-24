import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
from models.deep_lob import TemporalFusionDeepLOB, create_model


def export_model_to_torchscript():
    """
    Export the Temporal Fusion DeepLOB model to TorchScript format.
    
    The model accepts input of shape (batch, 1000, 32)
    and produces output of shape (batch, 901, 2).
    
    The exported model will be saved as 'model_jit.pt' in the root directory.
    """
    print("Starting TorchScript export...")
    
    # Create the model
    model = create_model()
    model.eval()  # Set to evaluation mode
    
    # Create dummy input with the correct shape
    # Input: (batch_size, sequence_length=1000, features=32)
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1000, 32)
    
    print(f"Model input shape: {dummy_input.shape}")
    
    # Test the model forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Expected output shape: (batch, 901, 2)")
        
        # Verify the output shape matches requirements
        assert output.shape == (batch_size, 901, 2), \
            f"Output shape {output.shape} does not match expected (batch, 901, 2)"
    
    # Define the TorchScript export path
    torchscript_path = 'model_jit.pt'
    
    print(f"\nExporting model to TorchScript format...")
    
    # Method 1: Trace the model (recommended for models without control flow)
    print("Method 1: Tracing the model...")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Save the traced model
        traced_model.save(torchscript_path)
        print(f"Traced model saved to {torchscript_path}")
        
        # Verify the traced model
        loaded_model = torch.jit.load(torchscript_path)
        with torch.no_grad():
            traced_output = loaded_model(dummy_input)
        
        print(f"Traced model output shape: {traced_output.shape}")
        
        # Check if outputs match
        if torch.allclose(output, traced_output, rtol=1e-3, atol=1e-5):
            print("Original model and traced model outputs match within tolerance!")
        else:
            print("Warning: Original model and traced model outputs differ")
            print(f"Max absolute difference: {torch.max(torch.abs(output - traced_output))}")
            print(f"Mean absolute difference: {torch.mean(torch.abs(output - traced_output))}")
            
    except Exception as e:
        print(f"Tracing failed: {e}")
        print("Falling back to scripting...")
        
        # Method 2: Script the model (for models with control flow)
        try:
            scripted_model = torch.jit.script(model)
            
            # Save the scripted model
            scripted_model.save(torchscript_path)
            print(f"Scripted model saved to {torchscript_path}")
            
            # Verify the scripted model
            loaded_model = torch.jit.load(torchscript_path)
            with torch.no_grad():
                scripted_output = loaded_model(dummy_input)
            
            print(f"Scripted model output shape: {scripted_output.shape}")
            
            # Check if outputs match
            if torch.allclose(output, scripted_output, rtol=1e-3, atol=1e-5):
                print("Original model and scripted model outputs match within tolerance!")
            else:
                print("Warning: Original model and scripted model outputs differ")
                print(f"Max absolute difference: {torch.max(torch.abs(output - scripted_output))}")
                print(f"Mean absolute difference: {torch.mean(torch.abs(output - scripted_output))}")
                
        except Exception as e2:
            print(f"Scripting also failed: {e2}")
            raise
    
    # Test the exported model with different batch sizes
    print("\nTesting exported model with different batch sizes...")
    
    for test_batch_size in [1, 2, 4]:
        test_input = torch.randn(test_batch_size, 1000, 32)
        
        with torch.no_grad():
            # Original model
            original_output = model(test_input)
            
            # Loaded TorchScript model
            loaded_model = torch.jit.load(torchscript_path)
            torchscript_output = loaded_model(test_input)
            
            print(f"Batch size {test_batch_size}:")
            print(f"  Original shape: {original_output.shape}")
            print(f"  TorchScript shape: {torchscript_output.shape}")
            
            if torch.allclose(original_output, torchscript_output, rtol=1e-3, atol=1e-5):
                print(f"  ✓ Outputs match")
            else:
                print(f"  ✗ Outputs differ")
                print(f"    Max diff: {torch.max(torch.abs(original_output - torchscript_output)):.6f}")
    
    # Test the predict method (which ensures float32 output)
    print("\nTesting predict method...")
    with torch.no_grad():
        predict_output = model.predict(dummy_input)
        print(f"Predict output dtype: {predict_output.dtype}")
        print(f"Predict output shape: {predict_output.shape}")
        
        # Verify it's float32
        assert predict_output.dtype == torch.float32, \
            f"Predict output dtype {predict_output.dtype} is not float32"
    
    print(f"\nTorchScript export completed successfully!")
    print(f"Model saved as: {torchscript_path}")
    
    # Provide usage example
    print("\n" + "="*50)
    print("TORCHSCRIPT USAGE EXAMPLE:")
    print("="*50)
    print("In Python:")
    print("  model = torch.jit.load('model_jit.pt')")
    print("  input_tensor = torch.randn(1, 1000, 32)")
    print("  output = model(input_tensor)")
    print("")
    print("In C++ (libtorch):")
    print("  #include <torch/script.h>")
    print("  torch::jit::script::Module module;")
    print("  module = torch::jit::load(\"model_jit.pt\");")
    print("  std::vector<torch::jit::IValue> inputs;")
    print("  inputs.push_back(torch::randn({1, 1000, 32}));")
    print("  at::Tensor output = module.forward(inputs).toTensor();")
    print("="*50)
    
    return True


if __name__ == "__main__":
    try:
        success = export_model_to_torchscript()
        if success:
            print("\n" + "="*50)
            print("TORCHSCRIPT EXPORT SUMMARY:")
            print("="*50)
            print(f"Model saved as: model_jit.pt")
            print(f"Input shape: (batch, 1000, 32)")
            print(f"Output shape: (batch, 901, 2)")
            print(f"Export method: Tracing (fallback to scripting if needed)")
            print(f"Output dtype: float32 (ensured by predict method)")
            print("="*50)
            sys.exit(0)
        else:
            print("Export failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)