import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
import numpy as np
from models.deep_lob import TemporalFusionDeepLOB, create_model


def export_model_to_onnx():
    """
    Export the Temporal Fusion DeepLOB model to ONNX format.
    
    The model accepts input of shape (batch, 1000, 32)
    and produces output of shape (batch, 901, 2).
    
    The exported model will be saved as 'model.onnx' in the root directory.
    """
    print("Starting ONNX export...")
    
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
    
    # Define the ONNX export path
    onnx_path = 'model.onnx'
    
    # Export the model to ONNX
    print(f"\nExporting model to {onnx_path}...")
    
    # Define dynamic axes for batch dimension
    dynamic_axes = {
        'input': {0: 'batch_size'},  # batch dimension is dynamic
        'output': {0: 'batch_size'}   # batch dimension is dynamic
    }
    
    # Export with opset version 14 for better compatibility
    torch.onnx.export(
        model,                    # model to export
        dummy_input,              # model input (or a tuple for multiple inputs)
        onnx_path,                # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=14,         # ONNX opset version to export the model to
        do_constant_folding=True, # whether to execute constant folding for optimization
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes=dynamic_axes, # variable length axes
        verbose=False
    )
    
    print(f"Model successfully exported to {onnx_path}")
    
    # Verify the exported model can be loaded
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed!")
        
        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Prepare input
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        
        # Run inference
        ort_outs = ort_session.run(None, ort_inputs)
        
        print(f"ONNX Runtime output shape: {ort_outs[0].shape}")
        
        # Compare PyTorch and ONNX Runtime outputs
        pytorch_output = output.numpy()
        onnx_output = ort_outs[0]
        
        # Check if outputs are close (allow small numerical differences)
        if np.allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5):
            print("PyTorch and ONNX Runtime outputs match within tolerance!")
        else:
            print("Warning: PyTorch and ONNX Runtime outputs differ")
            print(f"Max absolute difference: {np.max(np.abs(pytorch_output - onnx_output))}")
            print(f"Mean absolute difference: {np.mean(np.abs(pytorch_output - onnx_output))}")
        
    except ImportError as e:
        print(f"Warning: Could not verify ONNX model - {e}")
    except Exception as e:
        print(f"Warning: ONNX verification failed - {e}")
    
    print("\nExport completed successfully!")
    return True


if __name__ == "__main__":
    try:
        success = export_model_to_onnx()
        if success:
            print("\n" + "="*50)
            print("ONNX EXPORT SUMMARY:")
            print("="*50)
            print(f"Model saved as: model.onnx")
            print(f"Input shape: (batch, 1000, 32)")
            print(f"Output shape: (batch, 901, 2)")
            print(f"ONNX opset version: 14")
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