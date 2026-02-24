import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import torch
import torch.nn as nn
from src.models.deep_lob import TemporalFusionDeepLOB

def load_pytorch_model():
    """Load the PyTorch model from checkpoint"""
    checkpoint_path = './models/final/best_model.pt'
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model instance
        model = TemporalFusionDeepLOB()
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        print(f"Loaded PyTorch model from {checkpoint_path}")
        print(f"Model architecture: {model.__class__.__name__}")
        
        return model
        
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_onnx_inference():
    """Test ONNX model inference for compatibility"""
    print("\n" + "="*60)
    print("Testing ONNX Model Inference")
    print("="*60)
    
    import onnxruntime as ort
    
    onnx_path = 'model.onnx'
    
    try:
        # Initialize ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        ort_session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        print(f"Loaded ONNX model from {onnx_path}")
        print(f"Input name: {input_name}, shape: {ort_session.get_inputs()[0].shape}")
        print(f"Output name: {output_name}, shape: {ort_session.get_outputs()[0].shape}")
        
        # Create a test tensor with shape (1, 1000, 32)
        test_tensor = np.random.randn(1, 1000, 32).astype(np.float32)
        print(f"\nCreated test tensor with shape: {test_tensor.shape}")
        print(f"Tensor dtype: {test_tensor.dtype}")
        
        # Run inference
        ort_inputs = {input_name: test_tensor}
        output = ort_session.run([output_name], ort_inputs)[0]
        
        print(f"\nONNX Inference successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        
        # Verify the output shape is (1, 901, 2)
        if output.shape == (1, 901, 2):
            print("✓ ONNX Output shape is correct: (1, 901, 2)")
        else:
            print(f"✗ ONNX Output shape is incorrect: {output.shape}, expected (1, 901, 2)")
            return False
            
        # Verify the output is float32
        if output.dtype == np.float32:
            print("✓ ONNX Output dtype is correct: float32")
        else:
            print(f"✗ ONNX Output dtype is incorrect: {output.dtype}, expected float32")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Running prerun.py to verify model inference with torch.inference_mode()...")
    
    # Test PyTorch model with torch.inference_mode()
    print("\n" + "="*60)
    print("Testing PyTorch Model with torch.inference_mode()")
    print("="*60)
    
    model = load_pytorch_model()
    if model is None:
        print("Failed to load PyTorch model")
        return 1
    
    # Create a test tensor with shape (1, 1000, 32)
    test_tensor = torch.randn(1, 1000, 32, dtype=torch.float32)
    print(f"\nCreated PyTorch test tensor with shape: {test_tensor.shape}")
    print(f"Tensor dtype: {test_tensor.dtype}")
    
    # Test with torch.inference_mode() as requested in PRD
    with torch.inference_mode():
        try:
            output = model(test_tensor)
            
            print(f"\nPyTorch inference with torch.inference_mode() successful!")
            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            
            # Verify the output shape is (1, 901, 2)
            if output.shape == (1, 901, 2):
                print("✓ PyTorch Output shape is correct: (1, 901, 2)")
            else:
                print(f"✗ PyTorch Output shape is incorrect: {output.shape}, expected (1, 901, 2)")
                return 1
                
            # Verify the output is float32
            if output.dtype == torch.float32:
                print("✓ PyTorch Output dtype is correct: torch.float32")
            else:
                print(f"✗ PyTorch Output dtype is incorrect: {output.dtype}, expected torch.float32")
                return 1
                
            # Test the model's predict method (which should also use torch.inference_mode internally)
            print(f"\nTesting model.predict() method...")
            predict_output = model.predict(test_tensor)
            print(f"model.predict() output shape: {predict_output.shape}")
            print(f"model.predict() output dtype: {predict_output.dtype}")
            
        except Exception as e:
            print(f"Error during PyTorch inference: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Test ONNX inference for compatibility
    onnx_success = test_onnx_inference()
    if not onnx_success:
        return 1
    
    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)
    print("\nSummary:")
    print("✓ PyTorch model loaded and tested with torch.inference_mode()")
    print("✓ Input shape correctly mapped: (1, 1000, 32) -> (1, 901, 2)")
    print("✓ ONNX model compatible and produces same output shape")
    print("✓ All outputs are float32 precision")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)