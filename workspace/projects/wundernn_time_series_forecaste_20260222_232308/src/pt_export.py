import torch
import os
import sys
import argparse
from pathlib import Path

# Ensure we can import our model
sys.path.insert(0, os.path.abspath('.'))
from src.models.deep_lob import create_model as create_deep_lob
from src.models.t_kan import create_model as create_t_kan


def export(model_type='deep_lob', checkpoint_path=None):
    print(f"💎 Guru Export: Freezing {model_type.upper()} into TorchScript...")

    # Determine input shape based on model type
    if model_type == 'deep_lob':
        model = create_deep_lob()
        input_shape = (1, 1000, 40)  # Fixed: DeepLOB expects 40 features, not 32
    elif model_type == 't_kan':
        model = create_t_kan()
        input_shape = (1, 1000, 40)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Use provided checkpoint or default path
    if checkpoint_path:
        model_path = Path(checkpoint_path)
    else:
        model_path = Path('models/final/best_model.pt')
    
    export_path = Path('model_jit.pt')

    if not model_path.exists():
        print(f"❌ Error: {model_path} not found. Did you finish training?")
        print(f"Creating a model with random weights for testing...")
        # Continue with random weights for testing
    else:
        # Load the state dict
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(*input_shape)

    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Model test forward pass successful")
        print(f"✅ Input shape: {dummy_input.shape}")
        print(f"✅ Output shape: {output.shape}")
        
        # Verify output shape
        assert output.shape == (1, 901, 2), f"Output shape mismatch: {output.shape}"

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    # Save for submission
    traced_model.save(str(export_path))
    print(f"✅ Success! TorchScript model saved to: {export_path}")
    print(f"📦 Model size: {os.path.getsize(export_path) / 1024:.1f} KB")
    
    # Test loading the traced model
    loaded_model = torch.jit.load(str(export_path))
    with torch.no_grad():
        loaded_output = loaded_model(dummy_input)
    
    if torch.allclose(output, loaded_output, rtol=1e-3, atol=1e-5):
        print("✅ Traced model loads and runs correctly!")
    else:
        print("⚠️ Warning: Traced model output differs from original")


def main():
    parser = argparse.ArgumentParser(description='Export model to TorchScript format')
    parser.add_argument('--model', type=str, default='deep_lob', 
                       choices=['deep_lob', 't_kan'],
                       help='Model type to export')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file')
    
    args = parser.parse_args()
    export(args.model, args.checkpoint)


if __name__ == "__main__":
    main()