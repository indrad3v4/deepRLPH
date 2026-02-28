import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
# STRICT ALIGNMENT: Testing T-KAN only
from src.models.t_kan import TemporalKAN


def main():
    print("Running prerun.py to verify model inference with torch.inference_mode()...")

    print("\n" + "=" * 60)
    print("Testing PyTorch Model (T-KAN) with torch.inference_mode()")
    print("=" * 60)

    checkpoint_path = 'best_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return 1

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = TemporalKAN(input_dim=40)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print("✅ Successfully loaded TemporalKAN weights")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    test_tensor = torch.randn(1, 1000, 40, dtype=torch.float32)
    print(f"Created PyTorch test tensor with shape: {test_tensor.shape}")

    with torch.inference_mode():
        try:
            output = model(test_tensor)
            print(f"PyTorch inference successful!")
            print(f"Output shape: {output.shape}")

            if output.shape == (1, 901, 2):
                print("✅ PyTorch Output shape is correct: (1, 901, 2)")
            else:
                print(f"❌ PyTorch Output shape is incorrect: {output.shape}")
                return 1

        except Exception as e:
            print(f"❌ Error during PyTorch inference: {e}")
            return 1

    # Verify JIT Model exists
    if not os.path.exists('model_jit.pt'):
        print("❌ Error: model_jit.pt not found. Did the deploy() function run?")
        return 1
    else:
        print("✅ Verified model_jit.pt exists for packaging.")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)