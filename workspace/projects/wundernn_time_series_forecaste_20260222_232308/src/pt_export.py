import torch
import os
import sys
from pathlib import Path

# Ensure we can import our model
sys.path.insert(0, os.path.abspath('.'))
from src.models.deep_lob import create_model


def export():
    print("💎 Guru Export: Freezing the Beast into TorchScript...")

    # Paths
    model_path = Path('models/final/best_model.pt')
    export_path = Path('model_jit.pt')

    if not model_path.exists():
        print(f"❌ Error: {model_path} not found. Did you finish training?")
        return

    # 1. Load the state dict
    model = create_model()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Create dummy input for tracing (batch, seq, features)
    dummy_input = torch.randn(1, 1000, 40)

    # 3. Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    # 4. Save for submission
    traced_model.save(str(export_path))
    print(f"✅ Success! TorchScript model saved to: {export_path}")
    print(f"📦 Model size: {os.path.getsize(export_path) / 1024:.1f} KB")


if __name__ == "__main__":
    export()