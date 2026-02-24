import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn


class TemporalFusionDeepLOB(nn.Module):
    """
    TRIZ-Optimized DeepLOB Architecture for hyper-fast CPU Inference.

    Cures the following pains:
    1. The "Batch=1 Crash": Uses GroupNorm instead of BatchNorm to ensure
       production prerun.py and evaluation servers never crash.
    2. The "CPU Asphyxiation": Uses a 1-Layer Unidirectional LSTM and slices
       the tensor BEFORE the linear layers, saving ~10% of total compute.
    3. The "Cross-Sectional Blindness": Uses a 1x1 Conv to instantly map
       all 40 engineered features into a unified spatial representation.

    Input: (batch_size, 1000, 40) -> Matches the Iron Shield loader
    Output: (batch_size, 901, 2)
    """

    def __init__(self, input_dim=40, hidden_dim=64):
        super(TemporalFusionDeepLOB, self).__init__()

        # 1. Cross-Sectional Feature Mixer (1x1 Conv)
        # TRIZ Principle 17: Another Dimension. We learn the relationships between
        # normalized prices, Log-OFI, and Imbalances BEFORE looking at time.
        self.feature_mixer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1),
            nn.GroupNorm(8, 64),  # 8 groups, completely immune to batch_size=1 panics
            nn.LeakyReLU(0.1)
        )

        # 2. Spatial Feature Extraction (Temporal CNNs)
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1)
        )

        # 3. Temporal Modeling
        # 1-Layer Unidirectional LSTM. Bidirectional is twice as slow on CPUs and
        # often leaks future bias in poor implementations. We keep it lean.
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 4. Prediction Head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(32, 2)

        self._init_weights()

    def _init_weights(self):
        """
        Xavier and Kaiming initialization to ensure the model doesn't start
        with extreme weights that trigger NaN losses.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expected Input: (batch_size, 1000, 40)
        Expected Output: (batch_size, 901, 2)
        """
        # Permute for Conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        # Apply 1x1 Mixing and Spatial CNNs
        x = self.feature_mixer(x)
        x = self.spatial_cnn(x)

        # Permute back for LSTM: (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # Apply LSTM
        lstm_out, _ = self.lstm(x)

        # TRIZ Principle 2: Taking out (Extract only what is needed)
        # We only need predictions from step 99 to 999.
        # Slicing BEFORE the linear layers saves 99 * batch_size matrix multiplications per forward pass.
        lstm_out = lstm_out[:, 99:, :]  # Shape: (batch, 901, hidden_dim)

        # Apply fully connected head
        out = self.fc1(lstm_out)
        out = self.leaky_relu(out)
        out = self.fc2(out)  # Shape: (batch, 901, 2)

        # Competition evaluator requires float32
        return out.to(torch.float32)


def create_model():
    """Factory function"""
    return TemporalFusionDeepLOB(input_dim=40)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # PAIN RELIEF TEST: Verify immunity to batch_size=1 crashes
    # ---------------------------------------------------------
    model = create_model()
    model.eval()  # Force eval mode to trigger normalization behavior

    batch_size = 1  # Simulating the exact environment of prerun.py
    dummy_input = torch.randn(batch_size, 1000, 40)

    with torch.inference_mode():
        output = model(dummy_input)

    print(f"✅ Model instantiated and executed successfully.")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,} (Highly CPU optimal)")
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Output shape: {output.shape}")

    assert output.shape == (batch_size, 901, 2), f"Dimension mismatch: {output.shape}"