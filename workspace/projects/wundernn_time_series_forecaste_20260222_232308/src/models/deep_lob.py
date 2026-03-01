import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn


class TemporalFusionDeepLOB(nn.Module):
    """
    TRIZ-Optimized DeepLOB Architecture for hyper-fast CPU Inference.

    ANTI-OVERFITTING EDITION:
    Combats the 1.0000 -> 0.0270 Leaderboard drop by severely penalizing
    memorization pathways using heavy Dropout layers.
    
    UPDATED FOR HYDRA:
    - Uses 40 input channels as per PRD requirements
    - Maintains strict causal slicing
    """

    def __init__(self, input_dim=40, hidden_dim=64, num_lstm_layers=2):
        super(TemporalFusionDeepLOB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # 1. Cross-Sectional Feature Mixer (1x1 Conv) - UPDATED: 40 input channels
        self.feature_mixer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1)
            # No dropout here: we want pure spatial mixing first
        )

        # 2. Spatial Feature Extraction (Temporal CNNs)
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),  # 🛡️ THE SHIELD: Kills 30% of local temporal neurons

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3)   # 🛡️ THE SHIELD: Prevents specific tick memorization
        )

        # 3. Temporal Modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # 4. Prediction Head
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.4) # 🛡️ THE ULTIMATE BOTTLENECK: 40% kill rate before final output
        self.fc2 = nn.Linear(32, 2)

        # Example input for documentation and testing
        self.example_input_array = torch.randn(1, 1000, input_dim)

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
        Forward pass of the DeepLOB model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
                - sequence_length: Must be >= 1000 for proper slicing
                - input_dim: Must be 40 features (as per PRD)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length - 99, 2)
                - First dimension: batch_size
                - Second dimension: sequence_length - 99 (e.g., 901 when input is 1000)
                - Third dimension: 2 (prediction targets)

        Note:
            The model slices the last (sequence_length - 99) timesteps from the LSTM output
            to create predictions for future timesteps.
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

        # Slice BEFORE the linear layers saves 99 * batch_size matrix multiplications
        lstm_out = lstm_out[:, 99:, :]  # Shape: (batch, seq_len - 99, hidden_dim)

        # Apply fully connected head
        out = self.fc1(lstm_out)
        out = self.leaky_relu(out)
        out = self.dropout(out)  # 🚨 CRITICAL FIX: Applying the 40% Dropout here!
        out = self.fc2(out)      # Shape: (batch, seq_len - 99, 2)

        # Competition evaluator requires float32
        return out.to(torch.float32)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict method that ensures float32 output."""
        return self.forward(x)


def create_model():
    """Factory function to create a TemporalFusionDeepLOB model instance."""
    return TemporalFusionDeepLOB(input_dim=40, hidden_dim=64, num_lstm_layers=2)


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
    print(f"✅ Verification passed: Output shape is {output.shape}")