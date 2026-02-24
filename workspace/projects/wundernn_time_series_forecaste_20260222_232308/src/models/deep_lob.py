import sys
import os

sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn


class TemporalFusionDeepLOB(nn.Module):
    """
    TRIZ-Optimized DeepLOB Architecture for CPU Inference
    Input: (batch_size, 1000, 40)
    Output: (batch_size, 901, 2)
    """

    def __init__(self, input_dim=40, hidden_dim=64):
        super(TemporalFusionDeepLOB, self).__init__()

        # 1. Cross-Sectional Feature Mixer (1x1 Conv)
        self.feature_mixer = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1)
        )

        # 2. Spatial Feature Extraction
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1)
        )

        # 3. Temporal Modeling (Lightning fast 1-Layer LSTM)
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
        # Conv1d Permute: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.feature_mixer(x)
        x = self.spatial_cnn(x)

        # LSTM Permute: (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)

        # Slice BEFORE linear math to save CPU
        lstm_out = lstm_out[:, 99:, :]

        out = self.fc1(lstm_out)
        out = self.leaky_relu(out)
        out = self.fc2(out)

        return out.to(torch.float32)


def create_model():
    return TemporalFusionDeepLOB(input_dim=40)


if __name__ == "__main__":
    model = create_model()
    model.eval()

    batch_size = 1
    dummy_input = torch.randn(batch_size, 1000, 40)

    with torch.inference_mode():
        output = model(dummy_input)

    print(f"Model created successfully. Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 901, 2)