import sys
import os
import subprocess
from pathlib import Path

# UNIVERSAL PATH INJECTOR
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import argparse
import json

warnings.filterwarnings('ignore')

from src.models.registry import get_model
from src.data.loader import OFIDataLoader
from src.losses.weighted_pearson import (
    WeightedPearsonCorrelationLoss,
    weighted_pearson_corr,
)


class TrainingPipeline:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: str = "cpu",
        lr: float = 3e-4,
        patience: int = 4,
        max_epochs: int = 20,
        target_pearson: float = 0.35,  # kept only for logging
        teacher_forcing_prob: float = 0.5,
        causal_training: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.target_pearson = target_pearson
        self.teacher_forcing_prob = teacher_forcing_prob
        self.causal_training = causal_training

        self.best_val_pearson = -float("inf")
        self.no_improvement_count = 0

        # Loss: weighted Pearson + MSE hybrid
        self.pearson_criterion = WeightedPearsonCorrelationLoss(reduction="mean")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )

    # --------------------------------------------------------------------- #
    # Training / validation
    # --------------------------------------------------------------------- #

    def train_epoch(self):
        self.model.train()
        total_pearson_loss = 0.0
        total_hybrid_loss = 0.0

        # For metric: collect only LAST timestep predictions (leaderboard-style)
        last_preds = []
        last_targets = []

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            # Optional teacher forcing for causal training
            if (
                self.causal_training
                and self.teacher_forcing_prob > 0.0
                and self.model.training
            ):
                preds = self._forward_with_teacher_forcing(features, targets)
            else:
                preds = self.model(features)

            # Align sequence length if needed
            if preds.shape[1] != targets.shape[1]:
                targets = targets[:, -preds.shape[1] :, :]

            p_loss = self.pearson_criterion(preds, targets)
            mse_loss = F.mse_loss(preds, targets)

            pearson_weight = 0.7
            mse_weight = 0.3
            hybrid_loss = pearson_weight * p_loss + mse_weight * mse_loss

            hybrid_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_pearson_loss += p_loss.item()
            total_hybrid_loss += hybrid_loss.item()

            # metric on last timestep only
            last_preds.append(preds[:, -1, :].detach().cpu())
            last_targets.append(targets[:, -1, :].detach().cpu())

        batches = len(self.train_loader)
        avg_hybrid = total_hybrid_loss / max(batches, 1)

        last_preds_tensor = torch.cat(last_preds, dim=0)
        last_targets_tensor = torch.cat(last_targets, dim=0)
        train_pearson_last = weighted_pearson_corr(
            last_preds_tensor, last_targets_tensor
        )

        return avg_hybrid, train_pearson_last

    def _forward_with_teacher_forcing(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with (placeholder) teacher forcing for causal training.
        Currently just forwards features; hook is here for future refinement.
        """
        # Example place to modify features based on targets if needed.
        if torch.rand(1).item() < self.teacher_forcing_prob:
            _ = features  # no-op for now
        return self.model(features)

    def validate(self):
        self.model.eval()
        total_p_loss = 0.0
        total_hybrid_loss = 0.0

        last_preds = []
        last_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                f = batch["features"].to(self.device)
                t = batch["targets"].to(self.device)

                p = self.model(f)
                if p.shape[1] != t.shape[1]:
                    t = t[:, -p.shape[1] :, :]

                p_loss = self.pearson_criterion(p, t)
                mse_loss = F.mse_loss(p, t)

                pearson_weight = 0.7
                mse_weight = 0.3
                hybrid_loss = pearson_weight * p_loss + mse_weight * mse_loss

                total_p_loss += p_loss.item()
                total_hybrid_loss += hybrid_loss.item()

                last_preds.append(p[:, -1, :].detach().cpu())
                last_targets.append(t[:, -1, :].detach().cpu())

        batches = len(self.val_loader)
        avg_hybrid = total_hybrid_loss / max(batches, 1)

        last_preds_tensor = torch.cat(last_preds, dim=0)
        last_targets_tensor = torch.cat(last_targets, dim=0)
        val_pearson_last = weighted_pearson_corr(
            last_preds_tensor, last_targets_tensor
        )

        return avg_hybrid, val_pearson_last

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #

    def run(self):
        print(f"🚀 Starting Training Pipeline | Target Pearson (info): {self.target_pearson}")
        print(f"📊 Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.causal_training:
            print(
                f"🔒 Causal training enabled with teacher forcing probability: {self.teacher_forcing_prob}"
            )

        for epoch in range(1, self.max_epochs + 1):
            train_hybrid, train_pearson = self.train_epoch()
            val_hybrid, val_pearson = self.validate()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{self.max_epochs} | "
                f"Train Loss: {train_hybrid:.4f} | Train Pearson: {train_pearson:.4f} | "
                f"Val Loss: {val_hybrid:.4f} | Val Pearson: {val_pearson:.4f} | "
                f"LR: {lr:.6f}"
            )

            # EARLY STOP only on plateau of Val Pearson, not on hitting a target KPI
            if val_pearson > self.best_val_pearson:
                self.best_val_pearson = val_pearson
                self.no_improvement_count = 0
                torch.save(self.model.state_dict(), "best_model.pt")
                print("  🌟 NEW BEST! Saved weights to best_model.pt")
            else:
                self.no_improvement_count += 1
                print(
                    f"  ⚠️ No improvement ({self.no_improvement_count}/{self.patience})"
                )

            if self.no_improvement_count >= self.patience:
                print(
                    f"\n🛑 EARLY STOPPING: Plateau reached at Val Pearson = {self.best_val_pearson:.4f}"
                )
                break

        self.deploy()

    # --------------------------------------------------------------------- #
    # Deployment / packaging
    # --------------------------------------------------------------------- #

    def deploy(self):
        print("\n" + "=" * 50 + "\n📦 PACKAGING SUBMISSION\n" + "=" * 50)

        if os.path.exists("best_model.pt"):
            self.model.load_state_dict(
                torch.load("best_model.pt", map_location=self.device)
            )
            self.model.eval()
            print("✅ Loaded best weights.")

            print("⚡ Generating TorchScript JIT graph...")
            if hasattr(self.model, "input_dim"):
                input_dim = self.model.input_dim
            else:
                input_dim = 32  # fallback
            dummy = torch.randn(1, 1000, input_dim, device=self.device)
            with torch.no_grad():
                traced = torch.jit.trace(self.model, dummy)
                torch.jit.save(traced, "model_jit.pt")
            print("✅ Saved model_jit.pt")

        if os.path.exists("prerun.py"):
            print("\n🧪 Executing prerun.py sanity check...")
            res = subprocess.run(["python", "prerun.py"], capture_output=True, text=True)
            if res.returncode == 0:
                print("✅ Prerun Passed")
            else:
                print(f"❌ Prerun Failed:\n{res.stderr}")

        if os.path.exists("scripts/make_submission.py"):
            print("\n🗜️ Creating submission.zip...")
            subprocess.run(["python", "scripts/make_submission.py", "."])
            print("✅ Pipeline Complete.")


# ------------------------------------------------------------------------- #
# Data loaders and config
# ------------------------------------------------------------------------- #

def create_data_loaders(train_file, valid_file, batch_size=32, test_mode=False):
    """Create data loaders for training and validation."""
    train_ds = OFIDataLoader(train_file, 1000, 32, True, "cpu", 20000, test_mode)
    val_ds = OFIDataLoader(valid_file, 1000, 32, False, "cpu", 5000, test_mode)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


def load_config(config_path=None):
    """Load training configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "model": "causal_lob_forecaster",
            "input_dim": 40,
            "horizon": 5,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "max_epochs": 20,
            "patience": 4,          # 4 epochs without improvement -> stop
            "target_pearson": 0.35,
            "device": "cpu",
            "causal_mask": True,
            "causal_training": True,
            "teacher_forcing_prob": 0.5,
        }
    return config


def main():
    parser = argparse.ArgumentParser(description="Train time series forecasting model")
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode with small dataset",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run fast development test",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="causal_lob_forecaster",
        help="Model name from registry",
    )
    parser.add_argument("--input_dim", type=int, default=40, help="Input dimension")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon")
    parser.add_argument(
        "--target_pearson",
        type=float,
        default=0.35,
        help="Target pearson correlation (info only)",
    )
    parser.add_argument(
        "--causal_mask",
        action="store_true",
        help="Enable causal masking (default: True for causal_lob_forecaster)",
    )
    parser.add_argument(
        "--causal_training",
        action="store_true",
        help="Enable causal training with teacher forcing",
    )
    parser.add_argument(
        "--teacher_forcing_prob",
        type=float,
        default=0.5,
        help="Teacher forcing probability (0.0 to 1.0)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.model != "causal_lob_forecaster":
        config["model"] = args.model
    if args.input_dim != 40:
        config["input_dim"] = args.input_dim
    if args.horizon != 5:
        config["horizon"] = args.horizon
    if args.target_pearson != 0.35:
        config["target_pearson"] = args.target_pearson
    if args.causal_mask:
        config["causal_mask"] = True
    if args.causal_training:
        config["causal_training"] = True
    if args.teacher_forcing_prob != 0.5:
        config["teacher_forcing_prob"] = args.teacher_forcing_prob

    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create model using registry
    print(
        f"\n🏗️  Creating model: {config['model']} with "
        f"input_dim={config['input_dim']}, horizon={config['horizon']}, "
        f"causal_mask={config.get('causal_mask', False)}"
    )
    model_kwargs = {
        "input_dim": config["input_dim"],
        "horizon": config["horizon"],
    }
    if "causal_mask" in config:
        model_kwargs["causal_mask"] = config["causal_mask"]

    import src.models.causal_lob_forecaster  # ensure module is loaded
    model = get_model(config["model"], **model_kwargs)

    # Create data loaders
    train_file = "data/raw/train.parquet"
    valid_file = "data/raw/valid.parquet"

    batch_size = 4 if args.test_mode or args.fast_dev_run else config["batch_size"]
    train_loader, val_loader = create_data_loaders(
        train_file,
        valid_file,
        batch_size=batch_size,
        test_mode=args.test_mode or args.fast_dev_run,
    )

    # Create and run pipeline
    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        lr=config["learning_rate"],
        patience=config["patience"],
        max_epochs=2
        if args.test_mode or args.fast_dev_run
        else config["max_epochs"],
        target_pearson=config["target_pearson"],
        teacher_forcing_prob=config.get("teacher_forcing_prob", 0.5),
        causal_training=config.get("causal_training", False),
    )

    pipeline.run()


if __name__ == "__main__":
    main()
