# -*- coding: utf-8 -*-
"""End-to-end WunderNN integration test utilities.

This is a high-level smoke test that:
- Creates a temporary ML competition-style project (WunderNN-like time series)
- Runs schema ingestion + PRD refinement
- Executes a tiny PRD loop with a mocked DeepSeek client
- Runs submission packaging (solution.py + dummy model.onnx)

The goal is to verify that the whole deepRLPH pipeline stays wired for
WunderNN-style projects without depending on real external APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from orchestrator import RalphOrchestrator, ProjectConfig, MLConfig


class DummyDeepseekClient:
    """Minimal DeepSeek stub for end-to-end tests.

    It returns deterministic, small responses that allow the PRD generator
    and execution engine to run without external calls.
    """

    def __init__(self) -> None:
        self.calls = []

    async def call_agent(self, system_prompt: str, user_message: str, thinking_budget: int = 1000, temperature: float = 0.0) -> Dict[str, Any]:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
        })
        # Return a tiny JSON blob that looks like a clarified task / config
        fake = {
            "status": "success",
            "response": json.dumps(
                {
                    "problem_type": "time_series_forecasting",
                    "ml_framework": "PyTorch",
                    "model_type": "Transformer",
                    "training_preset": {"batch_size": 4, "epochs": 1, "learning_rate": 0.0005},
                    "eval_metric": "weighted_pearson",
                    "metric_target": 0.1,
                    "checklist": ["dummy item"],
                }
            ),
        }
        return fake


def create_dummy_wundernn_project(tmp_path: Path) -> RalphOrchestrator:
    """Create a minimal WunderNN-style project in a temp workspace."""

    deepseek_client = DummyDeepseekClient()
    orchestrator = RalphOrchestrator(workspace_dir=tmp_path, deepseek_client=deepseek_client)

    ml_cfg = MLConfig(
        competition_source="wundernn.io",
        competition_url="https://wundernn.io/competitions/demo",
        dataset_files=["train.parquet"],
        problem_type="time_series_forecasting",
        sequence_length=128,
        num_features=8,
        target_variable="target",
        model_type="Transformer",
        ml_framework="PyTorch",
        batch_size=4,
        epochs=1,
        learning_rate=0.0005,
        eval_metric="weighted_pearson",
    )

    cfg = ProjectConfig(
        name="wundernn_demo",
        domain="time_series_forecasting",
        description="Dummy WunderNN-style time series competition",
        project_type="ml_competition",
        ml_config=ml_cfg,
        metadata={
            "project_type": "ml_competition",
            "competition_url": ml_cfg.competition_url,
            "eval_metric": "weighted_pearson",
            "metric_target": 0.1,
        },
    )

    result = orchestrator.create_project(cfg)
    assert result["status"] == "success"

    project_dir = Path(result["path"])
    # Create dummy dataset + model + solution for submission pipeline
    (project_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (project_dir / "data" / "raw" / "train.parquet").write_bytes(b"PARQUET")

    (project_dir / "models" / "final").mkdir(parents=True, exist_ok=True)
    (project_dir / "models" / "final" / "model.onnx").write_bytes(b"ONNX")

    (project_dir / "solution.py").write_text(
        """class PredictionModel:\n    def predict(self, df):\n        return [0.0] * len(df)\n""",
        encoding="utf-8",
    )

    return orchestrator
