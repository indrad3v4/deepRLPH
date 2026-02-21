# -*- coding: utf-8 -*-
"""
prd_generator.py - PRD Generator from Technical Brief (PR-001)

Supports KPI-aware ML competitions and standard API/LLM/Web projects.
Deroasted version: Fixes relative imports, avoids side-effect mutation,
and enforces universal submission rules for ML projects.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Absolute import to avoid "no known parent package" errors in orchestrator
from prd_model import PRDModel, normalize_prd

logger = logging.getLogger("PRDGenerator")


# ---------------------------------------------------------------------------
# Helper utilities for KPI + context extraction
# ---------------------------------------------------------------------------

def _extract_kpi_info(project_meta: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """Return canonical (metric_key, target) without mutating the input directly."""
    # We use .get to look for common KPI keys
    metric = (
            project_meta.get("kpi_metric")
            or project_meta.get("eval_metric")
            or "weighted_pearson"
    )

    m_lower = str(metric).lower()
    metric_key = "weighted_pearson" if "pearson" in m_lower else str(metric)

    target_raw = project_meta.get("kpi_target", project_meta.get("metric_target"))
    try:
        kpi_target = float(target_raw) if target_raw is not None else None
    except (TypeError, ValueError):
        kpi_target = None

    return metric_key, kpi_target


def _get_first_parquet_summary(project_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ctx = project_meta.get("ingested_context") or {}
    summaries = ctx.get("binary_summaries") or {}
    for _rel, summary in summaries.items():
        if summary.get("format") == "parquet":
            return summary
    return None


@dataclass
class UserStory:
    """Single PRD item builder."""
    id: str
    title: str
    description: str
    acceptance_criteria: List[str]
    verification: str
    why: str
    files_touched: List[str]
    status: str = "todo"
    attempts: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self):
        return asdict(self)


class PRDGenerator:
    """Convert technical brief → PRD backlog"""

    def __init__(self):
        self.stories: List[UserStory] = []

    def generate(
            self,
            technical_brief: Dict[str, Any],
            domain: str = "llm-app",
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate PRD from clarified task."""
        clarified_task = technical_brief.get("clarified_task", "")
        key_reqs = technical_brief.get("key_requirements", [])
        project_meta = project_metadata or technical_brief.get("project_metadata") or {}

        # Detect project type
        project_type_meta = project_meta.get("project_type", "")
        is_ml_competition = (project_type_meta == "ml_competition") or ("time_series" in domain.lower())

        if is_ml_competition:
            logger.info("   Detected: ML Competition")
            stories = self._decompose_ml_competition(
                task=clarified_task,
                reqs=key_reqs,
                project_meta=project_meta,
            )
        else:
            logger.info("   Using generic decomposition")
            stories = self._decompose_generic(clarified_task, key_reqs)

        # Build raw PRD dict using legacy shape for the normalizer
        raw_prd = {
            "task": clarified_task,
            "domain": domain,
            "total_items": len(stories),
            "user_stories": [s.to_dict() for s in stories],
            "project_metadata": project_meta,
        }

        # Normalize via PRDModel to guarantee canonical structure on disk
        return normalize_prd(raw_prd)

    def _decompose_ml_competition(
            self,
            task: str,
            reqs: List[str],
            project_meta: Optional[Dict[str, Any]] = None,
    ) -> List[UserStory]:
        """ML Competition decomposition (Universal & TRIZ-Optimized)."""
        project_meta = project_meta or {}
        metric_key, kpi_target = _extract_kpi_info(project_meta)

        # Pull data context if available
        ds_summary = _get_first_parquet_summary(project_meta) or {}
        row_count = ds_summary.get("row_count", "10M+")
        col_count = ds_summary.get("column_count", "37")

        stories = [
            UserStory(
                id="ML-001",
                title="Implement Custom Weighted Pearson Loss",
                description="Create a loss function that maximizes correlation directly.",
                why="Standard MSE doesn't optimize for the competition's success metric.",
                acceptance_criteria=[
                    "Implements weighted average of Pearson correlations",
                    "Handles amplitude weighting w_i = |y_i|",
                    "Includes output clipping to [-6, 6]"
                ],
                verification="pytest tests/test_loss.py -v",
                files_touched=["src/losses/weighted_pearson.py", "tests/test_loss.py"]
            ),
            UserStory(
                id="ML-002",
                title="Set Up Data Loader for LOB Sequences",
                description="Load 1000-step sequences and implement warm-up logic.",
                why="Correct data ingestion is foundational for sequence modeling.",
                acceptance_criteria=[
                    f"Loads sequences of shape (batch, 1000, {col_count})",
                    "Implements warm-up steps 0-98 and prediction steps 99-999",
                    "Respects independent sequence IDs"
                ],
                verification="pytest tests/test_data.py -v",
                files_touched=["src/data/loader.py", "tests/test_data.py"]
            ),
            UserStory(
                id="ML-003",
                title="Implement Microstructure Signal Engineering",
                description="Engineer VBI, Spread dynamics, and VPIN signals.",
                why="Raw features need microstructure context for high precision.",
                acceptance_criteria=[
                    "Computes Volume Bucket Imbalance (VBI)",
                    "Computes Order Flow Toxicity (VPIN)",
                    "Handles edge cases like zero volume or NaN"
                ],
                verification="pytest tests/test_features.py -v",
                files_touched=["src/features/microsignals.py", "tests/test_features.py"]
            ),
            UserStory(
                id="ML-004",
                title="Implement CNN Spatial Feature Extractor",
                description="CNN layers to extract bid/ask relationships.",
                why="Captures spatial dependencies across price and volume levels.",
                acceptance_criteria=[
                    "Specific kernel sizes for 32 input features",
                    "Unit tests validate output dimensions"
                ],
                verification="pytest tests/test_model.py::test_cnn -v",
                files_touched=["src/models/cnn_extractor.py", "tests/test_model.py"]
            ),
            UserStory(
                id="ML-005",
                title="Implement Variable Selection and Attention",
                description="Implement VSN layers and Temporal Self-Attention.",
                why="Dynamically weighs feature importance across 1000 steps.",
                acceptance_criteria=[
                    "VSN weighs raw vs engineered signals",
                    "Self-attention captures non-linear dependencies"
                ],
                verification="pytest tests/test_model.py::test_vsn_attention -v",
                files_touched=["src/models/variable_selection.py", "src/models/attention.py"]
            ),
            UserStory(
                id="ML-006",
                title="Assemble TemporalFusionDeepLOB Model",
                description="Integrate all components into a single PyTorch module.",
                why="Final cohesive model for dual-target (t0, t1) forecasting.",
                acceptance_criteria=[
                    "Integrated CNN, VSN, and Attention",
                    "Correct dual-head output for price movements"
                ],
                verification="pytest tests/test_full_model.py -v",
                files_touched=["src/models/temporal_fusion_deeplob.py", "tests/test_full_model.py"]
            ),
            UserStory(
                id="ML-007",
                title="Implement Training Loop and Early Stopping",
                description="Trainer with intellectual learning rate and metric tracking.",
                why="Ensures convergence and prevents overfitting.",
                acceptance_criteria=[
                    f"Tracks {metric_key} on validation set",
                    "Implements early stopping with patience",
                    "Logs training metrics per epoch"
                ],
                verification="pytest tests/test_training.py -v",
                files_touched=["src/training/trainer.py", "tests/test_training.py"]
            ),
            UserStory(
                id="ML-008",
                title="Implement ONNX Export and Root-Level Inference Script",
                description="Create root solution.py and package for submission.",
                why="Ensures code runs in competition environment and builds submission.zip.",
                acceptance_criteria=[
                    "Export model to 'models/final/model.onnx'",
                    "Create 'solution.py' in the ROOT directory (not src/)",
                    "Verify the package with the submission script"
                ],
                # ⛓️ The Universal Command: Test + Zip
                verification="pytest tests/test_onnx.py -v && python scripts/make_submission.py .",
                # 🏠 The Universal Path: solution.py at the root
                files_touched=["src/onnx/export.py", "solution.py", "tests/test_onnx.py"]
            )
        ]
        return stories

    def _decompose_generic(self, task: str, reqs: List[str]) -> List[UserStory]:
        """Fallback for non-ML projects."""
        return [
            UserStory(
                id="ITEM-001",
                title="Initial Setup",
                description="Base project structure.",
                why="Required to begin development.",
                acceptance_criteria=["Root main.py exists"],
                verification="pytest tests/",
                files_touched=["main.py"]
            )
        ]


def generate_prd(
        technical_brief: Dict[str, Any],
        domain: str = "llm-app",
        project_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to generate PRD."""
    gen = PRDGenerator()
    return gen.generate(technical_brief, domain, project_metadata)