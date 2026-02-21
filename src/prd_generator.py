# -*- coding: utf-8 -*-
"""
prd_generator.py - PRD Generator from Technical Brief (PR-001)

Supports KPI-aware ML competitions and standard API/LLM/Web projects.

Enhancements in this revision:
- Accepts optional project_metadata from orchestrator
- For ML competitions, injects metric/KPI hints into the decomposed stories
- Uses PRDModel as a single, normalized schema for all PRD output
- Reads dataset / ONNX summaries from ingested context when available
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .prd_model import PRDModel, normalize_prd

logger = logging.getLogger("PRDGenerator")


# ---------------------------------------------------------------------------
# Helper utilities for KPI + context extraction
# ---------------------------------------------------------------------------


def _extract_kpi_info(project_meta: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    """Return canonical (metric_key, target) and normalise metadata in-place.

    - metric_key is a snake_case identifier like "weighted_pearson".
    - target is a float or None if not provided.
    """

    metric = (
        project_meta.get("kpi_metric")
        or project_meta.get("eval_metric")
        or "weighted_pearson"
    )

    m_lower = str(metric).lower()
    if m_lower.startswith("weighted") and "pearson" in m_lower:
        metric_key = "weighted_pearson"
    else:
        metric_key = str(metric)

    target_raw = project_meta.get("kpi_target", project_meta.get("metric_target"))
    try:
        kpi_target = float(target_raw) if target_raw is not None else None
    except (TypeError, ValueError):
        kpi_target = None

    # Normalise back into metadata for downstream consumers
    project_meta.setdefault("eval_metric", metric_key)
    project_meta["kpi_metric"] = metric_key
    if kpi_target is not None:
        project_meta["kpi_target"] = kpi_target

    return metric_key, kpi_target


def _get_first_parquet_summary(project_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ctx = project_meta.get("ingested_context") or {}
    summaries = ctx.get("binary_summaries") or {}
    for _rel, summary in summaries.items():
        if summary.get("format") == "parquet":
            return summary
    return None


def _get_first_onnx_summary(project_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ctx = project_meta.get("ingested_context") or {}
    summaries = ctx.get("binary_summaries") or {}
    for _rel, summary in summaries.items():
        # Heuristic: ONNX summaries have "total_parameters" or "model_type_hint"
        if "total_parameters" in summary or summary.get("model_type_hint"):
            return summary
    return None


@dataclass
class UserStory:
    """Single PRD item (legacy internal representation).

    NOTE: External `prd.json` shape is controlled by PRDModel.
    This dataclass remains as a convenient builder API.
    """

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
        """Generate PRD from clarified task.

        Args:
            technical_brief: Output from TaskClarifier
            domain: Project domain
            project_metadata: Extra context from orchestrator (competition url, eval metric, etc.)
        """

        clarified_task = technical_brief.get("clarified_task", "")
        key_reqs = technical_brief.get("key_requirements", [])
        project_meta = project_metadata or technical_brief.get("project_metadata") or {}

        logger.info("📝 PRD Generator Starting...")
        logger.info(f"   Task: {clarified_task[:80]}...")
        logger.info(f"   Domain: {domain}")
        if project_meta:
            logger.info(f"   Project meta keys: {list(project_meta.keys())}")

        # Enhanced domain detection
        domain_lower = domain.lower()
        task_lower = clarified_task.lower()

        # ML Competition detection
        is_ml_competition = any(
            [
                "time_series" in domain_lower,
                "ml_competition" in domain_lower,
                "ml_model" in domain_lower,
                "forecasting" in task_lower,
                "kaggle" in task_lower,
                "wundernn" in task_lower,
                "competition" in task_lower
                and ("train" in task_lower or "model" in task_lower),
                "onnx" in task_lower,
                "pearson correlation" in task_lower,
                project_meta.get("project_type") == "ml_competition",
            ]
        )

        # Normalise KPI metadata early for ML competitions
        if is_ml_competition:
            _extract_kpi_info(project_meta)

        # Decompose based on domain
        if is_ml_competition:
            logger.info("   Detected: ML Competition")
            stories = self._decompose_ml_competition(
                task=clarified_task,
                reqs=key_reqs,
                project_meta=project_meta,
            )
        elif "api" in domain_lower or "backend" in domain_lower:
            logger.info("   Detected: API/Backend")
            stories = self._decompose_api(clarified_task, key_reqs)
        elif "llm" in domain_lower or "chatbot" in domain_lower:
            logger.info("   Detected: LLM App")
            stories = self._decompose_llm(clarified_task, key_reqs)
        elif "web" in domain_lower:
            logger.info("   Detected: Web App")
            stories = self._decompose_web(clarified_task, key_reqs)
        else:
            logger.warning("   Using generic decomposition")
            stories = self._decompose_generic(clarified_task, key_reqs)

        logger.info(f"✅ Generated {len(stories)} PRD items (pre-normalization)")

        # Build raw PRD dict using legacy shape
        raw_prd = {
            "task": clarified_task,
            "domain": domain,
            "total_items": len(stories),
            "user_stories": [s.to_dict() for s in stories],
            "project_metadata": project_meta,
        }

        # DR-03 / DR-05: normalize through PRDModel so that `prd.json` has
        # a single, canonical structure regardless of generator version.
        prd_normalized = normalize_prd(raw_prd)
        logger.info(
            "📐 PRD normalized via PRDModel: %d items", prd_normalized.get("total_items", 0)
        )

        return prd_normalized

    def _decompose_ml_competition(
        self,
        task: str,
        reqs: List[str],
        project_meta: Optional[Dict[str, Any]] = None,
    ) -> List[UserStory]:
        """ML Competition decomposition (10 atomic items).

        Injects metric name / target into evaluation stories if available and
        uses dataset / ONNX summaries from ingested context when present.
        """

        project_meta = project_meta or {}

        # KPI info
        metric_key, kpi_target = _extract_kpi_info(project_meta)
        metric_label = metric_key  # keep snake_case for now (matches tests)

        # Context summaries if available
        ds_summary = _get_first_parquet_summary(project_meta) or {}
        onnx_summary = _get_first_onnx_summary(project_meta) or {}

        row_count = ds_summary.get("row_count")
        col_count = ds_summary.get("column_count")
        total_params = onnx_summary.get("total_parameters")
        model_type_hint = onnx_summary.get("model_type_hint")

        if row_count is not None:
            train_shape_line = f"train.parquet row count verified: {row_count} sequences"
        else:
            train_shape_line = "train.parquet row count verified (non-empty)."

        if col_count is not None:
            train_cols_line = f"Feature count verified: {col_count} columns in train.parquet"
        else:
            train_cols_line = "Feature count verified for train.parquet."

        if kpi_target is not None:
            eval_kpi_line = (
                f"Achieves {metric_label}  {kpi_target:.2f} on validation set (primary KPI)."
            )
        else:
            eval_kpi_line = f"Reports {metric_label} on validation set as primary KPI."

        if total_params is not None:
            baseline_param_line = f"Baseline ONNX model parameters: {total_params} (from summariser)."
        else:
            baseline_param_line = "Baseline ONNX model parameters computed via summariser."

        if model_type_hint:
            baseline_type_line = f"Baseline architecture hint: {model_type_hint}."
        else:
            baseline_type_line = "Baseline architecture inspected from ONNX graph."

        stories = [
            # stories unchanged ...
        ]

        return stories

    # rest of file unchanged ...


def generate_prd(
    technical_brief: Dict[str, Any],
    domain: str = "llm-app",
    project_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to generate PRD (normalized)."""
    gen = PRDGenerator()
    return gen.generate(technical_brief, domain, project_metadata)
