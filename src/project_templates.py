# -*- coding: utf-8 -*-
"""
project_templates.py - Stage 5 templates + UX-ready descriptors for deepRLPH

Goals (Этап 5 — Шаблоны и UX‑полировка):
- Central registry of opinionated project templates (API, ML competition, etc.).
- UX-level descriptions that the wizard/UI can show as "cards" or presets.
- TRIZ: separate stable template knowledge from per-run user data, so the
  wizard only applies templates instead of re-encoding patterns.

This module is deliberately UI-agnostic: it returns plain dicts so that
Tkinter, CLI, or future web UI can consume the same templates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class ProjectTemplate:
    """Single high-level template for a deepRLPH project.

    Fields are intentionally aligned with ProjectWizard.project_data and
    ProjectConfig so the wiring is trivial:
    - project_type: "api" or "ml" (wizard field)
    - name_suffix: suggested suffix for project name
    - description: pre-filled description text
    - tags: default tags
    - defaults: extra config hints (domain, framework, kpi_metric, etc.)
    """

    template_id: str
    label: str
    short_label: str
    description: str
    project_type: str  # "api" | "ml"
    tags: List[str]
    defaults: Dict[str, Any]

    def to_public_dict(self) -> Dict[str, Any]:
        """Return data that is safe to show directly in UX."""

        data = asdict(self)
        # Do not expose anything heavyweight here; defaults are fine.
        return data


# ---------------- registry ----------------

TEMPLATES: Dict[str, ProjectTemplate] = {}


def _register(template: ProjectTemplate) -> None:
    TEMPLATES[template.template_id] = template


# API / backend archetype
_register(
    ProjectTemplate(
        template_id="api-fastapi-service",
        label="Backend API (FastAPI)",
        short_label="FastAPI service",
        description=(
            "Opinionated template for a backend API service built with FastAPI, "
            "PostgreSQL, tests and linting. Good default for typical product APIs."
        ),
        project_type="api",
        tags=["api", "backend", "fastapi", "postgresql"],
        defaults={
            "domain": "backend_api",
            "architecture": "clean_architecture",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "kpi_metric": "tests_passed_ratio",
            "kpi_target": 1.0,
        },
    )
)


# ML competition / time series archetype (e.g. Wundernn, Kaggle)
_register(
    ProjectTemplate(
        template_id="ml-timeseries-competition",
        label="ML Competition (Time Series)",
        short_label="ML time series",
        description=(
            "Template for ML competitions with time series / tabular signals. "
            "Uses PyTorch, RNN/Transformer-style models, and a regression metric."
        ),
        project_type="ml",
        tags=["ml", "competition", "timeseries", "wundernn"],
        defaults={
            "domain": "time_series_forecasting",
            "ml_framework": "PyTorch",
            "model_type": "Transformer",
            "architecture": "ml_pipeline",
            "framework": "PyTorch",
            "database": "None",
            "kpi_metric": "weighted_pearson",
            "kpi_target": 0.85,
            "batch_size": 64,
            "epochs": 80,
            "learning_rate": 0.0005,
        },
    )
)


# LLM application archetype
_register(
    ProjectTemplate(
        template_id="llm-app-orchestrator",
        label="LLM App / Orchestrator",
        short_label="LLM app",
        description=(
            "Template for LLM-centric applications (tools, agents, orchestrators). "
            "Targets high test coverage and robust prompt/agent wiring."
        ),
        project_type="api",
        tags=["llm", "agents", "orchestrator"],
        defaults={
            "domain": "llm-app",
            "architecture": "clean_architecture",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "kpi_metric": "tests_passed_ratio",
            "kpi_target": 1.0,
        },
    )
)


# ---------------- public API ----------------


def list_project_templates() -> List[Dict[str, Any]]:
    """Return all registered templates as UX-friendly dicts.

    Intended usage in UI:
    - show cards with label/short_label/description/tags
    - when user clicks, call apply_template_to_project_data(...)
    """

    return [tpl.to_public_dict() for tpl in TEMPLATES.values()]


def get_project_template(template_id: str) -> Dict[str, Any]:
    """Get a single template by id.

    Raises KeyError if not found so callers can surface a clear UX error.
    """

    return TEMPLATES[template_id].to_public_dict()


def apply_template_to_project_data(template_id: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply template defaults onto a ProjectWizard.project_data-like dict.

    This is a pure function (does not mutate input) so it is safe to call
    from UI code when the user switches templates.
    """

    tpl = TEMPLATES[template_id]

    # Start from a shallow copy of project_data
    merged: Dict[str, Any] = dict(project_data or {})

    # Core wizard fields
    merged.setdefault("name", merged.get("name", ""))
    merged.setdefault("description", "")
    merged.setdefault("tags", ",".join(tpl.tags))
    merged["project_type"] = tpl.project_type

    # Inject defaults into a dedicated sub-dict so consumers can wire them
    # into final_config / metadata as needed.
    defaults = dict(tpl.defaults)
    merged.setdefault("template_id", tpl.template_id)
    merged["template_defaults"] = defaults

    # Keep existing AI suggestions / PRD backlog if already computed.
    return merged
