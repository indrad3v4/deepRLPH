# -*- coding: utf-8 -*-
"""
prompt_generator.py - Meta-prompting helpers for Project Refinement (Phase 2)

Existing responsibilities:
- Phase 2B: Expand AI config → PRD backlog (wizard)

New in Stage 2:
- add `build_project_refinement_brief` that turns
  wizard config + ingested context into a single
  "Project Refinement" brief for orchestrator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ProjectRefinementBrief:
    """High-level brief for Project Refinement instead of raw task text."""

    project_name: str
    description: str
    domain: str
    architecture: str
    framework: str
    kpi_metric: str
    kpi_target: float
    prd_backlog: Dict[str, Any]
    project_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "description": self.description,
            "domain": self.domain,
            "architecture": self.architecture,
            "framework": self.framework,
            "kpi_metric": self.kpi_metric,
            "kpi_target": self.kpi_target,
            "prd_backlog": self.prd_backlog,
            "project_metadata": self.project_metadata,
        }


class PromptGenerator:
    """Existing PromptGenerator with new helper for Project Refinement.

    NOTE: Only the new method is defined here, to be merged with
    the existing implementation in the repo.
    """

    def __init__(self, deepseek_client):
        self.deepseek_client = deepseek_client

    async def expand_to_prd_backlog(self, config: Dict[str, Any], project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Existing Phase 2B entrypoint (placeholder for context).

        In the real file this is already implemented; we keep the
        signature here so imports stay consistent.
        """

        raise NotImplementedError

    def build_project_refinement_brief(self, project_data: Dict[str, Any]) -> ProjectRefinementBrief:
        """Build a structured brief for Project Refinement.

        This replaces the loose "task" string from the Task Refinement
        tab with a project-scoped description and the PRD backlog.
        """

        cfg = project_data.get("final_config", {})
        prd_backlog = project_data.get("prd_backlog") or {}

        # Everything the agents need to understand the project scope.
        metadata = {
            "project_type": cfg.get("project_type"),
            "tags": (cfg.get("tags") or "").split(",") if cfg.get("tags") else [],
            "doc_files": project_data.get("doc_files", []),
            "dataset_files": project_data.get("dataset_files", []),
            "baseline_files": project_data.get("baseline_files", []),
        }

        return ProjectRefinementBrief(
            project_name=cfg.get("name", project_data.get("name", "Unnamed Project")),
            description=cfg.get("description", project_data.get("description", "")),
            domain=cfg.get("domain", "llm-app"),
            architecture=cfg.get("architecture", "clean_architecture"),
            framework=cfg.get("framework", "FastAPI"),
            kpi_metric=str(cfg.get("kpi_metric", "tests_passed_ratio")),
            kpi_target=float(cfg.get("kpi_target", 1.0)),
            prd_backlog=prd_backlog,
            project_metadata=metadata,
        )
