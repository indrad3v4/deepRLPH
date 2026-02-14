# -*- coding: utf-8 -*-
"""Package init for deepRLPH core modules.

Stage 4 (observability & UX machine):
- exposes `open_observability_dashboard` at package level for easy import
  from external tools / RalphUI.

Stage 5 (templates & UX polish):
- exposes project template helpers so UIs can offer one-click presets.
"""

from pathlib import Path
from typing import Any

from .observability_dashboard import open_observability_dashboard  # noqa: F401
from .project_templates import (  # noqa: F401
    list_project_templates,
    get_project_template,
    apply_template_to_project_data,
)


__all__ = [
    "open_observability_dashboard",
    "list_project_templates",
    "get_project_template",
    "apply_template_to_project_data",
]


def get_project_dir_from_orchestrator(orchestrator: Any) -> Path:
    """Best-effort helper to get current project directory from orchestrator.

    This avoids tight coupling: orchestrator only needs to expose
    `current_project_dir` attribute (as already used in RalphUI).
    """

    proj_dir = getattr(orchestrator, "current_project_dir", None)
    if proj_dir is None:
        raise ValueError("Orchestrator has no current_project_dir set")
    return Path(proj_dir)
