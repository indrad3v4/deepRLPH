# -*- coding: utf-8 -*-
"""
ui_refinement_switch.py - Wire Project Refinement as primary flow (Stage 2)

This file shows how RalphUI should evolve from a "Task Refinement"-centric
UI to a "Project Refinement"-centric one WITHOUT deleting the old tab.

Key ideas (TRIZ):
- Do not remove the old Task Refinement; encapsulate the new behavior
  as a separate view and switch default focus.
- Reuse existing PRD preview and execution flow, but feed it from
  project-level PRD/backlog generated in the wizard.

NOTE: This is provided as a patch-style example. Integrate these
changes into `src/ui/setup_window.py` manually if needed.
"""

from __future__ import annotations

from typing import Dict, Any

from refinement_modes import RefinementMode, DEFAULT_REFINEMENT_CONFIG


def get_default_refinement_tab_index() -> int:
    """Return the index of the tab that should be selected by default.

    0 = Welcome, 1 = Projects, 2 = Task Refinement, 3 = Execution, 4 = Validation, 5 = Logs

    Stage 2: we want Project Refinement to live conceptually between
    Projects and Execution, so it will reuse the old "Task Refinement"
    slot but change the label and behavior.
    """

    if DEFAULT_REFINEMENT_CONFIG.default_mode is RefinementMode.PROJECT:
        # Keep physical index but semantics are "Project Refinement" now.
        return 2
    return 2


def build_project_refinement_placeholder_text(project: Dict[str, Any]) -> str:
    """Helper to show a project-scoped refinement hint in the text box."""

    if not project:
        return "Select a project on the Projects tab, then refine its PRD backlog here."

    name = project.get("name", "Unnamed Project")
    domain = project.get("domain", "")
    framework = project.get("framework", "")

    return (
        f"Project Refinement for: {name}\n"
        f"Domain: {domain} | Framework: {framework}\n\n"
        "Describe high-level changes you want to make to the PRD/backlog, "
        "or leave empty to use the existing PRD from the Project Wizard."
    )
