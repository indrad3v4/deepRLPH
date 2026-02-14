# -*- coding: utf-8 -*-
"""
refinement_modes.py - Project Refinement vs Task Refinement (TRIZ switch)

Stage 2 goal:
- Move refinement focus from single ad-hoc "task" text box
  to project-scoped refinement using the wizard config + PRD backlog.

This module encodes the refinement mode and routing logic, so
UI and orchestrator can favor Project Refinement without breaking
existing Task Refinement entrypoint.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Literal


class RefinementMode(str, Enum):
    PROJECT = "project"
    TASK = "task"


@dataclass
class RefinementConfig:
    """Current refinement strategy.

    Used by UI + orchestrator to decide which refinement flow
    to prioritize / display as default.
    """

    default_mode: RefinementMode = RefinementMode.PROJECT
    # In the future we can add flags like
    # allow_task_override: bool = True


# Global default config for now – later this can be read from
# a config.json / user settings file.
DEFAULT_REFINEMENT_CONFIG = RefinementConfig()
