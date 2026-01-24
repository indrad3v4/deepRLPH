# -*- coding: utf-8 -*-
"""
RALPH - Multi-Agent Development System
Orchestrated by AI agents for autonomous software development
"""

__version__ = "2.0"
__author__ = "indradev_"

from .orchestrator import (
    RalphOrchestrator,
    ProjectConfig,
    ProjectDomain,
    ArchitectureType,
    get_orchestrator,
)

__all__ = [
    "RalphOrchestrator",
    "ProjectConfig",
    "ProjectDomain",
    "ArchitectureType",
    "get_orchestrator",
]