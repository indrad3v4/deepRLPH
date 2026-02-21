# -*- coding: utf-8 -*-
"""
agent_profiles.py - Persistent agent profile store for Ralph-style multi-agent loop.

Responsibility:
- Define AgentProfile (role/expertise/KPI focus for one DeepSeek agent).
- Persist profiles to agents.json (machine-readable) and AGENTS.md (LLM/human-readable).
- Provide simple APIs to load/save/update profiles per project.

This is intentionally generic; DeepSeek-specific initialization and
KPI-based evolution are orchestrated from RalphOrchestrator / ExecutionEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import textwrap
import logging

logger = logging.getLogger("AgentProfiles")


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


@dataclass
class AgentProfile:
    """Stable description of one Ralph agent.

    This is the long-lived "personality" + focus of a worker that can be
    reused across executions. It is designed to be serializable to JSON and
    easily embedded into LLM system prompts.
    """

    id: str
    name: str
    role: str
    expertise: List[str] = field(default_factory=list)
    target_files: List[str] = field(default_factory=list)
    metric_focus: Optional[str] = None
    kpi_history: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentProfile":
        return cls(**data)

    def to_prompt_blob(self) -> str:
        """Render this profile as a compact text block for system prompts.

        This is used by ExecutionEngine to specialize DeepSeek calls per agent.
        """
        expertise_str = ", ".join(self.expertise) if self.expertise else "generalist"
        files_str = ", ".join(self.target_files) if self.target_files else "any project files"
        notes_str = "\n".join(f"- {n}" for n in self.notes) if self.notes else "(no special notes yet)"

        metric_line = (
            f"Primary KPI focus: {self.metric_focus}." if self.metric_focus else "Primary KPI focus: project-level success."
        )

        return textwrap.dedent(
            f"""
            [AGENT PROFILE]
            ID: {self.id}
            Name: {self.name}
            Role: {self.role}
            Expertise: {expertise_str}
            Typical files: {files_str}
            {metric_line}

            Historical learnings:
            {notes_str}
            """
        ).strip()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AgentProfileStore:
    """Per-project store for agent profiles.

    Files:
    - <project_dir>/agents.json : canonical structured representation
    - <project_dir>/AGENTS.md   : human/LLM-facing summary, Ralph-style
    """

    def __init__(self, project_dir: Path, agents: Optional[Dict[str, AgentProfile]] = None) -> None:
        self.project_dir = Path(project_dir)
        self.agents: Dict[str, AgentProfile] = agents or {}

        self._json_path = self.project_dir / "agents.json"
        self._md_path = self.project_dir / "AGENTS.md"

    # ------------------------
    # Loading / saving
    # ------------------------

    @classmethod
    def load_for_project(cls, project_dir: Path) -> "AgentProfileStore":
        """Load existing profiles for a project or create an empty store.

        This method is safe to call even if files do not yet exist.
        """
        project_dir = Path(project_dir)
        store = cls(project_dir)

        # Load JSON if present
        if store._json_path.exists():
            try:
                with open(store._json_path, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                for agent_id, data in raw.get("agents", {}).items():
                    try:
                        store.agents[agent_id] = AgentProfile.from_dict(data)
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning("Failed to load agent profile %s: %s", agent_id, e)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Error reading agents.json: %s", e)

        # Ensure there is at least a minimal AGENTS.md template
        if not store._md_path.exists():
            store._write_markdown()
        else:
            # No parsing for now; AGENTS.md is append-only human/LLM context.
            logger.debug("AGENTS.md already exists at %s", store._md_path)

        return store

    def save(self) -> None:
        """Persist current profiles to agents.json and regenerate AGENTS.md."""
        payload = {"agents": {aid: a.to_dict() for aid, a in self.agents.items()}}
        try:
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("💾 Saved agent profiles to %s", self._json_path)
        except Exception as e:  # pragma: no cover - filesystem issues
            logger.error("Error writing agents.json: %s", e)

        self._write_markdown()

    # ------------------------
    # CRUD-style helpers
    # ------------------------

    def upsert_agent(self, profile: AgentProfile) -> None:
        """Create or replace an agent profile in the store."""
        self.agents[profile.id] = profile

    def get(self, agent_id: str) -> Optional[AgentProfile]:
        return self.agents.get(agent_id)

    def all(self) -> List[AgentProfile]:
        return list(self.agents.values())

    def record_learning(
        self,
        agent_id: str,
        summary: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a short learning/KPI observation for the agent.

        This is intended to be called by ExecutionEngine after runs, using
        metrics from metrics_history.json / execution summary.
        """
        metrics = metrics or {}
        profile = self.agents.get(agent_id)
        if not profile:
            logger.warning("record_learning called for unknown agent_id=%s", agent_id)
            return

        profile.kpi_history.append({"summary": summary, "metrics": metrics})
        profile.notes.append(summary)

    # ------------------------
    # Internal helpers
    # ------------------------

    def _write_markdown(self) -> None:
        """Regenerate AGENTS.md from current profiles.

        The format is intentionally simple and Ralph-friendly: it can be fed
        directly into an LLM as context about how different agents behave.
        """
        lines: List[str] = []

        lines.append("# AGENTS.md - Ralph Agent Profiles")
        lines.append("")
        lines.append("This file stores long-lived knowledge about specialized agents for this project.")
        lines.append("Each agent is a small, focused worker in the Ralph loop.")
        lines.append("")

        if not self.agents:
            lines.append("## Agents")
            lines.append("")
            lines.append("No agents have been defined yet. They will appear here once initialized.")
        else:
            lines.append("## Agents")
            lines.append("")
            for agent in self.all():
                lines.append(f"### {agent.name} ({agent.id})")
                lines.append("")
                lines.append(f"**Role**: {agent.role}")
                if agent.expertise:
                    lines.append(f"**Expertise**: {', '.join(agent.expertise)}")
                if agent.target_files:
                    lines.append(f"**Typical files**: {', '.join(agent.target_files)}")
                if agent.metric_focus:
                    lines.append(f"**Primary KPI**: {agent.metric_focus}")
                if agent.notes:
                    lines.append("")
                    lines.append("**Recent learnings:**")
                    for note in agent.notes[-10:]:  # keep last 10 for brevity
                        lines.append(f"- {note}")
                lines.append("")

        try:
            with open(self._md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info("💾 Wrote AGENTS.md to %s", self._md_path)
        except Exception as e:  # pragma: no cover - filesystem issues
            logger.error("Error writing AGENTS.md: %s", e)
