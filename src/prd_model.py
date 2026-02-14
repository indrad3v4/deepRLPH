# -*- coding: utf-8 -*-
"""
prd_model.py - Canonical PRD data model and normalizer (DR-03, DR-05)

Unifies multiple PRD shapes into a single, validated JSON structure:
- Existing PRDGenerator output with `user_stories`
- Wizard-style backlog with `backlog` + `item_id` + `verification_command`

This becomes the single source of truth for `prd.json` files.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PRDUserStory(BaseModel):
    """Canonical representation of a single PRD item.

    The goal is to be tolerant to upstream variants while keeping
    downstream execution strict and predictable.
    """

    id: str = Field(..., description="Stable identifier like ML-001 or ITEM-001")
    title: str = Field(..., description="Short human-readable title")
    description: str = Field("", description="Longer functional description / why")
    acceptance_criteria: List[str] = Field(default_factory=list)
    verification: str = Field("", description="CLI command or manual check description")
    why: str = Field("", description="Business or technical motivation")
    files_touched: List[str] = Field(default_factory=list)

    # Execution-related fields (used by Execution Engine)
    status: str = Field("todo", description="todo | in_progress | done | blocked")
    attempts: int = 0
    errors: List[str] = Field(default_factory=list)

    @validator("status", pre=True, always=True)
    def _normalize_status(cls, value: str) -> str:  # type: ignore[override]
        if not value:
            return "todo"
        v = str(value).lower()
        allowed = {"todo", "in_progress", "in-progress", "done", "blocked"}
        if v not in allowed:
            return "todo"
        # normalize dash vs underscore
        return v.replace("-", "_")

    def to_dict(self) -> Dict[str, Any]:
        """Return plain dict, keeping only canonical fields.

        Using `dict()` instead of `asdict` to leverage Pydantic's serialization.
        """

        return self.dict()


class PRDModel(BaseModel):
    """Canonical PRD representation used for all `prd.json` files.

    This model intentionally mirrors the existing PRDGenerator output
    but enforces consistency and provides a migration path from
    wizard-style backlogs.
    """

    task: str = Field("", description="Clarified task or project brief")
    domain: str = Field("llm-app", description="Project domain, e.g. llm-app, backend_api, time_series")
    total_items: int = 0
    user_stories: List[PRDUserStory] = Field(default_factory=list)
    project_metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("total_items", always=True)
    def _sync_total_items(cls, value: int, values: Dict[str, Any]) -> int:  # type: ignore[override]
        """Ensure `total_items` always matches the number of stories."""

        stories = values.get("user_stories") or []
        return len(stories)

    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> "PRDModel":
        """Create PRDModel from a loose dict.

        Supported inputs:
        1) Existing PRDGenerator output with keys:
           {task, domain, total_items, user_stories, project_metadata}
        2) Wizard-style backlog with keys:
           {task?, domain?, backlog: [...], execution_plan?, definition_of_done?, project_metadata?/metadata?}
        """

        if "user_stories" in data:
            # Assume near-canonical shape; Pydantic will validate details.
            return cls(**data)

        # Fallback: backlog-style structure from wizard (Phase 2B)
        backlog = data.get("backlog") or []
        task = data.get("task") or data.get("description") or ""
        domain = data.get("domain", "llm-app")
        meta = data.get("project_metadata") or data.get("metadata") or {}

        stories: List[PRDUserStory] = []
        for idx, item in enumerate(backlog, start=1):
            # Prefer explicit IDs but be tolerant.
            sid = (
                item.get("item_id")
                or item.get("id")
                or f"ITEM-{idx:03d}"
            )
            verification = (
                item.get("verification_command")
                or item.get("verification")
                or ""
            )
            files = (
                item.get("files_touched")
                or item.get("files")
                or []
            )

            story = PRDUserStory(
                id=str(sid),
                title=item.get("title", "Untitled"),
                description=item.get("description") or item.get("why", ""),
                acceptance_criteria=item.get("acceptance_criteria") or [],
                verification=verification,
                why=item.get("why") or "",
                files_touched=list(files),
                status=item.get("status", "todo"),
                attempts=int(item.get("attempts", 0) or 0),
                errors=item.get("errors") or [],
            )
            stories.append(story)

        return cls(
            task=task,
            domain=domain,
            total_items=len(stories),
            user_stories=stories,
            project_metadata=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return normalized dict suitable for `prd.json` on disk."""

        # We intentionally do not include any helper/alias fields here.
        return {
            "task": self.task,
            "domain": self.domain,
            "total_items": len(self.user_stories),
            "user_stories": [s.to_dict() for s in self.user_stories],
            "project_metadata": self.project_metadata,
        }


def normalize_prd(prd: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an arbitrary PRD-like dict into canonical shape.

    Used by both backend (orchestrator / PRDGenerator) and future UI flows
    to guarantee `prd.json` stays structurally consistent across versions.
    """

    model = PRDModel.from_raw(prd)
    return model.to_dict()
