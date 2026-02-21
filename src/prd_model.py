# -*- coding: utf-8 -*-
"""
prd_model.py - Canonical PRD data model and PRD execution backlog

Unifies multiple PRD shapes into a single, validated JSON structure AND
provides an in-memory execution backlog model used by the orchestrator
loop and tests (PRDItem / PRDBacklog).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ============================================================================
# CANONICAL PRD (Pydantic) - used for prd.json normalisation
# ============================================================================


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

        Using `dict()` instead of `model_dump` to stay compatible with tests
        that call `dict()` on the Pydantic model (Pydantic v2 still supports
        this method but emits a deprecation warning).
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


# ============================================================================
# EXECUTION BACKLOG (dataclasses) - used heavily by tests & agents
# ============================================================================


class PRDItemStatus(str, Enum):
    """Status for a PRD backlog item.

    Mirrors the semantics used in tests: PENDING → IN_PROGRESS → TESTING →
    PASS / FAIL, with ability to reset for retry.
    """

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    TESTING = "TESTING"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class PRDItem:
    """Single executable PRD backlog item (ITEM-001, etc.).

    This is intentionally a lightweight dataclass so it can be used from
    non-Pydantic code (agent coordinator, execution engine, tests).
    """

    item_id: str
    title: str
    priority: int = 1

    acceptance_criteria: List[str] = field(default_factory=list)
    verification_command: str = ""
    files_touched: List[str] = field(default_factory=list)

    status: PRDItemStatus = PRDItemStatus.PENDING
    agent_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    attempt_count: int = 0
    error_log: List[str] = field(default_factory=list)

    def mark_in_progress(self, agent_id: str) -> None:
        """Transition PENDING → IN_PROGRESS and bump attempt counter."""

        if self.status not in {PRDItemStatus.PENDING, PRDItemStatus.FAIL}:
            raise ValueError(f"Cannot transition from {self.status} to IN_PROGRESS")
        import time

        self.status = PRDItemStatus.IN_PROGRESS
        self.agent_id = agent_id
        self.start_time = time.time()
        self.end_time = None
        self.attempt_count += 1

    def mark_testing(self) -> None:
        """Transition IN_PROGRESS → TESTING."""

        if self.status is not PRDItemStatus.IN_PROGRESS:
            raise ValueError("Cannot transition to TESTING from non IN_PROGRESS state")
        self.status = PRDItemStatus.TESTING

    def mark_pass(self) -> None:
        """Transition TESTING/IN_PROGRESS → PASS and freeze end_time."""

        if self.status not in {PRDItemStatus.TESTING, PRDItemStatus.IN_PROGRESS}:
            raise ValueError("Cannot mark PASS from current state")
        import time

        self.status = PRDItemStatus.PASS
        self.end_time = time.time()

    def mark_fail(self, error: str) -> None:
        """Transition IN_PROGRESS/TESTING → FAIL and record error."""

        if self.status not in {PRDItemStatus.TESTING, PRDItemStatus.IN_PROGRESS}:
            raise ValueError("Cannot mark FAIL from current state")
        import time

        self.status = PRDItemStatus.FAIL
        self.end_time = time.time()
        self.error_log.append(error)

    def reset_for_retry(self) -> None:
        """Reset a failed item back to PENDING, preserving attempts + errors."""

        if self.status is not PRDItemStatus.FAIL:
            raise ValueError("Can only reset items that are in FAIL state")
        self.status = PRDItemStatus.PENDING
        self.agent_id = None
        self.start_time = None
        self.end_time = None

    def get_duration(self) -> float:
        """Return wall-clock duration in seconds (0 if not finished)."""

        import time

        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return max(0.0, end - self.start_time)

    # Serialization helpers used in tests
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "priority": self.priority,
            "acceptance_criteria": list(self.acceptance_criteria),
            "verification_command": self.verification_command,
            "files_touched": list(self.files_touched),
            "status": self.status.value,
            "agent_id": self.agent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attempt_count": self.attempt_count,
            "error_log": list(self.error_log),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PRDItem":
        return cls(
            item_id=data["item_id"],
            title=data.get("title", ""),
            priority=data.get("priority", 1),
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            verification_command=data.get("verification_command", ""),
            files_touched=list(data.get("files_touched", [])),
            status=PRDItemStatus(data.get("status", "PENDING")),
            agent_id=data.get("agent_id"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            attempt_count=int(data.get("attempt_count", 0)),
            error_log=list(data.get("error_log", [])),
        )


@dataclass
class PRDBacklog:
    """Thread-safe manager around a list of PRDItem objects.

    The tests exercise priority ordering, status transitions and
    save/load functionality. We keep implementation minimal but
    correct for those behaviours.
    """

    project_id: str
    items: List[PRDItem] = field(default_factory=list)
    created_at: str = ""

    # internal lock for thread-safety
    _lock: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        from datetime import datetime
        from threading import RLock

        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self._lock is None:
            self._lock = RLock()

    # basic operations -----------------------------------------------------

    def add_item(self, item: PRDItem) -> None:
        with self._lock:
            self.items.append(item)

    def get_next_pending(self) -> Optional[PRDItem]:
        """Return highest-priority PENDING item (smaller number = higher).

        If multiple items have the same priority, preserve insertion order.
        """

        with self._lock:
            pending = [it for it in self.items if it.status == PRDItemStatus.PENDING]
            if not pending:
                return None
            # min by priority; list.index for stable tie‑break
            best = min(pending, key=lambda it: (it.priority, self.items.index(it)))
            return best

    def has_pending_items(self) -> bool:
        with self._lock:
            return any(it.status == PRDItemStatus.PENDING for it in self.items)

    def mark_in_progress(self, item_id: str, agent_id: str) -> bool:
        with self._lock:
            for it in self.items:
                if it.item_id == item_id:
                    it.mark_in_progress(agent_id)
                    return True
        return False

    def mark_complete(self, item_id: str, success: bool, error: str = "") -> bool:
        with self._lock:
            for it in self.items:
                if it.item_id == item_id:
                    if success:
                        if it.status != PRDItemStatus.PASS:
                            it.mark_pass()
                    else:
                        it.mark_fail(error or "Unknown error")
                    return True
        return False

    def get_failed_items(self) -> List[PRDItem]:
        with self._lock:
            return [it for it in self.items if it.status == PRDItemStatus.FAIL]

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self.items)
            cnt_pass = sum(1 for it in self.items if it.status == PRDItemStatus.PASS)
            cnt_fail = sum(1 for it in self.items if it.status == PRDItemStatus.FAIL)
            cnt_pending = sum(1 for it in self.items if it.status == PRDItemStatus.PENDING)
            progress_pct = (cnt_pass / total * 100.0) if total else 0.0
            return {
                "total": total,
                "pass": cnt_pass,
                "fail": cnt_fail,
                "pending": cnt_pending,
                "progress_pct": progress_pct,
            }

    # persistence ----------------------------------------------------------

    def save_to_file(self, path: Any) -> None:
        """Save backlog state to JSON file."""

        import json
        from pathlib import Path

        p = Path(path)
        data = {
            "project_id": self.project_id,
            "created_at": self.created_at,
            "items": [it.to_dict() for it in self.items],
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_from_file(cls, path: Any) -> "PRDBacklog":
        """Load backlog state from JSON file."""

        import json
        from pathlib import Path

        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        items = [PRDItem.from_dict(d) for d in raw.get("items", [])]
        return cls(project_id=raw.get("project_id", "unknown"), items=items, created_at=raw.get("created_at", ""))
