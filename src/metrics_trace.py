# -*- coding: utf-8 -*-
"""
metrics_trace.py - Unified metrics + JSONL trace logger for Ralph loop (Stage 3)

Goals for Stage 3 (Executable Loop & Metrics):
- Single place to write JSONL trace of all execution events.
- Single place to accumulate and evaluate KPI metrics per project.
- TRIZ: remove duplication between per-module logging by elevating
  metrics/trace to a small dedicated "meta-system".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class ExecutionTraceEvent:
    """Single JSONL trace event.

    Type examples:
    - execution_start / execution_end
    - agent_start / agent_finish
    - item_start / item_complete / item_fail
    - metric_update
    """

    type: str
    timestamp: str
    data: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class MetricsTraceLogger:
    """Write execution events as JSONL + keep in-memory KPI snapshot."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.trace_file = self.project_dir / "execution_trace.jsonl"
        self._kpi_snapshot: Dict[str, Any] = {}

        self.trace_file.parent.mkdir(parents=True, exist_ok=True)

    # --------------- low-level helpers ---------------

    def _write_event(self, event_type: str, **data: Any) -> None:
        evt = ExecutionTraceEvent(
            type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
        )
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(evt.to_json() + "\n")
        except Exception:
            # Trace must never crash the main loop; fail silently.
            return

    # --------------- high-level API used by ExecutionEngine ---------------

    def execution_start(self, execution_id: str, total_items: int) -> None:
        self._write_event("execution_start", execution_id=execution_id, total_items=total_items)

    def execution_end(self, execution_id: str, status: str, completed: int, failed: int) -> None:
        self._write_event(
            "execution_end",
            execution_id=execution_id,
            status=status,
            completed=completed,
            failed=failed,
        )

    def agent_start(self, agent_id: str, assigned_items: int) -> None:
        self._write_event("agent_start", agent_id=agent_id, assigned_items=assigned_items)

    def agent_finish(self, agent_id: str, completed_items: int, failed_items: int, duration_seconds: float) -> None:
        self._write_event(
            "agent_finish",
            agent_id=agent_id,
            completed_items=completed_items,
            failed_items=failed_items,
            duration_seconds=duration_seconds,
        )

    def item_start(self, agent_id: str, story_id: str, title: str) -> None:
        self._write_event(
            "item_start",
            agent_id=agent_id,
            story_id=story_id,
            title=title,
        )

    def item_complete(self, agent_id: str, story_id: str, duration_seconds: float, files_created: List[str]) -> None:
        self._write_event(
            "item_complete",
            agent_id=agent_id,
            story_id=story_id,
            duration_seconds=duration_seconds,
            files_created=files_created,
        )

    def item_fail(self, agent_id: str, story_id: str, error_message: str, attempt: int) -> None:
        self._write_event(
            "item_fail",
            agent_id=agent_id,
            story_id=story_id,
            error_message=error_message,
            attempt=attempt,
        )

    def metric_update(self, agent_id: str, metric_name: str, value: float, target: float) -> None:
        """Record KPI change and keep last value in-memory for quick access."""

        self._kpi_snapshot[metric_name] = {
            "value": value,
            "target": target,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._write_event(
            "metric_update",
            agent_id=agent_id,
            metric=metric_name,
            value=value,
            target=target,
        )

    # --------------- KPI accessors ---------------

    def get_latest_kpi(self, metric_name: str) -> Optional[Dict[str, Any]]:
        return self._kpi_snapshot.get(metric_name)

