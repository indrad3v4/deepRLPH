# -*- coding: utf-8 -*-
"""
prd_model.py - PRD Item Status Tracking Model

Core data model for tracking execution state of PRD items through entire lifecycle.

ITEM-001: PRD Item Status Tracking Model
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import threading
import logging

logger = logging.getLogger(__name__)


class PRDItemStatus(str, Enum):
    """Status of a PRD item in execution lifecycle."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    TESTING = "TESTING"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class PRDItem:
    """Single PRD item with tracking metadata.
    
    Attributes:
        item_id: Unique identifier (e.g., "ITEM-001")
        title: Short descriptive title
        priority: Priority level (1=highest)
        status: Current execution status
        acceptance_criteria: List of acceptance criteria strings
        verification_command: Shell command to verify completion
        files_touched: List of file paths expected to be modified
        start_time: ISO timestamp when item execution started
        end_time: ISO timestamp when item completed/failed
        attempt_count: Number of execution attempts
        error_log: Error messages from failed attempts
        agent_id: ID of agent currently/last handling this item
    """
    item_id: str
    title: str
    priority: int
    status: PRDItemStatus = PRDItemStatus.PENDING
    acceptance_criteria: List[str] = field(default_factory=list)
    verification_command: str = ""
    files_touched: List[str] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    attempt_count: int = 0
    error_log: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PRDItem":
        """Create from dict (JSON deserialization)."""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = PRDItemStatus(data['status'])
        return cls(**data)
    
    def mark_in_progress(self, agent_id: str) -> None:
        """Mark item as in progress."""
        if self.status != PRDItemStatus.PENDING:
            raise ValueError(
                f"Cannot transition from {self.status} to IN_PROGRESS. "
                f"Must be PENDING."
            )
        self.status = PRDItemStatus.IN_PROGRESS
        self.agent_id = agent_id
        self.start_time = datetime.now().isoformat()
        self.attempt_count += 1
        logger.info(f"[{self.item_id}] Started by {agent_id} (attempt {self.attempt_count})")
    
    def mark_testing(self) -> None:
        """Mark item as in testing phase."""
        if self.status != PRDItemStatus.IN_PROGRESS:
            raise ValueError(
                f"Cannot transition from {self.status} to TESTING. "
                f"Must be IN_PROGRESS."
            )
        self.status = PRDItemStatus.TESTING
        logger.info(f"[{self.item_id}] Running verification: {self.verification_command}")
    
    def mark_pass(self) -> None:
        """Mark item as passed."""
        if self.status not in (PRDItemStatus.IN_PROGRESS, PRDItemStatus.TESTING):
            raise ValueError(
                f"Cannot transition from {self.status} to PASS. "
                f"Must be IN_PROGRESS or TESTING."
            )
        self.status = PRDItemStatus.PASS
        self.end_time = datetime.now().isoformat()
        logger.info(f"✅ [{self.item_id}] PASS (duration: {self.get_duration():.1f}s)")
    
    def mark_fail(self, error: str) -> None:
        """Mark item as failed."""
        if self.status not in (PRDItemStatus.IN_PROGRESS, PRDItemStatus.TESTING):
            raise ValueError(
                f"Cannot transition from {self.status} to FAIL. "
                f"Must be IN_PROGRESS or TESTING."
            )
        self.status = PRDItemStatus.FAIL
        self.end_time = datetime.now().isoformat()
        self.error_log.append(f"[Attempt {self.attempt_count}] {error}")
        logger.error(f"❌ [{self.item_id}] FAIL: {error}")
    
    def reset_for_retry(self) -> None:
        """Reset status to PENDING for retry."""
        if self.status != PRDItemStatus.FAIL:
            raise ValueError(f"Can only retry FAIL items, not {self.status}")
        self.status = PRDItemStatus.PENDING
        self.start_time = None
        self.end_time = None
        logger.info(f"[{self.item_id}] Reset for retry (attempt {self.attempt_count + 1})")
    
    def get_duration(self) -> float:
        """Get execution duration in seconds."""
        if not self.start_time:
            return 0.0
        start = datetime.fromisoformat(self.start_time)
        if self.end_time:
            end = datetime.fromisoformat(self.end_time)
        else:
            end = datetime.now()
        return (end - start).total_seconds()


class PRDBacklog:
    """Manager for collection of PRD items with thread-safe operations.
    
    Attributes:
        items: List of all PRD items
        project_id: Associated project identifier
        created_at: ISO timestamp of backlog creation
        _lock: Thread lock for concurrent agent access
    """
    
    def __init__(self, project_id: str, items: Optional[List[PRDItem]] = None):
        self.project_id = project_id
        self.items: List[PRDItem] = items or []
        self.created_at = datetime.now().isoformat()
        self._lock = threading.Lock()
    
    def add_item(self, item: PRDItem) -> None:
        """Add item to backlog."""
        with self._lock:
            self.items.append(item)
            logger.info(f"Added {item.item_id} to backlog (total: {len(self.items)})")
    
    def get_next_pending(self) -> Optional[PRDItem]:
        """Get next pending item by priority (thread-safe)."""
        with self._lock:
            pending = [item for item in self.items if item.status == PRDItemStatus.PENDING]
            if not pending:
                return None
            # Sort by priority (1 = highest)
            pending.sort(key=lambda x: x.priority)
            return pending[0]
    
    def get_item_by_id(self, item_id: str) -> Optional[PRDItem]:
        """Get item by ID."""
        with self._lock:
            for item in self.items:
                if item.item_id == item_id:
                    return item
            return None
    
    def mark_in_progress(self, item_id: str, agent_id: str) -> bool:
        """Mark item as in progress (thread-safe)."""
        with self._lock:
            item = self.get_item_by_id(item_id)
            if not item:
                return False
            try:
                item.mark_in_progress(agent_id)
                return True
            except ValueError as e:
                logger.warning(f"Failed to mark {item_id} in progress: {e}")
                return False
    
    def mark_complete(self, item_id: str, success: bool, error: str = "") -> bool:
        """Mark item as complete (pass/fail)."""
        with self._lock:
            item = self.get_item_by_id(item_id)
            if not item:
                return False
            try:
                if success:
                    item.mark_pass()
                else:
                    item.mark_fail(error)
                return True
            except ValueError as e:
                logger.warning(f"Failed to mark {item_id} complete: {e}")
                return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            stats = {
                "total": len(self.items),
                "pending": sum(1 for i in self.items if i.status == PRDItemStatus.PENDING),
                "in_progress": sum(1 for i in self.items if i.status == PRDItemStatus.IN_PROGRESS),
                "testing": sum(1 for i in self.items if i.status == PRDItemStatus.TESTING),
                "pass": sum(1 for i in self.items if i.status == PRDItemStatus.PASS),
                "fail": sum(1 for i in self.items if i.status == PRDItemStatus.FAIL),
                "progress_pct": 0.0,
            }
            if stats["total"] > 0:
                stats["progress_pct"] = (stats["pass"] / stats["total"]) * 100
            return stats
    
    def has_pending_items(self) -> bool:
        """Check if any pending items remain."""
        with self._lock:
            return any(item.status == PRDItemStatus.PENDING for item in self.items)
    
    def get_failed_items(self) -> List[PRDItem]:
        """Get all failed items."""
        with self._lock:
            return [item for item in self.items if item.status == PRDItemStatus.FAIL]
    
    def save_to_file(self, path: Path) -> None:
        """Save backlog state to JSON file."""
        with self._lock:
            data = {
                "project_id": self.project_id,
                "created_at": self.created_at,
                "items": [item.to_dict() for item in self.items],
                "statistics": self.get_statistics(),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved backlog state to {path}")
    
    @classmethod
    def load_from_file(cls, path: Path) -> "PRDBacklog":
        """Load backlog state from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = [PRDItem.from_dict(item_data) for item_data in data['items']]
        backlog = cls(project_id=data['project_id'], items=items)
        backlog.created_at = data['created_at']
        logger.info(f"Loaded backlog from {path} ({len(items)} items)")
        return backlog
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        with self._lock:
            return {
                "project_id": self.project_id,
                "created_at": self.created_at,
                "items": [item.to_dict() for item in self.items],
                "statistics": self.get_statistics(),
            }
