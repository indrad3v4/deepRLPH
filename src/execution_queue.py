# -*- coding: utf-8 -*-
"""
execution_queue.py - PRD Items to Execution Queue

ITEM-016: Convert PRD items to execution queue for agent processing.
"""

from collections import deque
from typing import Optional, Deque
import logging
import threading

from src.prd_model import PRDBacklog, PRDItem, PRDItemStatus

logger = logging.getLogger(__name__)


class ExecutionQueue:
    """Thread-safe queue for PRD item execution."""
    
    def __init__(self, backlog: PRDBacklog):
        self.backlog = backlog
        self._queue: Deque[PRDItem] = deque()
        self._lock = threading.Lock()
        self._current_item: Optional[PRDItem] = None
    
    def populate_from_backlog(self) -> None:
        """Populate queue with pending items from backlog."""
        with self._lock:
            pending_items = [
                item for item in self.backlog.items
                if item.status == PRDItemStatus.PENDING
            ]
            # Sort by priority (1 = highest)
            pending_items.sort(key=lambda x: x.priority)
            
            self._queue.clear()
            self._queue.extend(pending_items)
            
            logger.info(f"Populated queue with {len(self._queue)} pending items")
    
    def get_next(self) -> Optional[PRDItem]:
        """Get next item from queue."""
        with self._lock:
            if self._queue:
                item = self._queue.popleft()
                self._current_item = item
                logger.info(f"Dequeued {item.item_id} (queue size: {len(self._queue)})")
                return item
            return None
    
    def peek_next(self) -> Optional[PRDItem]:
        """Peek at next item without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None
    
    def get_current(self) -> Optional[PRDItem]:
        """Get currently executing item."""
        with self._lock:
            return self._current_item
    
    def mark_current_complete(self, success: bool, error: str = "") -> bool:
        """Mark current item as complete and clear it."""
        with self._lock:
            if not self._current_item:
                return False
            
            item_id = self._current_item.item_id
            success = self.backlog.mark_complete(item_id, success, error)
            self._current_item = None
            return success
    
    def reprioritize(self, item_id: str, new_priority: int) -> bool:
        """Change priority of queued item."""
        with self._lock:
            for item in self._queue:
                if item.item_id == item_id:
                    item.priority = new_priority
                    # Re-sort queue
                    items = list(self._queue)
                    items.sort(key=lambda x: x.priority)
                    self._queue.clear()
                    self._queue.extend(items)
                    logger.info(f"Reprioritized {item_id} to priority {new_priority}")
                    return True
            return False
    
    def add_item(self, item: PRDItem) -> None:
        """Add item to queue maintaining priority order."""
        with self._lock:
            # Find insertion point
            inserted = False
            for i, queued_item in enumerate(self._queue):
                if item.priority < queued_item.priority:
                    # Insert before this item (lower number = higher priority)
                    self._queue.insert(i, item)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(item)
            
            logger.info(f"Added {item.item_id} to queue (priority {item.priority})")
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0
    
    def clear(self) -> None:
        """Clear queue."""
        with self._lock:
            self._queue.clear()
            self._current_item = None
            logger.info("Cleared execution queue")
