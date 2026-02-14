# -*- coding: utf-8 -*-
"""
item_selector.py - Next Item Selection Logic

ITEM-020: Select next PRD item to work on based on priority and dependencies.
"""

import logging
from typing import Optional, List, Dict, Any

from src.prd_model import PRDBacklog, PRDItem, PRDItemStatus

logger = logging.getLogger(__name__)


class ItemSelector:
    """Select next PRD item based on priority and dependencies."""
    
    def __init__(self, backlog: PRDBacklog):
        self.backlog = backlog
        self._dependencies: Dict[str, List[str]] = {}
    
    def add_dependency(self, item_id: str, depends_on: str) -> None:
        """Add dependency: item_id depends on depends_on."""
        if item_id not in self._dependencies:
            self._dependencies[item_id] = []
        self._dependencies[item_id].append(depends_on)
        logger.info(f"{item_id} now depends on {depends_on}")
    
    def select_next(self) -> Optional[PRDItem]:
        """Select next item to work on.
        
        Selection criteria:
        1. Status must be PENDING
        2. All dependencies must be PASS
        3. Select by priority (lower number = higher priority)
        """
        pending_items = [
            item for item in self.backlog.items
            if item.status == PRDItemStatus.PENDING
        ]
        
        if not pending_items:
            logger.info("No pending items to select")
            return None
        
        # Filter by dependencies
        available_items = [
            item for item in pending_items
            if self._check_dependencies(item)
        ]
        
        if not available_items:
            logger.warning("Pending items exist but none have satisfied dependencies")
            return None
        
        # Sort by priority (1 = highest)
        available_items.sort(key=lambda x: x.priority)
        
        selected = available_items[0]
        logger.info(
            f"Selected {selected.item_id} (priority {selected.priority}) "
            f"from {len(available_items)} available items"
        )
        
        return selected
    
    def _check_dependencies(self, item: PRDItem) -> bool:
        """Check if item's dependencies are satisfied."""
        if item.item_id not in self._dependencies:
            return True  # No dependencies
        
        depends_on = self._dependencies[item.item_id]
        
        for dep_id in depends_on:
            dep_item = self.backlog.get_item_by_id(dep_id)
            if not dep_item:
                logger.warning(f"Dependency {dep_id} not found in backlog")
                return False
            
            if dep_item.status != PRDItemStatus.PASS:
                logger.debug(
                    f"{item.item_id} blocked by {dep_id} "
                    f"(status: {dep_item.status})"
                )
                return False
        
        return True
    
    def get_blocked_items(self) -> List[PRDItem]:
        """Get items blocked by dependencies."""
        pending_items = [
            item for item in self.backlog.items
            if item.status == PRDItemStatus.PENDING
        ]
        
        return [
            item for item in pending_items
            if not self._check_dependencies(item)
        ]
    
    def get_available_items(self) -> List[PRDItem]:
        """Get all available items (pending with satisfied dependencies)."""
        pending_items = [
            item for item in self.backlog.items
            if item.status == PRDItemStatus.PENDING
        ]
        
        available = [
            item for item in pending_items
            if self._check_dependencies(item)
        ]
        
        available.sort(key=lambda x: x.priority)
        return available
    
    def clear_dependencies(self) -> None:
        """Clear all dependencies."""
        self._dependencies.clear()
        logger.info("Cleared all item dependencies")
