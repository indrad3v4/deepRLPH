# -*- coding: utf-8 -*-
"""
verification_registry.py - Verification Command Registry

ITEM-015: Central registry for verification commands mapped to PRD items.
"""

from typing import Dict, Optional, List
import logging
import re

logger = logging.getLogger(__name__)


class VerificationRegistry:
    """Registry for verification commands per PRD item."""
    
    def __init__(self):
        self._commands: Dict[str, str] = {}
        self._patterns: Dict[str, str] = {}
    
    def register(self, item_id: str, command: str) -> None:
        """Register verification command for item."""
        if not command:
            raise ValueError(f"Cannot register empty command for {item_id}")
        
        self._commands[item_id] = command
        logger.info(f"Registered verification for {item_id}: {command}")
    
    def register_pattern(self, pattern: str, command: str) -> None:
        """Register command pattern for items matching regex.
        
        Example:
            registry.register_pattern(r"ITEM-\d{3}", "make test")
        """
        self._patterns[pattern] = command
        logger.info(f"Registered pattern '{pattern}' → '{command}'")
    
    def get(self, item_id: str) -> Optional[str]:
        """Get verification command for item."""
        # Check exact match first
        if item_id in self._commands:
            return self._commands[item_id]
        
        # Check patterns
        for pattern, command in self._patterns.items():
            if re.match(pattern, item_id):
                logger.info(f"Matched {item_id} to pattern '{pattern}'")
                return command
        
        return None
    
    def has_command(self, item_id: str) -> bool:
        """Check if item has registered verification command."""
        return self.get(item_id) is not None
    
    def list_all(self) -> Dict[str, str]:
        """List all registered commands."""
        return dict(self._commands)
    
    def remove(self, item_id: str) -> bool:
        """Remove registered command."""
        if item_id in self._commands:
            del self._commands[item_id]
            logger.info(f"Removed verification for {item_id}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered commands."""
        self._commands.clear()
        self._patterns.clear()
        logger.info("Cleared all verification commands")


# Global registry instance
_global_registry: Optional[VerificationRegistry] = None


def get_global_registry() -> VerificationRegistry:
    """Get global verification registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = VerificationRegistry()
        # Register default patterns
        _global_registry.register_pattern(r"ITEM-\d+", "make test")
    return _global_registry


def register_verification(item_id: str, command: str) -> None:
    """Convenience function to register verification command."""
    get_global_registry().register(item_id, command)
