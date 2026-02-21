# -*- coding: utf-8 -*-
"""
trace_logger.py - BE-008.1: JSONL Execution Trace Logger

Logs execution events as JSONL (one JSON object per line) for debugging
and analysis. Thread-safe with file locking.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import fcntl  # For file locking on Unix
import platform

logger = logging.getLogger("TraceLogger")


class TraceLogger:
    """JSONL trace logger for execution events"""
    
    def __init__(self, trace_file: Path, execution_id: str):
        """
        Initialize trace logger.
        
        Args:
            trace_file: Path to .jsonl file
            execution_id: Unique execution identifier
        """
        self.trace_file = Path(trace_file)
        self.execution_id = execution_id
        self._lock = threading.Lock()
        self._is_windows = platform.system() == "Windows"
        
        # Ensure parent directory exists
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write header event
        self.log_event("execution_start", {
            "execution_id": execution_id,
            "start_time": datetime.now().isoformat(),
        })
        
        logger.info(f"📝 Trace logger initialized: {self.trace_file}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an event to JSONL file.
        
        Args:
            event_type: Event type (agent_start, item_complete, etc.)
            data: Event data (will be merged with standard fields)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "execution_id": self.execution_id,
            "event": event_type,
            **data
        }
        
        self._write_line(event)
    
    def _write_line(self, event: Dict[str, Any]) -> None:
        """Write event as JSON line (thread-safe)"""
        with self._lock:
            try:
                with open(self.trace_file, 'a', encoding='utf-8') as f:
                    # File locking (Unix only, Windows handles differently)
                    if not self._is_windows:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    json_line = json.dumps(event, default=str)
                    f.write(json_line + '\n')
                    f.flush()
                    
                    if not self._is_windows:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        
            except Exception as e:
                logger.error(f"Failed to write trace event: {e}")
    
    # Convenience methods for common events
    
    def agent_start(self, agent_id: str, assigned_items: int) -> None:
        """Log agent start event"""
        self.log_event("agent_start", {
            "agent_id": agent_id,
            "assigned_items": assigned_items,
        })
    
    def item_start(self, agent_id: str, item_id: str, item_title: str) -> None:
        """Log PRD item start"""
        self.log_event("item_start", {
            "agent_id": agent_id,
            "item_id": item_id,
            "item_title": item_title,
        })
    
    def item_complete(
        self,
        agent_id: str,
        item_id: str,
        duration_seconds: float,
        files_created: list
    ) -> None:
        """Log PRD item completion"""
        self.log_event("item_complete", {
            "agent_id": agent_id,
            "item_id": item_id,
            "duration_seconds": duration_seconds,
            "files_created": files_created,
        })
    
    def item_fail(
        self,
        agent_id: str,
        item_id: str,
        error_message: str,
        attempt: int
    ) -> None:
        """Log PRD item failure"""
        self.log_event("item_fail", {
            "agent_id": agent_id,
            "item_id": item_id,
            "error_message": error_message,
            "attempt": attempt,
        })
    
    def metric_update(
        self,
        agent_id: str,
        metric_name: str,
        metric_value: float,
        target: Optional[float] = None
    ) -> None:
        """Log metric measurement"""
        self.log_event("metric_update", {
            "agent_id": agent_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "target": target,
        })
    
    def agent_finish(
        self,
        agent_id: str,
        completed_items: int,
        failed_items: int,
        duration_seconds: float
    ) -> None:
        """Log agent finish event"""
        self.log_event("agent_finish", {
            "agent_id": agent_id,
            "completed_items": completed_items,
            "failed_items": failed_items,
            "duration_seconds": duration_seconds,
        })
    
    def execution_end(self, status: str, total_duration: float) -> None:
        """Log execution end"""
        self.log_event("execution_end", {
            "status": status,
            "total_duration_seconds": total_duration,
            "end_time": datetime.now().isoformat(),
        })
