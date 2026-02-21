# -*- coding: utf-8 -*-
"""
orchestrator_extensions.py - PRD Execution Loop Extensions

ITEM-002: Orchestrator PRD Execution Loop Integration

Extensions to RalphOrchestrator for executing PRD backlogs.
Separate file to avoid modifying the large orchestrator.py during Phase 1.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from prd_model import PRDBacklog, PRDItem, PRDItemStatus

logger = logging.getLogger(__name__)


class PRDExecutionMixin:
    """Mixin for RalphOrchestrator to add PRD backlog execution.
    
    Usage:
        class RalphOrchestrator(PRDExecutionMixin, ...):
            ...
    
    This mixin provides execute_prd_backlog() method that:
    1. Loads PRDBacklog from items
    2. Iteratively executes items with single agent
    3. Runs verification commands
    4. Updates backlog state
    5. Reports progress
    """
    
    async def execute_prd_backlog(
        self,
        backlog: PRDBacklog,
        project_dir: Path,
        log_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute PRD backlog items sequentially.
        
        Args:
            backlog: PRDBacklog with items to execute
            project_dir: Project directory path
            log_callback: Optional callback for log messages
            progress_callback: Optional callback for progress (0-100)
        
        Returns:
            Dict with execution results:
            - status: "success" | "partial" | "failed"
            - completed_count: Number of passed items
            - failed_count: Number of failed items
            - backlog_state: Final backlog state
        """
        logger.info("🚀 Starting PRD backlog execution")
        logger.info(f"   Project: {project_dir}")
        logger.info(f"   Total items: {len(backlog.items)}")
        
        await self._log_callback(log_callback, "🚀 Starting PRD backlog execution")
        await self._log_callback(log_callback, f"   Total items: {len(backlog.items)}")
        
        execution_start = datetime.now()
        iteration = 0
        max_iterations = len(backlog.items) * 2  # Allow retries
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get next pending item
            item = backlog.get_next_pending()
            if not item:
                await self._log_callback(log_callback, "✅ No more pending items")
                break
            
            await self._log_callback(
                log_callback,
                f"\n📌 Iteration {iteration}: {item.item_id} - {item.title}"
            )
            
            # Execute item
            success = await self._execute_prd_item(
                backlog=backlog,
                item=item,
                project_dir=project_dir,
                log_callback=log_callback,
            )
            
            # Update progress
            stats = backlog.get_statistics()
            progress = stats["progress_pct"]
            await self._progress_callback(progress_callback, progress)
            
            # Check if done
            if not backlog.has_pending_items():
                await self._log_callback(log_callback, "🎉 All items completed")
                break
        
        # Final statistics
        duration = (datetime.now() - execution_start).total_seconds()
        stats = backlog.get_statistics()
        
        await self._log_callback(log_callback, "\n📊 Execution Summary:")
        await self._log_callback(log_callback, f"   ✅ Passed: {stats['pass']}")
        await self._log_callback(log_callback, f"   ❌ Failed: {stats['fail']}")
        await self._log_callback(log_callback, f"   ⏱️  Duration: {duration:.1f}s")
        
        # Save final state
        state_file = project_dir / "prd_backlog_state.json"
        backlog.save_to_file(state_file)
        await self._log_callback(log_callback, f"   💾 State saved: {state_file}")
        
        # Determine final status
        if stats["fail"] > 0 and stats["pass"] == 0:
            status = "failed"
        elif stats["fail"] > 0:
            status = "partial"
        else:
            status = "success"
        
        return {
            "status": status,
            "completed_count": stats["pass"],
            "failed_count": stats["fail"],
            "total_items": stats["total"],
            "duration_seconds": duration,
            "backlog_state": backlog.to_dict(),
        }
    
    async def _execute_prd_item(
        self,
        backlog: PRDBacklog,
        item: PRDItem,
        project_dir: Path,
        log_callback: Optional[Callable] = None,
    ) -> bool:
        """Execute single PRD item.
        
        Returns:
            True if item passed, False if failed
        """
        agent_id = "agent_1"  # Phase 1: single agent
        
        # Mark as in progress
        backlog.mark_in_progress(item.item_id, agent_id)
        
        try:
            # Get execution engine
            if not hasattr(self, 'execution_engine') or not self.execution_engine:
                await self._log_callback(log_callback, "   Creating execution engine...")
                from execution_engine import ExecutionEngine
                self.execution_engine = ExecutionEngine(
                    project_dir=project_dir,
                    deepseek_client=getattr(self, 'deepseek_client', None),
                    agent_coordinator=getattr(self, 'agent_coordinator', None),
                )
            
            # Get agent coordinator
            if not hasattr(self, 'agent_coordinator') or not self.agent_coordinator:
                await self._log_callback(log_callback, "   Creating agent coordinator...")
                from agent_coordinator import AgentCoordinator
                self.agent_coordinator = AgentCoordinator(workspace=project_dir / "workspace")
            
            # Assign to agent (generates code)
            await self._log_callback(log_callback, f"   🤖 Agent {agent_id}: Generating code...")
            
            code_result = await self.agent_coordinator.assign_prd_item(
                item=item,
                agent_id=agent_id,
                project_dir=project_dir,
                deepseek_client=getattr(self, 'deepseek_client', None),
            )
            
            if code_result["status"] != "success":
                error = code_result.get("error", "Unknown error")
                await self._log_callback(log_callback, f"   ❌ Code generation failed: {error}")
                backlog.mark_complete(item.item_id, success=False, error=error)
                return False
            
            files_created = code_result.get("files_created", [])
            await self._log_callback(
                log_callback,
                f"   💾 Created {len(files_created)} file(s)"
            )
            
            # Run verification if command exists
            if item.verification_command and item.verification_command.strip():
                await self._log_callback(log_callback, "   🧪 Running verification...")
                
                verify_result = await self.execution_engine.run_verification_for_prd(
                    command=item.verification_command,
                    project_dir=project_dir,
                )
                
                if verify_result["success"]:
                    await self._log_callback(log_callback, f"   ✅ {item.item_id} PASSED")
                    backlog.mark_complete(item.item_id, success=True)
                    return True
                else:
                    error = verify_result.get("error", "Verification failed")
                    await self._log_callback(log_callback, f"   ❌ {item.item_id} FAILED: {error}")
                    backlog.mark_complete(item.item_id, success=False, error=error)
                    return False
            else:
                # No verification, mark as passed
                await self._log_callback(
                    log_callback,
                    "   ⚠️  No verification command, marking as passed"
                )
                backlog.mark_complete(item.item_id, success=True)
                return True
        
        except Exception as e:
            error = f"Execution error: {str(e)}"
            logger.exception(f"Error executing {item.item_id}")
            await self._log_callback(log_callback, f"   ❌ {error}")
            backlog.mark_complete(item.item_id, success=False, error=error)
            return False
    
    async def _log_callback(self, callback: Optional[Callable], message: str):
        """Call log callback if provided."""
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
    
    async def _progress_callback(self, callback: Optional[Callable], progress: float):
        """Call progress callback if provided."""
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)
