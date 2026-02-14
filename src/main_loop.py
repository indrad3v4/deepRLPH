# -*- coding: utf-8 -*-
"""
main_loop.py - Full Loop with Progress Reporting

ITEM-021: Complete execution loop with progress reporting.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.prd_model import PRDBacklog, PRDItem, PRDItemStatus
from src.execution_queue import ExecutionQueue
from src.agent_iteration import AgentIteration
from src.verification_runner import VerificationRunner
from src.fix_loop import FixLoop
from src.item_selector import ItemSelector
from src.prd_export import export_prd_to_markdown

logger = logging.getLogger(__name__)


class MainLoop:
    """Main execution loop for PRD items."""
    
    def __init__(
        self,
        backlog: PRDBacklog,
        agent_iteration: AgentIteration,
        verification_runner: VerificationRunner,
        workspace_dir: Path,
        max_iterations: int = 100,
        max_retry_attempts: int = 3
    ):
        self.backlog = backlog
        self.execution_queue = ExecutionQueue(backlog)
        self.agent_iteration = agent_iteration
        self.verification_runner = verification_runner
        self.fix_loop = FixLoop(agent_iteration, verification_runner, max_retry_attempts)
        self.item_selector = ItemSelector(backlog)
        self.workspace_dir = workspace_dir
        self.max_iterations = max_iterations
        
        self.iteration_count = 0
        self.start_time: Optional[datetime] = None
    
    def run(self) -> Dict[str, Any]:
        """Run main execution loop.
        
        Returns:
            Dict with:
                - success: bool (True if all items passed)
                - iterations: int
                - duration_seconds: float
                - statistics: Dict
        """
        logger.info("🚀 Starting main execution loop")
        logger.info(f"Max iterations: {self.max_iterations}")
        
        self.start_time = datetime.now()
        
        # Populate initial queue
        self.execution_queue.populate_from_backlog()
        
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {self.iteration_count}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            # Select next item
            item = self.item_selector.select_next()
            
            if not item:
                logger.info("No more available items to process")
                break
            
            # Execute iteration
            success = self._execute_iteration(item)
            
            # Report progress
            self._report_progress()
            
            # Check if done
            if self._is_complete():
                logger.info("✅ All items completed successfully!")
                break
            
            # Check if blocked
            if self._is_blocked():
                logger.warning("⚠️ All remaining items are blocked or failed")
                break
        
        # Final report
        return self._generate_final_report()
    
    def _execute_iteration(self, item: PRDItem) -> bool:
        """Execute single iteration for item.
        
        Returns:
            True if item passed, False otherwise
        """
        context = self._build_context()
        
        # Execute through agent
        logger.info(f"\n🔧 Executing {item.item_id}: {item.title}")
        agent_result = self.agent_iteration.execute_item(item, context)
        
        if not agent_result["success"]:
            error = agent_result["error"]
            logger.error(f"❌ Agent execution failed: {error}")
            self.backlog.mark_complete(item.item_id, success=False, error=error)
            return False
        
        # Verify
        logger.info(f"\n🧪 Verifying {item.item_id}...")
        verify_result = self.verification_runner.verify_item(item)
        
        if verify_result.success:
            logger.info(f"✅ {item.item_id} PASSED")
            self.backlog.mark_complete(item.item_id, success=True)
            return True
        
        # Try to fix
        logger.warning(f"❌ {item.item_id} verification failed, attempting fixes...")
        fix_result = self.fix_loop.attempt_fix(item, verify_result, context)
        
        if fix_result["success"]:
            logger.info(f"✅ {item.item_id} FIXED after {fix_result['attempts']} attempts")
            self.backlog.mark_complete(item.item_id, success=True)
            return True
        else:
            error = f"Failed after {fix_result['attempts']} fix attempts"
            logger.error(f"❌ {item.item_id} could not be fixed: {error}")
            self.backlog.mark_complete(item.item_id, success=False, error=error)
            return False
    
    def _build_context(self) -> Dict[str, Any]:
        """Build execution context."""
        return {
            "project_id": self.backlog.project_id,
            "workspace_dir": str(self.workspace_dir),
            "iteration": self.iteration_count,
            "max_iterations": self.max_iterations,
        }
    
    def _report_progress(self) -> None:
        """Report current progress."""
        stats = self.backlog.get_statistics()
        
        logger.info("\n" + "="*60)
        logger.info("PROGRESS REPORT")
        logger.info("="*60)
        logger.info(f"Iteration: {self.iteration_count}/{self.max_iterations}")
        logger.info(f"Total Items: {stats['total']}")
        logger.info(f"✅ Passed: {stats['pass']}")
        logger.info(f"⏳ Pending: {stats['pending']}")
        logger.info(f"🔄 In Progress: {stats['in_progress']}")
        logger.info(f"🧪 Testing: {stats['testing']}")
        logger.info(f"❌ Failed: {stats['fail']}")
        logger.info(f"Progress: {stats['progress_pct']:.1f}%")
        logger.info("="*60 + "\n")
        
        # Export to markdown
        md_path = self.workspace_dir / "prd_progress.md"
        export_prd_to_markdown(self.backlog, md_path)
        logger.info(f"Exported progress to {md_path}")
    
    def _is_complete(self) -> bool:
        """Check if all items are complete."""
        return not self.backlog.has_pending_items()
    
    def _is_blocked(self) -> bool:
        """Check if execution is blocked."""
        # No pending items at all
        if not self.backlog.has_pending_items():
            return True
        
        # All pending items are blocked by dependencies
        available = self.item_selector.get_available_items()
        return len(available) == 0
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final execution report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        stats = self.backlog.get_statistics()
        
        all_passed = stats['pass'] == stats['total']
        
        logger.info("\n" + "="*60)
        logger.info("FINAL REPORT")
        logger.info("="*60)
        logger.info(f"Status: {'✅ SUCCESS' if all_passed else '❌ INCOMPLETE'}")
        logger.info(f"Iterations: {self.iteration_count}/{self.max_iterations}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Items Passed: {stats['pass']}/{stats['total']}")
        logger.info(f"Items Failed: {stats['fail']}/{stats['total']}")
        logger.info(f"Progress: {stats['progress_pct']:.1f}%")
        logger.info("="*60)
        
        # Export final state
        final_md_path = self.workspace_dir / "prd_final_report.md"
        export_prd_to_markdown(self.backlog, final_md_path)
        logger.info(f"Final report exported to {final_md_path}")
        
        return {
            "success": all_passed,
            "iterations": self.iteration_count,
            "duration_seconds": duration,
            "statistics": stats
        }
