# -*- coding: utf-8 -*-
"""
fix_loop.py - Fix Loop

ITEM-019: Retry failed items with fix attempts.
"""

import logging
from typing import Dict, Any, Optional

from src.prd_model import PRDItem, PRDItemStatus
from src.agent_iteration import AgentIteration
from src.verification_runner import VerificationRunner, VerificationResult

logger = logging.getLogger(__name__)


class FixLoop:
    """Retry failed items with fix attempts."""
    
    def __init__(
        self,
        agent_iteration: AgentIteration,
        verification_runner: VerificationRunner,
        max_retry_attempts: int = 3
    ):
        self.agent_iteration = agent_iteration
        self.verification_runner = verification_runner
        self.max_retry_attempts = max_retry_attempts
    
    def attempt_fix(
        self,
        item: PRDItem,
        verification_result: VerificationResult,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt to fix failed item.
        
        Returns:
            Dict with:
                - success: bool (True if fixed)
                - attempts: int
                - final_result: VerificationResult
        """
        logger.info(f"Starting fix loop for {item.item_id}")
        
        attempts = 0
        current_error = verification_result.error or "Verification failed"
        
        while attempts < self.max_retry_attempts:
            attempts += 1
            logger.info(f"Fix attempt {attempts}/{self.max_retry_attempts} for {item.item_id}")
            
            # Reset item for retry
            item.reset_for_retry()
            
            # Add error context to help agent fix the issue
            fix_context = self._create_fix_context(context, current_error, attempts)
            
            # Execute fix through agent
            agent_result = self.agent_iteration.execute_item(item, fix_context)
            
            if not agent_result["success"]:
                current_error = agent_result["error"]
                logger.warning(f"Fix attempt {attempts} agent execution failed: {current_error}")
                continue
            
            # Verify the fix
            verify_result = self.verification_runner.verify_item(item)
            
            if verify_result.success:
                logger.info(f"✅ {item.item_id} fixed after {attempts} attempts")
                return {
                    "success": True,
                    "attempts": attempts,
                    "final_result": verify_result
                }
            else:
                current_error = verify_result.error or "Verification still failing"
                logger.warning(f"Fix attempt {attempts} verification failed: {current_error}")
        
        # All attempts exhausted
        logger.error(f"❌ {item.item_id} could not be fixed after {attempts} attempts")
        return {
            "success": False,
            "attempts": attempts,
            "final_result": verification_result
        }
    
    def _create_fix_context(self, context: Dict[str, Any], error: str, attempt: int) -> Dict[str, Any]:
        """Create context for fix attempt including error information."""
        fix_context = dict(context)
        fix_context["fix_attempt"] = attempt
        fix_context["previous_error"] = error
        fix_context["instruction_prefix"] = (
            f"FIXING PREVIOUS FAILURE (Attempt {attempt}):\n"
            f"Previous error: {error}\n\n"
            f"Please analyze the error and fix the implementation.\n\n"
        )
        return fix_context
    
    def should_retry(self, item: PRDItem) -> bool:
        """Check if item should be retried."""
        if item.status != PRDItemStatus.FAIL:
            return False
        
        if item.attempt_count >= self.max_retry_attempts:
            logger.info(f"{item.item_id} reached max retry attempts ({self.max_retry_attempts})")
            return False
        
        return True
