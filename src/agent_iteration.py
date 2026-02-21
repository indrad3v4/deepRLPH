# -*- coding: utf-8 -*-
"""
agent_iteration.py - Agent Iteration Loop

ITEM-017: Execute single PRD item through agent.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

from src.prd_model import PRDItem, PRDItemStatus
from src.agent_coordinator import AgentCoordinator
from src.execution_engine import ExecutionEngine
from src.prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)


class AgentIteration:
    """Execute single PRD item through agent."""
    
    def __init__(
        self,
        agent_coordinator: AgentCoordinator,
        execution_engine: ExecutionEngine,
        prompt_generator: PromptGenerator,
        agent_id: str = "deepseek-coder"
    ):
        self.agent_coordinator = agent_coordinator
        self.execution_engine = execution_engine
        self.prompt_generator = prompt_generator
        self.agent_id = agent_id
    
    def execute_item(self, item: PRDItem, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single PRD item.
        
        Returns:
            Dict with:
                - success: bool
                - output: str (agent output)
                - error: Optional[str]
                - files_modified: List[str]
        """
        logger.info(f"Starting execution of {item.item_id}: {item.title}")
        
        # Mark item as in progress
        item.mark_in_progress(self.agent_id)
        
        try:
            # Generate prompt for item
            prompt = self._generate_item_prompt(item, context)
            
            # Execute through agent coordinator
            agent_result = self.agent_coordinator.execute_task(
                task_description=prompt,
                context=context
            )
            
            if agent_result.get("success"):
                logger.info(f"✅ {item.item_id} agent execution succeeded")
                return {
                    "success": True,
                    "output": agent_result.get("output", ""),
                    "error": None,
                    "files_modified": agent_result.get("files_modified", [])
                }
            else:
                error_msg = agent_result.get("error", "Unknown error")
                logger.error(f"❌ {item.item_id} agent execution failed: {error_msg}")
                return {
                    "success": False,
                    "output": agent_result.get("output", ""),
                    "error": error_msg,
                    "files_modified": []
                }
        
        except Exception as e:
            error_msg = f"Exception during execution: {str(e)}"
            logger.error(f"❌ {item.item_id} exception: {e}", exc_info=True)
            return {
                "success": False,
                "output": "",
                "error": error_msg,
                "files_modified": []
            }
    
    def _generate_item_prompt(self, item: PRDItem, context: Dict[str, Any]) -> str:
        """Generate prompt for PRD item."""
        prompt_parts = [
            f"## Task: {item.item_id} - {item.title}",
            "",
            "### Acceptance Criteria",
        ]
        
        for i, criterion in enumerate(item.acceptance_criteria, 1):
            prompt_parts.append(f"{i}. {criterion}")
        
        prompt_parts.extend([
            "",
            "### Context",
            f"Project: {context.get('project_id', 'Unknown')}",
            f"Priority: {item.priority}",
        ])
        
        if item.files_touched:
            prompt_parts.extend([
                "",
                "### Files to Modify",
            ])
            for file_path in item.files_touched:
                prompt_parts.append(f"- {file_path}")
        
        if item.verification_command:
            prompt_parts.extend([
                "",
                "### Verification",
                f"After implementation, this will be verified with: `{item.verification_command}`",
            ])
        
        prompt_parts.extend([
            "",
            "### Instructions",
            "Implement the above requirements. Ensure all acceptance criteria are met.",
            "Write clean, tested, documented code.",
        ])
        
        return "\n".join(prompt_parts)
