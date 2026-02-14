# -*- coding: utf-8 -*-
"""
verification_runner.py - Test & Verification Step

ITEM-018: Run tests and verification after agent execution.
"""

import subprocess
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from src.prd_model import PRDItem
from src.verification_registry import get_global_registry

logger = logging.getLogger(__name__)


class VerificationResult:
    """Result of verification run."""
    
    def __init__(
        self,
        success: bool,
        output: str,
        error: Optional[str] = None,
        exit_code: Optional[int] = None
    ):
        self.success = success
        self.output = output
        self.error = error
        self.exit_code = exit_code
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code
        }


class VerificationRunner:
    """Run verification commands for PRD items."""
    
    def __init__(self, workspace_dir: Path, timeout: int = 300):
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self.registry = get_global_registry()
    
    def verify_item(self, item: PRDItem) -> VerificationResult:
        """Run verification for PRD item."""
        logger.info(f"Verifying {item.item_id}...")
        
        # Mark item as testing
        item.mark_testing()
        
        # Get verification command
        command = self._get_verification_command(item)
        if not command:
            logger.warning(f"No verification command for {item.item_id}, skipping")
            return VerificationResult(
                success=True,
                output="No verification command registered",
                error=None
            )
        
        # Run command
        return self._run_command(command)
    
    def _get_verification_command(self, item: PRDItem) -> Optional[str]:
        """Get verification command for item."""
        # Check item's own verification command first
        if item.verification_command:
            return item.verification_command
        
        # Check registry
        return self.registry.get(item.item_id)
    
    def _run_command(self, command: str) -> VerificationResult:
        """Execute verification command."""
        logger.info(f"Running: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            success = result.returncode == 0
            
            if success:
                logger.info(f"✅ Verification passed (exit code {result.returncode})")
            else:
                logger.error(f"❌ Verification failed (exit code {result.returncode})")
            
            return VerificationResult(
                success=success,
                output=result.stdout,
                error=result.stderr if not success else None,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            error_msg = f"Verification timed out after {self.timeout}s"
            logger.error(f"❌ {error_msg}")
            return VerificationResult(
                success=False,
                output="",
                error=error_msg,
                exit_code=-1
            )
        
        except Exception as e:
            error_msg = f"Exception running verification: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return VerificationResult(
                success=False,
                output="",
                error=error_msg,
                exit_code=-1
            )
    
    def run_custom_verification(
        self,
        command: str,
        description: str = "Custom verification"
    ) -> VerificationResult:
        """Run custom verification command."""
        logger.info(f"{description}: {command}")
        return self._run_command(command)
