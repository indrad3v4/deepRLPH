# -*- coding: utf-8 -*-
"""
execution_engine_extensions.py - PRD Verification Extensions

ITEM-003: Verification Command Integration

Extensions to ExecutionEngine for running PRD item verification commands.
"""

import asyncio
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PRDVerificationMixin:
    """Mixin for ExecutionEngine to add PRD verification support.
    
    Usage:
        class ExecutionEngine(PRDVerificationMixin, ...):
            ...
    
    This mixin provides run_verification_for_prd() method that:
    1. Executes shell verification commands
    2. Captures stdout/stderr
    3. Returns success/failure with output
    4. Handles timeouts and errors
    """
    
    async def run_verification_for_prd(
        self,
        command: str,
        project_dir: Path,
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Run verification command for PRD item.
        
        Args:
            command: Shell command to execute (e.g., "pytest tests/")
            project_dir: Project directory to run command in
            timeout: Timeout in seconds (default: 600)
        
        Returns:
            Dict with:
            - success: True if command succeeded (exit code 0)
            - output: Combined stdout/stderr
            - error: Error message if failed
            - exit_code: Command exit code
        """
        logger.info(f"Running verification: {command}")
        logger.info(f"   Working directory: {project_dir}")
        
        try:
            # Run command in project directory
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    shell=True,
                    cwd=str(project_dir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                ),
            )
            
            # Combine stdout and stderr
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr
            
            # Check exit code
            if result.returncode == 0:
                logger.info("   ✅ Verification passed")
                return {
                    "success": True,
                    "output": output,
                    "exit_code": result.returncode,
                }
            else:
                logger.warning(f"   ❌ Verification failed with code {result.returncode}")
                return {
                    "success": False,
                    "error": f"Command failed with exit code {result.returncode}",
                    "output": output,
                    "exit_code": result.returncode,
                }
        
        except subprocess.TimeoutExpired:
            error = f"Verification timed out after {timeout}s"
            logger.error(f"   ❌ {error}")
            return {
                "success": False,
                "error": error,
                "output": "",
                "exit_code": -1,
            }
        
        except Exception as e:
            error = f"Verification error: {str(e)}"
            logger.exception("Verification execution failed")
            return {
                "success": False,
                "error": error,
                "output": "",
                "exit_code": -1,
            }
