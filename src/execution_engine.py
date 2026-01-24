# -*- coding: utf-8 -*-
"""
execution_engine.py - Real orchestration with DeepSeek integration
Connects UI → Orchestrator → DeepSeek API → Code Generation
"""

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger("ExecutionEngine")


class ExecutionEngine:
    """
    Real orchestration engine.

    Coordinates:
    - DeepSeek API calls
    - Agent task distribution
    - Code generation
    - File output
    """

    def __init__(
            self,
            project_dir: Path,
            deepseek_client,  # Injected from main
            agent_coordinator,  # Injected from main
            log_callback: Optional[Callable] = None,
            progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize execution engine.

        Args:
            project_dir: Path to project workspace
            deepseek_client: DeepseekClient instance
            agent_coordinator: AgentCoordinator instance
            log_callback: Function to log messages (logs to UI)
            progress_callback: Function to update progress (0-100)
        """
        self.project_dir = project_dir
        self.output_dir = project_dir / "workspace" / "output" / "generated_code"
        self.deepseek = deepseek_client
        self.coordinator = agent_coordinator
        self.log_callback = log_callback or self._default_log
        self.progress_callback = progress_callback or self._default_progress

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ ExecutionEngine initialized")
        logger.info(f"   Output dir: {self.output_dir}")

    def _default_log(self, message: str):
        """Default logging (fallback)"""
        logger.info(message)

    def _default_progress(self, value: float):
        """Default progress (fallback)"""
        pass

    async def execute(
            self,
            task_description: str,
            num_agents: int = 4,
            duration_hours: int = 1,
            thinking_budget: int = 5000,
    ) -> Dict[str, Any]:
        """
        Execute multi-agent orchestration with real DeepSeek calls.

        Args:
            task_description: Main project task
            num_agents: Number of parallel agents
            duration_hours: Execution duration
            thinking_budget: Tokens for extended thinking

        Returns:
            Results with generated code paths
        """
        try:
            self._log(f"🚀 Starting RALPHlooping")
            self._log(f"   Agents: {num_agents} | Duration: {duration_hours}h | Thinking: {thinking_budget:,} tokens")
            self._log("")

            # Step 1: Create subtasks
            self._log("📋 Creating subtasks for agents...")
            subtasks = self.coordinator.create_subtasks(task_description, num_agents)
            self._progress(10)
            self._log(f"   Created {len(subtasks)} subtasks")
            self._log("")

            # Step 2: Call agents in parallel
            self._log(f"🤖 Spawning {num_agents} agents...")
            self._log(f"   Calling DeepSeek API (deepseek-v3.2)...")

            results = await self.deepseek.coordinate_agents(
                system_prompt=self._get_system_prompt(),
                subtasks=subtasks,
                num_agents=num_agents,
                thinking_budget=thinking_budget,
            )
            self._progress(60)
            self._log(f"✅ Agents completed")
            self._log("")

            # Step 3: Aggregate and save results
            self._log("📦 Aggregating results...")
            aggregated = await self.coordinator.aggregate_results(results)
            self._progress(80)
            self._log(f"   Successful agents: {aggregated['successful_agents']}/{aggregated['total_agents']}")
            self._log(f"   Code artifacts: {len(aggregated['artifacts'])}")
            self._log("")

            # Step 4: Extract and save code
            self._log("💾 Saving generated code...")
            code_files = await self._save_artifacts(results)
            self._progress(95)
            self._log(f"   Saved {len(code_files)} files")
            self._log("")

            # Step 5: Generate report
            self._log("📊 Generating report...")
            self.coordinator.report_coordination()
            self._progress(100)
            self._log("")
            self._log("✅ RALPHlooping Complete!")
            self._log(f"   Total tokens used: {aggregated['statistics']['total_tokens']:,}")
            self._log(f"   Output directory: {self.output_dir}")

            return {
                "status": "success",
                "results": aggregated,
                "code_files": code_files,
                "output_dir": str(self.output_dir),
            }

        except Exception as e:
            self._log(f"❌ Execution failed: {str(e)}")
            logger.error(f"Execution error: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    async def _save_artifacts(self, results: list) -> list:
        """Extract and save code artifacts from agent responses"""
        saved_files = []

        for agent_id, response in enumerate(results, 1):
            if response.get("status") != "success":
                continue

            # Extract code blocks from response
            response_text = response.get("response", "")
            agent_dir = self.output_dir / f"agent_{agent_id}"
            agent_dir.mkdir(parents=True, exist_ok=True)

            # Parse code blocks from markdown
            import re
            code_blocks = re.findall(
                r'```(\w+)?\n(.*?)\n```',
                response_text,
                re.DOTALL
            )

            for block_idx, (lang, code) in enumerate(code_blocks, 1):
                # Infer filename from content
                filename = self._infer_filename(code, lang, block_idx)

                file_path = agent_dir / filename
                try:
                    file_path.write_text(code, encoding='utf-8')
                    saved_files.append(str(file_path))
                    self._log(f"   💾 {filename} ({len(code)} bytes)")
                except Exception as e:
                    logger.error(f"Error saving {filename}: {e}")

        return saved_files

    def _infer_filename(self, code: str, lang: str, block_idx: int) -> str:
        """Infer filename from code content"""
        # Try to detect class/function name
        import re

        # Python class
        match = re.search(r'^class\s+(\w+)', code, re.MULTILINE)
        if match:
            return f"{match.group(1).lower()}.py"

        # Python function
        match = re.search(r'^def\s+(\w+)', code, re.MULTILINE)
        if match:
            return f"{match.group(1)}.py"

        # Fallback
        lang_ext = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'sql': 'sql',
            'html': 'html',
            'css': 'css',
        }
        ext = lang_ext.get(lang, 'txt')
        return f"code_block_{block_idx}.{ext}"

    def _get_system_prompt(self) -> str:
        """Get system prompt for agents"""
        return """You are an expert software architect and developer participating in a multi-agent system.

Your task is to generate high-quality, production-ready code for your assigned module.

Requirements:
1. Write clean, well-documented Python code
2. Include proper error handling and logging
3. Add type hints and docstrings
4. Write comprehensive unit tests (>80% coverage)
5. Follow PEP 8 style guidelines
6. Consider performance and security
7. Format code in markdown code blocks

Output ONLY code blocks. No explanations outside of code comments."""

    def _log(self, message: str):
        """Log message to callback"""
        try:
            self.log_callback(message)
        except Exception as e:
            logger.error(f"Log callback error: {e}")

    def _progress(self, value: float):
        """Update progress bar"""
        try:
            self.progress_callback(min(100, max(0, value)))
        except Exception as e:
            logger.error(f"Progress callback error: {e}")


# Async wrapper for UI threading
def run_execution_async(
        project_dir: Path,
        deepseek_client,
        agent_coordinator,
        task_description: str,
        num_agents: int,
        duration_hours: int,
        thinking_budget: int,
        log_callback,
        progress_callback,
) -> Dict[str, Any]:
    """Run execution engine in event loop (for UI threading)"""
    try:
        engine = ExecutionEngine(
            project_dir=project_dir,
            deepseek_client=deepseek_client,
            agent_coordinator=agent_coordinator,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        # Run async in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            engine.execute(
                task_description=task_description,
                num_agents=num_agents,
                duration_hours=duration_hours,
                thinking_budget=thinking_budget,
            )
        )
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Async execution error: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}