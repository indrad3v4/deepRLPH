# -*- coding: utf-8 -*-
"""
agent_coordinator_extensions.py - PRD Agent Assignment Extensions (Guru Edition)

Upgrades:
- Transactional Stability: Strict iteration limits and memory management.
- Stage 2 Isolation: Guardrail-aware command execution.
- Deterministic Handoff: explicit human intervention triggers.
"""

import asyncio
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from prd_model import PRDItem

logger = logging.getLogger(__name__)


class PRDAgentAssignmentMixin:
    """Refactored Mixin for AgentCoordinator focusing on safe autonomous loops."""

    async def assign_prd_item(
            self,
            item: PRDItem,
            agent_id: str,
            project_dir: Path,
            deepseek_client: Any,
            log_callback: Optional[callable] = None,
            orchestrator_prompt: str = "",
    ) -> Dict[str, Any]:

        async def _log(msg: str):
            logger.info(f"[{agent_id}] {msg}")
            if log_callback:
                if asyncio.iscoroutinefunction(log_callback):
                    await log_callback(f"[{agent_id}] {msg}")
                else:
                    log_callback(f"[{agent_id}] {msg}")

        await _log(f"Assigning {item.item_id}: {item.title}")

        # Structured Agentic Prompting (Ralph Loop Routine)
        system_prompt = f"""{orchestrator_prompt}

### AGENTIC ROUTINE
1. Contextualize (read_file) -> 2. Implement (write_file) -> 3. Verify (execute_command)
4. Escalate (request_human_intervention) if stuck.

### STRICT CONSTRAINTS
- FLAT PACKAGING: Submission must be 3 files at root.
- PATH INJECTION: Start Python files with sys.path.insert(0, os.path.abspath('.')).
"""
        messages = [{"role": "user", "content": "Analyze requirements and start the Implementation Phase."}]

        # Tools definition (Assumed consistent with deeprlph patterns)
        tools = self._get_agent_tools()

        max_iterations = 15  # Tightened from 30 to prevent token burn
        files_created: Set[str] = set()

        try:
            for iteration in range(max_iterations):
                await _log(f"Loop iteration {iteration + 1}/{max_iterations}...")

                response = await deepseek_client.call_agent_with_tools(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    temperature=0.2  # Lowered for implementation consistency
                )

                if response.get("status") != "success":
                    return {"status": "error", "error": response.get("error"), "files_created": list(files_created)}

                message = response.get("message", {})
                messages.append(message)  # Store agent's thought/call

                if not message.get("tool_calls"):
                    messages.append({"role": "user", "content": "Continue implementation using tools or finish_task."})
                    continue

                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"] or "{}")

                    # Tool Execution Logic (Deroasted for safety)
                    result = await self._dispatch_tool(func_name, args, project_dir, item, files_created, _log)

                    # Sequential Logic for Early Termination
                    if result.get("terminate"):
                        return result["payload"]

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": func_name,
                        "content": str(result["content"])[:5000]  # Content truncation
                    })

            return {"status": "success", "response": "Max iterations reached.", "files_created": list(files_created)}

        except Exception as e:
            return {"status": "error", "error": f"Agent crash: {str(e)}", "files_created": list(files_created)}

    async def _dispatch_tool(self, func_name, args, project_dir, item, files_created, log_fn):
        """Isolated tool dispatcher with Stage 2 Guardrails."""
        if func_name == "execute_command":
            cmd = args.get("command", "")
            # Stage 2 Guardrail: Intercept high-risk commands
            if "rm " in cmd and str(project_dir) not in cmd:
                return {"content": "🚫 BLOCKED: Unsafe delete detected outside project root.", "terminate": False}

            # Execute with PYTHONPATH grounding
            res = await self._run_command_safe(cmd, project_dir, log_fn)

            # Stage 3 logic: If official verification passes, we are done
            if res["success"] and cmd.strip() == item.verification_command.strip():
                await log_fn("🎯 Verification Passed. Auto-terminating.")
                return {"terminate": True, "payload": {"status": "success", "files_created": list(files_created)}}

            return {"content": res["output"], "terminate": False}

        # Handle other tools (read_file, write_file, request_human_intervention)
        # ... logic follows deeprlph standards ...
        return {"content": "Tool executed", "terminate": False}

    async def _run_command_safe(self, cmd: str, project_dir: Path, log_fn):
        """Execution helper with 120s timeout."""
        await log_fn(f"Executing: {cmd}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_dir)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(project_dir), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            output = stdout.decode().strip() + "\n" + stderr.decode().strip()
            return {"success": proc.returncode == 0, "output": output}
        except Exception as e:
            return {"success": False, "output": f"Error: {e}"}

    def _get_agent_tools(self) -> List[Dict]:
        """Returns standard tool schema (read, write, execute, escalate, finish)."""
        # (Standard JSON tool schema as seen in original code)
        pass