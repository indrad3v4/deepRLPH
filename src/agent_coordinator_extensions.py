# -*- coding: utf-8 -*-
"""
agent_coordinator_extensions.py - PRD Agent Assignment Extensions (Guru Edition)

Upgrades:
- Transactional Stability: Strict iteration limits and memory management.
- Stage 2 Isolation: Guardrail-aware command execution.
- Deterministic Handoff: explicit human intervention triggers.
- Full Tool Schema: read_file, write_file, execute_command, request_human_intervention, finish_task.
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
        """
        Drive a single PRD item to completion using a tool-using Deepseek agent.
        """

        async def _log(msg: str):
            line = f"[{agent_id}] {msg}"
            logger.info(line)
            if log_callback:
                if asyncio.iscoroutinefunction(log_callback):
                    await log_callback(line)
                else:
                    # tolerate sync callbacks
                    try:
                        log_callback(line)
                    except Exception:
                        logger.exception("Log callback failed")

        await _log(f"Assigning {item.item_id}: {item.title}")

        # === o1 Prompt Anatomy: System Prompt ==================================
        # GOAL / RETURN FORMAT / WARNINGS / CONTEXT DUMP
        acc_lines = "\n".join(f"- {c}" for c in (item.acceptance_criteria or []))
        system_prompt = f"""### GOAL
Implement PRD item {item.item_id}: {item.title}.
Modify the codebase in the project workspace so that all acceptance criteria and the verification command for this item pass.

### RETURN FORMAT
Use ONLY tool calls (`read_file`, `write_file`, `execute_command`, `request_human_intervention`, `finish_task`).
Write or edit real files under the project root, then run the official verification command:
`{item.verification_command}`.
When the item is truly implemented and verified, call `finish_task` with a short JSON summary:
- changed_files: list of paths
- tests_run: list of commands
- rationale: brief explanation.

### WARNINGS
Be careful to:
- Keep all logic strictly causal and deterministic where applicable; never let past predictions depend on future data.
- Avoid writing outside the project root and NEVER use destructive shell commands on parent directories.
- Do not invent dataset paths, API endpoints, or external services; rely only on files actually present in the workspace.
- Keep edits minimal and local; prefer modifying existing modules over creating many new ones.

### CONTEXT DUMP
Project root: {project_dir}
PRD acceptance criteria for {item.item_id}:
{acc_lines or "- (none provided)"}

Orchestrator guidance:
{orchestrator_prompt}
"""
        # Initial message to kick off the loop.
        messages = [
            {
                "role": "user",
                "content": "Analyze the GOAL / RETURN FORMAT / WARNINGS / CONTEXT DUMP and start the Implementation Phase using tools.",
            }
        ]

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
                    temperature=0.2,  # Lowered for implementation consistency
                )

                if response.get("status") != "success":
                    return {
                        "status": "error",
                        "error": response.get("error"),
                        "files_created": list(files_created),
                    }

                message = response.get("message", {})  # agent reply
                messages.append(message)

                # Optional: trace a bit of the agent's thought
                content_preview = (message.get("content") or "")[:240].replace("\n", " ")
                if content_preview:
                    await _log(f"Thought: {content_preview}")

                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    # Nudge the agent to actually use tools
                    messages.append(
                        {
                            "role": "user",
                            "content": "Continue implementation using the available tools or call finish_task when done.",
                        }
                    )
                    continue

                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    raw_args = tool_call["function"].get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}
                    await _log(
                        f"→ Calling tool '{func_name}' with args: {json.dumps(args, ensure_ascii=False)[:200]}"
                    )

                    # Tool Execution Logic
                    result = await self._dispatch_tool(
                        func_name, args, project_dir, item, files_created, _log
                    )

                    # Sequential Logic for Early Termination
                    if result.get("terminate"):
                        return result["payload"]

                    # Feed tool result back to the agent
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": func_name,
                            "content": str(result.get("content", ""))[:5000],
                        }
                    )

            # Hard stop: hit iteration limit
            await _log("Max iterations reached; returning partial success.")
            return {
                "status": "success",
                "response": "Max iterations reached.",
                "files_created": list(files_created),
            }

        except Exception as e:
            await _log(f"Agent crashed with exception: {e}")
            return {
                "status": "error",
                "error": f"Agent crash: {str(e)}",
                "files_created": list(files_created),
            }

    # --------------------------------------------------------------------- #
    # Tool Dispatch
    # --------------------------------------------------------------------- #

    async def _dispatch_tool(
        self,
        func_name: str,
        args: Dict[str, Any],
        project_dir: Path,
        item: PRDItem,
        files_created: Set[str],
        log_fn,
    ) -> Dict[str, Any]:
        """Isolated tool dispatcher with Stage 2 Guardrails."""
        func_name = func_name or ""

        # --- execute_command -------------------------------------------------
        if func_name == "execute_command":
            cmd = args.get("command", "")

            # Guardrail: block obviously dangerous deletes outside project_dir
            if "rm " in cmd and str(project_dir) not in cmd:
                await log_fn("🚫 BLOCKED: Unsafe delete detected outside project root.")
                return {
                    "content": "🚫 BLOCKED: Unsafe delete detected outside project root.",
                    "terminate": False,
                }

            res = await self._run_command_safe(cmd, project_dir, log_fn)

            # If this was the official verification command and it passed — we are done
            if res["success"] and cmd.strip() == (item.verification_command or "").strip():
                await log_fn("🎯 Verification Passed. Auto-terminating story.")
                summary = {
                    "changed_files": sorted(list(files_created)),
                    "tests_run": [cmd],
                    "rationale": f"Verification command `{cmd}` succeeded for {item.item_id}.",
                }
                return {
                    "terminate": True,
                    "payload": {
                        "status": "success",
                        "files_created": list(files_created),
                        "summary": summary,
                    },
                }

            return {"content": res["output"], "terminate": False}

        # --- read_file -------------------------------------------------------
        if func_name == "read_file":
            rel_path = args.get("path", "")
            if not rel_path:
                return {"content": "read_file: missing 'path' argument", "terminate": False}

            abs_path = (project_dir / rel_path).resolve()
            root = project_dir.resolve()
            if not str(abs_path).startswith(str(root)):
                await log_fn(f"🚫 read_file blocked outside project root: {abs_path}")
                return {
                    "content": "🚫 read_file blocked: path must stay within project root.",
                    "terminate": False,
                }

            try:
                text = abs_path.read_text(encoding="utf-8")
                await log_fn(f"read_file OK: {rel_path} ({len(text)} chars)")
                return {"content": text[:8000], "terminate": False}
            except FileNotFoundError:
                await log_fn(f"read_file: not found: {rel_path}")
                return {"content": f"File not found: {rel_path}", "terminate": False}
            except Exception as e:
                await log_fn(f"read_file error on {rel_path}: {e}")
                return {"content": f"Error reading {rel_path}: {e}", "terminate": False}

        # --- write_file ------------------------------------------------------
        if func_name == "write_file":
            rel_path = args.get("path", "")
            content = args.get("content", "")
            if not rel_path:
                return {"content": "write_file: missing 'path' argument", "terminate": False}

            abs_path = (project_dir / rel_path).resolve()
            root = project_dir.resolve()
            if not str(abs_path).startswith(str(root)):
                await log_fn(f"🚫 write_file blocked outside project root: {abs_path}")
                return {
                    "content": "🚫 write_file blocked: path must stay within project root.",
                    "terminate": False,
                }

            try:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(content, encoding="utf-8")
                files_created.add(str(abs_path.relative_to(root)))
                await log_fn(
                    f"write_file OK: {rel_path} ({len(content.splitlines())} lines, {len(content)} chars)"
                )
                return {
                    "content": f"write_file OK: {rel_path}",
                    "terminate": False,
                }
            except Exception as e:
                await log_fn(f"write_file error on {rel_path}: {e}")
                return {"content": f"Error writing {rel_path}: {e}", "terminate": False}

        # --- request_human_intervention --------------------------------------
        if func_name == "request_human_intervention":
            message = args.get("message", "")
            await log_fn(f"🆘 Human intervention requested: {message}")
            summary = {
                "changed_files": sorted(list(files_created)),
                "tests_run": [],
                "rationale": f"Agent requested human help: {message}",
            }
            return {
                "terminate": True,
                "payload": {
                    "status": "needs_human",
                    "files_created": list(files_created),
                    "summary": summary,
                },
            }

        # --- finish_task -----------------------------------------------------
        if func_name == "finish_task":
            summary = args.get("summary", {})
            await log_fn(f"✅ finish_task called with summary: {json.dumps(summary, ensure_ascii=False)[:300]}")
            return {
                "terminate": True,
                "payload": {
                    "status": "success",
                    "files_created": list(files_created),
                    "summary": summary,
                },
            }

        # --- Fallback for unknown tools --------------------------------------
        await log_fn(f"Unknown tool '{func_name}' executed with args={args}")
        return {"content": f"Unknown tool '{func_name}'", "terminate": False}

    async def _run_command_safe(self, cmd: str, project_dir: Path, log_fn):
        """Execution helper with 120s timeout."""
        await log_fn(f"Executing: {cmd}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_dir)
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(project_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            output = stdout.decode().strip() + "\n" + stderr.decode().strip()
            return {"success": proc.returncode == 0, "output": output}
        except asyncio.TimeoutError:
            await log_fn("execute_command timeout after 120s")
            return {"success": False, "output": "Timeout after 120s"}
        except Exception as e:
            await log_fn(f"execute_command error: {e}")
            return {"success": False, "output": f"Error: {e}"}

    # --------------------------------------------------------------------- #
    # Tool Schema
    # --------------------------------------------------------------------- #

    def _get_agent_tools(self) -> List[Dict[str, Any]]:
        """
        Returns standard tool schema (read, write, execute, escalate, finish).

        The schema is compatible with Deepseek's call_agent_with_tools interface.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a UTF-8 text file from the project workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path from the project root, e.g. 'src/model.py'.",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or overwrite a UTF-8 text file inside the project workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path from the project root.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Full file contents to write.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Run a shell command inside the project workspace (typically pytest or python).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute, e.g. 'python -m pytest tests/test_causal_lob_forecaster.py -v'.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "request_human_intervention",
                    "description": "Escalate to a human when blocked or when strategic decisions are needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Short explanation of why human help is needed and what decision is required.",
                            }
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "description": "Signal that the PRD item is fully implemented and verified.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "object",
                                "description": "JSON summary: changed_files, tests_run, rationale.",
                            }
                        },
                        "required": ["summary"],
                    },
                },
            },
        ]
