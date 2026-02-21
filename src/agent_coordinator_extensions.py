# -*- coding: utf-8 -*-
"""
agent_coordinator_extensions.py - PRD Agent Assignment Extensions

ITEM-004: Agent Assignment Protocol

Extensions to AgentCoordinator for assigning PRD items to agents.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from prd_model import PRDItem

logger = logging.getLogger(__name__)


class PRDAgentAssignmentMixin:
    """Mixin for AgentCoordinator to add PRD item assignment.

    Usage:
        class AgentCoordinator(PRDAgentAssignmentMixin, ...):
            ...

    This mixin provides assign_prd_item() method that executes the
    autonomous 'Ralph Loop' by providing the LLM with Native Tools.
    """

    async def assign_prd_item(
            self,
            item: PRDItem,
            agent_id: str,
            project_dir: Path,
            deepseek_client: Any,
            log_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Assign PRD item to agent for implementation using a tool-calling loop.

        Args:
            item: PRD item to implement
            agent_id: ID of agent (e.g., "agent_1")
            project_dir: Project directory path
            deepseek_client: DeepSeek client instance
            log_callback: UI log callback

        Returns:
            Dict with status, files_created, and response/error.
        """

        async def _log(msg: str):
            logger.info(f"[{agent_id}] {msg}")
            if log_callback:
                if asyncio.iscoroutinefunction(log_callback):
                    await log_callback(f"[{agent_id}] {msg}")
                else:
                    log_callback(f"[{agent_id}] {msg}")

        await _log(f"Assigning {item.item_id}: {item.title}")

        if not deepseek_client:
            return {
                "status": "error",
                "error": "DeepSeek client not initialized",
                "files_created": [],
            }

        prompt = self._generate_prd_item_prompt(item)

        system_prompt = (
            "You are a senior software engineer operating in an autonomous loop (Ralph Loop). "
            "You MUST use the provided tools to read files, write code, and run tests. "
            "Your workflow: "
            "1. Read necessary context (read_file). "
            "2. Write implementation and tests (write_file). "
            "3. Run the verification command (execute_command). "
            "4. If verification fails, read the error and FIX your code. "
            "5. You cannot finish until verification passes. "
            "6. Call finish_task ONLY when you are completely done and tests pass."
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        # Define the tools the agent can use natively
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file in the project",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string", "description": "Path relative to project root"}
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file. Overwrites if exists. Creates directories if needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string", "description": "Path relative to project root"},
                            "content": {"type": "string", "description": "Full file content"}
                        },
                        "required": ["filepath", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a shell command in the project root (e.g. pytest, python script.py)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The bash command to run"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "description": "Mark the task as completely finished. Call this ONLY after tests pass.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Summary of what was done"}
                        },
                        "required": ["summary"]
                    }
                }
            }
        ]

        max_iterations = 15
        files_created = set()

        try:
            # The True Ralph Loop 🔄
            for iteration in range(max_iterations):
                await _log(f"Loop iteration {iteration + 1}/{max_iterations}...")

                # We will implement this new method in deepseek_client.py next!
                response = await deepseek_client.call_agent_with_tools(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    temperature=0.3
                )

                if response.get("status") != "success":
                    await _log(f"❌ API Error: {response.get('error')}")
                    return {"status": "error", "error": response.get("error"), "files_created": list(files_created)}

                message = response.get("message", {})
                messages.append(message)  # Add assistant's message to history

                # Check if the agent called any tools
                if not message.get("tool_calls"):
                    await _log("⚠️ Agent didn't call any tools, nudging...")
                    messages.append({
                        "role": "user",
                        "content": "You must use the provided tools to interact with the project. If you are done, use finish_task."
                    })
                    continue

                finished = False

                # Execute the tools requested by the agent
                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]
                    call_id = tool_call["id"]

                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = {}

                    await _log(f"🛠️ Tool call: {func_name}")
                    tool_result = ""

                    if func_name == "read_file":
                        filepath = project_dir / args.get("filepath", "")
                        try:
                            if filepath.exists() and filepath.is_file():
                                tool_result = filepath.read_text(encoding="utf-8")
                            else:
                                tool_result = f"Error: File {filepath.name} does not exist."
                        except Exception as e:
                            tool_result = f"Error reading file: {e}"

                    elif func_name == "write_file":
                        filepath = project_dir / args.get("filepath", "")
                        content = args.get("content", "")
                        try:
                            filepath.parent.mkdir(parents=True, exist_ok=True)
                            filepath.write_text(content, encoding="utf-8")
                            files_created.add(str(filepath))
                            tool_result = f"Successfully wrote to {args.get('filepath')}"
                            await _log(f"   💾 Saved {args.get('filepath')}")
                        except Exception as e:
                            tool_result = f"Error writing file: {e}"

                    elif func_name == "execute_command":
                        cmd = args.get("command", "")
                        await _log(f"   💻 Executing: {cmd}")
                        try:
                            proc = await asyncio.create_subprocess_shell(
                                cmd,
                                cwd=str(project_dir),
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
                            out_str = stdout.decode('utf-8', errors='replace')
                            err_str = stderr.decode('utf-8', errors='replace')

                            tool_result = f"Exit code: {proc.returncode}\n"
                            if out_str: tool_result += f"STDOUT:\n{out_str}\n"
                            if err_str: tool_result += f"STDERR:\n{err_str}\n"

                            if proc.returncode == 0:
                                await _log(f"   ✅ Command passed")
                            else:
                                await _log(f"   ❌ Command failed (code {proc.returncode})")

                        except asyncio.TimeoutError:
                            tool_result = "Error: Command timed out after 120 seconds."
                        except Exception as e:
                            tool_result = f"Error executing command: {e}"

                    elif func_name == "finish_task":
                        finished = True
                        summary = args.get("summary", "")
                        await _log(f"   🏁 Task finished: {summary}")
                        tool_result = "Task marked as finished."

                    else:
                        tool_result = f"Unknown tool: {func_name}"

                    # Feed the result of the tool back into the conversation context
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": str(tool_result)[:4000]  # Truncate to avoid context limits
                    })

                if finished:
                    return {
                        "status": "success",
                        "files_created": list(files_created),
                        "response": "Task finished successfully via tool call."
                    }

            await _log(f"⚠️ Reached max iterations ({max_iterations}) without finish_task.")
            return {
                "status": "error",
                "error": "Max loop iterations reached without completing the task.",
                "files_created": list(files_created)
            }

        except Exception as e:
            error = f"Agent loop error: {str(e)}"
            logger.exception(f"[{agent_id}] Assignment failed")
            return {
                "status": "error",
                "error": error,
                "files_created": list(files_created),
            }

    def _generate_prd_item_prompt(self, item: PRDItem) -> str:
        """Generate prompt for PRD item implementation."""
        acceptance = "\n".join([f"  - {c}" for c in item.acceptance_criteria])
        files = ", ".join(item.files_touched) if item.files_touched else "(to be determined)"

        return f"""Implement the following PRD item:

**Item ID**: {item.item_id}
**Title**: {item.title}
**Priority**: {item.priority}

**Acceptance Criteria**:
{acceptance}

**Verification Command**: {item.verification_command}

**Files to Modify**: {files}

Use your tools to write the code and run the verification command. You MUST run `{item.verification_command}` using the `execute_command` tool.
If it fails, read the error and correct your code using `write_file`.
Only call `finish_task` when the verification command succeeds and all acceptance criteria are met!
"""
