# -*- coding: utf-8 -*-
"""
agent_coordinator_extensions.py - PRD Agent Assignment Extensions

ITEM-004: Agent Assignment Protocol
Upgraded with OpenAI Guardrails & Human Intervention Patterns.
"""

import asyncio
import logging
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

from prd_model import PRDItem

logger = logging.getLogger(__name__)


class PRDAgentAssignmentMixin:
    """Mixin for AgentCoordinator to add PRD item assignment."""

    async def assign_prd_item(
            self,
            item: PRDItem,
            agent_id: str,
            project_dir: Path,
            deepseek_client: Any,
            log_callback: Optional[callable] = None,
            orchestrator_prompt: str = "",  # 🆕 THE MISSING WIRE RECONNECTED!
    ) -> Dict[str, Any]:

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

        file_prompt = self._generate_prd_item_prompt(item, project_dir)

        # 🚀 OpenAI Pattern: Structured Routine & Strict Constraints
        system_prompt = f"""{orchestrator_prompt}

{file_prompt}

### AGENTIC ROUTINE
You are an elite AI worker. You must execute your work through this strict routine:
1. **Contextualize**: Use `read_file` to audit the existing codebase.
2. **Implement**: Use `write_file` to write production-grade code.
3. **Verify**: Use `execute_command` to run the exact verification command.
4. **Escalate**: If verification fails 3 times, DO NOT guess blindly. Use the `request_human_intervention` tool to escalate.
5. **Complete**: If verification passes, use `finish_task`.

### STRICT CONSTRAINTS
- STRICT DOCKER PATHING: In solution.py, load the model using strictly: `onnx_path = 'model.onnx'`.
- FLAT PACKAGING: The final submission must be exactly 3 files at root.
- PATH INJECTION: Every Python file MUST start with: `import sys, os; sys.path.insert(0, os.path.abspath('.'))`
- GUARDRAIL AWARENESS: Heavy commands (like model training) will be automatically blocked by the system unless you append `--test_mode` or `--fast_dev_run`.
"""

        messages = [
            {"role": "user", "content": "Begin execution."}
        ]

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
                    "description": "Execute a shell command in the project root",
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
                    "name": "request_human_intervention",
                    "description": "Escalate to the Human Strategist if you are stuck in a loop or cannot resolve an error.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Detailed technical reason for escalation"}
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "description": "Mark the task as completely finished.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Summary of changes"}
                        },
                        "required": ["summary"]
                    }
                }
            }
        ]

        max_iterations = 30
        files_created = set()

        try:
            for iteration in range(max_iterations):
                await _log(f"Loop iteration {iteration + 1}/{max_iterations}...")

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
                messages.append(message)

                if not message.get("tool_calls"):
                    await _log("⚠️ Agent didn't call any tools, nudging...")
                    messages.append({
                        "role": "user",
                        "content": "You must use the provided tools. Use finish_task if done, or request_human_intervention if stuck."
                    })
                    continue

                finished = False

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

                        # ==========================================
                        # 🚀 OPENAI PATTERN: TOOL SAFEGUARDS (GUARDRAILS)
                        # ==========================================
                        if "train.py" in cmd and "--test_mode" not in cmd and "--fast_dev_run" not in cmd:
                            tool_result = (
                                "🚫 GUARDRAIL BLOCKED: You attempted to run a full training loop without "
                                "a test flag. This will cause a 600s timeout. You MUST append '--test_mode' "
                                "or '--fast_dev_run' to bypass heavy data loading."
                            )
                            await _log(f"   🛡️ Guardrail intercepted unsafe command: {cmd}")
                        else:
                            if cmd.startswith("pytest "):
                                cmd = "python -m " + cmd

                            await _log(f"   💻 Executing: {cmd}")
                            try:
                                env = os.environ.copy()
                                env["PYTHONPATH"] = str(project_dir)

                                proc = await asyncio.create_subprocess_shell(
                                    cmd,
                                    cwd=str(project_dir),
                                    env=env,
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
                                    clean_cmd = cmd.replace("python -m ", "").strip()
                                    expected_cmd = item.verification_command.replace("python -m ", "").strip()

                                    if clean_cmd == expected_cmd:
                                        await _log("   🎯 Official Verification Command passed! Auto-terminating loop.")
                                        return {
                                            "status": "success",
                                            "files_created": list(files_created),
                                            "response": "Auto-terminated due to successful official verification."
                                        }
                                else:
                                    await _log(f"   ❌ Command failed (code {proc.returncode})")

                            except asyncio.TimeoutError:
                                tool_result = "Error: Command timed out after 120 seconds."
                            except Exception as e:
                                tool_result = f"Error executing command: {e}"

                    # ==========================================
                    # 🚀 OPENAI PATTERN: HUMAN INTERVENTION HANDOFF
                    # ==========================================
                    elif func_name == "request_human_intervention":
                        reason = args.get("reason", "No reason provided")
                        await _log(f"   🙋‍♂️ Escalating to Human: {reason}")
                        return {
                            "status": "failed",  # Triggers the HITL UI logic
                            "error": f"HUMAN INTERVENTION REQUESTED: {reason}",
                            "files_created": list(files_created)
                        }

                    elif func_name == "finish_task":
                        finished = True
                        summary = args.get("summary", "")
                        await _log(f"   🏁 Task finished: {summary}")
                        tool_result = "Task marked as finished."

                    else:
                        tool_result = f"Unknown tool: {func_name}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": str(tool_result)[:6000]
                    })

                if finished:
                    return {
                        "status": "success",
                        "files_created": list(files_created),
                        "response": "Task finished successfully via tool call."
                    }

            await _log(f"⚠️ Reached max iterations ({max_iterations}). Forcing evaluation of generated code.")
            return {
                "status": "success",
                "response": "Max iterations reached. Proceeding to official validation.",
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

    def _generate_prd_item_prompt(self, item: PRDItem, project_dir: Path) -> str:
        acceptance = "\n".join([f"  - {c}" for c in item.acceptance_criteria])
        files = ", ".join(item.files_touched) if item.files_touched else "(to be determined)"

        try:
            tree_paths = []
            for root, dirs, filenames in os.walk(project_dir):
                if 'venv' in root or '.git' in root or '__pycache__' in root:
                    continue
                level = root.replace(str(project_dir), '').count(os.sep)
                if level < 3:
                    indent = ' ' * 4 * (level)
                    tree_paths.append(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 4 * (level + 1)
                    for f in filenames:
                        if f.endswith('.py') or f.endswith('.parquet') or f.endswith('.md'):
                            tree_paths.append(f"{subindent}{f}")
            tree_output = "\n".join(tree_paths)
        except Exception:
            tree_output = "(Directory structure unavailable)"

        existing_context = ""
        if item.files_touched:
            existing_context = "\n### CURRENT FILE CONTENTS ###\n"
            for f in item.files_touched:
                fpath = project_dir / f
                if fpath.exists():
                    try:
                        content = fpath.read_text(encoding='utf-8')
                        existing_context += f"\n--- {f} ---\n```python\n{content}\n```\n"
                    except Exception:
                        pass

        return f"""**Item ID**: {item.item_id}
**Title**: {item.title}

**Acceptance Criteria**:
{acceptance}

**Verification Command**: `{item.verification_command}`

**Files to Modify**: {files}

### WORKSPACE STRUCTURE ###
```text
{tree_output}
{existing_context}
"""