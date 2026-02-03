# -*- coding: utf-8 -*-
"""
execution_engine.py - Sequential Ralph Loop Implementation

✅ REWRITTEN: Follows snarktank/ralph pattern
  - Sequential story-by-story execution (not parallel)
  - Pick ONE story at a time (highest priority, passes: false)
  - Implement → Verify → Update PRD → Append progress → Repeat
  - Clean separation: each story = one LLM call

Connects Orchestrator → DeepSeek → Code Generation → Verification
"""

import asyncio
import logging
import json
import subprocess
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime
import re

logger = logging.getLogger("ExecutionEngine")


class ExecutionEngine:
    """
    Sequential Ralph Loop Engine.

    Implements:
    1. Pick next story (passes: false)
    2. Call DeepSeek to implement
    3. Save generated code
    4. Run verification command
    5. Update prd.json with result
    6. Append to progress.txt
    7. Repeat until all pass or max iterations
    """

    def __init__(
            self,
            project_dir: Path,
            deepseek_client,
            agent_coordinator,
    ):
        """Initialize execution engine"""
        self.project_dir = Path(project_dir)
        self.workspace_dir = self.project_dir / "workspace"
        self.output_dir = self.workspace_dir / "output" / "generated_code"
        self.prd_file = self.project_dir / "prd.json"
        self.progress_file = self.project_dir / "progress.txt"
        
        self.deepseek = deepseek_client
        self.coordinator = agent_coordinator
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ ExecutionEngine initialized (Sequential Ralph Loop)")
        logger.info(f"   Project: {self.project_dir}")
        logger.info(f"   Output: {self.output_dir}")

    async def execute(
            self,
            execution_id: str,
            orchestrator_prompt: str,
            prd_partitions: Dict[str, List[Dict[str, Any]]],
            num_agents: int,
            progress_callback: Optional[Callable] = None,
            log_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute sequential Ralph loop on PRD stories.

        Args:
            execution_id: Unique execution ID
            orchestrator_prompt: System prompt for agents
            prd_partitions: PRD stories partitioned by agent (we'll flatten and execute sequentially)
            num_agents: Number of agents (used for progress calculation)
            progress_callback: UI progress update function
            log_callback: UI log message function

        Returns:
            Execution results with status, completed/failed items
        """
        # 🔍 DEBUG: Log function entry IMMEDIATELY
        logger.info("🔍 [DEBUG] ExecutionEngine.execute() CALLED")
        logger.info(f"🔍 [DEBUG] execution_id: {execution_id}")
        logger.info(f"🔍 [DEBUG] num_agents: {num_agents}")
        logger.info(f"🔍 [DEBUG] log_callback type: {type(log_callback)}")
        logger.info(f"🔍 [DEBUG] progress_callback type: {type(progress_callback)}")
        logger.info(f"🔍 [DEBUG] log_callback is None: {log_callback is None}")
        logger.info(f"🔍 [DEBUG] progress_callback is None: {progress_callback is None}")
        
        # Store callbacks
        logger.info("🔍 [DEBUG] Setting up callbacks...")
        self.log_callback = log_callback or self._default_log
        self.progress_callback = progress_callback or self._default_progress
        logger.info(f"🔍 [DEBUG] self.log_callback set to: {type(self.log_callback)}")
        logger.info(f"🔍 [DEBUG] self.progress_callback set to: {type(self.progress_callback)}")
        
        # Test callback immediately
        logger.info("🔍 [DEBUG] Testing log callback...")
        try:
            await self._log("🔍 [TEST] Log callback test message")
            logger.info("🔍 [DEBUG] Log callback test succeeded")
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Log callback test FAILED: {e}", exc_info=True)
        
        try:
            logger.info("🔍 [DEBUG] Sending first real log message...")
            await self._log(f"🚀 Starting Sequential Ralph Loop")
            await self._log(f"   Execution ID: {execution_id}")
            await self._log("")
            
            logger.info("🔍 [DEBUG] Loading PRD...")
            # Load PRD
            await self._log("📋 Loading PRD from prd.json...")
            prd = self._load_prd()
            logger.info(f"🔍 [DEBUG] PRD loaded: {prd is not None}")
            
            if not prd:
                logger.error("🔍 [DEBUG] PRD is None - returning error")
                return {"status": "error", "error": "PRD file not found or empty"}
            
            total_stories = len(prd.get("user_stories", []))
            logger.info(f"🔍 [DEBUG] Total stories in PRD: {total_stories}")
            await self._log(f"   Total stories: {total_stories}")
            await self._log("")
            
            # Sequential loop
            iteration = 0
            max_iterations = total_stories * 2  # Allow retries
            completed_items = []
            failed_items = []
            
            logger.info("🔍 [DEBUG] Starting main loop...")
            await self._log("🔄 Starting story-by-story execution...")
            await self._log("")
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"🔍 [DEBUG] Loop iteration {iteration}/{max_iterations}")
                
                # Pick next story
                story = self._pick_next_story(prd)
                logger.info(f"🔍 [DEBUG] Picked story: {story.get('id') if story else 'None'}")
                
                if not story:
                    await self._log("✅ All stories completed!")
                    logger.info("🔍 [DEBUG] No more stories - breaking loop")
                    break
                
                story_id = story.get("id", f"story-{iteration}")
                story_title = story.get("title", "Untitled")
                
                logger.info(f"🔍 [DEBUG] Executing story: {story_id} - {story_title}")
                await self._log(f"📌 Iteration {iteration}: {story_id} - {story_title}")
                await self._log("")
                
                # Execute story
                logger.info(f"🔍 [DEBUG] Calling _execute_story for {story_id}...")
                result = await self._execute_story(
                    story=story,
                    orchestrator_prompt=orchestrator_prompt,
                    iteration=iteration
                )
                logger.info(f"🔍 [DEBUG] _execute_story returned: {result.get('status')}")
                
                # Update progress
                completed_count = len([s for s in prd["user_stories"] if s.get("status") == "done"])
                progress = (completed_count / total_stories) * 100
                logger.info(f"🔍 [DEBUG] Progress: {progress:.1f}% ({completed_count}/{total_stories})")
                await self._progress(progress)
                
                if result["status"] == "success":
                    completed_items.append({
                        "id": story_id,
                        "title": story_title,
                        "iteration": iteration,
                        "files": result.get("files", [])
                    })
                    await self._log(f"   ✅ Story {story_id} PASSED")
                    logger.info(f"🔍 [DEBUG] Story {story_id} marked as success")
                else:
                    failed_items.append({
                        "id": story_id,
                        "title": story_title,
                        "iteration": iteration,
                        "error": result.get("error", "Unknown error")
                    })
                    await self._log(f"   ❌ Story {story_id} FAILED: {result.get('error', 'Unknown')}")
                    logger.info(f"🔍 [DEBUG] Story {story_id} marked as failed")
                
                await self._log("")
                
                # Check if all done
                remaining = len([s for s in prd["user_stories"] if s.get("status") == "todo"])
                logger.info(f"🔍 [DEBUG] Remaining stories: {remaining}")
                if remaining == 0:
                    await self._log("🎉 All stories completed successfully!")
                    logger.info("🔍 [DEBUG] All done - breaking loop")
                    break
            
            # Final results
            logger.info("🔍 [DEBUG] Execution complete, generating summary...")
            await self._progress(100)
            await self._log("")
            await self._log("📊 Execution Summary:")
            await self._log(f"   ✅ Completed: {len(completed_items)}")
            await self._log(f"   ❌ Failed: {len(failed_items)}")
            await self._log(f"   🔄 Iterations: {iteration}")
            await self._log("")
            
            status = "success" if len(failed_items) == 0 else "partial" if len(completed_items) > 0 else "failed"
            logger.info(f"🔍 [DEBUG] Final status: {status}")
            
            return {
                "status": status,
                "execution_id": execution_id,
                "completed_items": completed_items,
                "failed_items": failed_items,
                "total_iterations": iteration,
                "agents": {"agent_1": {"completed_items": completed_items, "failed_items": failed_items}},
            }
            
        except Exception as e:
            logger.error(f"🔍 [DEBUG] EXCEPTION in execute(): {e}", exc_info=True)
            await self._log(f"❌ Execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "execution_id": execution_id,
            }

    async def _execute_story(
            self,
            story: Dict[str, Any],
            orchestrator_prompt: str,
            iteration: int
    ) -> Dict[str, Any]:
        """Execute a single PRD story"""
        story_id = story.get("id", "unknown")
        logger.info(f"🔍 [DEBUG] _execute_story called for {story_id}")
        
        try:
            # 1. Generate implementation prompt
            await self._log("   🤖 Calling DeepSeek to generate code...")
            logger.info(f"🔍 [DEBUG] Creating prompt for {story_id}...")
            
            user_prompt = self._create_story_prompt(story)
            logger.info(f"🔍 [DEBUG] Prompt created, length: {len(user_prompt)}")
            
            # Call DeepSeek
            logger.info(f"🔍 [DEBUG] Calling DeepSeek API...")
            response = await self._call_deepseek(
                system_prompt=orchestrator_prompt,
                user_prompt=user_prompt
            )
            logger.info(f"🔍 [DEBUG] DeepSeek returned {len(response)} chars")
            
            await self._log(f"   💬 Received response ({len(response)} chars)")
            
            # 2. Extract and save code
            await self._log("   💾 Saving generated code...")
            logger.info(f"🔍 [DEBUG] Extracting and saving code...")
            files = await self._save_code(story_id, response, iteration)
            await self._log(f"   📁 Saved {len(files)} files")
            logger.info(f"🔍 [DEBUG] Saved {len(files)} files")
            
            # 3. Run verification
            await self._log("   🧪 Running verification...")
            verification = story.get("verification", "")
            logger.info(f"🔍 [DEBUG] Verification command: {verification[:100] if verification else 'None'}")
            
            if verification and verification.strip():
                logger.info(f"🔍 [DEBUG] Running verification...")
                verify_result = await self._run_verification(verification)
                logger.info(f"🔍 [DEBUG] Verification result: {verify_result['success']}")
                
                if verify_result["success"]:
                    await self._log("   ✅ Verification PASSED")
                    # Update PRD
                    self._update_prd_story(story_id, status="done", passes=True)
                    # Append to progress
                    await self._append_progress(
                        story_id=story_id,
                        status="success",
                        learning=f"Completed {story.get('title', 'story')}. Verification passed."
                    )
                    logger.info(f"🔍 [DEBUG] {story_id} marked as success")
                    return {"status": "success", "files": files}
                else:
                    await self._log(f"   ❌ Verification FAILED: {verify_result['error']}")
                    # Update PRD with error
                    story["attempts"] = story.get("attempts", 0) + 1
                    story["errors"] = story.get("errors", []) + [verify_result["error"]]
                    self._update_prd_story(story_id, status="failed", passes=False, attempts=story["attempts"], errors=story["errors"])
                    # Append to progress
                    await self._append_progress(
                        story_id=story_id,
                        status="failed",
                        learning=f"Failed verification: {verify_result['error']}"
                    )
                    logger.info(f"🔍 [DEBUG] {story_id} marked as failed")
                    return {"status": "failed", "error": verify_result["error"], "files": files}
            else:
                # No verification command - assume success
                await self._log("   ⚠️  No verification command specified, marking as done")
                logger.info(f"🔍 [DEBUG] No verification for {story_id}, marking as done")
                self._update_prd_story(story_id, status="done", passes=True)
                await self._append_progress(
                    story_id=story_id,
                    status="success",
                    learning=f"Completed {story.get('title', 'story')} (no verification)."
                )
                return {"status": "success", "files": files}
                
        except Exception as e:
            logger.error(f"🔍 [DEBUG] EXCEPTION in _execute_story for {story_id}: {e}", exc_info=True)
            await self._log(f"   ❌ Error: {str(e)}")
            self._update_prd_story(story_id, status="failed", passes=False)
            return {"status": "error", "error": str(e)}

    def _create_story_prompt(self, story: Dict[str, Any]) -> str:
        """Create prompt for implementing a single story"""
        story_id = story.get("id", "unknown")
        title = story.get("title", "Untitled")
        description = story.get("description", "")
        acceptance = "\n".join([f"  - {c}" for c in story.get("acceptance_criteria", [])])
        why = story.get("why", "")
        files = ", ".join(story.get("files_touched", []))
        
        return f"""Implement the following PRD story:

**Story ID**: {story_id}
**Title**: {title}

**Description**:
{description}

**Why**: {why}

**Acceptance Criteria**:
{acceptance}

**Files to create/modify**: {files}

**Instructions**:
1. Generate complete, working code for this story
2. Include all necessary imports, type hints, and docstrings
3. Add error handling and logging where appropriate
4. Ensure code is production-ready and follows best practices
5. Format code in markdown code blocks with language tags
6. Include file paths as comments at the top of each code block

**Output Format**:
```python
# File: path/to/file.py
# Your code here
```

Generate the implementation now."""

    async def _call_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        """Call DeepSeek API"""
        logger.info("🔍 [DEBUG] _call_deepseek entered")
        try:
            if not self.deepseek:
                logger.error("🔍 [DEBUG] DeepSeek client is None!")
                return """# Error: DeepSeek client not initialized
print("DeepSeek client is None - cannot generate code")
"""
            
            logger.info("🔍 [DEBUG] DeepSeek client exists, calling chat()...")
            # Call DeepSeek (async)
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.deepseek.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000,
                )
            )
            
            logger.info(f"🔍 [DEBUG] DeepSeek response received: {type(response)}")
            return response.get("content", "")
            
        except Exception as e:
            logger.error(f"🔍 [DEBUG] EXCEPTION in _call_deepseek: {e}", exc_info=True)
            return f"# Error calling DeepSeek: {str(e)}"

    async def _save_code(self, story_id: str, response: str, iteration: int) -> List[str]:
        """Extract code blocks from response and save to files"""
        logger.info(f"🔍 [DEBUG] _save_code called for {story_id}")
        saved_files = []
        
        # Extract code blocks: ```language\ncode\n```
        code_blocks = re.findall(r"```(\w+)?\n(.*?)\n```", response, re.DOTALL)
        logger.info(f"🔍 [DEBUG] Found {len(code_blocks)} code blocks")
        
        if not code_blocks:
            # No code blocks found, save entire response as text
            fallback_file = self.output_dir / f"{story_id}_iter{iteration}.txt"
            fallback_file.write_text(response, encoding="utf-8")
            saved_files.append(str(fallback_file))
            logger.info(f"🔍 [DEBUG] No code blocks, saved as: {fallback_file}")
            return saved_files
        
        for block_idx, (lang, code) in enumerate(code_blocks, 1):
            # Try to extract file path from comment
            file_path = self._extract_file_path(code)
            
            if not file_path:
                # Infer from code content or use default
                file_path = self._infer_filename(code, lang, story_id, block_idx)
            
            # Save relative to project root (not output_dir)
            target_file = self.project_dir / file_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                target_file.write_text(code, encoding="utf-8")
                saved_files.append(str(target_file))
                await self._log(f"      💾 {file_path}")
                logger.info(f"🔍 [DEBUG] Saved file: {file_path}")
            except Exception as e:
                logger.error(f"🔍 [DEBUG] Error saving {file_path}: {e}")
        
        return saved_files

    def _extract_file_path(self, code: str) -> Optional[str]:
        """Extract file path from code comment (# File: path/to/file.py)"""
        match = re.search(r"^#\s*[Ff]ile:\s*(.+?)\s*$", code, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    def _infer_filename(self, code: str, lang: str, story_id: str, block_idx: int) -> str:
        """Infer filename from code content"""
        # Python class
        match = re.search(r"^class\s+(\w+)", code, re.MULTILINE)
        if match:
            return f"src/{match.group(1).lower()}.py"
        
        # Python function
        match = re.search(r"^def\s+(\w+)", code, re.MULTILINE)
        if match:
            return f"src/{match.group(1)}.py"
        
        # Fallback
        lang_ext = {"python": "py", "javascript": "js", "typescript": "ts", "sql": "sql", "bash": "sh", "shell": "sh"}
        ext = lang_ext.get(lang, "txt")
        return f"src/{story_id}_block{block_idx}.{ext}"

    async def _run_verification(self, command: str) -> Dict[str, Any]:
        """Run verification command from PRD"""
        logger.info(f"🔍 [DEBUG] _run_verification: {command}")
        try:
            # Run in project directory
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.project_dir),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            )
            
            logger.info(f"🔍 [DEBUG] Verification exit code: {result.returncode}")
            
            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {
                    "success": False,
                    "error": f"Command failed with code {result.returncode}: {result.stderr[:200]}"
                }
                
        except subprocess.TimeoutExpired:
            logger.error("🔍 [DEBUG] Verification timed out")
            return {"success": False, "error": "Verification timed out (60s)"}
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Verification exception: {e}", exc_info=True)
            return {"success": False, "error": f"Verification error: {str(e)}"}

    def _load_prd(self) -> Optional[Dict[str, Any]]:
        """Load PRD from prd.json"""
        logger.info(f"🔍 [DEBUG] _load_prd: {self.prd_file}")
        logger.info(f"🔍 [DEBUG] PRD file exists: {self.prd_file.exists()}")
        
        if not self.prd_file.exists():
            logger.error(f"🔍 [DEBUG] PRD file not found: {self.prd_file}")
            return None
        
        try:
            with open(self.prd_file, 'r', encoding='utf-8') as f:
                prd = json.load(f)
            logger.info(f"🔍 [DEBUG] PRD loaded successfully, {len(prd.get('user_stories', []))} stories")
            return prd
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Error loading PRD: {e}", exc_info=True)
            return None

    def _pick_next_story(self, prd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pick next story to execute (first with status='todo')"""
        for story in prd.get("user_stories", []):
            if story.get("status") == "todo":
                logger.info(f"🔍 [DEBUG] Picked story: {story.get('id')}")
                return story
        logger.info("🔍 [DEBUG] No more 'todo' stories found")
        return None

    def _update_prd_story(self, story_id: str, status: str, passes: bool, attempts: int = 0, errors: List[str] = None):
        """Update PRD story status"""
        logger.info(f"🔍 [DEBUG] _update_prd_story: {story_id} → {status}")
        try:
            prd = self._load_prd()
            if not prd:
                return
            
            for story in prd["user_stories"]:
                if story.get("id") == story_id:
                    story["status"] = status
                    if attempts > 0:
                        story["attempts"] = attempts
                    if errors:
                        story["errors"] = errors
                    break
            
            # Save updated PRD
            with open(self.prd_file, 'w', encoding='utf-8') as f:
                json.dump(prd, f, indent=2)
            logger.info(f"🔍 [DEBUG] PRD updated successfully")
            
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Error updating PRD: {e}", exc_info=True)

    async def _append_progress(self, story_id: str, status: str, learning: str):
        """Append to progress.txt (Ralph pattern)"""
        logger.info(f"🔍 [DEBUG] _append_progress: {story_id} - {status}")
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"\n[{timestamp}] {story_id} - {status.upper()}\n{learning}\n"
            
            with open(self.progress_file, 'a', encoding='utf-8') as f:
                f.write(entry)
            logger.info("🔍 [DEBUG] Progress appended successfully")
                
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Error appending to progress: {e}", exc_info=True)

    async def _log(self, message: str):
        """Log message to UI"""
        try:
            # Always log to console first
            logger.info(f"[UI LOG] {message}")
            
            if asyncio.iscoroutinefunction(self.log_callback):
                await self.log_callback(message)
            else:
                self.log_callback(message)
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Log callback error: {e}", exc_info=True)

    async def _progress(self, value: float):
        """Update progress bar"""
        try:
            logger.info(f"[UI PROGRESS] {value:.1f}%")
            
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(value)
            else:
                self.progress_callback(value)
        except Exception as e:
            logger.error(f"🔍 [DEBUG] Progress callback error: {e}", exc_info=True)

    def _default_log(self, message: str):
        """Default log (fallback)"""
        logger.info(f"[DEFAULT LOG] {message}")

    def _default_progress(self, value: float):
        """Default progress (fallback)"""
        logger.info(f"[DEFAULT PROGRESS] {value:.1f}%")