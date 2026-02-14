# -*- coding: utf-8 -*-
"""
execution_engine.py - Multi-agent Ralph Loop (ENHANCED)

Adds:
- Asyncio-based worker pool with shared work queue (work-stealing style)
- Per-agent DeepSeek system prompts from AgentProfile
- Story-level KPI "rewards" recorded back into AgentProfileStore
- BE-008: JSONL trace logging for all execution events
- BE-009: Multi-pattern metric extraction
"""

import asyncio
import logging
import json
import subprocess
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime
import re
import math

from agent_profiles import AgentProfileStore, AgentProfile

logger = logging.getLogger("ExecutionEngine")


class ExecutionEngine:
    """Multi-agent Ralph Loop Engine with domain metrics + trace logging.

    Responsibilities:
    1. Distribute TODO stories across N asyncio agents via a shared work queue
    2. For each story: call DeepSeek with per-agent system+user prompts
    3. Save generated code
    4. Run verification command
    5. **BE-009:** Extract metrics using multiple regex patterns
    6. Update prd.json and metrics_history.json
    7. **BE-008:** Log all events to JSONL trace
    8. Append to progress.txt
    9. Record per-story reward into AgentProfileStore (RL-like learning)
    """

    def __init__(
        self,
        project_dir: Path,
        deepseek_client,
        agent_coordinator,
    ):
        self.project_dir = Path(project_dir)
        self.workspace_dir = self.project_dir / "workspace"
        self.output_dir = self.workspace_dir / "output" / "generated_code"
        self.prd_file = self.project_dir / "prd.json"
        self.progress_file = self.project_dir / "progress.txt"
        # Per-project metric config + history
        self.metrics_config_file = self.project_dir / "metrics_config.json"
        self.metrics_history_file = self.project_dir / "metrics_history.json"

        self.deepseek = deepseek_client
        self.coordinator = agent_coordinator

        # BE-008: Trace logger (set by orchestrator)
        self.trace_logger = None

        # PR-Agents: persistent per-project agent profiles
        self.agent_profile_store: Optional[AgentProfileStore] = AgentProfileStore.load_for_project(
            self.project_dir
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metric config if exists
        self.metric_config = self._load_metrics_config()

        logger.info("✅ ExecutionEngine initialized (Multi-agent Ralph Loop + metrics + traces)")
        logger.info(f"   Project: {self.project_dir}")
        logger.info(f"   Output: {self.output_dir}")
        if self.metric_config:
            logger.info(
                "   Metric target: %s >= %s",
                self.metric_config.get("name"),
                self.metric_config.get("target"),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        execution_id: str,
        orchestrator_prompt: str,
        prd_partitions: Dict[str, List[Dict[str, Any]]],
        num_agents: int,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Run the multi-agent Ralph loop until stories complete.

        NOTE: prd_partitions is currently ignored; we operate on prd.json and
        distribute TODO stories dynamically via a shared asyncio.Queue.
        """
        self.log_callback = log_callback or self._default_log
        self.progress_callback = progress_callback or self._default_progress

        await self._log("🚀 Starting Multi-agent Ralph Loop (asyncio pool)")
        await self._log(f"   Execution ID: {execution_id}")
        await self._log("")

        prd = self._load_prd()
        if not prd:
            return {"status": "error", "error": "PRD file not found or empty"}

        all_stories = prd.get("user_stories", [])
        todo_stories = [s for s in all_stories if s.get("status") == "todo"]
        total_stories = len(all_stories)
        total_todo = len(todo_stories)

        await self._log(f"   Total stories: {total_stories}")
        await self._log(f"   TODO stories: {total_todo}")
        await self._log("")

        if total_todo == 0:
            await self._progress(100)
            return {
                "status": "success",
                "execution_id": execution_id,
                "completed_items": [],
                "failed_items": [],
                "total_iterations": 0,
                "agents": {},
            }

        # ------------------------------------------------------------------
        # Initialize / ensure agent profiles
        # ------------------------------------------------------------------
        if not self.agent_profile_store:
            self.agent_profile_store = AgentProfileStore.load_for_project(self.project_dir)

        store = self.agent_profile_store
        assert store is not None  # for type checkers

        # Ensure we have at least num_agents profiles: agent_1, agent_2, ...
        for i in range(1, max(1, num_agents) + 1):
            agent_id = f"agent_{i}"
            if store.get(agent_id) is None:
                metric_focus = self.metric_config.get("name") if self.metric_config else None
                profile = AgentProfile(
                    id=agent_id,
                    name=f"Agent {i}",
                    role="Generalist implementation agent",
                    expertise=["backend", "tests", "refactoring"],
                    target_files=["src/"],
                    metric_focus=metric_focus,
                )
                store.upsert_agent(profile)

        agent_ids = [f"agent_{i}" for i in range(1, max(1, num_agents) + 1)]
        agent_profiles: Dict[str, AgentProfile] = {
            aid: store.get(aid) for aid in agent_ids if store.get(aid) is not None
        }

        # Compose per-agent system prompts (orchestrator + profile blob)
        system_prompts: Dict[str, str] = {}
        for aid, profile in agent_profiles.items():
            system_prompts[aid] = (
                f"{orchestrator_prompt}\n\n" f"{profile.to_prompt_blob()}"
            )

        # ------------------------------------------------------------------
        # Shared work queue with recommended agent for potential skill routing
        # ------------------------------------------------------------------
        queue: asyncio.Queue = asyncio.Queue()

        # Assign recommended agent in simple round-robin for now
        jobs: List[Dict[str, Any]] = []
        for idx, story in enumerate(todo_stories):
            iteration = idx + 1
            recommended_agent = agent_ids[idx % len(agent_ids)] if agent_ids else "agent_1"
            jobs.append(
                {
                    "story": story,
                    "iteration": iteration,
                    "recommended_agent": recommended_agent,
                }
            )
            await queue.put(jobs[-1])

        # For trace logger: how many items "назначены" каждому агенту
        if self.trace_logger:
            assigned_counts: Dict[str, int] = {aid: 0 for aid in agent_ids}
            for job in jobs:
                assigned_counts[job["recommended_agent"]] += 1
            for aid in agent_ids:
                self.trace_logger.agent_start(aid, assigned_items=assigned_counts.get(aid, 0))

        completed_items: List[Dict[str, Any]] = []
        failed_items: List[Dict[str, Any]] = []
        per_agent_completed: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in agent_ids}
        per_agent_failed: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in agent_ids}
        completed_count = 0

        loop_start = datetime.now()

        # ------------------------------------------------------------------
        # Worker coroutine
        # ------------------------------------------------------------------

        async def worker(agent_id: str) -> None:
            nonlocal completed_count
            profile = agent_profiles[agent_id]
            system_prompt = system_prompts[agent_id]
            agent_start_time = datetime.now()

            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    break

                story = job["story"]
                iteration = job["iteration"]
                story_id = story.get("id", f"story-{iteration}")
                story_title = story.get("title", "Untitled")

                await self._log(
                    f"📌 [agent={agent_id}] Iteration {iteration}: {story_id} - {story_title}"
                )

                # BE-008: Log item start
                if self.trace_logger:
                    self.trace_logger.item_start(agent_id, story_id, story_title)

                start_time = datetime.now()
                result = await self._execute_story(
                    story=story,
                    orchestrator_prompt=system_prompt,
                    iteration=iteration,
                    agent_id=agent_id,
                    agent_profile=profile,
                )
                duration = (datetime.now() - start_time).total_seconds()

                metric_value = result.get("metric_value")

                if result["status"] == "success":
                    item_record = {
                        "id": story_id,
                        "title": story_title,
                        "iteration": iteration,
                        "files": result.get("files", []),
                        "agent_id": agent_id,
                    }
                    completed_items.append(item_record)
                    per_agent_completed[agent_id].append(item_record)
                    completed_count += 1

                    await self._log(
                        f"   ✅ [agent={agent_id}] Story {story_id} PASSED"
                    )

                    # BE-008: Log item complete
                    if self.trace_logger:
                        self.trace_logger.item_complete(
                            agent_id,
                            story_id,
                            duration_seconds=duration,
                            files_created=result.get("files", []),
                        )

                    # RL-like reward: success → positive, scaled by metric if present
                    reward = 1.0
                    if metric_value is not None and self.metric_config:
                        target = float(self.metric_config.get("target", 0.0))
                        reward = metric_value - target

                    if self.agent_profile_store:
                        summary = (
                            f"Story {story_id} succeeded by {agent_id}, "
                            f"reward={reward:.4f}, metric={metric_value}"
                        )
                        self.agent_profile_store.record_learning(
                            agent_id,
                            summary,
                            metrics={
                                "story_id": story_id,
                                "status": "success",
                                "metric_value": metric_value,
                                "reward": reward,
                            },
                        )
                else:
                    error_msg = result.get("error", "Unknown error")
                    item_record = {
                        "id": story_id,
                        "title": story_title,
                        "iteration": iteration,
                        "error": error_msg,
                        "agent_id": agent_id,
                    }
                    failed_items.append(item_record)
                    per_agent_failed[agent_id].append(item_record)

                    await self._log(
                        f"   ❌ [agent={agent_id}] Story {story_id} FAILED: {error_msg}"
                    )

                    # BE-008: Log item fail
                    if self.trace_logger:
                        self.trace_logger.item_fail(
                            agent_id,
                            story_id,
                            error_message=error_msg,
                            attempt=story.get("attempts", 1),
                        )

                    # RL-like reward: failure → negative
                    if self.agent_profile_store:
                        reward = -1.0
                        summary = (
                            f"Story {story_id} failed by {agent_id}, "
                            f"reward={reward:.4f}, error={error_msg[:120]}"
                        )
                        self.agent_profile_store.record_learning(
                            agent_id,
                            summary,
                            metrics={
                                "story_id": story_id,
                                "status": "failed",
                                "metric_value": metric_value,
                                "reward": reward,
                            },
                        )

                # Update global progress (based on completed_count vs total_todo)
                progress = (
                    (completed_count / total_todo) * 100 if total_todo else 100
                )
                await self._progress(progress)

                # KPI: if metric target configured, log latest metric
                if self.metric_config:
                    latest_metric = self._get_latest_metric_value()
                    if latest_metric is not None:
                        await self._log(
                            f"   📈 [agent={agent_id}] Current {self.metric_config['name']}: "
                            f"{latest_metric:.4f}"
                        )

                        if self.trace_logger:
                            self.trace_logger.metric_update(
                                agent_id,
                                self.metric_config["name"],
                                latest_metric,
                                target=float(
                                    self.metric_config.get("target", 0.0)
                                ),
                            )

                await self._log("")
                queue.task_done()

            # Agent finished all available work
            agent_duration = (datetime.now() - agent_start_time).total_seconds()
            if self.trace_logger:
                self.trace_logger.agent_finish(
                    agent_id,
                    completed_items=len(per_agent_completed[agent_id]),
                    failed_items=len(per_agent_failed[agent_id]),
                    duration_seconds=agent_duration,
                )

        # Spawn workers
        workers = [
            asyncio.create_task(worker(aid)) for aid in agent_ids
        ]

        # Add sentinel None jobs so each worker can exit cleanly
        for _ in agent_ids:
            await queue.put(None)

        # Wait for all work to be processed
        await queue.join()
        await asyncio.gather(*workers, return_exceptions=True)

        loop_duration = (datetime.now() - loop_start).total_seconds()

        await self._progress(100)
        await self._log("")
        await self._log("📊 Execution Summary:")
        await self._log(f"   ✅ Completed: {len(completed_items)}")
        await self._log(f"   ❌ Failed: {len(failed_items)}")
        await self._log(f"   🔄 Stories attempted: {len(completed_items) + len(failed_items)}")
        await self._log(f"   ⏱  Duration: {loop_duration:.1f}s")

        # Persist updated agent profiles (with new learnings)
        if self.agent_profile_store:
            try:
                self.agent_profile_store.save()
            except Exception as e:  # pragma: no cover
                logger.error("Error saving agent profiles: %s", e)

        status: str
        if failed_items and not completed_items:
            status = "failed"
        elif failed_items:
            status = "partial"
        else:
            status = "success"

        agents_summary: Dict[str, Any] = {}
        for aid in agent_ids:
            agents_summary[aid] = {
                "completed_items": per_agent_completed[aid],
                "failed_items": per_agent_failed[aid],
                "profile_id": aid,
            }

        return {
            "status": status,
            "execution_id": execution_id,
            "completed_items": completed_items,
            "failed_items": failed_items,
            "total_iterations": len(completed_items) + len(failed_items),
            "agents": agents_summary,
        }

    # ------------------------------------------------------------------
    # Story execution (extended with agent info + metric return)
    # ------------------------------------------------------------------

    async def _execute_story(
        self,
        story: Dict[str, Any],
        orchestrator_prompt: str,
        iteration: int,
        agent_id: str,
        agent_profile: AgentProfile,
    ) -> Dict[str, Any]:
        story_id = story.get("id", "unknown")
        try:
            await self._log(
                f"   🤖 [agent={agent_id}] Calling DeepSeek to generate code..."
            )

            base_user_prompt = self._create_story_prompt(story)

            # Enrich user prompt with agent assignment for better specialization
            user_prompt = (
                f"You are the assigned agent for this story.\n"
                f"Agent ID: {agent_id}\n"
                f"Agent Name: {agent_profile.name}\n"
                f"Role: {agent_profile.role}\n"
                f"Primary KPI focus: {agent_profile.metric_focus or 'project-level success'}.\n"
                f"This specific story is currently assigned to you; implement it according to your profile.\n\n"
                f"{base_user_prompt}"
            )

            response = await self._call_deepseek(
                system_prompt=orchestrator_prompt,
                user_prompt=user_prompt,
            )

            await self._log(
                f"   💬 [agent={agent_id}] Received response ({len(response)} chars)"
            )

            files = await self._save_code(story_id, response, iteration)
            await self._log(
                f"   💾 [agent={agent_id}] Saved {len(files)} file(s)"
            )

            verification = story.get("verification", "")
            metric_value: Optional[float] = None

            if verification and verification.strip():
                await self._log(
                    f"   🧪 [agent={agent_id}] Running verification..."
                )
                verify_result = await self._run_verification(verification)

                # Metric-aware evaluation (BE-009: multi-pattern extraction)
                metric_pass = True
                if self.metric_config and verify_result.get("output"):
                    metric_value = self._extract_metric(
                        verify_result["output"], self.metric_config
                    )
                    if metric_value is not None:
                        metric_pass = metric_value >= float(
                            self.metric_config.get("target", 0.0)
                        )
                        self._append_metric_history(metric_value)

                if verify_result["success"] and metric_pass:
                    self._update_prd_story(story_id, status="done", passes=True)
                    await self._append_progress(
                        story_id=story_id,
                        status="success",
                        learning=(
                            f"[agent={agent_id}] Verification passed. "
                            f"Metric: {metric_value if metric_value is not None else 'N/A'}"
                        ),
                    )
                    return {
                        "status": "success",
                        "files": files,
                        "metric_value": metric_value,
                    }

                # Verification or metric failed
                error_msg = verify_result.get("error", "Verification failed")
                if not metric_pass and metric_value is not None:
                    error_msg = (
                        f"Metric {self.metric_config['name']}={metric_value:.4f} "
                        f"< target {self.metric_config['target']}"
                    )

                story["attempts"] = story.get("attempts", 0) + 1
                story.setdefault("errors", []).append(error_msg)
                self._update_prd_story(
                    story_id,
                    status="failed",
                    passes=False,
                    attempts=story["attempts"],
                    errors=story["errors"],
                )
                await self._append_progress(
                    story_id=story_id,
                    status="failed",
                    learning=f"[agent={agent_id}] {error_msg}",
                )
                return {
                    "status": "failed",
                    "error": error_msg,
                    "files": files,
                    "metric_value": metric_value,
                }

            # No verification => mark as done but log warning
            await self._log(
                f"   ⚠️  [agent={agent_id}] No verification command specified, marking story as done"
            )
            self._update_prd_story(story_id, status="done", passes=True)
            await self._append_progress(
                story_id=story_id,
                status="success",
                learning=(
                    f"[agent={agent_id}] Completed {story.get('title', 'story')} "
                    f"(no verification)."
                ),
            )
            return {"status": "success", "files": files, "metric_value": metric_value}

        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Error executing story %s", story_id)
            await self._log(f"   ❌ [agent={agent_id}] Error: {str(e)}")
            self._update_prd_story(story_id, status="failed", passes=False)
            return {
                "status": "error",
                "error": str(e),
                "metric_value": metric_value,
            }

    # ------------------------------------------------------------------
    # DeepSeek + code saving (unchanged apart from logging prefixes)
    # ------------------------------------------------------------------

    def _create_story_prompt(self, story: Dict[str, Any]) -> str:
        title = story.get("title", "Untitled")
        description = story.get("description", "")
        acceptance = "\n".join(
            [f"  - {c}" for c in story.get("acceptance_criteria", [])]
        )
        why = story.get("why", "")
        files = ", ".join(story.get("files_touched", []))

        return f"""Implement the following PRD story:

Title: {title}

Description:
{description}

Why:
{why}

Acceptance Criteria:
{acceptance}

Files to create/modify: {files}

Instructions:
1. Generate complete, working code for this story.
2. Include all necessary imports, type hints, and docstrings.
3. Add error handling and logging where appropriate.
4. Ensure code is production-ready and follows best practices.
5. Output code in markdown code blocks with language tags.
6. At the top of each code block, include a line: `# File: path/to/file.py`.
"""

    async def _call_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        """Call DeepSeek via the async DeepseekClient using reasoner-compatible API.

        Uses DeepseekClient.call_agent, which supports extended thinking and
        works with the configured model (e.g. deepseek-reasoner).
        """
        if not self.deepseek:
            return (
                "# Error: Deepseek client not initialized\n"
                "print('Deepseek client is None - cannot generate code')\n"
            )

        # DeepseekClient.call_agent is async and returns a dict with a
        # 'response' field containing the model output text.
        result = await self.deepseek.call_agent(
            system_prompt=system_prompt,
            user_message=user_prompt,
            temperature=0.7,
        )
        return result.get("response", "")

    async def _save_code(self, story_id: str, response: str, iteration: int) -> List[str]:
        saved_files: List[str] = []

        code_blocks = re.findall(
            r"```(\w+)?\n(.*?)\n```", response, re.DOTALL
        )
        if not code_blocks:
            fallback_file = self.output_dir / f"{story_id}_iter{iteration}.txt"
            fallback_file.write_text(response, encoding="utf-8")
            saved_files.append(str(fallback_file))
            return saved_files

        for block_idx, (lang, code) in enumerate(code_blocks, 1):
            file_path = self._extract_file_path(code)
            if not file_path:
                file_path = self._infer_filename(code, lang, story_id, block_idx)

            target_file = self.project_dir / file_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                target_file.write_text(code, encoding="utf-8")
                saved_files.append(str(target_file))
                await self._log(f"      💾 {file_path}")
            except Exception as e:  # pragma: no cover - filesystem issues
                logger.error("Error saving %s: %s", file_path, e)

        return saved_files

    def _extract_file_path(self, code: str) -> Optional[str]:
        match = re.search(r"^#\s*[Ff]ile:\s*(.+?)\s*$", code, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _infer_filename(
        self, code: str, lang: str, story_id: str, block_idx: int
    ) -> str:
        match = re.search(r"^class\s+(\w+)", code, re.MULTILINE)
        if match:
            return f"src/{match.group(1).lower()}.py"

        match = re.search(r"^def\s+(\w+)", code, re.MULTILINE)
        if match:
            return f"src/{match.group(1)}.py"

        lang_ext = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "sql": "sql",
            "bash": "sh",
            "shell": "sh",
        }
        ext = lang_ext.get(lang or "", "txt")
        return f"src/{story_id}_block{block_idx}.{ext}"

    async def _run_verification(self, command: str) -> Dict[str, Any]:
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.project_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                ),
            )
            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            return {
                "success": False,
                "error": f"Command failed with code {result.returncode}: {result.stderr[:400]}",
                "output": result.stdout,
            }
        except subprocess.TimeoutExpired:
            return {"success": False,
                    "error": "Verification timed out (600s)",
                    "output": ""}
        except Exception as e:  # pragma: no cover
            return {"success": False,
                    "error": f"Verification error: {str(e)}",
                    "output": ""}

    # ------------------------------------------------------------------
    # PRD + metrics helpers
    # ------------------------------------------------------------------

    def _load_prd(self) -> Optional[Dict[str, Any]]:
        if not self.prd_file.exists():
            logger.error("PRD file not found: %s", self.prd_file)
            return None
        try:
            with open(self.prd_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error loading PRD: %s", e)
            return None

    def _update_prd_story(
        self,
        story_id: str,
        status: str,
        passes: bool,
        attempts: int = 0,
        errors: Optional[List[str]] = None,
    ) -> None:
        prd = self._load_prd()
        if not prd:
            return
        try:
            for story in prd["user_stories"]:
                if story.get("id") == story_id:
                    story["status"] = status
                    if attempts:
                        story["attempts"] = attempts
                    if errors:
                        story["errors"] = errors
                    break
            with open(self.prd_file, "w", encoding="utf-8") as f:
                json.dump(prd, f, indent=2)
        except Exception as e:
            logger.error("Error updating PRD: %s", e)

    def _load_metrics_config(self) -> Optional[Dict[str, Any]]:
        if not self.metrics_config_file.exists():
            return None
        try:
            with open(self.metrics_config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg or None
        except Exception as e:
            logger.error("Error loading metrics_config.json: %s", e)
            return None

    def _append_metric_history(self, value: float) -> None:
        history: Dict[str, Any] = {"values": []}
        if self.metrics_history_file.exists():
            try:
                with open(self.metrics_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = {"values": []}
        history.setdefault("values", []).append(
            {"timestamp": datetime.utcnow().isoformat(), "value": value}
        )
        try:
            with open(self.metrics_history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error("Error writing metrics_history.json: %s", e)

    def _get_latest_metric_value(self) -> Optional[float]:
        if not self.metrics_history_file.exists():
            return None
        try:
            with open(self.metrics_history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not history.get("values"):
                return None
            return history["values"][-1]["value"]
        except Exception:
            return None

    def _extract_metric(self, output: str, config: Dict[str, Any]) -> Optional[float]:
        """BE-009: Extract metric from verification output using multiple patterns.

        Supports two config formats:

        1. Single pattern (backward compatible):
        {
            "name": "weighted_pearson",
            "pattern": "Weighted Pearson: (?P<value>[-+]?\\d*\\.\\d+)",
            "target": 0.35
        }

        2. Multiple patterns (BE-009):
        {
            "name": "weighted_pearson",
            "patterns": [
                {"regex": "Weighted Pearson: (?P<value>[-+]?\\d*\\.\\d+)", "group": "value"},
                {"regex": "WPC: (?P<score>[-+]?\\d*\\.\\d+)", "group": "score"},
                {"regex": "Score: (?P<val>[-+]?\\d*\\.\\d+)", "group": "val"}
            ],
            "target": 0.35
        }

        Tries each pattern in order until one matches.
        """
        # BE-009: Try multiple patterns if configured
        if "patterns" in config and isinstance(config["patterns"], list):
            for pattern_config in config["patterns"]:
                regex = pattern_config.get("regex")
                group_name = pattern_config.get("group", "value")

                if not regex:
                    continue

                try:
                    match = re.search(regex, output)
                    if match:
                        # Try named group first, fall back to group(1)
                        try:
                            raw = match.group(group_name)
                        except (IndexError, AttributeError):
                            raw = (
                                match.group(1)
                                if match.lastindex and match.lastindex >= 1
                                else None
                            )

                        if raw is None:
                            continue

                        value = float(raw)
                        if not math.isnan(value) and not math.isinf(value):
                            logger.debug(
                                "Extracted metric using pattern %s: %s = %f",
                                regex[:50],
                                config.get("name"),
                                value,
                            )
                            return value
                except Exception as e:
                    logger.debug("Pattern %s failed: %s", regex[:50], e)
                    continue

            # No pattern matched
            logger.warning(
                "No pattern matched for metric %s in output (tried %d patterns)",
                config.get("name"),
                len(config["patterns"]),
            )
            return None

        # Backward compatible: single pattern
        pattern = config.get("pattern")
        if not pattern:
            return None

        try:
            match = re.search(pattern, output)
            if not match:
                return None
            raw = (
                match.group("value")
                if "value" in match.groupdict()
                else match.group(1)
            )
            value = float(raw)
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        except Exception:
            return None

    async def _append_progress(self, story_id: str, status: str, learning: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n[{timestamp}] {story_id} - {status.upper()}\n{learning}\n"
        try:
            with open(self.progress_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            logger.error("Error writing progress.txt: %s", e)

    async def _log(self, message: str) -> None:
        logger.info("[UI LOG] %s", message)
        try:
            if asyncio.iscoroutinefunction(self.log_callback):
                await self.log_callback(message)
            else:
                self.log_callback(message)
        except Exception as e:  # pragma: no cover
            logger.error("Log callback error: %s", e)

    async def _progress(self, value: float) -> None:
        logger.info("[UI PROGRESS] %.1f%%", value)
        try:
            if asyncio.iscoroutinefunction(self.progress_callback):
                await self.progress_callback(value)
            else:
                self.progress_callback(value)
        except Exception as e:  # pragma: no cover
            logger.error("Progress callback error: %s", e)

    def _default_log(self, message: str) -> None:
        logger.info("[DEFAULT LOG] %s", message)

    def _default_progress(self, value: float) -> None:
        logger.info("[DEFAULT PROGRESS] %.1f%%", value)
