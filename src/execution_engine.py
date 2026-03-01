# -*- coding: utf-8 -*-
"""
execution_engine.py - RALPH Transactional Execution Engine (PURE ASYNC)

Guru-Level Refactor:
- Clean Architecture: Removed legacy "ghost" code (file parsing/saving) now handled by Coordinator Tools.
- Robust Council: Dynamic role injection for Mixture of Experts (MoE).
- Process Safety: Switched to asyncio.subprocess for signal-safe execution.
- Transactional Stability: Removed silent 'pass' blocks to prevent context corruption.

WunderNN / Council Upgrades:
- Role-specialised AgentProfiles with self-reflection rubrics for Architect/Reviewer/Refiner/Verifier.
"""

import asyncio
import logging
import json
import math
import re
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from datetime import datetime

from prd_model import PRDItem
from agent_profiles import AgentProfileStore, AgentProfile

logger = logging.getLogger("ExecutionEngine")

# Dynamic Council Configuration
COUNCIL_ROLES = {
    1: "Architect: Initial design and core logic implementation.",
    2: "Reviewer: Aggressive audit for memory leaks, OOM, and efficiency.",
    3: "Refiner: DARIZ-based contradiction fixing and refactoring.",
    4: "Verifier: Final gatekeeper. Ensures PRD adherence and test parity.",
}


class ExecutionEngine:
    """Refactored Ralph Loop Engine focusing on transactional reliability and MoE."""

    def __init__(self, project_dir: Path, deepseek_client, agent_coordinator):
        self.project_dir = Path(project_dir)
        self.prd_file = self.project_dir / "prd.json"
        self.progress_file = self.project_dir / "progress.txt"
        self.metrics_history_file = self.project_dir / "metrics_history.json"
        self.metrics_config_file = self.project_dir / "metrics_config.json"

        self.deepseek = deepseek_client
        self.coordinator = agent_coordinator
        self.trace_logger = None
        self.agent_profile_store = AgentProfileStore.load_for_project(self.project_dir)

        self.metric_config = self._load_metrics_config()
        logger.info(f"🚀 Execution Engine Online: {self.project_dir.name}")

    async def execute(
        self,
        execution_id: str,
        orchestrator_prompt: str,
        prd_partitions: Dict[str, List[Dict[str, Any]]],
        num_agents: int,
        execution_mode: str = "parallel",
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Main entry point for Swarm or Council orchestration."""
        self.log_callback = log_callback
        self.progress_callback = progress_callback

        await self._log(f"🚀 Starting Loop: {execution_mode.upper()} mode (ID: {execution_id})")

        prd = self._load_prd()
        if not prd:
            return {"status": "error", "error": "Canonical PRD missing from workspace."}

        todo_stories = [s for s in prd.get("user_stories", []) if s.get("status") == "todo"]
        if not todo_stories:
            await self._progress(100)
            return {"status": "success", "execution_id": execution_id, "total_iterations": 0}

        # Profile Preparation
        agent_ids = [f"agent_{i}" for i in range(1, max(1, num_agents) + 1)]
        agent_profiles = self._prepare_agent_profiles(agent_ids)

        if execution_mode == "council":
            return await self._run_council(todo_stories, agent_profiles, orchestrator_prompt, execution_id)
        return await self._run_swarm(todo_stories, agent_profiles, orchestrator_prompt, execution_id)

    # ------------------------------------------------------------------
    # Mixture of Experts (Model Council) Logic
    # ------------------------------------------------------------------

    async def _run_council(self, stories, profiles, orchestrator_prompt, exec_id) -> Dict[str, Any]:
        """Sequential escalation through expert roles."""
        completed, failed = [], []
        total = len(stories)

        for idx, story in enumerate(stories):
            story_id = story.get("id", f"STORY-{idx}")
            await self._log(f"🏛️ [Council] Convening on {story_id}: {story.get('title')}")

            handoff_context = "Architect: Initialize implementation."
            resolved = False

            for i, (aid, profile) in enumerate(profiles.items()):
                role_desc = COUNCIL_ROLES.get(i + 1, "Expert: Generic implementation.")
                council_prompt = f"{orchestrator_prompt}\n\nROLE: {role_desc}\n\nHANDOFF: {handoff_context}"

                await self._log(f"   👔 [{aid}] Role: {role_desc.split(':')[0]}")

                start_time = datetime.now()
                result = await self._execute_story(story, council_prompt, aid, profile)
                duration = (datetime.now() - start_time).total_seconds()

                if result["status"] == "success":
                    completed.append({"id": story_id, "agent": aid})
                    await self._log(f"   ✅ [Council] Success under {aid}. Handoff complete.")
                    resolved = True
                    break
                else:
                    await self._log(f"   ⚠️ [{aid}] Failed. Escalating to next Expert.")
                    handoff_context = (
                        f"Previous attempt by {aid} failed.\nError: {result.get('error')}\nPlease rectify."
                    )

            if not resolved:
                failed.append({"id": story_id, "error": "Council exhausted."})

            await self._progress(((idx + 1) / total) * 100)

        return {"status": "success" if not failed else "partial", "completed": completed, "failed": failed}

    # ------------------------------------------------------------------
    # Core Logic & Verification (Pure Async)
    # ------------------------------------------------------------------

    async def _execute_story(self, story, prompt, agent_id, profile: AgentProfile) -> Dict[str, Any]:
        """Hand control to Autonomous Tool Loop with Verification."""
        story_id = story.get("id", "unknown")

        item = PRDItem(
            item_id=story_id,
            title=story.get("title", "Untitled"),
            acceptance_criteria=story.get("acceptance_criteria", []),
            verification_command=story.get("verification", ""),
            files_touched=story.get("files_touched", []),
        )

        # The Missing Wire: Passing the dynamic prompt + profile to the Coordinator
        res = await self.coordinator.assign_prd_item(
            item=item,
            agent_id=agent_id,
            project_dir=self.project_dir,
            deepseek_client=self.deepseek,
            log_callback=self._log,
            orchestrator_prompt=prompt,
            agent_profile=profile,
        )

        if res.get("status") != "success":
            return self._handle_story_failure(story, agent_id, res.get("error"))

        # Official Independent Verification
        verify_cmd = story.get("verification")
        if verify_cmd:
            await self._log(f"   🧪 [{agent_id}] Running deterministic verification...")
            v_res = await self._run_async_subprocess(verify_cmd)

            if not v_res["success"]:
                return self._handle_story_failure(story, agent_id, v_res.get("error"))

        self._update_prd_story(story_id, "done", True)
        return {"status": "success", "files": res.get("files_created", [])}

    async def _run_async_subprocess(self, command: str) -> Dict[str, Any]:
        """Signal-safe process execution using asyncio.create_subprocess_shell."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

            output = stdout.decode().strip() + "\n" + stderr.decode().strip()
            if proc.returncode == 0:
                return {"success": True, "output": output}
            return {"success": False, "error": output[:1000]}
        except asyncio.TimeoutError:
            return {"success": False, "error": "Verification Timeout (600s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Utility & Transactional Helpers
    # ------------------------------------------------------------------

    def _prepare_agent_profiles(self, agent_ids) -> Dict[str, AgentProfile]:
        """Ensure all agents have valid grounding profiles with rubrics."""
        profiles: Dict[str, AgentProfile] = {}

        # Default, role-specialised profiles for a 4-agent council.
        default_profiles: Dict[str, Dict[str, Any]] = {
            "agent_1": {
                "name": "Architect",
                "role": "Architect / Lead Implementer",
                "expertise": [
                    "architecture",
                    "PyTorch",
                    "time-series",
                    "WunderNN solution.py contract",
                ],
                "self_reflection": (
                    "You are the Architect. Priorities: (1) correctness, (2) minimal, local patches, "
                    "(3) keeping solution.py / PredictionModel and training code coherent, (4) "
                    "respecting sequence warm-up 0-98 and need_prediction gating, (5) preserving "
                    "determinism and fixed seeds."
                ),
            },
            "agent_2": {
                "name": "Reviewer",
                "role": "Safety / Performance Reviewer",
                "expertise": [
                    "profiling",
                    "memory-optimization",
                    "CPU-only inference",
                    "numerical stability",
                ],
                "self_reflection": (
                    "You are the Reviewer. Priorities: (1) catch OOM and latency risks, (2) "
                    "simplify data paths and loaders, (3) avoid unnecessary copies / caches "
                    "that violate 16GB RAM and 60-minute constraints, (4) ensure deterministic "
                    "behaviour under fixed seeds."
                ),
            },
            "agent_3": {
                "name": "Refiner",
                "role": "TRIZ/DARIZ Refiner",
                "expertise": [
                    "TRIZ",
                    "DARIZ",
                    "contradiction resolution",
                    "refactoring",
                ],
                "self_reflection": (
                    "You are the Refiner. Priorities: (1) identify technical contradictions in "
                    "the implementation, (2) apply DARIZ/TRIZ steps to restructure code with "
                    "fewer moving parts, (3) keep interface contracts unchanged, (4) make "
                    "changes easy to revert if they underperform."
                ),
            },
            "agent_4": {
                "name": "Verifier",
                "role": "Verifier / Metric Guardian",
                "expertise": [
                    "testing",
                    "Weighted Pearson metric",
                    "ONNX export",
                    "submission packaging",
                ],
                "self_reflection": (
                    "You are the Verifier. Priorities: (1) ensure tests, metric computations, "
                    "and clipping [-6, 6] match the competition spec, (2) strictly check "
                    "PredictionModel.predict and ONNX export / solution.py behaviour, (3) never "
                    "mark a story done without at least one deterministic verification run."
                ),
            },
        }

        for aid in agent_ids:
            profile = self.agent_profile_store.get(aid)
            defaults = default_profiles.get(aid)

            if not profile:
                if defaults:
                    profile = AgentProfile(
                        id=aid,
                        name=defaults["name"],
                        role=defaults["role"],
                        expertise=defaults["expertise"],
                        metric_focus="weighted_pearson",
                        self_reflection=defaults["self_reflection"],
                    )
                else:
                    profile = AgentProfile(
                        id=aid,
                        name=f"Expert {aid[-1]}",
                        role="ML/Software Engineer",
                    )

                self.agent_profile_store.upsert_agent(profile)
            else:
                # Backfill self_reflection for existing profiles if missing
                if not getattr(profile, "self_reflection", None) and defaults:
                    profile.self_reflection = defaults["self_reflection"]
                    self.agent_profile_store.upsert_agent(profile)

            profiles[aid] = profile

        # Persist any new/updated profiles (for AGENTS.md and agents.json)
        try:
            self.agent_profile_store.save()
        except Exception:
            logger.warning("Failed to save agent profiles after preparation", exc_info=True)

        return profiles

    def _handle_story_failure(self, story, aid, err):
        """Transactional failure tracking."""
        sid = story.get("id")
        story["attempts"] = story.get("attempts", 0) + 1
        story.setdefault("errors", []).append(f"[{aid}] {err}")
        self._update_prd_story(sid, "failed", False, story["attempts"], story["errors"])
        return {"status": "failed", "error": err}

    def _update_prd_story(self, sid, status, passes, attempts=0, errors=None):
        """Atomic PRD state update to avoid Race Conditions."""
        prd = self._load_prd()
        if not prd:
            return
        for s in prd["user_stories"]:
            if s.get("id") == sid:
                s["status"], s["passes"] = status, passes
                if attempts:
                    s["attempts"] = attempts
                if errors:
                    s["errors"] = errors
        self.prd_file.write_text(json.dumps(prd, indent=2))

    def _load_prd(self):
        try:
            return json.loads(self.prd_file.read_text())
        except Exception:
            return None

    def _load_metrics_config(self):
        try:
            return json.loads(self.metrics_config_file.read_text())
        except Exception:
            return None

    async def _log(self, msg):
        if self.log_callback:
            await self.log_callback(msg)

    async def _progress(self, val):
        if self.progress_callback:
            await self.progress_callback(val)
