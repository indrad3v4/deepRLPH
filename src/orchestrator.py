# -*- coding: utf-8 -*-

"""
orchestrator.py - RALPH Main Orchestrator Engine (FULL INTEGRATION - GURU EDITION)

Architectural Refactor:
- Pure Async Pipeline: Replaced nested asyncio.run() with awaits to prevent deadlocks.
- Transactional State: Atomic PRD writes and Project Discovery.
- BE-006: AI-generated project-specific prompts instead of stubs.
- BE-002 & BE-005: Deep grounding via SchemaIntegrator and Context Ingestor.

Additional Enhancements (Council v2):
- Project-wide Context Caching:
  - Cache a compact PROJECT CONTEXT summary per project in config.metadata["context_summary"].
  - Reuse cached summary in DARIZ runs to avoid repeated filesystem scans.
- Attached Files Awareness:
  - Normalize and persist user-attached files in config.metadata["attached_files"] at project creation.
  - Surface these files explicitly into DARIZ input so agents know where rules/specs live.
- DARIZ Prompt Upgrade:
  - Use an o1-style GOAL / RETURN FORMAT / WARNINGS / CONTEXT DUMP layout.
  - Embed TRIZ-style multi-stage reasoning instructions as internal guidance.
  - Still enforce JSON-only PRD output for stable downstream parsing.

Competition-Aware Enhancements (WunderNN):
- Competition Brief Wiring:
  - Persist a long-form competition brief in config.metadata["competition_brief"], defaulting to description.
  - Surface eval_metric / metric_target into DARIZ + council prompts.
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import json
import logging
import sys
import asyncio
import re

# Internal Components
from context_ingestor import ingest_project_context
from schema_integrator import SchemaIntegrator
from agent_profiles import AgentProfileStore

logger = logging.getLogger("Orchestrator")


# ============================================================================
# DATA MODELS (Transactional & Type-Safe)
# ============================================================================


class ProjectType(str, Enum):
    API_DEVELOPMENT = "api_dev"
    ML_COMPETITION = "ml_competition"
    DATA_PIPELINE = "data_pipeline"
    LLM_APP = "llm_app"


class ArchitectureType(str, Enum):
    CLEAN = "clean_architecture"
    MVC = "mvc"
    MICROSERVICES = "microservices"
    HEXAGONAL = "hexagonal"


@dataclass
class MLConfig:
    competition_source: str = ""
    problem_type: str = "time_series_forecasting"
    model_type: str = "LSTM"
    ml_framework: str = "PyTorch"
    eval_metric: str = "score"
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MLConfig":
        return cls(**data)


@dataclass
class ProjectConfig:
    name: str
    domain: str
    description: str = ""
    project_type: str = "api_dev"
    architecture: str = "clean_architecture"
    framework: str = "FastAPI"
    language: str = "Python"
    database: str = "PostgreSQL"
    parallel_agents: int = 4
    ml_config: Optional[MLConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.ml_config:
            data["ml_config"] = self.ml_config.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectConfig":
        if data.get("ml_config"):
            data["ml_config"] = MLConfig.from_dict(data["ml_config"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExecutionState:
    execution_id: str
    project_id: str
    start_time: str
    status: str = "running"
    total_items: int = 0
    completed_items: int = 0
    logs: List[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


# ============================================================================
# WORKSPACE & ORCHESTRATOR
# ============================================================================


class WorkspaceManager:
    """Manages atomic project workspace initialization."""

    def __init__(self, workspace_dir: Path):
        self.workspace = workspace_dir.resolve()

    def create_project_workspace(self, project_id: str, config: ProjectConfig) -> Path:
        project_dir = self.workspace / "projects" / project_id
        dirs = [
            "workspace/config",
            "workspace/input",
            "workspace/output/logs",
            "src",
            "logs",
            "data/raw",
        ]
        if config.project_type == ProjectType.ML_COMPETITION:
            dirs.extend(["models/final", "submissions"])

        for d in dirs:
            (project_dir / d).mkdir(parents=True, exist_ok=True)
        return project_dir


class RalphOrchestrator:
    """The central brain of deepRLPH. Purely asynchronous transactional engine."""

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        deepseek_client=None,
        agent_coordinator=None,
        execution_engine=None,
    ):
        self.workspace = Path(workspace_dir or Path.home() / ".ralph").resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.workspace_manager = WorkspaceManager(self.workspace)

        self.deepseek_client = deepseek_client
        self.agent_coordinator = agent_coordinator
        self.execution_engine = execution_engine

        self.current_config: Optional[ProjectConfig] = None
        self.current_project_dir: Optional[Path] = None
        self.execution_log: List[str] = []
        self.executions: Dict[str, ExecutionState] = {}

        self.schema_integrator = SchemaIntegrator()
        self.agent_profile_store: Optional[AgentProfileStore] = None

    # --------------------------------------------------------------------- #
    # PROJECT LIFECYCLE
    # --------------------------------------------------------------------- #

    def create_project(self, config: ProjectConfig) -> Dict[str, Any]:
        """Synchronous setup of the project structure, with metadata normalization."""
        try:
            self.current_config = config
            project_id = self._generate_project_id(config.name)
            project_dir = self.workspace_manager.create_project_workspace(project_id, config)
            self.current_project_dir = project_dir

            # Normalize and persist attached files (if UI provided them) so DARIZ & context
            # ingest can consistently surface them later without re-discovery.
            metadata = config.metadata or {}
            attached = metadata.get("attached_files", [])
            normalized_attached: List[str] = []

            for raw in attached:
                try:
                    p = Path(raw)
                    # If UI already copied them into the project directory, store as relative paths
                    if str(p).startswith(str(project_dir)):
                        rel = str(p.relative_to(project_dir))
                        normalized_attached.append(rel)
                    else:
                        # Otherwise, keep as-is (UI must make sure they are readable)
                        normalized_attached.append(str(p))
                except Exception:
                    normalized_attached.append(str(raw))

            metadata["attached_files"] = normalized_attached
            # Initialize a context_summary cache slot (filled later in refine_task / DARIZ)
            metadata.setdefault("context_summary", "")

            # Seed competition_brief from description if UI/template did not provide one explicitly.
            if "competition_brief" not in metadata:
                desc = metadata.get("description") or config.description or ""
                metadata["competition_brief"] = desc

            config.metadata = metadata

            (project_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
            self.agent_profile_store = AgentProfileStore.load_for_project(project_dir)

            self._log("✅ Project created: %s", project_id)
            return {"status": "success", "project_id": project_id, "path": str(project_dir)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def refine_task(
        self,
        project_id: str,
        raw_task: str,
        dataset_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Grounding-first Refinement Phase (PURE ASYNC)."""
        if not self.current_config:
            return {"status": "error", "error": "No session active."}

        try:
            # 1. BE-002: Schema Detection
            if dataset_path:
                schema_res = self.schema_integrator.inspect_and_save(
                    self.current_project_dir,
                    dataset_path,
                )
                if schema_res["status"] == "success":
                    self.current_config.metadata["dataset_schemas"] = schema_res["schemas"]

            # 2. PRD-01: Ingest Context (with lightweight caching of the text summary)
            context_summary = (self.current_config.metadata or {}).get("context_summary", "")
            if not context_summary:
                # Fresh ingest
                ctx = ingest_project_context(self.current_project_dir)
                context_summary = ctx.to_text_summary()
                self.current_config.metadata["ingested_context"] = ctx.to_dict()
                self.current_config.metadata["context_summary"] = context_summary

                # Persist updated config with cached context for future DARIZ runs
                (self.current_project_dir / "config.json").write_text(
                    json.dumps(self.current_config.to_dict(), indent=2)
                )

            # 3. AI Clarification (Pure Await)
            from task_clarifier import TaskClarifier
            from prd_generator import PRDGenerator

            clarifier = TaskClarifier(self.deepseek_client)
            enriched_task = f"{raw_task}\n\nCONTEXT:\n{context_summary}"

            brief = await clarifier.clarify(enriched_task, self.current_config.domain)
            if brief.get("status") != "success":
                return brief

            # 4. BE-005: PRD Generation
            gen = PRDGenerator(self.deepseek_client)
            prd = await gen.generate(brief, self.current_config.domain, self.current_config.metadata)

            (self.current_project_dir / "prd.json").write_text(json.dumps(prd, indent=2))
            return {"status": "success", "prd": prd}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # --------------------------------------------------------------------- #
    # EXECUTION LOOP (Model Council)
    # --------------------------------------------------------------------- #

    async def execute_prd_loop(
        self,
        prd: Dict,
        num_agents: int = 4,
        execution_mode: str = "parallel",
        log_callback: Optional[Callable[[str], Any]] = None,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> Dict:
        """Transactional execution loop with Mode Routing."""
        try:
            canonical_prd = self._normalize_prd_for_engine(prd)
            execution_id = f"exec_{self._generate_project_id('')}"

            # Atomic State: Save PRD before spawning agents
            (self.current_project_dir / "prd.json").write_text(json.dumps(canonical_prd, indent=2))

            # BE-006: Call AI for specific system instructions
            orchestrator_prompt = await self._generate_dynamic_orchestrator_prompt(
                canonical_prd,
                self.current_config,
            )

            if not self.execution_engine:
                from execution_engine import ExecutionEngine

                self.execution_engine = ExecutionEngine(
                    self.current_project_dir,
                    self.deepseek_client,
                    self.agent_coordinator,
                )

            self._log("⚙️  Deploying %d agents in %s mode...", num_agents, execution_mode)

            # Execute through engine
            return await self.execution_engine.execute(
                execution_id=execution_id,
                orchestrator_prompt=orchestrator_prompt,
                prd_partitions={"stories": canonical_prd["user_stories"]},
                num_agents=num_agents,
                execution_mode=execution_mode,
                progress_callback=progress_callback,
                log_callback=log_callback,
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # --------------------------------------------------------------------- #
    # DARIZ ENGINE (Crash → PRD Fix via TRIZ-style reasoning)
    # --------------------------------------------------------------------- #

    async def run_dariz_fix(
        self,
        crash_snapshot: Dict[str, Any],
        strategy_text: str = "",
        auto_execute: bool = False,
    ) -> Dict[str, Any]:
        """Centralized DARIZ engine.

        Takes a crash snapshot + optional user strategy, calls DeepSeek with the
        Kukalev-style DARIZ prompt, parses JSON PRD, normalizes it and writes
        prd.json. Returns normalized PRD and a suggested execution mode.
        """
        if not self.current_project_dir:
            return {"status": "error", "error": "No active project for DARIZ fix."}

        try:
            # Use cached context summary when available; fall back to ingest.
            context_summary = (self.current_config.metadata or {}).get("context_summary", "")
            if not context_summary:
                try:
                    ctx = await ingest_project_context(self.current_project_dir)
                    context_summary = ctx.to_text_summary()
                    self.current_config.metadata["ingested_context"] = ctx.to_dict()
                    self.current_config.metadata["context_summary"] = context_summary
                    (self.current_project_dir / "config.json").write_text(
                        json.dumps(self.current_config.to_dict(), indent=2)
                    )
                except Exception as e:
                    logger.warning("DARIZ: context ingest failed: %s", e)
                    context_summary = ""

            metadata = self.current_config.metadata or {}
            competition_brief = metadata.get("competition_brief", "")
            eval_metric = metadata.get("eval_metric", "")
            metric_target = metadata.get("metric_target", "")

            # Build an optional competition-awareness block for the system prompt.
            competition_block_lines: List[str] = []
            if competition_brief:
                competition_block_lines.append("### COMPETITION BRIEF\n" + competition_brief.strip())
            if eval_metric or metric_target:
                competition_block_lines.append(
                    "### METRIC & TARGET\n"
                    + f"- Eval metric: {eval_metric or 'n/a'}\n- Target: {metric_target or 'n/a'}"
                )
            competition_block = "\n\n".join(competition_block_lines).strip()

            exit_code = crash_snapshot.get("exit_code")
            output_tail = crash_snapshot.get("output_tail", "")
            terminal_dump = crash_snapshot.get("terminal_dump") or crash_snapshot.get("terminal", "")
            exec_logs = crash_snapshot.get("exec_logs", "")
            user_strategy = (crash_snapshot.get("user_strategy") or strategy_text or "").strip()

            attached_files = (self.current_config.metadata or {}).get("attached_files", [])

            # === DARIZ System Prompt (o1 + TRIZ meta-instructions) =====================
            system_prompt = """
### GOAL
You are the DARIZ-Kukalev troubleshooting engine for deepRLPH.
Given a failing execution trace, project context, and human strategy, propose a small, executable PRD
(backlog of 1–4 stories) that will likely fix the problem in the next council iteration.

### RETURN FORMAT
Return ONLY one JSON object with either a 'user_stories' or 'backlog' list.
Each story MUST have:
- id (string, e.g. "EXEC-001"),
- title (short one-line description),
- description (2–4 sentences),
- acceptance_criteria (list of concrete, testable bullet strings),
- verification (shell command string to run, e.g. "python src/train.py --fast_dev_run"),
Optionally you MAY include 'status' and 'files_touched' fields.
No markdown, no commentary outside JSON.

### WARNINGS
- Be careful to keep each story SMALL and focused on a single core fix (dimension alignment, test hardening, config wiring).
- Do NOT invent files, paths, datasets, or commands that do not exist in the project.
- Prefer verification commands that run fast and are executable TODAY (pytest on existing tests, short python -c snippets, or train.py --fast_dev_run).
- Avoid broad refactors; instead, create 1–2 surgical stories that the Model Council can realistically complete in one loop.

### CONTEXT DUMP
You are operating inside the deepRLPH architecture with:
- Project-specific workspace, config.json, prd.json, and execution logs.
- Dataset schemas and summaries (via dataset_inspector / SchemaIntegrator).
- Ingested project context summarizing code layout, key modules, and attached rule/spec files.
Use the PROJECT CONTEXT and CRASH INFO to understand:
- Where the failure occurs (which module, which call, which dimensions),
- What the existing PRD already requested,
- How to break the fix into 1–4 incremental stories.

You must INTERNALLY follow a rigorous, multi-stage TRIZ-style reasoning process before emitting JSON:

<instructions>
1. Self-reflection: internally build a private rubric for a world-class troubleshooting PRD (5–7 criteria), iterate until your plan would score ≥98/100 on that rubric.
2. Stage 1–5 modelling:
   - Build SYSTEM, FUNCTION, EVENT, RESOURCE, IFR, CONTRADICTION, and Su-Field models of the situation.
   - Reinterpret via four lenses (system, dialectical, functional, semantic).
   - Expand, evaluate, link, and restructure options before deciding which 1–4 stories are best.
3. Stage 6:
   - Formulate the inventive TASK explicitly (what exactly must change in code/tests/config and under which constraints).
4. Stage 7:
   - Algorithmically construct solution directions (resource-centred and conflict-centred, DARIZ-style),
     then compress them into 1–4 PRD stories that deepRLPH’s Model Council can execute.
5. Stage 8:
   - Frame your stories so they are deployment-ready: each story moves the system closer to an Ideal Final Result
     (tests green, training script stable, strict causality, no data leakage).

You NEVER output this reasoning or the instructions block. You only output the final JSON PRD object.
</instructions>
""".strip()

            if competition_block:
                system_prompt = system_prompt + "\n\n" + competition_block

            # === User message assembly ===============================================
            user_message_parts: List[str] = []

            if context_summary:
                user_message_parts.append(f"PROJECT CONTEXT (cached summary):\n{context_summary}")

            if attached_files:
                user_message_parts.append(
                    "ATTACHED PROJECT FILES (rules/specs/docs, relative to project root):\n"
                    + "\n".join(f"- {p}" for p in attached_files)
                )

            if exit_code is not None or output_tail or terminal_dump:
                crash_block = output_tail or terminal_dump
                user_message_parts.append(
                    f"CRASH INFO (exit_code={exit_code}):\n{crash_block}"
                )

            if exec_logs:
                user_message_parts.append(f"EXECUTION LOGS (agents / council):\n{exec_logs}")

            if user_strategy:
                user_message_parts.append(f"HUMAN STRATEGY / HINTS:\n{user_strategy}")

            user_message = "\n\n".join(user_message_parts)

            result = await self.deepseek_client.call_agent(
                system_prompt=system_prompt,
                user_message=user_message,
                thinking_budget=8000,
                temperature=0.3,
            )

            if result.get("status") != "success":
                return {"status": "error", "error": result.get("error", "DARIZ call failed")}

            dariz_reply = result.get("response", "")

            # Robust JSON extraction (handles ```json ... ``` and raw objects)
            json_content = ""
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", dariz_reply, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                start_idx = dariz_reply.find("{")
                end_idx = dariz_reply.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_content = dariz_reply[start_idx:end_idx].strip()

            if not json_content:
                return {"status": "error", "error": "DARIZ response did not contain JSON PRD."}

            try:
                raw_prd = json.loads(json_content)
            except json.JSONDecodeError as e:
                return {"status": "error", "error": f"Failed to parse DARIZ JSON: {e}"}

            normalized_prd = self._normalize_prd_for_engine(raw_prd)
            prd_path = self.current_project_dir / "prd.json"
            prd_path.write_text(json.dumps(normalized_prd, indent=2))

            # By default DARIZ suggests a council-style deep reasoning fix with 4 agents
            return {
                "status": "success",
                "prd": normalized_prd,
                "mode": "council",
                "num_agents": 4,
                "auto_execute": bool(auto_execute),
                "raw_response": dariz_reply,
            }
        except Exception as e:
            logger.error("DARIZ run_dariz_fix failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    # --------------------------------------------------------------------- #
    # ORCHESTRATOR PROMPT FOR MODEL COUNCIL
    # --------------------------------------------------------------------- #

    async def _generate_dynamic_orchestrator_prompt(self, prd: Dict, config: ProjectConfig) -> str:
        """BE-006: AI analysis of context to generate grounded instructions
        for the Model Council (agent roles)."""

        context_summary = (config.metadata or {}).get("context_summary", "")
        schemas = json.dumps(config.metadata.get("dataset_schemas", "none"))

        metadata = config.metadata or {}
        competition_brief = metadata.get("competition_brief", "")
        eval_metric = metadata.get("eval_metric", "")
        metric_target = metadata.get("metric_target", "")

        competition_lines: List[str] = []
        if competition_brief:
            competition_lines.append("### COMPETITION BRIEF\n" + competition_brief.strip())
        if eval_metric or metric_target:
            competition_lines.append(
                "### METRIC & TARGET\n"
                + f"- Eval metric: {eval_metric or 'n/a'}\n- Target: {metric_target or 'n/a'}"
            )
        competition_section = "\n\n".join(competition_lines).strip()

        # Keep this light; main heavy TRIZ thinking lives in DARIZ. Here we nudge
        # council agents toward small, surgical implementation plus fast verification.
        user_message = f"""
### GOAL
The Model Council must implement the current PRD stories for project '{config.name}' so that
all acceptance criteria and verification commands pass.

### RETURN FORMAT
You do NOT return text here; your job is to follow these instructions internally when using tools
(read_file, write_file, execute_command, request_human_intervention, finish_task).

### WARNINGS
- Each story should consume at most ~15 tool calls: spend ≤5 on reading, then propose and apply
  concrete edits via write_file, and finally run the story's verification command once.
- Prefer minimal, local changes over broad refactors.
- Do not invent files or commands; use only those that exist in this repository.

### CONTEXT DUMP
- PROJECT CONTEXT SUMMARY:
{context_summary or '(none cached)'}

- DATASET SCHEMAS:
{schemas}
""".strip()

        if competition_section:
            user_message = user_message + "\n\n" + competition_section

        result = await self.deepseek_client.call_agent(
            system_prompt=(
                "You are a system architect for deepRLPH. Rewrite the user guidance as a short, "
                "direct paragraph for autonomous coding agents. Do not add new requirements."
            ),
            user_message=user_message,
            thinking_budget=4000,
        )
        return result.get(
            "response",
            "Proceed with implementation using tools: read_file, write_file, execute_command, request_human_intervention, finish_task.",
        )

    # --------------------------------------------------------------------- #
    # DISCOVERY, SUBMISSION, UTILITIES
    # --------------------------------------------------------------------- #

    def list_projects(self) -> List[Dict]:
        """Synchronous project discovery for GUI."""
        projects_dir = self.workspace / "projects"
        if not projects_dir.exists():
            return []

        def _scan() -> List[Dict[str, Any]]:
            found: List[Dict[str, Any]] = []
            for p_path in projects_dir.iterdir():
                if not p_path.is_dir() or not (p_path / "config.json").exists():
                    continue
                try:
                    data = json.loads((p_path / "config.json").read_text())
                    found.append(
                        {
                            "name": data.get("name", p_path.name),
                            "project_id": p_path.name,
                            "path": str(p_path),
                            "created_at": data.get("timestamp", ""),
                        }
                    )
                except Exception:
                    continue
            return found

        projects = _scan()
        return sorted(projects, key=lambda x: x["created_at"], reverse=True)

    def create_submission(self, project_id: str) -> Dict:
        """Strict packaging for competition."""
        try:
            from make_submission import create_submission as package_zip

            if package_zip():
                return {
                    "status": "success",
                    "zip_path": str(self.current_project_dir / "submissions/submission.zip"),
                }
            return {"status": "error", "error": "Zip failed."}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _normalize_prd_for_engine(self, prd: Dict) -> Dict:
        stories = prd.get("user_stories") or prd.get("backlog") or prd.get("items") or []
        normalized = [
            {
                "id": s.get("id", "ITEM-0"),
                "title": s.get("title", "Untitled"),
                "description": s.get("description", ""),
                "acceptance_criteria": s.get("acceptance_criteria", []),
                "verification": s.get("verification", ""),
                "status": s.get("status", "todo"),
            }
            for s in stories
        ]
        return {"user_stories": normalized, "total_items": len(normalized)}

    def _log(self, msg: str, *args: Any) -> None:
        formatted = msg % args if args else msg
        logger.info(formatted)
        self.execution_log.append(formatted)

    def _generate_project_id(self, name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())[:20]
        return f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_execution_log(self) -> List[str]:
        return self.execution_log

    def clear_execution_log(self) -> None:
        self.execution_log.clear()


def get_orchestrator(**kwargs: Any) -> RalphOrchestrator:
    return RalphOrchestrator(**kwargs)
