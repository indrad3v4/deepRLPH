# -*- coding: utf-8 -*-

"""
orchestrator.py - RALPH Main Orchestrator Engine (FULL INTEGRATION - GURU EDITION)

Architectural Refactor:
- Pure Async Pipeline: Replaced nested asyncio.run() with awaits to prevent deadlocks.
- Transactional State: Atomic PRD writes and Project Discovery.
- BE-006: AI-generated project-specific prompts instead of stubs.
- BE-002 & BE-005: Deep grounding via SchemaIntegrator and Context Ingestor.
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

    def create_project(self, config: ProjectConfig) -> Dict[str, Any]:
        """Synchronous setup of the project structure."""
        try:
            self.current_config = config
            project_id = self._generate_project_id(config.name)
            project_dir = self.workspace_manager.create_project_workspace(project_id, config)
            self.current_project_dir = project_dir

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

            # 2. PRD-01: Ingest Context
            ctx = ingest_project_context(self.current_project_dir)
            self.current_config.metadata["ingested_context"] = ctx.to_dict()

            # 3. AI Clarification (Pure Await)
            from task_clarifier import TaskClarifier
            from prd_generator import PRDGenerator

            clarifier = TaskClarifier(self.deepseek_client)
            enriched_task = f"{raw_task}\n\nCONTEXT:\n{ctx.to_text_summary()}"

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

    async def execute_prd_loop(
        self,
        prd: Dict,
        num_agents: int = 4,
        execution_mode: str = "parallel",
        log_callback: Optional[Callable[[str], Any]] = None,
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
                log_callback=log_callback,
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
            # Grounded context for DARIZ (DR4-03 partial): reuse ingest_project_context
            try:
                ctx = ingest_project_context(self.current_project_dir)
                context_summary = ctx.to_text_summary()
            except Exception as e:
                logger.warning("DARIZ: context ingest failed: %s", e)
                context_summary = ""

            exit_code = crash_snapshot.get("exit_code")
            output_tail = crash_snapshot.get("output_tail", "")
            terminal_dump = crash_snapshot.get("terminal_dump") or crash_snapshot.get("terminal", "")
            exec_logs = crash_snapshot.get("exec_logs", "")
            user_strategy = (crash_snapshot.get("user_strategy") or strategy_text or "").strip()

            system_prompt = (
                "You are DARIZ-Kukalev 8-step troubleshooting engine for deepRLPH. "
                "You receive: (1) crash traceback, (2) execution logs, (3) project context, (4) human strategy. "
                "Your job is to propose a concrete, executable PRD fix for the next iteration.\n\n"
                "OUTPUT FORMAT (MANDATORY):\n"
                "Return ONLY one JSON object with either 'user_stories' or 'backlog' list. "
                "Each story must have: id, title, description, acceptance_criteria (list of strings),"
                " and verification (command or description). No markdown, no commentary outside JSON."
            )

            user_message_parts: List[str] = []
            if context_summary:
                user_message_parts.append(f"PROJECT CONTEXT:\n{context_summary}")
            if exit_code is not None or output_tail:
                user_message_parts.append(
                    f"CRASH INFO (exit_code={exit_code}):\n" f"{output_tail or terminal_dump}"
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
            }
        except Exception as e:
            logger.error("DARIZ run_dariz_fix failed: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _generate_dynamic_orchestrator_prompt(self, prd: Dict, config: ProjectConfig) -> str:
        """BE-006: AI analysis of context to generate grounded instructions."""
        context = f"Project: {config.name}\nSchemas: {json.dumps(config.metadata.get('dataset_schemas', 'none'))}"
        result = await self.deepseek_client.call_agent(
            system_prompt="You are a system architect. Generate specific implementation instructions.",
            user_message=f"Based on: {context}",
            thinking_budget=4000,
        )
        return result.get("response", "Proceed with implementation.")

    async def list_projects(self) -> List[Dict]:
        """Async project discovery."""
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

        loop = asyncio.get_running_loop()
        projects = await loop.run_in_executor(None, _scan)
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
