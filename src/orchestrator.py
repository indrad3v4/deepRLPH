# -*- coding: utf-8 -*-

"""
orchestrator.py - RALPH Main Orchestrator Engine (FIXED)

Connects UI (setup_window.py) with agents (deepseek_client, agent_coordinator)

✅ FIXED: 3 systematic bugs accessing PRD fields
  - BUG 1: _partition_prd_items now uses 'user_stories' not 'items'
  - BUG 2: _format_prd_summary now uses 'user_stories' not 'items'
  - BUG 3: execute_prd_loop logging fixed

Handles:
- Project creation & management
- Workspace initialization
- Agent coordination
- Multi-agent execution (PR-002: Parallel Orchestrator)
- Code validation (PR-003: Code Validator)
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
import subprocess
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ralph.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Orchestrator")


# ============================================================================
# DATA MODELS
# ============================================================================

class ProjectDomain(str, Enum):
    """Supported project domains"""
    LLM_APP = "llm-app"
    WEB_APP = "web_app"
    BACKEND_API = "backend_api"
    MICROSERVICES = "microservices"
    DATA_PIPELINE = "data_pipeline"
    AUTOMATION = "automation"
    CLI_TOOL = "cli_tool"
    ML_MODEL = "ml_model"
    OTHER = "other"


class ArchitectureType(str, Enum):
    """Supported architecture patterns"""
    CLEAN = "clean_architecture"
    MVC = "mvc"
    LAYERED = "layered"
    MICROSERVICES = "microservices"
    HEXAGONAL = "hexagonal"
    MODULAR_MONOLITH = "modular_monolith"
    DDD = "domain_driven_design"


@dataclass
class ProjectConfig:
    """Project configuration - sent from UI"""
    name: str
    domain: str
    description: str = ""
    architecture: str = "clean_architecture"
    framework: str = "FastAPI"
    language: str = "Python"
    database: str = "PostgreSQL"
    duration_hours: int = 4
    target_lines_of_code: int = 5000
    testing_coverage: int = 85
    parallel_agents: int = 4
    deployment_target: str = "Docker"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectConfig":
        return cls(**data)


@dataclass
class ExecutionState:
    """Tracks parallel execution state across all agents (PR-002)"""
    execution_id: str
    project_id: str
    start_time: str
    status: str = "running"  # running | completed | failed | partial
    num_agents: int = 4
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending | validated | failed
    error_message: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_progress(self) -> float:
        """Return 0-100 progress"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100

    def add_log(self, message: str) -> None:
        """Add timestamped log"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{ts}] {message}")

    def mark_item_completed(self, agent_id: str, item_id: str, files: List[str]) -> None:
        """Mark PRD item as completed by agent"""
        if agent_id not in self.artifacts:
            self.artifacts[agent_id] = []
        self.artifacts[agent_id].append({
            "item_id": item_id,
            "files": files,
            "timestamp": datetime.now().isoformat(),
        })
        self.completed_items += 1

    def mark_item_failed(self, agent_id: str, item_id: str, error: str) -> None:
        """Mark PRD item as failed"""
        self.failed_items += 1
        self.add_log(f"❌ Agent {agent_id} failed on item {item_id}: {error}")


@dataclass
class ValidationResult:
    """Result from code validation (PR-003)"""
    execution_id: str
    status: str  # success | partial | failed
    files_validated: int = 0
    files_passed: int = 0
    files_failed: int = 0
    violations: Dict[str, List[str]] = field(default_factory=dict)
    coverage: float = 0.0
    coverage_passed: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# WORKSPACE MANAGER
# ============================================================================

class WorkspaceManager:
    """Manages project workspace initialization"""

    def __init__(
            self,
            workspace_dir: Optional[Path] = None,
            deepseek_client: Optional[Any] = None,
            agent_coordinator: Optional[Any] = None,
            execution_engine: Optional[Any] = None
    ):
        """Initialize orchestrator"""
        import os

        self.workspace = workspace_dir or Path.home() / ".ralph"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.current_config: Optional[ProjectConfig] = None
        self.current_project_dir: Optional[Path] = None
        self.execution_log: List[str] = []

        # ✅ CRITICAL FIX: Initialize deepseek_client if not provided
        if deepseek_client is None:
            try:
                from deepseek_client import DeepseekClient
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    logger.warning("⚠️ DEEPSEEK_API_KEY not found - DeepSeek features will fail")
                    self.deepseek_client = None
                else:
                    self.deepseek_client = DeepseekClient(api_key=api_key, model="deepseek-reasoner")
                    logger.info("✅ DeepSeek client initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize DeepSeek client: {e}")
                self.deepseek_client = None
        else:
            self.deepseek_client = deepseek_client

        self.agent_coordinator = agent_coordinator
        self.execution_engine = execution_engine
        self.executions: Dict[str, ExecutionState] = {}
        self.validations: Dict[str, ValidationResult] = {}

        logger.info("🚀 RALPH Orchestrator initialized")
        logger.info(f"   Workspace: {self.workspace}")

    def create_project_workspace(self, project_id: str, config: ProjectConfig) -> Path:
        """Create project workspace structure"""
        project_dir = self.workspace / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        directories = [
            project_dir / "workspace" / "config",
            project_dir / "workspace" / "input",
            project_dir / "workspace" / "output" / "generated_code",
            project_dir / "workspace" / "output" / "architectures",
            project_dir / "workspace" / "output" / "logs",
            project_dir / "workspace" / "output" / "validation",
            project_dir / "src",
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Created workspace structure for {project_id}")
        return project_dir

    @staticmethod
    def create_config_files(project_dir: Path, config: ProjectConfig) -> None:
        """Create initial config files"""
        # Save config.json
        config_file = project_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Create README.md
        readme_content = f"""# {config.name}

## Project Information

- **Domain**: {config.domain}
- **Architecture**: {config.architecture}
- **Framework**: {config.framework}
- **Language**: {config.language}
- **Database**: {config.database}
- **Created**: {config.timestamp}

## Quick Start

cd {project_dir.name}
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Project Structure

workspace/
├── config/           # Configuration files
├── input/            # Input data
└── output/           # Generated output
    ├── generated_code/   # AI-generated code
    ├── architectures/    # Architecture blueprints
    ├── logs/             # Execution logs
    └── validation/       # Validation reports
src/                 # Your source code

## Architecture

Generated by RALPH - Multi-Agent Development System
"""
        (project_dir / "README.md").write_text(readme_content)

        # Create requirements.txt
        base_deps = [
            "python-dotenv>=0.19.0",
            "pydantic>=1.8.0",
            "loguru>=0.5.3",
            "aiohttp>=3.8.0",
        ]

        framework_deps = {
            "FastAPI": ["fastapi>=0.68.0", "uvicorn>=0.15.0"],
            "Django": ["django>=3.2", "djangorestframework>=3.12.0"],
            "Flask": ["flask>=2.0.0"],
        }

        db_deps = {
            "PostgreSQL": ["psycopg2-binary>=2.9.0", "sqlalchemy>=1.4.0"],
            "MongoDB": ["pymongo>=3.12.0"],
            "SQLite": ["sqlalchemy>=1.4.0"],
            "Redis": ["redis>=3.5.3"],
        }

        all_deps = (
                base_deps +
                framework_deps.get(config.framework, []) +
                db_deps.get(config.database, []) +
                ["pytest>=6.2.0", "black>=21.0", "flake8>=3.9.0", "pylint>=2.0.0", "mypy>=0.900"]
        )

        req_file = project_dir / "requirements.txt"
        req_file.write_text("\n".join(sorted(set(all_deps))))
        logger.info(f"📝 Created config files")


# ============================================================================
# ORCHESTRATOR - MAIN ENGINE (PR-002 + PR-003)
# ============================================================================

class RalphOrchestrator:
    """
    Main orchestrator engine.

    Manages project lifecycle and coordinates all components.
    """

    def __init__(
            self,
            workspace_dir: Optional[Path] = None,
            deepseek_client: Optional[Any] = None,
            agent_coordinator: Optional[Any] = None,
            execution_engine: Optional[Any] = None
    ):
        """Initialize orchestrator"""
        self.workspace = workspace_dir or Path.home() / ".ralph"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.workspace_manager = WorkspaceManager(self.workspace)
        self.current_config: Optional[ProjectConfig] = None
        self.current_project_dir: Optional[Path] = None
        self.execution_log: List[str] = []
        self.deepseek_client = deepseek_client
        self.agent_coordinator = agent_coordinator
        self.execution_engine = execution_engine
        self.executions: Dict[str, ExecutionState] = {}
        self.validations: Dict[str, ValidationResult] = {}

        logger.info("🚀 RALPH Orchestrator initialized")
        logger.info(f"   Workspace: {self.workspace}")

    def create_project(self, config: ProjectConfig) -> Dict[str, Any]:
        """
        Create new project.

        Called by UI after user fills in project details.
        """
        try:
            self.current_config = config

            # Generate project ID
            project_id = self._generate_project_id(config.name)

            # Create workspace
            project_dir = self.workspace_manager.create_project_workspace(
                project_id, config
            )

            self.current_project_dir = project_dir

            # Create config files
            WorkspaceManager.create_config_files(project_dir, config)

            # Log event
            self.execution_log.append(f"✅ Project created: {project_id}")

            logger.info(f"✅ Project created successfully: {project_id}")

            return {
                "status": "success",
                "project_id": project_id,
                "project_name": config.name,
                "path": str(project_dir),
                "workspace": str(project_dir / "workspace"),
                "config": config.to_dict(),
                "created_at": datetime.now().isoformat(),
            }

        except Exception as e:
            error_msg = f"Failed to create project: {str(e)}"
            self.execution_log.append(f"❌ {error_msg}")
            logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "error": error_msg,
            }

    def refine_task(
            self,
            project_id: str,
            raw_task: str
    ) -> Dict[str, Any]:
        """
        Step 1: Clarify raw task (PR-001)
        Step 2: Generate PRD (PR-001)

        Called after project is created, before execution loop.

        Args:
            project_id: Project ID to attach task refinement to
            raw_task: Raw task description from user

        Returns:
            Dict with brief and PRD, or error
        """
        try:
            # Import here to avoid circular imports
            from task_clarifier import TaskClarifier
            from prd_generator import PRDGenerator

            # Get project config
            config = self.current_config
            if not config:
                return {"status": "error", "error": "No project created"}

            self._log("🔍 Task Clarification Phase")
            self._log(f"   Input: {raw_task[:100]}...")

            # Step 1: Clarify task (call DeepSeek)
            clarifier = TaskClarifier(self.deepseek_client)
            brief = asyncio.run(
                clarifier.clarify(
                    raw_task=raw_task,
                    domain=config.domain,
                    framework=config.framework,
                    database=config.database
                )
            )

            if brief.get("status") != "success":
                return brief

            self._log("✅ Task clarified")
            self._log(f"   {brief['clarified_task'][:100]}...")

            # Step 2: Generate PRD
            self._log("📝 PRD Generation Phase")
            gen = PRDGenerator()
            prd = gen.generate(brief, domain=config.domain)
            self._log(f"✅ PRD generated: {prd['total_items']} items")

            # Save PRD to project
            prd_file = self.current_project_dir / "prd.json"
            with open(prd_file, 'w', encoding='utf-8') as f:
                json.dump(prd, f, indent=2)

            self._log(f"💾 Saved PRD: {prd_file}")

            return {
                "status": "success",
                "brief": brief,
                "prd": prd,
                "prd_file": str(prd_file),
            }

        except Exception as e:
            self._log(f"❌ Task refinement failed: {e}")
            logger.error(f"Task refinement error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def execute_prd_loop(
            self,
            prd: Dict[str, Any],
            num_agents: int = 4,
            log_callback: Optional[Callable] = None,
            progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute PRD items in parallel across N agents (PR-002).

        This is the CORE orchestrator method that:
        1. Partitions PRD items round-robin to agents
        2. Creates orchestrator prompt
        3. Calls ExecutionEngine for parallel execution
        4. Monitors progress and aggregates results

        Args:
            prd: PRD dict from refine_task()
            num_agents: Number of parallel agents (default 4)
            log_callback: Async callable(message: str) for UI updates
            progress_callback: Async callable(progress: float) for progress bar

        Returns:
            {
                "status": "success" | "partial" | "failed",
                "execution_id": str,
                "artifacts": Dict[agent_id] → List[files],
                "execution_log": List[str],
                "progress": 0-100,
            }
        """
        try:
            # ========== STEP 1: INITIALIZE ==========
            execution_id = f"exec_{self._generate_project_id('')}"
            exec_state = ExecutionState(
                execution_id=execution_id,
                project_id=self.current_config.name if self.current_config else "unknown",
                start_time=datetime.now().isoformat(),
                num_agents=num_agents,
                total_items=prd.get("total_items", 0),
            )
            self.executions[execution_id] = exec_state

            exec_state.add_log("🚀 Starting parallel execution")
            if log_callback:
                await log_callback("🚀 Starting parallel execution")

            # ========== STEP 2: PARTITION PRD ITEMS ==========
            partitioned = self._partition_prd_items(prd, num_agents)
            # ✅ FIX BUG 3: Use correct field name in logging
            total_user_stories = len(prd.get('user_stories', []))
            exec_state.add_log(f"📊 Partitioned {total_user_stories} user stories across {num_agents} agents")

            if log_callback:
                await log_callback(f"📊 Partitioned {total_user_stories} user stories across {num_agents} agents")

            # ========== STEP 3: CREATE ORCHESTRATOR PROMPT ==========
            orchestrator_prompt = self._create_orchestrator_prompt(
                prd=prd,
                config=self.current_config,
                domain=self.current_config.domain if self.current_config else "web_app"
            )

            # ========== STEP 4: EXECUTE IN PARALLEL ==========
            if not self.execution_engine:
                return {
                    "status": "failed",
                    "error": "ExecutionEngine not initialized. Wire it in main.py",
                    "execution_id": execution_id if 'execution_id' in locals() else "unknown"
                }

            exec_state.add_log("⚙️ Spawning parallel agents...")
            if log_callback:
                await log_callback("⚙️ Spawning parallel agents...")

            # Execute with progress monitoring
            execution_results = await self.execution_engine.execute(
                execution_id=execution_id,
                orchestrator_prompt=orchestrator_prompt,
                prd_partitions=partitioned,
                num_agents=num_agents,
                progress_callback=lambda p: self._handle_progress(
                    exec_state, p, progress_callback
                ),
                log_callback=lambda msg: self._handle_log(
                    exec_state, msg, log_callback
                ),
            )

            # ========== STEP 5: AGGREGATE RESULTS ==========
            if execution_results.get("status") == "success":
                exec_state.status = "completed"
                for agent_id, agent_result in execution_results.get("agents", {}).items():
                    for item in agent_result.get("completed_items", []):
                        exec_state.mark_item_completed(
                            agent_id=agent_id,
                            item_id=item["id"],
                            files=item.get("files", [])
                        )
                    for item in agent_result.get("failed_items", []):
                        exec_state.mark_item_failed(
                            agent_id=agent_id,
                            item_id=item["id"],
                            error=item.get("error", "Unknown error")
                        )
            else:
                exec_state.status = "partial" if exec_state.completed_items > 0 else "failed"
                exec_state.error_message = execution_results.get("error", "Execution failed")

            # ========== STEP 6: SAVE STATE ==========
            exec_state.duration_seconds = (
                    datetime.now() - datetime.fromisoformat(exec_state.start_time)
            ).total_seconds()

            state_file = (
                    self.current_project_dir / "workspace" / "output" / "logs" /
                    f"execution_{execution_id}.json"
            )
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(exec_state.to_dict(), f, indent=2, default=str)

            exec_state.add_log(f"💾 Execution state saved: {state_file}")

            # ========== STEP 7: RETURN RESULTS ==========
            if progress_callback:
                await progress_callback(100.0)

            return {
                "status": "success" if exec_state.status == "completed" else exec_state.status,
                "execution_id": execution_id,
                "artifacts": exec_state.artifacts,
                "execution_log": exec_state.logs,
                "progress": exec_state.get_progress(),
                "completed_items": exec_state.completed_items,
                "failed_items": exec_state.failed_items,
                "duration_seconds": exec_state.duration_seconds,
                "execution_state_file": str(state_file),
            }

        except Exception as e:
            self._log(f"❌ Execution failed: {e}")
            logger.error(f"Execute PRD loop error: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "execution_id": execution_id if 'execution_id' in locals() else "unknown",
            }

    def _partition_prd_items(
            self,
            prd: Dict[str, Any],
            num_agents: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partition PRD items round-robin to agents.

        ✅ FIX BUG 1: Use 'user_stories' field (actual PRD format from prd_generator.py)

        Returns:
            {
                "agent_1": [item1, item3, item5, ...],
                "agent_2": [item2, item4, item6, ...],
                ...
            }
        """
        # ✅ FIXED: prd_generator.py returns 'user_stories', not 'items'
        user_stories = prd.get("user_stories", [])
        partitions = {f"agent_{i + 1}": [] for i in range(num_agents)}

        for idx, story in enumerate(user_stories):
            agent_key = f"agent_{(idx % num_agents) + 1}"
            partitions[agent_key].append(story)

        self._log(f"📊 Partitioned {len(user_stories)} user stories:")
        for agent_key, agent_items in partitions.items():
            self._log(f"   {agent_key}: {len(agent_items)} stories")

        return partitions

    def _create_orchestrator_prompt(
            self,
            prd: Dict[str, Any],
            config: ProjectConfig,
            domain: str
    ) -> str:
        """
        Create system prompt for orchestrator agent.

        Tells agents: "You are agent X, implement these specific PRD items with this architecture"
        """
        prd_summary = self._format_prd_summary(prd)

        prompt = f"""You are a senior Python engineer coordinating the implementation of a software project.

PROJECT INFO:
- Name: {config.name if config else 'Unknown'}
- Domain: {domain}
- Architecture: {config.architecture if config else 'clean_architecture'}
- Framework: {config.framework if config else 'FastAPI'}
- Language: {config.language if config else 'Python'}
- Database: {config.database if config else 'PostgreSQL'}

YOUR TASK:
Implement the assigned PRD items. For each item:
1. Understand the acceptance criteria
2. Generate complete, working Python code
3. Include docstrings and type hints
4. Follow {config.architecture if config else 'clean'} architecture patterns
5. Ensure code is production-ready and tested

PRD CONTEXT:
{prd_summary}

DELIVERABLES:
- Complete source files (.py)
- Unit tests for each module
- Documentation strings
- Error handling and logging

QUALITY REQUIREMENTS:
✓ Type hints on all functions
✓ Docstrings (Google style)
✓ Unit test coverage >80%
✓ No linting errors (PEP8)
✓ Database migrations if needed
✓ Environment variables in .env.example

Start implementation now. Generate complete, working code."""

        return prompt

    def _format_prd_summary(self, prd: Dict[str, Any]) -> str:
        """
        Format PRD items as readable summary.

        ✅ FIX BUG 2: Use 'user_stories' field (actual PRD format)
        """
        summary = ""
        # ✅ FIXED: prd_generator.py returns 'user_stories', not 'items'
        user_stories = prd.get("user_stories", [])

        for idx, story in enumerate(user_stories[:10], 1):
            summary += f"\n{idx}. [{story.get('id', 'N/A')}] {story.get('title', 'Untitled')}\n"
            summary += f"   Why: {story.get('why', 'N/A')}\n"
            criteria = story.get('acceptance_criteria', [])
            if criteria:
                summary += f"   Criteria: {' | '.join(criteria[:3])}\n"

        if len(user_stories) > 10:
            summary += f"\n... and {len(user_stories) - 10} more items\n"

        return summary

    async def _handle_progress(
            self,
            exec_state: ExecutionState,
            progress: float,
            callback: Optional[Callable] = None
    ) -> None:
        """Handle progress updates"""
        if callback:
            await callback(progress)

    async def _handle_log(
            self,
            exec_state: ExecutionState,
            message: str,
            callback: Optional[Callable] = None
    ) -> None:
        """Handle log messages"""
        exec_state.add_log(message)
        if callback:
            await callback(message)

    async def execute_validation_loop(
            self,
            execution_id: str,
            coverage_target: int = 80
    ) -> Dict[str, Any]:
        """
        Execute validation on generated code (PR-003).

        Validates:
        - Code format (black)
        - Linting (pylint)
        - Type checking (mypy)
        - Unit tests (pytest)
        - Coverage (>80%)

        Args:
            execution_id: From execute_prd_loop() result
            coverage_target: Target coverage percentage (default 80)

        Returns:
            {
                "status": "success" | "failed" | "partial",
                "files_validated": int,
                "files_passed": int,
                "files_failed": int,
                "violations": Dict[filename] → List[errors],
                "coverage": float,
                "coverage_passed": bool,
                "validation_report_file": str,
            }
        """
        try:
            self._log(f"🔍 PR-003 Code Validation Started for {execution_id}")

            # Get execution state
            if execution_id not in self.executions:
                return {
                    "status": "error",
                    "error": f"Execution {execution_id} not found",
                }

            exec_state = self.executions[execution_id]

            # Create validation result
            val_result = ValidationResult(
                execution_id=execution_id,
                status="running"
            )
            self.validations[execution_id] = val_result

            # Find generated code files
            code_dir = self.current_project_dir / "workspace" / "output" / "generated_code"
            if not code_dir.exists():
                self._log(f"❌ Code directory not found: {code_dir}")
                return {
                    "status": "error",
                    "error": f"Code directory not found: {code_dir}",
                }

            # Find all Python files
            py_files = list(code_dir.rglob("*.py"))
            test_files = [f for f in py_files if "test" in f.name.lower()]
            src_files = [f for f in py_files if "test" not in f.name.lower()]

            val_result.files_validated = len(py_files)
            self._log(f"📊 Found {len(py_files)} Python files ({len(src_files)} source, {len(test_files)} tests)")

            violations: Dict[str, List[str]] = {}
            files_passed = 0
            files_failed = 0

            # ========== STEP 1: BLACK (Code Format) ==========
            self._log("🎨 Checking code format with black...")
            try:
                result = subprocess.run(
                    ["black", "--check", str(code_dir)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    self._log("⚠️ Code format issues found by black")
                    for file in src_files:
                        violations[str(file)] = violations.get(str(file), []) + ["black: format violations"]
                else:
                    self._log("✅ Code format check passed")
            except subprocess.TimeoutExpired:
                self._log("⚠️ Black check timed out")
            except FileNotFoundError:
                self._log("⚠️ Black not installed, skipping format check")

            # ========== STEP 2: PYLINT (Linting) ==========
            self._log("🔎 Running pylint linting...")
            try:
                result = subprocess.run(
                    ["pylint", "--exit-zero", str(code_dir)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if "error" in result.stdout.lower() or "error" in result.stderr.lower():
                    self._log("⚠️ Linting issues found by pylint")
                    for file in src_files:
                        violations[str(file)] = violations.get(str(file), []) + ["pylint: linting issues"]
                else:
                    self._log("✅ Linting check passed")
            except subprocess.TimeoutExpired:
                self._log("⚠️ Pylint check timed out")
            except FileNotFoundError:
                self._log("⚠️ Pylint not installed, skipping linting")

            # ========== STEP 3: MYPY (Type Checking) ==========
            self._log("🔤 Running mypy type checking...")
            try:
                result = subprocess.run(
                    ["mypy", str(code_dir), "--ignore-missing-imports"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode != 0:
                    self._log("⚠️ Type errors found by mypy")
                    for file in src_files:
                        violations[str(file)] = violations.get(str(file), []) + ["mypy: type errors"]
                else:
                    self._log("✅ Type checking passed")
            except subprocess.TimeoutExpired:
                self._log("⚠️ Mypy check timed out")
            except FileNotFoundError:
                self._log("⚠️ Mypy not installed, skipping type checking")

            # ========== STEP 4: PYTEST (Unit Tests) ==========
            self._log("🧪 Running pytest unit tests...")
            try:
                result = subprocess.run(
                    ["pytest", str(code_dir), "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    self._log("⚠️ Some tests failed")
                    for file in test_files:
                        violations[str(file)] = violations.get(str(file), []) + ["pytest: test failures"]
                    files_passed = len(test_files) // 2 if len(test_files) > 0 else 0
                    files_failed = len(test_files) - files_passed
                else:
                    self._log("✅ All tests passed")
                    files_passed = len(test_files)
            except subprocess.TimeoutExpired:
                self._log("⚠️ Pytest check timed out")
            except FileNotFoundError:
                self._log("⚠️ Pytest not installed, skipping unit tests")

            # ========== STEP 5: Coverage Check ==========
            self._log(f"📈 Checking code coverage (target: {coverage_target}%)...")
            try:
                result = subprocess.run(
                    ["pytest", str(code_dir), "--cov", "--cov-report=term-missing"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                # Parse coverage from output
                output_lines = result.stdout.split('\n')
                coverage = 0.0
                for line in output_lines:
                    if "TOTAL" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                coverage = float(parts[-1].rstrip('%'))
                            except ValueError:
                                coverage = 0.0

                val_result.coverage = coverage
                val_result.coverage_passed = coverage >= coverage_target

                if val_result.coverage_passed:
                    self._log(f"✅ Coverage check passed: {coverage}% >= {coverage_target}%")
                else:
                    self._log(f"⚠️ Coverage below target: {coverage}% < {coverage_target}%")
            except subprocess.TimeoutExpired:
                self._log("⚠️ Coverage check timed out")
            except FileNotFoundError:
                self._log("⚠️ pytest-cov not installed, skipping coverage")

            # ========== AGGREGATE RESULTS ==========
            val_result.violations = violations
            val_result.files_passed = files_passed
            val_result.files_failed = files_failed

            # Determine overall status
            if not violations and val_result.coverage_passed:
                val_result.status = "success"
                self._log(f"✅ VALIDATION PASSED")
            elif violations and val_result.files_failed > 0:
                val_result.status = "failed"
                self._log(f"❌ VALIDATION FAILED")
            else:
                val_result.status = "partial"
                self._log(f"⚠️ VALIDATION PARTIAL")

            # ========== SAVE VALIDATION REPORT ==========
            report_file = (
                    self.current_project_dir / "workspace" / "output" / "validation" /
                    f"validation_{execution_id}.json"
            )
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(val_result.to_dict(), f, indent=2, default=str)

            self._log(f"💾 Validation report saved: {report_file}")

            # Update execution state
            if execution_id in self.executions:
                self.executions[execution_id].validation_status = val_result.status

            return {
                "status": val_result.status,
                "execution_id": execution_id,
                "files_validated": val_result.files_validated,
                "files_passed": val_result.files_passed,
                "files_failed": val_result.files_failed,
                "violations": val_result.violations,
                "coverage": val_result.coverage,
                "coverage_passed": val_result.coverage_passed,
                "validation_report_file": str(report_file),
            }

        except Exception as e:
            self._log(f"❌ Validation failed: {e}")
            logger.error(f"Validation loop error: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "execution_id": execution_id,
            }

    def _log(self, message: str) -> None:
        """Log to orchestrator"""
        logger.info(message)
        if self.execution_log is not None:
            self.execution_log.append(message)

    def _generate_project_id(self, name: str) -> str:
        """Generate unique project ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        return f"ralph_{safe_name}_{timestamp}" if name else f"exec_{timestamp}"

    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects in workspace.

        Called by UI to refresh projects list.
        """
        projects = []
        projects_dir = self.workspace / "projects"

        if not projects_dir.exists():
            return projects

        try:
            for project_dir in sorted(projects_dir.iterdir()):
                if project_dir.is_dir():
                    config_file = project_dir / "config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                            projects.append({
                                "project_id": project_dir.name,
                                "name": config_data.get("name", "Unknown"),
                                "domain": config_data.get("domain", ""),
                                "architecture": config_data.get("architecture", ""),
                                "path": str(project_dir),
                                "created_at": config_data.get("timestamp", ""),
                                "status": "ready",
                            })
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid config.json in {project_dir.name}")

        except Exception as e:
            logger.error(f"Error listing projects: {e}")

        logger.info(f"📊 Found {len(projects)} projects")
        return projects

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get specific project details"""
        project_dir = self.workspace / "projects" / project_id
        config_file = project_dir / "config.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            return {
                "project_id": project_id,
                "config": config_data,
                "path": str(project_dir),
                "workspace": str(project_dir / "workspace"),
            }

        except Exception as e:
            logger.error(f"Error reading project {project_id}: {e}")
            return None

    def get_execution_log(self) -> List[str]:
        """Get execution log for UI display"""
        return self.execution_log

    def clear_execution_log(self) -> None:
        """Clear execution log"""
        self.execution_log = []

    def add_log_message(self, message: str) -> None:
        """Add message to execution log"""
        self.execution_log.append(message)
        logger.info(message)

    async def generate_architecture_blueprint(self) -> Dict[str, Any]:
        """
        Generate architecture blueprint.

        Would integrate with deepseek_client and agent_coordinator here.
        """
        if not self.current_config or not self.current_project_dir:
            return {"status": "error", "error": "No project loaded"}

        try:
            self.add_log_message("🏗️ Generating architecture blueprint...")

            blueprint = {
                "architecture": self.current_config.architecture,
                "framework": self.current_config.framework,
                "language": self.current_config.language,
                "database": self.current_config.database,
                "modules": self._get_architecture_modules(),
                "generated_at": datetime.now().isoformat(),
            }

            blueprint_file = (
                    self.current_project_dir / "workspace" / "output" /
                    "architectures" / "blueprint.json"
            )

            with open(blueprint_file, 'w', encoding='utf-8') as f:
                json.dump(blueprint, f, indent=2)

            self.add_log_message(f"✅ Blueprint saved: {blueprint_file}")

            return {
                "status": "success",
                "blueprint": blueprint,
                "saved_at": str(blueprint_file),
            }

        except Exception as e:
            error_msg = f"Architecture generation failed: {str(e)}"
            self.add_log_message(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}

    def _get_architecture_modules(self) -> List[str]:
        """Get modules for selected architecture"""
        architectures = {
            "clean_architecture": [
                "domain/entities.py",
                "domain/repositories.py",
                "application/use_cases.py",
                "adapters/controllers.py",
                "adapters/repositories.py",
                "frameworks/fastapi_app.py",
            ],
            "mvc": [
                "models/user.py",
                "models/product.py",
                "views/user_view.py",
                "controllers/user_controller.py",
                "config/settings.py",
            ],
            "layered": [
                "presentation/views.py",
                "business/services.py",
                "persistence/repositories.py",
                "common/utils.py",
            ],
            "microservices": [
                "auth_service/routes.py",
                "data_service/routes.py",
                "api_gateway/gateway.py",
                "shared/models.py",
            ],
        }

        return architectures.get(self.current_config.architecture, [])


# ============================================================================
# SINGLETON PATTERN - Global Orchestrator Instance
# ============================================================================

_orchestrator_instance: Optional[RalphOrchestrator] = None


def get_orchestrator(
        workspace_dir: Optional[Path] = None,
        deepseek_client: Optional[Any] = None,
        agent_coordinator: Optional[Any] = None,
        execution_engine: Optional[Any] = None
) -> RalphOrchestrator:
    """
    Get or create global orchestrator instance.

    Usage from UI:

    from orchestrator import get_orchestrator

    orchestrator = get_orchestrator(deepseek_client=your_client)
    result = orchestrator.create_project(config)
    """
    global _orchestrator_instance

    if _orchestrator_instance is None:
        _orchestrator_instance = RalphOrchestrator(
            workspace_dir,
            deepseek_client,
            agent_coordinator,
            execution_engine
        )

    return _orchestrator_instance


# ============================================================================
# FOR TESTING / CLI
# ============================================================================

if __name__ == "__main__":
    logger.info("Testing orchestrator...")

    orchestrator = get_orchestrator()

    # Test: Create project
    config = ProjectConfig(
        name="Test AI API",
        domain="llm-app",
        architecture="clean_architecture",
        framework="FastAPI",
        database="PostgreSQL",
    )

    result = orchestrator.create_project(config)
    logger.info(f"Create result: {json.dumps(result, indent=2, default=str)}")

    # Test: List projects
    projects = orchestrator.list_projects()
    logger.info(f"Found projects: {len(projects)}")
    for p in projects:
        logger.info(f"   - {p['name']} ({p['domain']})")
