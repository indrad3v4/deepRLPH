# -*- coding: utf-8 -*-

"""
orchestrator.py - RALPH Main Orchestrator Engine (ML COMPETITION ENHANCED + KPI CONFIG)

Adds metric-aware project creation and passes richer context into PRD generation.

Changes in this revision:
- When creating ML competition projects, auto-generate metrics_config.json under the project folder
- metrics_config.json schema: {"name", "pattern", "target"}
- refine_task now passes project metadata (including competition URL, eval metric) to PRDGenerator
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

class ProjectType(str, Enum):
    """Type of project to create"""
    API_DEVELOPMENT = "api_dev"
    ML_COMPETITION = "ml_competition"
    DATA_PIPELINE = "data_pipeline"
    LLM_APP = "llm_app"


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
    TIME_SERIES = "time_series_forecasting"
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


class MLFramework(str, Enum):
    """ML frameworks"""
    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"
    JAX = "JAX"
    SCIKIT_LEARN = "scikit-learn"


class MLModelType(str, Enum):
    """ML model architectures"""
    LSTM = "LSTM"
    GRU = "GRU"
    TRANSFORMER = "Transformer"
    CNN_LSTM = "CNN-LSTM"
    RNN = "RNN"
    CUSTOM = "Custom"


@dataclass
class MLConfig:
    """ML Competition specific configuration"""
    # Competition metadata
    competition_source: str = ""  # e.g., "wundernn.io", "kaggle", "custom"
    competition_url: str = ""

    # Dataset
    dataset_files: List[str] = field(default_factory=list)  # ["train.csv", "test.csv"]
    problem_type: str = "time_series_forecasting"  # classification, regression, etc.
    sequence_length: int = 1000
    num_features: int = 32
    target_variable: str = ""

    # Model
    model_type: str = "LSTM"
    ml_framework: str = "PyTorch"

    # Training
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    loss_function: str = "MSE"

    # Evaluation
    eval_metric: str = "R²"  # R², RMSE, MAE, Accuracy
    validation_split: float = 0.2
    cross_validation: str = "time_series_split"

    # Hardware
    gpu_required: bool = False
    estimated_training_hours: int = 24

    # Submission
    submission_format: str = "CSV"
    submission_deadline: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MLConfig":
        return cls(**data)


@dataclass
class ProjectConfig:
    """Project configuration - sent from UI (ENHANCED for ML)"""
    name: str
    domain: str
    description: str = ""

    # Project type determines which fields are used
    project_type: str = "api_dev"  # api_dev | ml_competition | data_pipeline | llm_app

    # API/Backend fields (used when project_type == "api_dev")
    architecture: str = "clean_architecture"
    framework: str = "FastAPI"
    language: str = "Python"
    database: str = "PostgreSQL"
    duration_hours: int = 4
    target_lines_of_code: int = 5000
    testing_coverage: int = 85
    parallel_agents: int = 4
    deployment_target: str = "Docker"

    # ML fields (used when project_type == "ml_competition")
    ml_config: Optional[MLConfig] = None

    # Flexible project metadata (competition URL, metric, presets, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

        # If ml_config is a dict, convert to MLConfig
        if isinstance(self.ml_config, dict):
            self.ml_config = MLConfig.from_dict(self.ml_config)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.ml_config:
            data['ml_config'] = self.ml_config.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectConfig":
        if 'ml_config' in data and data['ml_config'] is not None:
            data['ml_config'] = MLConfig.from_dict(data['ml_config'])
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
# WORKSPACE MANAGER (ENHANCED for ML)
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

        # ✅ CRITICAL FIX: Resolve to absolute path
        if workspace_dir:
            self.workspace = workspace_dir.resolve()
        else:
            self.workspace = (Path.home() / ".ralph").resolve()

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
        """Create project workspace structure (supports ML projects)"""
        project_dir = self.workspace / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # Base directories
        directories = [
            project_dir / "workspace" / "config",
            project_dir / "workspace" / "input",
            project_dir / "workspace" / "output" / "generated_code",
            project_dir / "workspace" / "output" / "architectures",
            project_dir / "workspace" / "output" / "logs",
            project_dir / "workspace" / "output" / "validation",
            project_dir / "src",
        ]

        # ML-specific directories
        if config.project_type == ProjectType.ML_COMPETITION or config.metadata.get("project_type") == "ml_competition":
            directories.extend([
                project_dir / "data" / "raw",
                project_dir / "data" / "processed",
                project_dir / "models" / "checkpoints",
                project_dir / "models" / "final",
                project_dir / "notebooks",
                project_dir / "submissions",
            ])

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Created workspace structure for {project_id}")
        return project_dir

    @staticmethod
    def create_config_files(project_dir: Path, config: ProjectConfig) -> None:
        """Create initial config files (supports ML projects + KPI config)"""
        # Save config.json
        config_file = project_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Create README.md
        if config.project_type == ProjectType.ML_COMPETITION or config.metadata.get("project_type") == "ml_competition":
            readme_content = WorkspaceManager._create_ml_readme(config)
        else:
            readme_content = WorkspaceManager._create_api_readme(config, project_dir)

        (project_dir / "README.md").write_text(readme_content)

        # Create requirements.txt
        WorkspaceManager._create_requirements(project_dir, config)

        # NEW: Auto-generate per-project metrics_config.json for ML competitions
        if config.project_type == ProjectType.ML_COMPETITION or config.metadata.get("project_type") == "ml_competition":
            WorkspaceManager._create_default_metrics_config(project_dir, config)

        logger.info(f"📝 Created config files")

    @staticmethod
    def _create_default_metrics_config(project_dir: Path, config: ProjectConfig) -> None:
        """Create a default metrics_config.json for ML competition projects.

        Uses project metadata or ml_config.eval_metric to define the primary KPI.
        """
        meta = config.metadata or {}
        metric_name = meta.get('eval_metric')
        if not metric_name and config.ml_config:
            metric_name = config.ml_config.eval_metric
        if not metric_name:
            metric_name = "score"

        # Basic patterns for common metrics; refined later per-project if needed
        patterns = {
            "R²": r"R2(?: Score)?: (?P<value>[-+]?\d*\.\d+)",
            "RMSE": r"RMSE: (?P<value>[-+]?\d*\.\d+)",
            "MAE": r"MAE: (?P<value>[-+]?\d*\.\d+)",
            "Accuracy": r"Accuracy: (?P<value>[-+]?\d*\.\d+)",
            "F1 Score": r"F1(?: Score)?: (?P<value>[-+]?\d*\.\d+)",
            "AUC-ROC": r"AUC-ROC: (?P<value>[-+]?\d*\.\d+)",
            "weighted_pearson": r"Weighted Pearson(?: Correlation Coefficient)?: (?P<value>[-+]?\d*\.\d+)",
            "score": r"Score: (?P<value>[-+]?\d*\.\d+)",
        }

        # Map some friendly names to internal keys
        name_key = metric_name
        if metric_name.lower().startswith("weighted") and "pearson" in metric_name.lower():
            name_key = "weighted_pearson"

        pattern = patterns.get(name_key, patterns["score"])

        # Conservative default target; user can edit per project
        default_target = float(meta.get('metric_target', 0.0))

        cfg = {
            "name": name_key,
            "pattern": pattern,
            "target": default_target,
        }

        metrics_file = project_dir / "metrics_config.json"
        if not metrics_file.exists():
            metrics_file.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            logger.info("📊 Created default metrics_config.json: %s", metrics_file)
        else:
            logger.info("📊 metrics_config.json already exists, not overwriting: %s", metrics_file)

    @staticmethod
    def _create_ml_readme(config: ProjectConfig) -> str:
        """Create README for ML competition project"""
        # Check both ml_config and metadata for ML info
        ml = config.ml_config
        meta = config.metadata

        comp_source = ml.competition_source if ml else meta.get('competition_url', 'N/A')
        comp_url = ml.competition_url if ml else meta.get('competition_url', 'N/A')
        problem_type = ml.problem_type if ml else meta.get('problem_type', 'N/A')
        ml_framework = ml.ml_framework if ml else meta.get('ml_framework', 'PyTorch')
        model_type = ml.model_type if ml else meta.get('model_type', 'LSTM')
        batch_size = ml.batch_size if ml else meta.get('batch_size', 64)
        epochs = ml.epochs if ml else meta.get('epochs', 100)
        learning_rate = ml.learning_rate if ml else meta.get('learning_rate', 0.001)
        eval_metric = ml.eval_metric if ml else meta.get('eval_metric', 'R²')

        return f"""# {config.name}

## ML Competition Project

- **Competition**: {comp_source}
- **URL**: {comp_url}
- **Problem Type**: {problem_type}
- **Framework**: {ml_framework}
- **Model**: {model_type}
- **Created**: {config.timestamp}

## Dataset

- **Sequence Length**: {ml.sequence_length if ml else 'N/A'}
- **Features**: {ml.num_features if ml else 'N/A'}
- **Target**: {ml.target_variable if ml else 'N/A'}
- **Files**: {', '.join(meta.get('dataset_files', []))}

## Training Config

- **Batch Size**: {batch_size}
- **Epochs**: {epochs}
- **Learning Rate**: {learning_rate}
- **Optimizer**: {ml.optimizer if ml else 'Adam'}
- **Loss**: {ml.loss_function if ml else 'MSE'}

## Evaluation

- **Metric**: {eval_metric}
- **Validation Split**: {ml.validation_split if ml else 0.2}
- **Cross-Validation**: {ml.cross_validation if ml else 'time_series_split'}

## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Place dataset files in data/raw/

# Train model
python src/train.py

# Generate predictions
python src/predict.py

# Create submission
python src/submit.py
```

## Project Structure

```
data/
├── raw/              # Original competition data
└── processed/        # Preprocessed data
models/
├── checkpoints/      # Training checkpoints
└── final/           # Final trained models
notebooks/           # Jupyter notebooks for EDA
submissions/         # Submission files
src/
├── data_loader.py   # Data loading and preprocessing
├── model.py         # Model architecture
├── train.py         # Training script
├── predict.py       # Inference script
└── submit.py        # Submission generator
```

## Notes

- **GPU**: {'Required' if ml and ml.gpu_required else 'Optional'}
- **Est. Training Time**: {ml.estimated_training_hours if ml else 24} hours
- **Submission Format**: {ml.submission_format if ml else 'CSV'}
- **Deadline**: {ml.submission_deadline if ml else 'TBD'}

Generated by RALPH - Multi-Agent ML Development System
"""

    @staticmethod
    def _create_api_readme(config: ProjectConfig, project_dir: Path) -> str:
        """Create README for API development project"""
        return f"""# {config.name}

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

    @staticmethod
    def _create_requirements(project_dir: Path, config: ProjectConfig) -> None:
        """Create requirements.txt based on project type"""
        base_deps = [
            "python-dotenv>=0.19.0",
            "pydantic>=1.8.0",
            "loguru>=0.5.3",
        ]

        if config.project_type == ProjectType.ML_COMPETITION or config.metadata.get("project_type") == "ml_competition":
            # ML dependencies
            ml = config.ml_config
            meta = config.metadata
            ml_framework = ml.ml_framework if ml else meta.get('ml_framework', 'PyTorch')

            ml_deps = [
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "scikit-learn>=1.0.0",
                "matplotlib>=3.4.0",
                "seaborn>=0.11.0",
                "jupyter>=1.0.0",
                "tqdm>=4.62.0",
            ]

            # Framework-specific
            if ml_framework == "PyTorch":
                ml_deps.extend([
                    "torch>=1.10.0",
                    "torchvision>=0.11.0",
                ])
            elif ml_framework == "TensorFlow":
                ml_deps.extend([
                    "tensorflow>=2.7.0",
                ])
            elif ml_framework == "JAX":
                ml_deps.extend([
                    "jax>=0.3.0",
                    "jaxlib>=0.3.0",
                ])

            all_deps = base_deps + ml_deps + [
                "pytest>=6.2.0",
                "black>=21.0",
                "flake8>=3.9.0",
            ]
        else:
            # API dependencies
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
                    ["pytest>=6.2.0", "black>=21.0", "flake8>=3.9.0", "pylint>=2.0.0", "mypy>=0.900", "aiohttp>=3.8.0"]
            )

        req_file = project_dir / "requirements.txt"
        req_file.write_text("\n".join(sorted(set(all_deps))))


# ============================================================================
# ORCHESTRATOR - MAIN ENGINE (PR-002 + PR-003 + ML SUPPORT)
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
        # ✅ CRITICAL FIX: Resolve to absolute path for consistent operations
        if workspace_dir:
            self.workspace = workspace_dir.resolve()
        else:
            self.workspace = (Path.home() / ".ralph").resolve()

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
        Create new project (API or ML).

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

            # Create config + README + requirements + metrics_config (for ML)
            WorkspaceManager.create_config_files(project_dir, config)

            # Log event
            project_type_display = config.metadata.get('project_type', config.project_type).upper().replace("_", " ")
            self.execution_log.append(f"✅ {project_type_display} project created: {project_id}")

            logger.info(f"✅ Project created successfully: {project_id}")

            return {
                "status": "success",
                "project_id": project_id,
                "project_name": config.name,
                "project_type": config.metadata.get('project_type', config.project_type),
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
        """Task clarification + PRD generation (PR-001, now project-aware)"""
        try:
            from task_clarifier import TaskClarifier
            from prd_generator import PRDGenerator

            config = self.current_config
            if not config:
                return {"status": "error", "error": "No project created"}

            self._log("🔍 Task Clarification Phase")
            self._log(f"   Input: {raw_task[:100]}...")

            clarifier = TaskClarifier(self.deepseek_client)

            # Pass project context into the clarifier so Deepseek can reason about competition/API specifics
            brief = asyncio.run(
                clarifier.clarify(
                    raw_task=raw_task,
                    domain=config.domain,
                    framework=config.framework,
                    database=config.database,
                    metadata=config.metadata,
                )
            )

            if brief.get("status") != "success":
                return brief

            # Attach project metadata and id into the brief for downstream PRD generation
            brief["project_id"] = project_id
            brief["project_metadata"] = config.metadata

            self._log("✅ Task clarified")
            self._log(f"   {brief['clarified_task'][:100]}...")

            self._log("📝 PRD Generation Phase")
            gen = PRDGenerator()
            prd = gen.generate(
                technical_brief=brief,
                domain=config.domain,
                project_metadata=config.metadata,
            )
            self._log(f"✅ PRD generated: {prd['total_items']} items")

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
        """Execute PRD items in parallel (PR-002)"""
        # 🔍 DEBUG: Log function entry
        logger.info("🔍 [DEBUG] execute_prd_loop() CALLED")
        logger.info(f"🔍 [DEBUG] num_agents: {num_agents}")
        logger.info(f"🔍 [DEBUG] log_callback type: {type(log_callback)}")
        logger.info(f"🔍 [DEBUG] progress_callback type: {type(progress_callback)}")

        try:
            execution_id = f"exec_{self._generate_project_id('')}"
            logger.info(f"🔍 [DEBUG] Generated execution_id: {execution_id}")

            exec_state = ExecutionState(
                execution_id=execution_id,
                project_id=self.current_config.name if self.current_config else "unknown",
                start_time=datetime.now().isoformat(),
                num_agents=num_agents,
                total_items=prd.get("total_items", 0),
            )
            self.executions[execution_id] = exec_state
            logger.info("🔍 [DEBUG] ExecutionState created")

            exec_state.add_log("🚀 Starting parallel execution")
            if log_callback:
                await log_callback("🚀 Starting parallel execution")

            logger.info("🔍 [DEBUG] Partitioning PRD items...")
            partitioned = self._partition_prd_items(prd, num_agents)
            total_user_stories = len(prd.get('user_stories', []))
            logger.info(f"🔍 [DEBUG] Partitioned {total_user_stories} stories")

            exec_state.add_log(f"📊 Partitioned {total_user_stories} user stories across {num_agents} agents")

            if log_callback:
                await log_callback(f"📊 Partitioned {total_user_stories} user stories across {num_agents} agents")

            logger.info("🔍 [DEBUG] Creating orchestrator prompt...")
            orchestrator_prompt = self._create_orchestrator_prompt(
                prd=prd,
                config=self.current_config,
                domain=self.current_config.domain if self.current_config else "web_app"
            )
            logger.info(f"🔍 [DEBUG] Orchestrator prompt created, length: {len(orchestrator_prompt)}")

            if not self.execution_engine:
                logger.info("🔍 [DEBUG] Creating ExecutionEngine for current project...")
                if not self.current_project_dir:
                    logger.error("❌ No project directory set!")
                    return {
                        "status": "failed",
                        "error": "No project loaded. Create a project first.",
                        "execution_id": execution_id
                    }

                from execution_engine import ExecutionEngine
                self.execution_engine = ExecutionEngine(
                    project_dir=self.current_project_dir,
                    deepseek_client=self.deepseek_client,
                    agent_coordinator=self.agent_coordinator,
                )
                logger.info("✅ ExecutionEngine created for project: %s", self.current_project_dir)

            logger.info(f"🔍 [DEBUG] ExecutionEngine exists: {type(self.execution_engine)}")

            exec_state.add_log("⚙️ Spawning parallel agents...")
            if log_callback:
                await log_callback("⚙️ Spawning parallel agents...")

            # ✅ FIX: Make callbacks async-compatible
            logger.info("🔍 [DEBUG] Creating async callback wrappers...")

            async def async_progress_callback(p):
                logger.info(f"🔍 [DEBUG] async_progress_callback called with {p}")
                await self._handle_progress(exec_state, p, progress_callback)

            async def async_log_callback(msg):
                logger.info(f"🔍 [DEBUG] async_log_callback called: {msg[:50]}...")
                await self._handle_log(exec_state, msg, log_callback)

            logger.info("🔍 [DEBUG] Async callbacks created")
            logger.info("🔍 [DEBUG] ========================================")
            logger.info("🔍 [DEBUG] CALLING execution_engine.execute()...")
            logger.info("🔍 [DEBUG] ========================================")
            logger.info(f"🔍 [DEBUG] Parameters:")
            logger.info(f"🔍 [DEBUG]   execution_id: {execution_id}")
            logger.info(f"🔍 [DEBUG]   num_agents: {num_agents}")
            logger.info(f"🔍 [DEBUG]   partitions: {len(partitioned)} agents")
            logger.info(f"🔍 [DEBUG]   prompt length: {len(orchestrator_prompt)}")

            execution_results = await self.execution_engine.execute(
                execution_id=execution_id,
                orchestrator_prompt=orchestrator_prompt,
                prd_partitions=partitioned,
                num_agents=num_agents,
                progress_callback=async_progress_callback,
                log_callback=async_log_callback,
            )

            logger.info("🔍 [DEBUG] ========================================")
            logger.info("🔍 [DEBUG] execution_engine.execute() RETURNED")
            logger.info("🔍 [DEBUG] ========================================")
            logger.info(f"🔍 [DEBUG] Result status: {execution_results.get('status')}")
            logger.info(f"🔍 [DEBUG] Result keys: {list(execution_results.keys())}")

            if execution_results.get("status") == "success":
                logger.info("🔍 [DEBUG] Processing success results...")
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
                logger.info("🔍 [DEBUG] Processing partial/failed results...")
                exec_state.status = "partial" if exec_state.completed_items > 0 else "failed"
                exec_state.error_message = execution_results.get("error", "Execution failed")

            exec_state.duration_seconds = (
                    datetime.now() - datetime.fromisoformat(exec_state.start_time)
            ).total_seconds()

            logger.info("🔍 [DEBUG] Saving execution state...")
            state_file = (
                    self.current_project_dir / "workspace" / "output" / "logs" /
                    f"execution_{execution_id}.json"
            )
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(exec_state.to_dict(), f, indent=2, default=str)

            exec_state.add_log(f"💾 Execution state saved: {state_file}")

            if progress_callback:
                await progress_callback(100.0)

            logger.info("🔍 [DEBUG] execute_prd_loop() COMPLETE")
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
            logger.error(f"🔍 [DEBUG] EXCEPTION in execute_prd_loop: {e}", exc_info=True)
            self._log(f"❌ Execution failed: {e}")
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
        """Partition PRD items round-robin to agents"""
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
        """Create system prompt for orchestrator agent"""
        prd_summary = self._format_prd_summary(prd)

        if config.metadata.get('project_type') == "ml_competition":
            meta = config.metadata
            prompt = f"""You are a senior ML engineer implementing a machine learning competition solution.

PROJECT INFO:
- Name: {config.name}
- Competition: {meta.get('competition_url', 'N/A')}
- Problem: {meta.get('problem_type', 'N/A')}
- Framework: {meta.get('ml_framework', 'PyTorch')}
- Model: {meta.get('model_type', 'LSTM')}
- Primary KPI: {meta.get('eval_metric', 'score')} (threshold in metrics_config.json)

YOUR TASK:
Implement the assigned PRD items. For each item:
1. Understand the acceptance criteria
2. Generate complete, working Python code
3. Include docstrings and type hints
4. Follow ML engineering best practices
5. Ensure code is reproducible and tested
6. Ensure evaluation scripts print the KPI exactly once using the pattern from metrics_config.json

PRD CONTEXT:
{prd_summary}

DELIVERABLES:
- Data loading and preprocessing modules
- Model architecture definition
- Training pipeline
- Evaluation metrics (with KPI logging)
- Prediction and submission generation
- Unit tests for data processing
- Documentation strings

QUALITY REQUIREMENTS:
✓ Type hints on all functions
✓ Docstrings (Google style)
✓ Reproducible training (seed setting)
✓ Model checkpointing
✓ Logging and monitoring
✓ Error handling
✓ GPU/CPU compatibility

Start implementation now. Generate complete, working code."""
        else:
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
        """Format PRD items as readable summary"""
        summary = ""
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
        logger.info(f"🔍 [DEBUG] _handle_progress: {progress}%")
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)

    async def _handle_log(
            self,
            exec_state: ExecutionState,
            message: str,
            callback: Optional[Callable] = None
    ) -> None:
        """Handle log messages"""
        logger.info(f"🔍 [DEBUG] _handle_log: {message[:100]}")
        exec_state.add_log(message)
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)

    async def execute_validation_loop(
            self,
            execution_id: str,
            coverage_target: int = 80
    ) -> Dict[str, Any]:
        """Execute validation on generated code (PR-003)"""
        pass

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
        """✅ FIXED: List all projects in workspace with enhanced logging"""
        projects = []
        projects_dir = self.workspace / "projects"

        logger.info(f"📂 Scanning for projects in: {projects_dir}")
        logger.info(f"   Workspace absolute path: {self.workspace.resolve()}")
        logger.info(f"   Projects dir exists: {projects_dir.exists()}")

        if not projects_dir.exists():
            logger.warning(f"⚠️ Projects directory does not exist: {projects_dir}")
            projects_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created projects directory: {projects_dir}")
            return projects

        try:
            # List all items in projects directory
            all_items = list(projects_dir.iterdir())
            logger.info(f"   Found {len(all_items)} items in projects directory")

            for project_dir in sorted(all_items):
                logger.info(f"   Checking: {project_dir.name} (is_dir: {project_dir.is_dir()})")

                if project_dir.is_dir():
                    config_file = project_dir / "config.json"
                    logger.info(f"      Looking for config: {config_file.exists()}")

                    if config_file.exists():
                        try:
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)

                            project_info = {
                                "project_id": project_dir.name,
                                "name": config_data.get("name", "Unknown"),
                                "domain": config_data.get("domain", ""),
                                "project_type": config_data.get("metadata", {}).get("project_type", config_data.get("project_type", "api_dev")),
                                "architecture": config_data.get("architecture", ""),
                                "framework": config_data.get("metadata", {}).get("ml_framework", config_data.get("framework", "")),
                                "path": str(project_dir),
                                "created_at": config_data.get("timestamp", ""),
                                "status": "ready",
                                "metadata": config_data.get("metadata", {}),
                            }
                            projects.append(project_info)
                            logger.info(f"      ✅ Added project: {project_info['name']}")

                        except json.JSONDecodeError as e:
                            logger.warning(f"      ⚠️ Invalid config.json in {project_dir.name}: {e}")
                        except Exception as e:
                            logger.error(f"      ❌ Error reading config in {project_dir.name}: {e}")
                    else:
                        logger.info(f"      ⚠️ No config.json found in {project_dir.name}")

        except Exception as e:
            logger.error(f"❌ Error listing projects: {e}", exc_info=True)

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
        """Generate architecture blueprint"""
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

    # Test: Create ML project
    ml_config = MLConfig(
        competition_source="wundernn.io",
        competition_url="https://wundernn.io",
        problem_type="time_series_forecasting",
        sequence_length=1000,
        num_features=32,
        model_type="LSTM",
        ml_framework="PyTorch",
        batch_size=64,
        epochs=100,
        eval_metric="R²",
    )

    config = ProjectConfig(
        name="Wundernn LOB Forecasting",
        domain="time_series_forecasting",
        project_type="ml_competition",
        ml_config=ml_config,
        metadata={
            "project_type": "ml_competition",
            "competition_url": "https://wundernn.io",
            "problem_type": "time_series_forecasting",
            "ml_framework": "PyTorch",
            "model_type": "LSTM",
            "eval_metric": "weighted_pearson",
            "metric_target": 0.35,
        },
    )

    result = orchestrator.create_project(config)
    logger.info(f"Create result: {json.dumps(result, indent=2, default=str)}")

    # Test: List projects
    projects = orchestrator.list_projects()
    logger.info(f"Found projects: {len(projects)}")
    for p in projects:
        logger.info(f"   - {p['name']} ({p['project_type']})")
