# -*- coding: utf-8 -*-

"""
orchestrator.py - RALPH Main Orchestrator Engine (FULL INTEGRATION)

Complete integration of:
- BE-002: Schema detection on dataset upload
- BE-005: Schema-aware PRD generation
- BE-007: Submission packaging
- BE-008: JSONL trace logging
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

from context_ingestor import ingest_project_context  # PRD-01
from schema_integrator import SchemaIntegrator  # BE-002.2

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
# DATA MODELS (unchanged from original)
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
# WORKSPACE MANAGER (unchanged - metrics_config creation already there)
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
            project_dir / "logs",  # NEW: For JSONL traces
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

        # NEW: Auto-generate per-project metrics_config.json for ALL projects
        WorkspaceManager._create_default_metrics_config(project_dir, config)

        logger.info(f"📝 Created config files")

    @staticmethod
    def _create_default_metrics_config(project_dir: Path, config: ProjectConfig) -> None:
        """Create a default metrics_config.json for any project.

        For ML competition projects, use project metadata or ml_config.eval_metric.
        For non-ML projects, default to a tests-based KPI so the loop is result-oriented
        even before you customize metrics per project.
        """
        meta = config.metadata or {}
        project_type_str = str(meta.get("project_type", config.project_type))

        # ------------------------------------------------------------------
        # ML COMPETITION PROJECTS: metric from eval_metric / metadata
        # ------------------------------------------------------------------
        if "ml_competition" in project_type_str:
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

        # ------------------------------------------------------------------
        # NON-ML PROJECTS: default to tests-based KPI
        # ------------------------------------------------------------------
        else:
            metric_name = meta.get('eval_metric', 'tests_passed_ratio')

            patterns = {
                # Expect your verification command to print e.g. "TESTS_OK: 1.0" when all tests pass
                "tests_passed_ratio": r"TESTS_OK:\s*(?P<value>[-+]?\d*\.?\d+)",
                # Or "TESTS_PASSED: 123"
                "tests_passed": r"TESTS_PASSED:\s*(?P<value>\d+)",
                # Or "COVERAGE: 87.5" for coverage percentage
                "coverage": r"COVERAGE:\s*(?P<value>[-+]?\d*\.?\d+)",
            }

            name_key = metric_name if metric_name in patterns else "tests_passed_ratio"
            pattern = patterns[name_key]

            # For generic projects, default target is "all tests pass" (ratio 1.0)
            if name_key == "coverage":
                default_target = float(meta.get('metric_target', 80.0))
            else:
                default_target = float(meta.get('metric_target', 1.0))

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
python scripts/make_submission.py .
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
# ORCHESTRATOR - MAIN ENGINE (FULL INTEGRATION)
# ============================================================================

class RalphOrchestrator:
    """
    Main orchestrator engine with full BE-002/005/007/008 integration.

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

        # BE-002: Schema integrator
        self.schema_integrator = SchemaIntegrator()

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

            # Create config + README + requirements + metrics_config
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
            raw_task: str,
            dataset_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Task clarification + PRD generation (ENHANCED with BE-002 schema detection).

        NEW (BE-002): Auto-detect dataset schema if dataset_path provided.
        NEW (PRD-01): Ingest local project context before calling DeepSeek.
        NEW (BE-005): Pass schemas to PRD generator for schema-aware stories.
        """
        try:
            from task_clarifier import TaskClarifier
            from prd_generator import PRDGenerator

            config = self.current_config
            if not config:
                return {"status": "error", "error": "No project created"}

            # =====================================================================
            # BE-002: SCHEMA DETECTION
            # =====================================================================
            if dataset_path or (self.current_project_dir / "data" / "raw").exists():
                self._log(
                    "📊 BE-002: Running dataset schema detection..."
                )
                schema_result = self.schema_integrator.inspect_and_save(
                    project_dir=self.current_project_dir,
                    dataset_path=dataset_path
                )

                if schema_result['status'] == 'success':
                    # Attach to metadata
                    config.metadata = self.schema_integrator.attach_to_metadata(
                        config.metadata,
                        schema_result['schemas']
                    )
                    self._log(
                        f"   ✅ Schema detected: {len(schema_result['schemas'])} dataset(s)"
                    )
                    self._log(
                        f"      Features: {config.metadata.get('num_features')}, "
                        f"Sequence length: {config.metadata.get('sequence_length')}"
                    )
                else:
                    self._log("   ⚠️  No datasets found for schema detection")

            # =====================================================================
            # PRD-01: CONTEXT INGESTION
            # =====================================================================
            context_summary_text = ""
            if self.current_project_dir and self.current_project_dir.exists():
                ctx = ingest_project_context(self.current_project_dir)
                context_summary = ctx.to_dict()
                context_summary_text = ctx.to_text_summary()

                # Attach to metadata (non-destructive merge)
                meta = dict(config.metadata or {})
                meta.setdefault("ingested_context", context_summary)
                config.metadata = meta

                self._log(
                    "📚 Project context attached to metadata (docs=%d, datasets=%d, code=%d, models=%d)",
                    len(context_summary.get("docs", [])),
                    len(context_summary.get("datasets", [])),
                    len(context_summary.get("code", [])),
                    len(context_summary.get("models", [])),
                )

            # =====================================================================
            # TASK CLARIFICATION
            # =====================================================================
            self._log("🔍 Task Clarification Phase")
            self._log(f"   Input: {raw_task[:100]}...")

            clarifier = TaskClarifier(self.deepseek_client)

            # Pass project context into the clarifier so Deepseek can reason
            # about real files in addition to high-level description.
            enriched_raw_task = raw_task
            if context_summary_text:
                enriched_raw_task = (
                    f"{raw_task}\n\n"  # original prompt first
                    "---\nPROJECT FILE CONTEXT (from local workspace):\n"  # separator
                    f"{context_summary_text}"
                )

            brief = asyncio.run(
                clarifier.clarify(
                    raw_task=enriched_raw_task,
                    domain=config.domain,
                    framework=config.framework,
                    database=config.database,
                )
            )

            if brief.get("status") != "success":
                return brief

            # Attach project metadata and id into the brief for downstream PRD generation
            brief["project_id"] = project_id
            brief["project_metadata"] = config.metadata

            self._log("✅ Task clarified")
            self._log(f"   {brief['clarified_task'][:100]}...")

            # =====================================================================
            # BE-005: SCHEMA-AWARE PRD GENERATION
            # =====================================================================
            self._log("📝 PRD Generation Phase (BE-005: schema-aware)")
            gen = PRDGenerator()
            prd = gen.generate(
                technical_brief=brief,
                domain=config.domain,
                project_metadata=config.metadata,  # Contains dataset_schemas
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
        """Execute PRD items in parallel (ENHANCED with BE-008 trace logging)"""
        logger.info("🔍 [DEBUG] execute_prd_loop() CALLED")

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

            # =====================================================================
            # BE-008: TRACE LOGGER INITIALIZATION
            # =====================================================================
            from trace_logger import TraceLogger

            trace_file = self.current_project_dir / "logs" / f"execution_{execution_id}.jsonl"
            trace_logger = TraceLogger(trace_file, execution_id)

            logger.info(f"📝 BE-008: Trace logging to {trace_file}")

            logger.info("🔍 [DEBUG] Partitioning PRD items...")
            partitioned = self._partition_prd_items(prd, num_agents)
            total_user_stories = len(prd.get('user_stories', []))
            logger.info(f"🔍 [DEBUG] Partitioned {total_user_stories} stories")

            exec_state.add_log(f"📊 Partitioned {total_user_stories} user stories across {num_agents} agents")

            if log_callback:
                await log_callback(f"📊 Partitioned {total_user_stories} user stories across {num_agents} agents")

            logger.info("🔍 [DEBUG] Creating orchestrator prompt...")
            orchestrator_prompt = await self._generate_dynamic_orchestrator_prompt(
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
                # BE-008: Attach trace logger to execution engine
                self.execution_engine.trace_logger = trace_logger

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

            logger.info("🔍 [DEBUG] CALLING execution_engine.execute()...")

            execution_results = await self.execution_engine.execute(
                execution_id=execution_id,
                orchestrator_prompt=orchestrator_prompt,
                prd_partitions=partitioned,
                num_agents=num_agents,
                progress_callback=async_progress_callback,
                log_callback=async_log_callback,
            )

            logger.info("🔍 [DEBUG] execution_engine.execute() RETURNED")
            logger.info(f"🔍 [DEBUG] Result status: {execution_results.get('status')}")

            # BE-008: Log execution end
            trace_logger.execution_end(
                status=execution_results.get('status', 'unknown'),
                total_duration=(datetime.now() - datetime.fromisoformat(exec_state.start_time)).total_seconds()
            )

            if execution_results.get("status") == "success":
                exec_state.status = "completed"
                exec_state.add_log("✅ Execution completed successfully")
            else:
                exec_state.status = "failed"
                exec_state.add_log(f"❌ Execution failed: {execution_results.get('error', 'Unknown')}")

            return execution_results

        except Exception as e:
            logger.error(f"❌ Execution error: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "execution_id": execution_id if 'execution_id' in locals() else "unknown"
            }

    def create_submission(self, project_id: str, output_name: Optional[str] = None) -> Dict[str, Any]:
        """BE-007: Create competition submission package.

        Validates and packages solution.py + model.onnx into submission.zip.

        Args:
            project_id: Project identifier
            output_name: Optional custom zip name

        Returns:
            Dict with status and zip_path
        """
        try:
            if not self.current_project_dir:
                return {
                    "status": "error",
                    "error": "No active project. Create a project first."
                }

            self._log("📦 BE-007: Creating submission package...")

            # Import packager
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
            from make_submission import SubmissionPackager

            packager = SubmissionPackager(self.current_project_dir)
            zip_path = packager.create_submission(output_name)

            self._log(f"✅ Submission created: {zip_path}")

            return {
                "status": "success",
                "zip_path": zip_path,
                "project_id": project_id,
            }

        except Exception as e:
            error_msg = f"Failed to create submission: {str(e)}"
            self._log(f"❌ {error_msg}")
            logger.error(error_msg, exc_info=True)
            return {
                "status": "error",
                "error": error_msg,
            }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _generate_project_id(self, name: str) -> str:
        """Generate unique project ID from name"""
        import re
        from datetime import datetime

        # Sanitize name
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_name}_{timestamp}"

    def _partition_prd_items(self, prd: Dict[str, Any], num_agents: int) -> Dict[str, List[Dict[str, Any]]]:
        """Partition PRD items across agents"""
        stories = prd.get('user_stories', [])
        partition_size = max(1, len(stories) // num_agents)

        partitions = {}
        for i in range(num_agents):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < num_agents - 1 else len(stories)
            agent_id = f"agent_{i + 1}"
            partitions[agent_id] = stories[start_idx:end_idx]

        return partitions

    async def _generate_dynamic_orchestrator_prompt(self, prd: Dict, config: ProjectConfig, domain: str) -> str:
        """Generate orchestrator system prompt"""
        return f"""You are a senior software engineer implementing a {domain} project.

Project: {config.name}
Description: {config.description}

Your task is to implement PRD stories with high quality, production-ready code.

Guidelines:
1. Write complete, working code with all imports
2. Add type hints and docstrings
3. Include error handling and logging
4. Follow best practices for {config.framework if hasattr(config, 'framework') else 'the stack'}
5. Ensure code is testable and maintainable
"""

    async def _handle_progress(self, exec_state: ExecutionState, progress: float, callback: Optional[Callable]):
        """Handle progress updates"""
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)

    async def _handle_log(self, exec_state: ExecutionState, message: str, callback: Optional[Callable]):
        """Handle log messages"""
        exec_state.add_log(message)
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)

    def _log(self, message: str, *args) -> None:
        """Internal logging"""
        if args:
            message = message % args
        logger.info(message)
        self.execution_log.append(message)
