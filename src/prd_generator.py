# -*- coding: utf-8 -*-
"""
prd_generator.py - PRD Generator from Technical Brief (PR-001)

✅ ENHANCED: ML Competition Support
  - Detects ML/time-series tasks
  - Generates 10 atomic items for competition workflow
  - Data download → Model training → ONNX export → Submission

Takes clarified task + brief, generates 5-10 actionable PRD items.
Each item is one agent iteration (fits in context window).
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json
import re

logger = logging.getLogger("PRDGenerator")


@dataclass
class UserStory:
    """Single PRD item"""
    id: str
    title: str
    description: str
    acceptance_criteria: List[str]
    verification: str
    why: str
    files_touched: List[str]
    status: str = "todo"
    attempts: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def to_dict(self):
        return asdict(self)


class PRDGenerator:
    """Convert technical brief → PRD backlog"""

    def __init__(self):
        self.stories: List[UserStory] = []

    def generate(self,
                 technical_brief: Dict[str, Any],
                 domain: str = "llm-app") -> Dict[str, Any]:
        """
        Generate PRD from clarified task.

        Args:
            technical_brief: Output from TaskClarifier
            domain: Project domain

        Returns:
            PRD dict with user_stories list
        """

        clarified_task = technical_brief.get("clarified_task", "")
        key_reqs = technical_brief.get("key_requirements", [])

        logger.info("📝 PRD Generator Starting...")
        logger.info(f"   Task: {clarified_task[:80]}...")
        logger.info(f"   Domain: {domain}")

        # ✅ Enhanced domain detection
        domain_lower = domain.lower()
        task_lower = clarified_task.lower()
        
        # ML Competition detection
        is_ml_competition = any([
            "time_series" in domain_lower,
            "ml_competition" in domain_lower,
            "ml_model" in domain_lower,
            "forecasting" in task_lower,
            "kaggle" in task_lower,
            "wundernn" in task_lower,
            "competition" in task_lower and ("train" in task_lower or "model" in task_lower),
            "onnx" in task_lower,
            "pearson correlation" in task_lower,
        ])

        # Decompose based on domain
        if is_ml_competition:
            logger.info("   Detected: ML Competition")
            stories = self._decompose_ml_competition(clarified_task, key_reqs)
        elif "api" in domain_lower or "backend" in domain_lower:
            logger.info("   Detected: API/Backend")
            stories = self._decompose_api(clarified_task, key_reqs)
        elif "llm" in domain_lower or "chatbot" in domain_lower:
            logger.info("   Detected: LLM App")
            stories = self._decompose_llm(clarified_task, key_reqs)
        elif "web" in domain_lower:
            logger.info("   Detected: Web App")
            stories = self._decompose_web(clarified_task, key_reqs)
        else:
            logger.warning("   Using generic decomposition")
            stories = self._decompose_generic(clarified_task, key_reqs)

        logger.info(f"✅ Generated {len(stories)} PRD items")

        prd = {
            "task": clarified_task,
            "domain": domain,
            "total_items": len(stories),
            "user_stories": [s.to_dict() for s in stories]
        }

        return prd

    def _decompose_ml_competition(self, task: str, reqs: List[str]) -> List[UserStory]:
        """ML Competition decomposition (10 atomic items)"""
        return [
            UserStory(
                id="ML-001",
                title="Environment setup and dependencies",
                description="Create Python 3.11 venv, install PyTorch, ONNX Runtime, Pandas, and competition requirements.",
                acceptance_criteria=[
                    "Python 3.11 virtual environment created",
                    "requirements.txt with pinned versions: torch>=2.0, onnxruntime, pandas, pyarrow, numpy, tqdm",
                    "All packages install without errors",
                    "Python version verified: 3.11.x",
                    ".gitignore includes venv/, __pycache__/, *.pyc"
                ],
                verification="python --version && pip list | grep -E '(torch|onnx|pandas)' && python -c 'import torch; print(torch.__version__)'",
                why="Match competition Docker environment (python:3.11-slim-bookworm). Ensure reproducibility.",
                files_touched=["requirements.txt", ".gitignore", "README.md"]
            ),
            UserStory(
                id="ML-002",
                title="Download and verify starter pack",
                description="Download competition data, extract to data/raw/, verify file integrity and data shapes.",
                acceptance_criteria=[
                    "Starter pack downloaded from competition URL",
                    "Files extracted: train.parquet, valid.parquet, utils.py, baseline.onnx",
                    "train.parquet shape verified: 10,721 sequences",
                    "valid.parquet shape verified: 1,444 sequences",
                    "Each sequence has 1000 timesteps confirmed",
                    "Script: scripts/download_data.py for reproducibility"
                ],
                verification="ls data/raw/datasets/ && python -c \"import pandas as pd; df = pd.read_parquet('data/raw/datasets/train.parquet'); print(f'Train: {df.shape}'); assert len(df) == 10721\"",
                why="Need training/validation data to start model development. Verify data integrity before training.",
                files_touched=["scripts/download_data.py", "data/raw/datasets/train.parquet", "data/raw/datasets/valid.parquet", "data/raw/utils.py"]
            ),
            UserStory(
                id="ML-003",
                title="PyTorch DataLoader implementation",
                description="Implement Dataset class for time-series sequences with 99-step context and 900-step prediction targets.",
                acceptance_criteria=[
                    "src/data_loader.py implements torch.utils.data.Dataset",
                    "Correctly splits sequences: context=[0:99], target=[99:999]",
                    "Returns (context, target) tuples with correct shapes",
                    "Supports both train and validation datasets",
                    "Optional data augmentation (noise, scaling) implemented",
                    "Unit tests pass: tests/test_data_loader.py",
                    "Docstrings with type hints on all methods"
                ],
                verification="python -c \"from src.data_loader import TimeSeriesDataset; ds = TimeSeriesDataset('data/raw/datasets/train.parquet'); print(f'Dataset size: {len(ds)}'); ctx, tgt = ds[0]; print(f'Context: {ctx.shape}, Target: {tgt.shape}'); assert ctx.shape[0] == 99\" && pytest tests/test_data_loader.py -v",
                why="Efficient batching and data loading for training. Correct context/target split is critical for model performance.",
                files_touched=["src/data_loader.py", "tests/test_data_loader.py"]
            ),
            UserStory(
                id="ML-004",
                title="Baseline model evaluation",
                description="Load baseline.onnx with ONNX Runtime, run inference on validation set, record Pearson Correlation score.",
                acceptance_criteria=[
                    "src/evaluate_baseline.py loads baseline.onnx successfully",
                    "Runs inference on all 1,444 validation sequences",
                    "Calculates Pearson Correlation using utils.py scoring function",
                    "Logs baseline score to models/baseline_score.txt",
                    "Measures and logs inference time (<10s requirement check)",
                    "Script outputs: 'Baseline Pearson Correlation: 0.XXXX, Time: Xs'"
                ],
                verification="python src/evaluate_baseline.py && cat models/baseline_score.txt",
                why="Establish performance target to beat. Baseline provides reference for model improvements.",
                files_touched=["src/evaluate_baseline.py", "models/baseline_score.txt"]
            ),
            UserStory(
                id="ML-005",
                title="Enhanced model architecture",
                description="Design and implement improved LSTM/GRU model with attention or Transformer encoder to beat baseline.",
                acceptance_criteria=[
                    "src/model.py implements PredictionModel class (competition interface)",
                    "Model architecture: Bidirectional LSTM + Multi-Head Attention OR GRU + Skip Connections",
                    "Input shape: (batch, 99, num_features), Output: (batch, 900)",
                    "Model parameters: 500K-2M (balance capacity vs speed)",
                    "Supports gradient checkpointing for memory efficiency",
                    "Unit tests verify forward pass: tests/test_model.py",
                    "Docstrings explain architecture choices"
                ],
                verification="python -c \"from src.model import PredictionModel; m = PredictionModel(); params = sum(p.numel() for p in m.parameters()); print(f'Parameters: {params:,}'); assert 500_000 <= params <= 2_000_000\" && pytest tests/test_model.py -v",
                why="Need improved architecture to beat baseline GRU. Attention/skip connections capture long-range dependencies in time-series.",
                files_touched=["src/model.py", "tests/test_model.py"]
            ),
            UserStory(
                id="ML-006",
                title="Training pipeline with checkpointing",
                description="Implement full training loop with AdamW optimizer, learning rate scheduling, early stopping, and model checkpointing.",
                acceptance_criteria=[
                    "src/train.py implements training loop",
                    "Loss: MSE or Huber loss",
                    "Optimizer: AdamW with cosine annealing LR scheduler",
                    "Batch size: 64 (configurable via CLI args)",
                    "Epochs: 100 max with early stopping (patience=10 on validation loss)",
                    "Saves best model checkpoint to models/checkpoints/best_model.pt",
                    "Logs train/val loss every epoch to CSV or TensorBoard",
                    "GPU/CPU auto-detection and utilization",
                    "Reproducibility: sets torch/numpy/random seeds",
                    "Progress bar with tqdm"
                ],
                verification="python src/train.py --epochs 5 --batch_size 32 --seed 42 && ls models/checkpoints/best_model.pt",
                why="Train model on 10,721 sequences efficiently. Checkpointing prevents loss of progress. Early stopping avoids overfitting.",
                files_touched=["src/train.py", "src/trainer.py", "config/train_config.yaml", "tests/test_training.py"]
            ),
            UserStory(
                id="ML-007",
                title="Model evaluation with metrics",
                description="Evaluate trained model on validation set, calculate Pearson Correlation (primary metric) plus RMSE, MAE, R².",
                acceptance_criteria=[
                    "src/evaluate.py loads trained PyTorch model from checkpoint",
                    "Runs inference on validation set (1,444 sequences)",
                    "Calculates Pearson Correlation using competition's utils.py",
                    "Also computes: RMSE, MAE, R² for analysis",
                    "Saves predictions to CSV: output/predictions.csv",
                    "Generates plots: predicted vs actual, residuals distribution",
                    "Outputs: 'Validation Pearson: 0.XXXX (Baseline: 0.YYYY, Beat: True/False)'",
                    "Evaluation results saved to models/evaluation_report.json"
                ],
                verification="python src/evaluate.py --model models/checkpoints/best_model.pt && cat models/evaluation_report.json",
                why="Competition metric is Pearson Correlation. Need to verify model beats baseline before ONNX export.",
                files_touched=["src/evaluate.py", "notebooks/evaluation_analysis.ipynb", "models/evaluation_report.json"]
            ),
            UserStory(
                id="ML-008",
                title="ONNX model export and optimization",
                description="Convert trained PyTorch model to ONNX format, verify inference speed <10s, validate output accuracy.",
                acceptance_criteria=[
                    "src/export_onnx.py converts PyTorch checkpoint to ONNX",
                    "Exported model: models/final/model.onnx",
                    "ONNX model tested with onnxruntime inference",
                    "Inference speed on validation set: <10 seconds",
                    "Output accuracy: PyTorch vs ONNX difference <1e-5 (tolerance)",
                    "Optional: model quantization for speed (fp16 or int8)",
                    "Script outputs: 'ONNX export successful. Inference: Xs, Accuracy: ✓'"
                ],
                verification="python src/export_onnx.py --checkpoint models/checkpoints/best_model.pt --output models/final/model.onnx && python src/test_onnx.py --model models/final/model.onnx",
                why="Competition requires ONNX format for fast inference. Must meet <10s time constraint.",
                files_touched=["src/export_onnx.py", "src/test_onnx.py", "models/final/model.onnx"]
            ),
            UserStory(
                id="ML-009",
                title="Competition submission package",
                description="Create solution.py implementing PredictionModel class interface, package with model.onnx into submission.zip.",
                acceptance_criteria=[
                    "solution.py at project root implements PredictionModel class",
                    "Loads models/final/model.onnx correctly",
                    "Implements .predict(datapoint: DataPoint) -> np.ndarray method",
                    "Uses utils.py DataPoint class (unchanged from starter pack)",
                    "scripts/make_submission.py creates submission.zip",
                    "submission.zip contains: solution.py, model.onnx (any other model artifacts)",
                    "Local test: python solution.py runs without errors",
                    "README updated with submission instructions"
                ],
                verification="python scripts/make_submission.py && unzip -l submission.zip && python solution.py",
                why="Competition submission format required. Must match interface spec exactly or submission fails.",
                files_touched=["solution.py", "scripts/make_submission.py", "submission.zip"]
            ),
            UserStory(
                id="ML-010",
                title="Documentation and reproducibility",
                description="Complete README, training config, EDA notebook, code quality checks (linting, tests).",
                acceptance_criteria=[
                    "README.md updated with: env setup, data download, training command, evaluation results, submission steps",
                    "config/train_config.yaml: all hyperparameters documented",
                    "notebooks/exploratory_analysis.ipynb: EDA on train/valid data",
                    "All Python files have docstrings and type hints",
                    "Code passes: black --check src/ && flake8 src/ --max-line-length=120",
                    "Tests pass: pytest tests/ -v --cov=src --cov-report=term-missing",
                    "Test coverage >80% on data_loader, model, training modules",
                    "Git repo clean: no uncommitted changes"
                ],
                verification="black --check src/ && flake8 src/ --max-line-length=120 && pytest tests/ -v --cov=src --cov-report=term && git status",
                why="Ensure reproducibility. Code quality for collaboration. Documentation for future reference.",
                files_touched=["README.md", "config/train_config.yaml", "notebooks/exploratory_analysis.ipynb", "src/*", "tests/*"]
            ),
        ]

    def _decompose_api(self, task: str, reqs: List[str]) -> List[UserStory]:
        """REST API decomposition"""
        return [
            UserStory(
                id="PS-001",
                title="Database schema and models",
                description="Define data models using Pydantic + SQLAlchemy. Create migration.",
                acceptance_criteria=[
                    "All entities modeled with type hints",
                    "Database migrations working",
                    "Unit tests for models (80%+ coverage)",
                    "Docstrings on all classes/methods"
                ],
                verification="pytest tests/test_models.py -v && python -m pytest --cov=src/models",
                why="Foundation for all endpoints. Data integrity.",
                files_touched=["src/models.py", "src/database.py", "tests/test_models.py"]
            ),
            UserStory(
                id="PS-002",
                title="Core CRUD endpoints",
                description="Implement GET, POST, PUT, DELETE endpoints with validation.",
                acceptance_criteria=[
                    "All 4 CRUD operations working",
                    "HTTP status codes correct (200, 201, 400, 404)",
                    "Input validation on POST/PUT",
                    "Error handling with proper messages",
                    "Tests for all endpoints"
                ],
                verification="pytest tests/test_api.py -v",
                why="Main API functionality users interact with.",
                files_touched=["src/api/routes.py", "tests/test_api.py"]
            ),
            UserStory(
                id="PS-003",
                title="Authentication & security",
                description="JWT tokens, password hashing, CORS, rate limiting.",
                acceptance_criteria=[
                    "JWT token generation and validation",
                    "Passwords hashed (bcrypt)",
                    "CORS configured for frontend",
                    "Rate limiting on auth endpoints",
                    "Security tests passing"
                ],
                verification="pytest tests/test_auth.py -v",
                why="Protect API from unauthorized access.",
                files_touched=["src/auth.py", "src/middleware.py", "tests/test_auth.py"]
            ),
            UserStory(
                id="PS-004",
                title="Documentation & deployment",
                description="OpenAPI/Swagger docs, Dockerfile, deployment guide.",
                acceptance_criteria=[
                    "Swagger docs at /docs endpoint",
                    "Dockerfile builds successfully",
                    "docker-compose.yml includes postgres",
                    "README with setup instructions",
                    "Health check endpoint working"
                ],
                verification="python -m pytest && docker build . && curl http://localhost:8000/health",
                why="Deploy to production, user documentation.",
                files_touched=["Dockerfile", "docker-compose.yml", "README.md", "src/main.py"]
            ),
        ]

    def _decompose_llm(self, task: str, reqs: List[str]) -> List[UserStory]:
        """LLM app decomposition"""
        return [
            UserStory(
                id="LPS-001",
                title="LLM client & config",
                description="Initialize Deepseek client, load API key, model selection.",
                acceptance_criteria=[
                    "Client initializes without errors",
                    "API key loaded from environment",
                    "Model can be changed via config",
                    "Health check works (can call API)",
                    "Tests passing"
                ],
                verification="pytest tests/test_llm_client.py -v",
                why="Foundation for all LLM calls.",
                files_touched=["src/llm_client.py", "src/config.py", "tests/test_llm_client.py"]
            ),
            UserStory(
                id="LPS-002",
                title="Prompt management",
                description="Prompt templates, variable substitution, prompt versioning.",
                acceptance_criteria=[
                    "Templates load from files",
                    "Variable injection working",
                    "Multiple prompts can coexist",
                    "Tests for template rendering",
                    "Documentation of all prompts"
                ],
                verification="pytest tests/test_prompts.py -v",
                why="Manage complex prompts cleanly.",
                files_touched=["src/prompts.py", "templates/", "tests/test_prompts.py"]
            ),
            UserStory(
                id="LPS-003",
                title="Response parsing & validation",
                description="Parse LLM output, validate structure, error handling.",
                acceptance_criteria=[
                    "Structured output parsing (JSON, markdown)",
                    "Graceful handling of malformed responses",
                    "Type validation on parsed data",
                    "Tests for edge cases",
                    "Logging of parse errors"
                ],
                verification="pytest tests/test_parser.py -v",
                why="Reliable data extraction from LLM.",
                files_touched=["src/parser.py", "tests/test_parser.py"]
            ),
            UserStory(
                id="LPS-004",
                title="Chat interface & memory",
                description="Multi-turn conversation, chat history, context management.",
                acceptance_criteria=[
                    "Store/retrieve chat history",
                    "Context window management",
                    "User sessions isolated",
                    "Tests for conversation flow",
                    "Memory usage reasonable"
                ],
                verification="pytest tests/test_chat.py -v",
                why="Enable multi-turn conversations.",
                files_touched=["src/chat.py", "src/storage.py", "tests/test_chat.py"]
            ),
        ]

    def _decompose_web(self, task: str, reqs: List[str]) -> List[UserStory]:
        """Web app decomposition"""
        return [
            UserStory(
                id="WPS-001",
                title="Frontend setup & routing",
                description="React/Vue setup, routing, basic layout.",
                acceptance_criteria=[
                    "React app initializes",
                    "Routing works (at least 3 pages)",
                    "Layout responsive",
                    "Tests for routing",
                    "Build succeeds"
                ],
                verification="npm run build && npm run test",
                why="UI foundation.",
                files_touched=["frontend/src/App.tsx", "frontend/src/routes/", "frontend/src/__tests__/"]
            ),
            UserStory(
                id="WPS-002",
                title="Backend API integration",
                description="Connect frontend to backend API.",
                acceptance_criteria=[
                    "API calls working",
                    "Error handling on failed requests",
                    "Loading states implemented",
                    "Tests for API integration",
                    "CORS working"
                ],
                verification="npm run test && pytest tests/test_api.py",
                why="Frontend talks to backend.",
                files_touched=["frontend/src/api/", "tests/"]
            ),
            UserStory(
                id="WPS-003",
                title="State management",
                description="Redux/Context state, user auth state.",
                acceptance_criteria=[
                    "Global state works",
                    "Auth state persists",
                    "Logging in/out works",
                    "Tests for state",
                    "No prop drilling"
                ],
                verification="npm run test -- --coverage",
                why="Manage app state cleanly.",
                files_touched=["frontend/src/store/", "frontend/src/__tests__/"]
            ),
        ]

    def _decompose_generic(self, task: str, reqs: List[str]) -> List[UserStory]:
        """Fallback generic decomposition"""
        logger.warning("⚠️ Using generic decomposition - consider adding domain-specific template")
        return [
            UserStory(
                id="G-001",
                title="Core functionality",
                description=f"Build: {task[:100]}",
                acceptance_criteria=[
                    "Feature implemented",
                    "Tests written",
                    "Code reviewed",
                    "Docstrings added",
                    "Ready for production"
                ],
                verification="pytest tests/ -v && black --check . && mypy .",
                why="Deliver the main feature.",
                files_touched=["src/main.py", "tests/"]
            ),
        ]


def generate_prd(technical_brief: Dict[str, Any],
                 domain: str = "llm-app") -> Dict[str, Any]:
    """Convenience function to generate PRD"""
    gen = PRDGenerator()
    return gen.generate(technical_brief, domain)