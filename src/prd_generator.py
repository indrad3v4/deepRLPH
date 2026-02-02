# -*- coding: utf-8 -*-
"""
prd_generator.py - PRD Generator from Technical Brief (PR-001)

Takes clarified task + brief, generates 5-7 actionable PRD items.
Each item is one agent iteration (fits in context window).
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json

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

        # Decompose based on domain
        if "api" in domain.lower() or "backend" in domain.lower():
            stories = self._decompose_api(clarified_task, key_reqs)
        elif "llm" in domain.lower() or "chatbot" in domain.lower():
            stories = self._decompose_llm(clarified_task, key_reqs)
        elif "web" in domain.lower():
            stories = self._decompose_web(clarified_task, key_reqs)
        else:
            stories = self._decompose_generic(clarified_task, key_reqs)

        logger.info(f"✅ Generated {len(stories)} PRD items")

        prd = {
            "task": clarified_task,
            "domain": domain,
            "total_items": len(stories),
            "user_stories": [s.to_dict() for s in stories]
        }

        return prd

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