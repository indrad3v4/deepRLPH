# -*- coding: utf-8 -*-
"""Agent Coordinator - Multi-agent task distribution and result aggregation"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# 🚀 IMPORT THE REAL RALPH LOOP MIXIN
from agent_coordinator_extensions import PRDAgentAssignmentMixin

logger = logging.getLogger("AgentCoordinator")


class AgentCoordinator(PRDAgentAssignmentMixin):
    """
    Coordinates multiple agents working on a single project.

    Distributes tasks, aggregates results, and manages multi-agent
    collaboration for autonomous software development.
    """

    def __init__(self, workspace: Path):
        """Initialize agent coordinator."""
        self.workspace = workspace
        self.output_dir = workspace / "output" / "generated_code"
        self.logs_dir = workspace / "output" / "logs"
        self.coordination_log = self.logs_dir / "coordination.log"
        self.agent_results: List[Dict[str, Any]] = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ AgentCoordinator initialized")
        logger.info(f"   Workspace: {workspace}")

    def create_subtasks(
            self,
            task_description: str,
            num_agents: int,
    ) -> List[str]:
        """Break down main task into subtasks for individual agents."""
        aspects = [
            "database schema, models, and ORM configuration",
            "API endpoints and route definitions",
            "authentication and security middleware",
            "business logic and service layer",
            "input validation and error handling",
            "logging and monitoring setup",
            "unit tests and integration tests",
            "documentation and API specifications",
        ]

        subtasks = []
        for i in range(num_agents):
            aspect = aspects[i % len(aspects)]

            subtask = f"""Your specialized role in the project:
Focus on implementing: {aspect}

Main project task: {task_description}

Requirements for your work:
1. Write production-ready, clean code
2. Follow the project architecture and design patterns
3. Include comprehensive docstrings and type hints
4. Write tests for all your code (>80% coverage)
5. Handle errors gracefully with proper logging
6. Coordinate with other agents by following naming conventions
7. Document any special setup or configuration needed
8. Consider performance and security implications
9. Make autonomous decisions - don't ask for clarification

Your output should include:
- Fully implemented code modules
- Comprehensive test cases"""

            subtasks.append(subtask)
        return subtasks

    async def aggregate_results(
            self,
            agent_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine and analyze results from multiple agents."""
        aggregated = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(agent_responses),
            "successful_agents": 0,
            "failed_agents": 0,
            "agents": [],
            "artifacts": [],
            "warnings": [],
            "statistics": {
                "total_thinking_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        for i, response in enumerate(agent_responses, 1):
            if response.get("status") == "success":
                aggregated["successful_agents"] += 1
                agent_info = {
                    "agent_id": i,
                    "status": "success",
                    "thinking_length": len(response.get("thinking", "")),
                    "response_length": len(response.get("response", "")),
                    "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                }
                
                usage = response.get("usage", {})
                aggregated["statistics"]["total_thinking_tokens"] += usage.get("completion_tokens", 0)
                aggregated["statistics"]["total_completion_tokens"] += usage.get("completion_tokens", 0)
                aggregated["statistics"]["total_tokens"] += usage.get("total_tokens", 0)

                aggregated["agents"].append(agent_info)
                artifacts = self._extract_artifacts(response.get("response", ""), agent_id=i)
                aggregated["artifacts"].extend(artifacts)
            else:
                aggregated["failed_agents"] += 1
                error_msg = response.get("error", "Unknown error")
                aggregated["warnings"].append(f"Agent {i} failed: {error_msg}")

        self._save_aggregation_results(aggregated)
        self.agent_results = aggregated
        return aggregated

    def _extract_artifacts(
            self,
            response_text: str,
            agent_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """Extract code blocks and artifacts from agent response."""
        artifacts = []
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response_text, re.DOTALL)

        for idx, (lang, code) in enumerate(code_blocks, 1):
            artifact = {
                "type": "code_block",
                "agent_id": agent_id,
                "block_number": idx,
                "language": lang or "text",
                "size_bytes": len(code.encode('utf-8')),
                "lines": len(code.split('\n')),
                "preview": code[:100].replace('\n', ' '),
            }
            artifacts.append(artifact)
        return artifacts

    def _save_aggregation_results(self, aggregated: Dict[str, Any]) -> Path:
        """Save aggregation results to file."""
        results_file = self.logs_dir / f"aggregation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2)
        return results_file

    def report_coordination(self) -> None:
        """Print comprehensive coordination report to logs"""
        pass

    async def save_agent_outputs(
            self,
            agent_id: int,
            artifacts: List[Dict[str, str]],
    ) -> List[Path]:
        """Save individual agent outputs to files."""
        saved_files = []
        agent_dir = self.output_dir / f"agent_{agent_id}"
        agent_dir.mkdir(parents=True, exist_ok=True)

        for artifact in artifacts:
            try:
                file_path = agent_dir / artifact.get("name", f"output_{len(saved_files)}.py")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(artifact.get("content", ""))
                saved_files.append(file_path)
            except IOError as e:
                logger.error(f"❌ Failed to save artifact: {e}")
        return saved_files

    def get_agent_statistics(self, agent_id: int) -> Dict[str, Any]:
        """Get statistics for a specific agent."""
        return next((a for a in self.agent_results.get("agents", []) if a["agent_id"] == agent_id), {})