# -*- coding: utf-8 -*-
"""
prd_generator.py - Dynamic AI-Driven PRD Generator

Dynamically generates PRD Backlogs based on project context, using
an o1-Anatomy prompt to ensure adaptive scoping and precise verifications.
"""

import logging
import json
from typing import Dict, Any, Optional

# Absolute import to avoid "no known parent package" errors in orchestrator
from prd_model import normalize_prd

logger = logging.getLogger("PRDGenerator")


class PRDGenerator:
    """Dynamic PRD generator powered by DeepSeek."""

    def __init__(self, deepseek_client: Any):
        self.deepseek = deepseek_client

    async def generate(
            self,
            technical_brief: Dict[str, Any],
            domain: str = "llm-app",
            project_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        project_meta = project_metadata or technical_brief.get("project_metadata") or {}
        task = technical_brief.get("clarified_task", "")

        # Extract KPI context
        kpi_metric = project_meta.get("kpi_metric") or project_meta.get("eval_metric") or "score"
        kpi_target = project_meta.get("kpi_target", project_meta.get("metric_target", 0.0))

        # Extract workspace context (what files already exist)
        ingested_context = project_meta.get("ingested_context", {})
        files_summary = {
            "datasets": ingested_context.get("datasets", []),
            "code": ingested_context.get("code", []),
            "models": ingested_context.get("models", []),
        }

        system_prompt = (
            "### Goal\n"
            "You are an elite Technical Product Manager. Your mission is to convert a technical brief into a strictly formatted, highly actionable JSON PRD Backlog for autonomous AI workers.\n\n"

            "### Return Format\n"
            "Return ONLY valid JSON. Do not wrap it in markdown code blocks. The JSON must match this exact schema:\n"
            "{\n"
            '  "total_items": <int>,\n'
            '  "user_stories": [\n'
            "    {\n"
            '      "id": "ML-001",\n'
            '      "title": "string",\n'
            '      "description": "string",\n'
            '      "acceptance_criteria": ["string", "string"],\n'
            '      "verification": "exact shell command (e.g., `pytest tests/test_file.py -v`)",\n'
            '      "why": "string",\n'
            '      "files_touched": ["path/to/file.py"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"

            "### Warnings & Strict Constraints\n"
            "- ADAPTIVE SCOPING: Read the Task carefully. If the task is a small iteration (e.g., 'Create run_pipeline.py'), ONLY generate stories for that specific delta. Do NOT regenerate foundational stories if they already exist in the Workspace Context.\n"
            "- PRECISE VERIFICATION: Every single story MUST have a functional `verification` command. Do NOT use specific function targets (`::test_name`) in pytest commands. Use file-level targets (e.g., `pytest tests/test_data.py -v`) or direct python execution (e.g., `python src/run_pipeline.py`).\n"
            f"- KPI AWARENESS: The project metric is '{kpi_metric}' with a target of {kpi_target}. Integrate this into relevant acceptance criteria.\n\n"

            "### Context Dump\n"
            "You are generating work for 'deepRLPH', an autonomous software construction engine. The workers need bite-sized, independent tasks. "
            "Use the provided workspace context to understand what files already exist so you do not ask agents to unnecessarily recreate existing components."
        )

        user_message = (
            f"TASK TO IMPLEMENT:\n{task}\n\n"
            f"DOMAIN:\n{domain}\n\n"
            f"EXISTING WORKSPACE FILES:\n{json.dumps(files_summary, indent=2)}\n\n"
            "Generate the JSON PRD Backlog now."
        )

        logger.info("🧠 Generating dynamic PRD via DeepSeek...")

        try:
            response = await self.deepseek.call_agent(
                system_prompt=system_prompt,
                user_message=user_message,
                thinking_budget=4000,
                temperature=0.2
            )

            if response.get("status") == "success":
                raw_text = response["response"]

                # Clean markdown formatting if present
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0]
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0]

                prd_data = json.loads(raw_text.strip())

                # Merge metadata
                prd_data["project_metadata"] = project_meta
                prd_data["task"] = task
                prd_data["domain"] = domain

                # Normalize and return
                return normalize_prd(prd_data)
            else:
                logger.error(f"❌ AI PRD Generation failed: {response.get('error')}")
                return self._fallback_prd(task, project_meta)

        except Exception as e:
            logger.error(f"❌ Exception in AI PRD Generation: {e}", exc_info=True)
            return self._fallback_prd(task, project_meta)

    def _fallback_prd(self, task: str, project_meta: dict) -> dict:
        logger.warning("⚠️ Falling back to simple generic PRD item")
        raw_prd = {
            "task": task,
            "domain": "unknown",
            "total_items": 1,
            "user_stories": [
                {
                    "id": "ITEM-001",
                    "title": "Implement Requested Task",
                    "description": task,
                    "acceptance_criteria": ["Implementation meets the task description"],
                    "verification": "pytest -v",
                    "why": "Fallback item due to AI generation failure",
                    "files_touched": [],
                    "status": "todo",
                    "attempts": 0,
                    "errors": []
                }
            ],
            "project_metadata": project_meta
        }
        return normalize_prd(raw_prd)


def generate_prd(
        technical_brief: Dict[str, Any],
        domain: str = "llm-app",
        project_metadata: Optional[Dict[str, Any]] = None,
        deepseek_client: Any = None,
) -> Dict[str, Any]:
    """Convenience function to generate PRD."""
    import asyncio
    gen = PRDGenerator(deepseek_client)
    return asyncio.run(gen.generate(technical_brief, domain, project_metadata))