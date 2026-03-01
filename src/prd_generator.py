# -*- coding: utf-8 -*-
"""
prd_generator.py - Grounded AI-Driven PRD Generator

Guru-Level Refactor:
- BE-005: Schema-aware story generation (using real data shapes).
- Stage 4: Integrated DARIZ fix support.
- Pure Async: Removed asyncio.run() deadlocks.
"""

import logging
import json
from typing import Dict, Any, Optional
from prd_model import normalize_prd

logger = logging.getLogger("PRDGenerator")


class PRDGenerator:
    """Dynamic PRD generator powered by DeepSeek with Data-Aware Grounding."""

    def __init__(self, deepseek_client: Any):
        self.deepseek = deepseek_client

    async def generate(
            self,
            technical_brief: Dict[str, Any],
            domain: str = "llm-app",
            project_metadata: Optional[Dict[str, Any]] = None,
            dariz_strategy: Optional[str] = None  # Added for Stage 4 Fixes
    ) -> Dict[str, Any]:

        project_meta = project_metadata or technical_brief.get("project_metadata") or {}
        task = technical_brief.get("clarified_task", "")

        # BE-005: Explicit Schema Extraction
        dataset_schemas = project_meta.get("dataset_schemas", {})
        schema_context = json.dumps(dataset_schemas, indent=2)

        # Extract workspace context
        ingested_context = project_meta.get("ingested_context", {})
        files_summary = {
            "datasets": ingested_context.get("datasets", []),
            "code": ingested_context.get("code", []),
            "models": ingested_context.get("models", []),
        }

        # Stage 4 Logic: Pivot prompt if this is a DARIZ fix
        goal_text = "convert a technical brief into a JSON PRD Backlog."
        if dariz_strategy:
            goal_text = f"translate this Kukalev DARIZ TRIZ fix into executable tasks: {dariz_strategy}"

        system_prompt = (
            "### Goal\n"
            f"You are an elite Technical Product Manager. Your mission is to {goal_text}\n\n"
            "### Return Format\n"
            "Return ONLY valid JSON. The JSON must match this exact schema:\n"
            "{\n"
            '  "total_items": <int>,\n'
            '  "user_stories": [\n'
            "    {\n"
            '      "id": "FIX-001",\n'
            '      "title": "string",\n'
            '      "description": "string",\n'
            '      "acceptance_criteria": ["string"],\n'
            '      "verification": "exact shell command",\n'
            '      "files_touched": ["path/to/file.py"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "### Grounding Constraints (BE-005)\n"
            f"- DATA SCHEMAS: {schema_context}\n"
            "- Use actual column names and sequence lengths in acceptance criteria.\n"
            "- PRECISE VERIFICATION: Every story MUST have a functional verification command."
        )

        user_message = (
            f"TASK:\n{task}\n\n"
            f"STRATEGY:\n{dariz_strategy if dariz_strategy else 'Implementation from scratch'}\n\n"
            f"WORKSPACE:\n{json.dumps(files_summary, indent=2)}"
        )

        try:
            response = await self.deepseek.call_agent(
                system_prompt=system_prompt,
                user_message=user_message,
                thinking_budget=4000,
                temperature=0.2
            )

            if response.get("status") == "success":
                # Robust extraction logic (Improved regex from Roast)
                raw_text = response["response"]
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL)
                json_text = json_match.group(1).strip() if json_match else raw_text.strip()

                prd_data = json.loads(json_text)
                prd_data["project_metadata"] = project_meta
                return normalize_prd(prd_data)

            return self._fallback_prd(task, project_meta)
        except Exception as e:
            logger.error(f"PRD Generation Error: {e}")
            return self._fallback_prd(task, project_meta)

    def _fallback_prd(self, task: str, project_meta: dict) -> dict:
        # Improved fallback with better grounding info
        return normalize_prd({
            "task": task,
            "user_stories": [{
                "id": "ITEM-001",
                "title": "Implementation Handoff",
                "description": f"Manual implementation of: {task}",
                "acceptance_criteria": ["Code passes verification"],
                "verification": "pytest -v",
                "files_touched": []
            }],
            "project_metadata": project_meta
        })


# DEROASTED: Function is now properly async and doesn't use asyncio.run
async def generate_prd_async(
        technical_brief: Dict[str, Any],
        domain: str = "llm-app",
        project_metadata: Optional[Dict[str, Any]] = None,
        deepseek_client: Any = None,
) -> Dict[str, Any]:
    """Pure async PRD generation."""
    gen = PRDGenerator(deepseek_client)
    return await gen.generate(technical_brief, domain, project_metadata)