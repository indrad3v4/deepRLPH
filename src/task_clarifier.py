# -*- coding: utf-8 -*-
"""
task_clarifier.py - Task Clarification Agent (PR-000)

Takes raw user task + UI context, calls DeepSeek to deepen understanding.
Outputs: Technical brief with clarified requirements.
"""

import asyncio
import logging
from typing import Dict, Any
import json

logger = logging.getLogger("TaskClarifier")


class TaskClarifier:
    """
    First agent in deepRLPH: clarify vague user tasks.

    Input: Raw task + project context (domain, framework, db)
    Output: Technical brief (clarified requirements)
    """

    def __init__(self, deepseek_client):
        """
        Args:
            deepseek_client: DeepseekClient instance (injected from orchestrator)
        """
        self.deepseek = deepseek_client

    async def clarify(self,
                      raw_task: str,
                      domain: str = "llm-app",
                      framework: str = "FastAPI",
                      database: str = "PostgreSQL") -> Dict[str, Any]:
        """
        Clarify a vague technical task.
        """

        logger.info("🔍 Task Clarification Agent Starting...")
        logger.info(f"   Raw task: {raw_task[:80]}...")
        logger.info(f"   Domain: {domain} | Framework: {framework} | DB: {database}")

        # ✅ ADD THIS CHECK - CRITICAL
        if not self.deepseek:
            logger.error("❌ DeepSeek client not initialized")
            return {
                "status": "error",
                "error": "DeepSeek client not initialized. Check DEEPSEEK_API_KEY in .env",
                "raw_task": raw_task,
                "clarified_task": raw_task
            }

        # Build prompt for clarification
        clarification_prompt = self._build_clarification_prompt(
            raw_task, domain, framework, database
        )

        try:
            # Call DeepSeek
            logger.info("📞 Calling DeepSeek for task analysis...")
            response = await self.deepseek.call_agent(
                system_prompt=self._get_system_prompt(),
                user_message=clarification_prompt,
                thinking_budget=3000
            )

            logger.info("✅ Task clarification complete")

            # Parse response
            brief = self._parse_response(response)

            return {
                "status": "success",
                "raw_task": raw_task,
                "clarified_task": brief.get("clarified_task"),
                "key_requirements": brief.get("key_requirements", []),
                "constraints": brief.get("constraints", []),
                "technical_brief": brief.get("technical_brief"),
                "estimated_effort_hours": brief.get("estimated_effort_hours", 8),
                "context": {
                    "domain": domain,
                    "framework": framework,
                    "database": database
                }
            }

        except Exception as e:
            logger.error(f"❌ Clarification failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "raw_task": raw_task,
                "clarified_task": raw_task  # Fallback
            }

    def _build_clarification_prompt(self,
                                    raw_task: str,
                                    domain: str,
                                    framework: str,
                                    database: str) -> str:
        """Build detailed prompt for clarification agent"""

        return f"""You are a technical requirements engineer. Your task is to clarify and deepen a vague technical task.

**USER'S RAW TASK:**
{raw_task}

**PROJECT CONTEXT:**
- Domain: {domain}
- Framework: {framework}
- Database: {database}

**YOUR JOB:**
Analyze the raw task and produce a detailed TECHNICAL BRIEF that answers:

1. **What** (Core functionality)
   - What does the system DO?
   - What problem does it SOLVE?
   - Main features/capabilities?

2. **Who** (Users & stakeholders)
   - Who uses it?
   - What are their pain points?
   - What's success for them?

3. **How** (Technical details)
   - What's the architecture pattern?
   - Any integrations needed?
   - Authentication approach?
   - Performance requirements?
   - Scale (users, data volume)?

4. **Constraints**
   - Security requirements?
   - Performance targets?
   - Deadline/timeline?
   - Budget constraints?
   - Compliance/regulatory?

5. **Success Criteria**
   - How do we know it's done?
   - What metrics matter?
   - Testing requirements?

**OUTPUT FORMAT:**

```
CLARIFIED TASK:
[1-2 sentences, concrete and specific]

KEY REQUIREMENTS:
- [Requirement 1]
- [Requirement 2]
- [Requirement 3]
- [...]

CONSTRAINTS:
- [Constraint 1]
- [Constraint 2]
- [...]

TECHNICAL BRIEF:
[Full 2-3 paragraph brief with all details]

ESTIMATED EFFORT:
[Hours needed to build, be realistic]
```

Generate the brief NOW. Be specific and detailed."""

    def _get_system_prompt(self) -> str:
        """System prompt for clarification agent"""
        return """You are an expert technical requirements engineer and product manager.
Your specialty is taking vague business requirements and turning them into detailed technical specifications.
Be thorough, ask the hard questions, and clarify ambiguities."""

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse agent response into structured brief"""

        response_text = response.get("response", "")

        # Extract sections (simple parsing)
        brief = {
            "clarified_task": self._extract_section(response_text, "CLARIFIED TASK"),
            "key_requirements": self._extract_list_section(response_text, "KEY REQUIREMENTS"),
            "constraints": self._extract_list_section(response_text, "CONSTRAINTS"),
            "technical_brief": self._extract_section(response_text, "TECHNICAL BRIEF"),
            "estimated_effort_hours": self._extract_effort(response_text),
        }

        return brief

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a named section from response"""
        import re
        pattern = rf"{section_name}:\s*(.*?)(?=\n\n|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_list_section(self, text: str, section_name: str) -> list:
        """Extract a bulleted list section"""
        section = self._extract_section(text, section_name)
        items = [line.strip("- •").strip() for line in section.split('\n')
                 if line.strip().startswith(('-', '•'))]
        return items

    def _extract_effort(self, text: str) -> int:
        """Extract estimated effort in hours"""
        import re
        match = re.search(r'(\d+)\s*(?:hours?|hrs?)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 8  # Default


# Async wrapper for UI threading
async def run_clarifier_async(deepseek_client,
                              raw_task: str,
                              domain: str = "llm-app",
                              framework: str = "FastAPI",
                              database: str = "PostgreSQL") -> Dict[str, Any]:
    """Run task clarification in event loop"""
    clarifier = TaskClarifier(deepseek_client)
    return await clarifier.clarify(raw_task, domain, framework, database)