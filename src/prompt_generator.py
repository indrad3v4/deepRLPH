# -*- coding: utf-8 -*-
"""
RALPH Prompt Generator - Meta-Prompting for AI Suggestions

Implements two-phase DeepSeek execution:
1. Phase 1: Generate specialized prompt based on project context
2. Phase 2: Execute specialized prompt to get implementation-ready specs
3. Phase 2B (NEW): Expand config into PRD backlog with acceptance criteria
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger("PromptGenerator")


class PromptGenerator:
    """Generate specialized prompts for AI suggestion generation"""

    def __init__(self, deepseek_client: Any):
        self.deepseek = deepseek_client

    async def generate_meta_prompt(self, project_data: Dict[str, Any]) -> str:
        """
        Phase 1: Generate specialized prompt based on project context.
        """
        project_type = project_data.get('project_type', 'api')
        description = project_data.get('description', '')
        files_summary = self._summarize_files(project_data)

        meta_prompt = f"""You are a technical architect analyzing this project:

Project Type: {project_type.upper()}
Description: {description}
Files Attached: {files_summary}

Your task: Generate a DETAILED system prompt that will produce implementation-ready specifications.

The specialized prompt you create MUST instruct the AI to provide:

1. **Exact Architecture** (not "Transformer" but "PatchTST with patch_size=16, num_layers=3, d_model=128...")
2. **Complete Training Config** (optimizer: AdamW, lr: 3e-4, scheduler: cosine_annealing, warmup_steps: 500...)
3. **Data Pipeline Specs** (preprocessing: RevIN, augmentation: [TimeWarp, MagnitudeWarp], validation: TimeSeriesSplit_5fold...)
4. **Actionable Implementation Tasks** with:
   - Task name
   - Detailed implementation steps
   - Acceptance criteria (measurable, testable)
   - File paths to modify
5. **Submission/Deployment Requirements**

For ML competitions, the prompt should ask for:
- Modern 2026 architectures (PatchTST, iTransformer, Mamba, TSMixer)
- Hardware-aware hyperparameters (batch size based on GPU memory)
- Realistic metric targets (baseline_target, stretch_target)
- Foundation model transfer learning strategies if applicable

For API projects, the prompt should ask for:
- Specific framework versions and extensions
- Database schema design
- API endpoint specifications
- Testing strategy with coverage targets

Output the specialized prompt as a single string that will be used as the system prompt in Phase 2.
"""

        try:
            logger.info("🧠 Generating meta-prompt (Phase 1)...")
            result = await self.deepseek.call_agent(
                system_prompt="You are a prompt engineering expert specializing in technical architecture.",
                user_message=meta_prompt,
                thinking_budget=3000,
                temperature=0.2
            )

            if result['status'] == 'success':
                specialized_prompt = result['response'].strip()
                logger.info(f"✅ Meta-prompt generated ({len(specialized_prompt)} chars)")
                return specialized_prompt
            else:
                logger.error(f"❌ Meta-prompt generation failed: {result.get('error')}")
                return self._get_fallback_prompt(project_type)

        except Exception as e:
            logger.error(f"❌ Exception in meta-prompt generation: {e}", exc_info=True)
            return self._get_fallback_prompt(project_type)

    async def execute_specialized_prompt(self, specialized_prompt: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Execute specialized prompt to get implementation-ready specs.
        """
        user_message = f"""Project Details:
{json.dumps(project_data, indent=2)}

Analyze this project and provide comprehensive, implementation-ready specifications.
Ensure all architectural decisions are specific (e.g., model names, exact hyperparameters).
Provide reasoning for key decisions.

Respond with valid JSON only, no markdown code blocks.
"""

        try:
            logger.info("🔬 Executing specialized prompt (Phase 2)...")
            result = await self.deepseek.call_agent(
                system_prompt=specialized_prompt,
                user_message=user_message,
                thinking_budget=5000,
                temperature=0.3
            )

            if result['status'] == 'success':
                response_text = result['response']
                try:
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()

                    suggestions = json.loads(response_text)
                    logger.info("✅ Implementation-ready specs generated")
                    return suggestions

                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse JSON: {e}")
                    return {'error': f'JSON parse error: {str(e)}', 'raw_response': response_text[:500]}
            else:
                logger.error(f"❌ Phase 2 execution failed: {result.get('error')}")
                return {'error': result.get('error', 'Unknown error')}

        except Exception as e:
            logger.error(f"❌ Exception in Phase 2: {e}", exc_info=True)
            return {'error': str(e)}

    async def expand_to_prd_backlog(self, config: Dict[str, Any], project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2B (NEW): Expand config into PRD backlog with acceptance criteria.
        Dynamically adapts rules and examples based on project type.
        """
        project_type = project_data.get('project_type', 'api')
        project_name = project_data.get('name', 'Unnamed Project')
        description = project_data.get('description', '')

        # Universal rules
        universal_rules = """
# Rules for Items
1. **Small scope**: Each item should take 1-2 hours max for an agent to implement
2. **Independent**: Items should not depend on each other's internal implementation
3. **Testable**: Every item MUST have automated verification (pytest command) or clear manual check
4. **Specific**: Avoid generic tasks. Be hyper-specific about functions and architectures.
5. **Ordered**: Priority 1 = foundational (e.g., data loading), higher priority = depends on earlier items
"""

        # Domain-specific rules and examples
        if project_type == 'ml':
            domain_rules = """
# CRITICAL SUBMISSION RULES FOR ML PROJECTS
The final item MUST be for submission packaging:
- The inference script MUST be named exactly `solution.py` and placed in the ROOT directory (e.g., "solution.py", NOT "src/solution.py" and NOT "src/inference.py").
- The verification command for this final item MUST include the packager tool exactly like this: `pytest tests/test_onnx.py -v && python scripts/make_submission.py .`
"""
            example_json = """{
  "backlog": [
    {
      "item_id": "ITEM-001",
      "title": "Implement Custom Weighted Pearson Loss",
      "why": "Core metric for competition evaluation",
      "priority": 1,
      "acceptance_criteria": [
        "Loss function matches competition formula",
        "Unit tests cover 5+ scenarios including edge cases"
      ],
      "verification_command": "pytest tests/test_loss.py -v",
      "verification_type": "automated",
      "files_touched": ["src/losses/weighted_pearson.py", "tests/test_loss.py"],
      "estimated_lines": 80
    },
    {
      "item_id": "ITEM-008",
      "title": "Implement ONNX Export and Submission Script",
      "why": "Ensure model runs in competition environment",
      "priority": 8,
      "acceptance_criteria": [
        "Inference script implements PredictionModel interface",
        "Creates valid submission.zip using packaging script"
      ],
      "verification_command": "pytest tests/test_onnx.py -v && python scripts/make_submission.py .",
      "verification_type": "automated",
      "files_touched": ["src/onnx/export.py", "solution.py", "tests/test_onnx.py"],
      "estimated_lines": 90
    }
  ],
  "execution_plan": "FOR EACH item: Implement -> Verify -> Commit.",
  "definition_of_done": ["All tests pass", "Metric target achieved"],
  "file_structure": {"Project/": ["src/losses/weighted_pearson.py", "solution.py", "tests/test_loss.py"]}
}"""
        else:
            domain_rules = """
# CRITICAL RULES FOR API/SOFTWARE PROJECTS
- Ensure a clear entry point (e.g., `main.py` or `app.py`) in the root directory.
- The verification commands should run the test suite (e.g., `pytest tests/ -v`).
"""
            example_json = """{
  "backlog": [
    {
      "item_id": "ITEM-001",
      "title": "Setup FastAPI Application and Database Models",
      "why": "Foundation for the API",
      "priority": 1,
      "acceptance_criteria": [
        "FastAPI app initializes correctly",
        "SQLAlchemy models match database schema",
        "Unit tests verify app creation and db connection"
      ],
      "verification_command": "pytest tests/test_main.py -v",
      "verification_type": "automated",
      "files_touched": ["main.py", "src/database/models.py", "tests/test_main.py"],
      "estimated_lines": 120
    }
  ],
  "execution_plan": "FOR EACH item: Implement -> Verify -> Commit.",
  "definition_of_done": ["All tests pass", "Coverage > 85%"],
  "file_structure": {"Project/": ["main.py", "src/database/models.py", "tests/test_main.py"]}
}"""

        # Build final expansion prompt
        expansion_prompt = f"""You are a senior technical PM converting high-level architecture into an executable PRD backlog.

# Project Context
Name: {project_name}
Type: {project_type.upper()}
Description: {description}

# High-Level Configuration
{json.dumps(config, indent=2)}

# Your Task
Convert this config into a PRD backlog of 5-8 small, testable items that fit in one agent iteration.
Each item must be independently implementable and verifiable.
{universal_rules}
{domain_rules}

# Required Output Format (JSON)
{example_json}

Provide ONLY valid JSON matching the format above. No markdown code blocks.
"""

        try:
            logger.info("📋 Expanding config to PRD backlog (Phase 2B)...")
            result = await self.deepseek.call_agent(
                system_prompt="You are an expert technical PM. Convert architecture into executable PRD items with acceptance criteria and verification commands.",
                user_message=expansion_prompt,
                thinking_budget=6000,
                temperature=0.2
            )

            if result['status'] == 'success':
                response_text = result['response']
                try:
                    if '```json' in response_text:
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif '```' in response_text:
                        response_text = response_text.split('```')[1].split('```')[0].strip()

                    prd_backlog = json.loads(response_text)

                    if 'backlog' not in prd_backlog or not isinstance(prd_backlog['backlog'], list):
                        logger.error("❌ PRD backlog missing 'backlog' array")
                        return {'error': 'Invalid PRD structure: missing backlog array'}

                    logger.info(f"✅ PRD backlog generated with {len(prd_backlog['backlog'])} items")
                    return prd_backlog

                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse PRD JSON: {e}")
                    return {'error': f'JSON parse error: {str(e)}', 'raw_response': response_text[:500]}
            else:
                logger.error(f"❌ PRD expansion failed: {result.get('error')}")
                return {'error': result.get('error', 'Unknown error')}

        except Exception as e:
            logger.error(f"❌ Exception in PRD expansion: {e}", exc_info=True)
            return {'error': str(e)}

    def _summarize_files(self, project_data: Dict[str, Any]) -> str:
        """Summarize attached files for context"""
        doc_files = project_data.get('doc_files', [])
        dataset_files = project_data.get('dataset_files', [])
        baseline_files = project_data.get('baseline_files', [])

        summary_parts = []
        if doc_files:
            summary_parts.append(f"{len(doc_files)} documentation files")
        if dataset_files:
            summary_parts.append(f"{len(dataset_files)} dataset files")
        if baseline_files:
            summary_parts.append(f"{len(baseline_files)} baseline code/model files")

        return ", ".join(summary_parts) if summary_parts else "none"

    def _get_fallback_prompt(self, project_type: str) -> str:
        """Fallback prompt if meta-prompt generation fails"""
        if project_type == 'ml':
            return """You are an expert ML engineer. Analyze the project and provide:
1. **Architecture**: Specific model name
2. **Training Config**: optimizer, scheduler, batch_size
3. **Data Pipeline**: validation strategy
4. **Implementation Tasks**: List of 5-7 tasks
Provide JSON with these fields. Be specific, not vague."""
        else:
            return """You are an expert software architect. Analyze the project and provide:
1. **Architecture**: Specific pattern
2. **Stack**: framework, database
3. **API Design**: key endpoints
4. **Implementation Tasks**: List of 5-7 tasks
Provide JSON with these fields. Be specific, not vague."""