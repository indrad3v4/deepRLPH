# -*- coding: utf-8 -*-
"""
code_generator_agent.py - PR-002: Code Generator Agent (3-4 hours)

Converts PRD items → executable Python code via DeepSeek extended thinking.
Handles code extraction, validation, and artifact management.

Architecture:
  CodeGeneratorAgent
  ├── async def generate(subtask_desc, context) → CodeArtifact
  ├── async def _call_deepseek_with_thinking(prompt) → response
  ├── def _extract_code_blocks(response) → List[CodeBlock]
  ├── def _validate_syntax(code) → bool
  └── async def save_artifacts(artifacts, output_dir) → List[Path]
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger("CodeGeneratorAgent")


@dataclass
class CodeBlock:
    """Extracted code snippet from LLM response."""
    language: str
    content: str
    line_start: int
    line_end: int
    confidence: float = 1.0  # 0-1, how confident we are this is valid code


@dataclass
class CodeArtifact:
    """Complete code generation result."""
    subtask_id: str
    subtask_title: str
    generated_at: datetime
    code_blocks: List[CodeBlock] = field(default_factory=list)
    raw_response: str = ""
    files_to_create: Dict[str, str] = field(default_factory=dict)  # {filepath: content}
    validation_errors: List[str] = field(default_factory=list)
    estimated_effort_hours: float = 0.0

    def is_valid(self) -> bool:
        """Check if artifact has no critical errors."""
        return len(self.validation_errors) == 0 and len(self.code_blocks) > 0


class CodeGeneratorAgent:
    """
    Generates production-ready Python code from PRD subtasks.

    Flow:
    1. Receive subtask (GG-001, GG-002, etc.)
    2. Call DeepSeek with extended thinking (v3.2+)
    3. Extract code blocks from response
    4. Validate Python syntax
    5. Return structured artifacts ready for file output
    """

    def __init__(
            self,
            deepseek_client,
            agent_coordinator,
            project_dir: Path,
            log_callback: Optional[callable] = None,
    ):
        """
        Initialize code generator agent.

        Args:
            deepseek_client: DeepseekClient instance (must have async methods)
            agent_coordinator: AgentCoordinator for task distribution
            project_dir: Root project directory
            log_callback: Function to log messages
        """
        self.deepseek = deepseek_client
        self.coordinator = agent_coordinator
        self.project_dir = project_dir
        self.log = log_callback or self._default_log
        self.model = "deepseek-chat"  # or "deepseek-v3.2" if extended thinking available

    def _default_log(self, msg: str, level: str = "INFO"):
        """Default logging function."""
        logger.log(getattr(logging, level), msg)

    async def generate(
            self,
            subtask_id: str,
            subtask_title: str,
            subtask_description: str,
            acceptance_criteria: List[str],
            context: Optional[Dict[str, Any]] = None,
    ) -> CodeArtifact:
        """
        Generate code for a single PRD subtask.

        Args:
            subtask_id: PR-002, GG-001, etc.
            subtask_title: "Code Generator Agent"
            subtask_description: Full description
            acceptance_criteria: List of AC bullets
            context: Optional context (previous artifacts, dependencies)

        Returns:
            CodeArtifact with generated code blocks
        """
        self.log(f"[{subtask_id}] Starting code generation for: {subtask_title}")

        artifact = CodeArtifact(
            subtask_id=subtask_id,
            subtask_title=subtask_title,
            generated_at=datetime.now(),
        )

        # Build prompt with extended thinking
        prompt = self._build_generation_prompt(
            subtask_id=subtask_id,
            title=subtask_title,
            description=subtask_description,
            criteria=acceptance_criteria,
            context=context,
        )

        try:
            # Call DeepSeek with extended thinking (streaming or batch)
            response = await self._call_deepseek_with_thinking(prompt)
            artifact.raw_response = response
            self.log(f"[{subtask_id}] Received response ({len(response)} chars)")

            # Extract code blocks
            code_blocks = self._extract_code_blocks(response)
            artifact.code_blocks = code_blocks
            self.log(f"[{subtask_id}] Extracted {len(code_blocks)} code blocks")

            # Validate syntax
            for block in code_blocks:
                errors = self._validate_syntax(block.content, block.language)
                if errors:
                    artifact.validation_errors.extend(errors)
                    block.confidence = 0.6
                else:
                    block.confidence = 0.95

            # Map code blocks to files
            artifact.files_to_create = self._map_to_files(
                code_blocks, subtask_id, context
            )

            # Estimate effort
            artifact.estimated_effort_hours = self._estimate_effort(
                len(code_blocks), len(artifact.files_to_create)
            )

            self.log(
                f"[{subtask_id}] Generation complete: "
                f"{len(artifact.code_blocks)} blocks, "
                f"{len(artifact.files_to_create)} files, "
                f"valid={artifact.is_valid()}"
            )

            return artifact

        except Exception as e:
            self.log(f"[{subtask_id}] ERROR: {str(e)}", "ERROR")
            artifact.validation_errors.append(f"Generation failed: {str(e)}")
            return artifact

    async def _call_deepseek_with_thinking(self, prompt: str) -> str:
        """
        Call DeepSeek API with extended thinking enabled.

        Note: This assumes DeepSeekClient supports:
        - async method
        - thinking_budget or extended_thinking parameter
        - Returns raw text response

        Args:
            prompt: Full generation prompt

        Returns:
            Raw response text (includes thinking + code)
        """
        try:
            # Try extended thinking (v3.2+)
            response = await self.deepseek.acall(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior Python engineer. Generate production-ready code. "
                            "Always use extended thinking to plan the implementation before writing."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=8000,
                # Extended thinking parameters (if supported):
                # thinking_budget=3000,
                # use_extended_thinking=True,
            )

            # Handle different response formats
            if isinstance(response, dict):
                return response.get("content", str(response))
            return str(response)

        except Exception as e:
            self.log(f"DeepSeek call failed: {str(e)}", "ERROR")
            raise

    def _build_generation_prompt(
            self,
            subtask_id: str,
            title: str,
            description: str,
            criteria: List[str],
            context: Optional[Dict] = None,
    ) -> str:
        """Build structured prompt for code generation."""
        context_str = ""
        if context:
            context_str = f"\n\nContext from prior work:\n{context.get('notes', '')}"

        criteria_str = "\n".join([f"  - {c}" for c in criteria])

        prompt = f"""
You are generating Python code for a PRD subtask.

**SUBTASK:** {subtask_id}: {title}

**DESCRIPTION:**
{description}

**ACCEPTANCE CRITERIA:**
{criteria_str}

**REQUIREMENTS:**
1. Write ONLY production-ready Python code (no pseudocode)
2. Include docstrings for all classes and functions
3. Use type hints (Python 3.9+)
4. Use async/await patterns where appropriate
5. Include error handling and logging
6. Include unit test stubs
7. Format all code in ```python code blocks
8. Specify the target filepath in comments: # → src/module/file.py

**OUTPUT FORMAT:**
- First, use extended thinking to plan the implementation
- Then provide complete code blocks with filepath comments
- After code, provide a brief implementation summary

Generate complete, working code now:
"""
        if context_str:
            prompt += context_str

        return prompt

    def _extract_code_blocks(self, response: str) -> List[CodeBlock]:
        """
        Extract all code blocks from LLM response.

        Handles:
        - ```python code blocks
        - ```python\n # → filepath\n code blocks
        - Multiple blocks per response
        """
        blocks = []

        # Pattern: ```language\ncode\n```
        pattern = r"```(\w+)\n(.*?)\n```"
        matches = re.finditer(pattern, response, re.DOTALL)

        for i, match in enumerate(matches):
            language = match.group(1)
            content = match.group(2)

            block = CodeBlock(
                language=language,
                content=content,
                line_start=response[:match.start()].count("\n"),
                line_end=response[:match.end()].count("\n"),
            )
            blocks.append(block)
            self.log(f"Extracted block {i + 1}: {language} ({len(content)} chars)")

        return blocks

    def _validate_syntax(self, code: str, language: str = "python") -> List[str]:
        """
        Validate Python code syntax.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if language.lower() not in ["python", "py"]:
            return [f"Unsupported language: {language}"]

        try:
            # Try to parse as Python AST
            import ast
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")

        # Additional checks
        if not code.strip():
            errors.append("Code block is empty")

        if code.count("TODO") > 0:
            errors.append("Code contains TODO markers (incomplete)")

        return errors

    def _map_to_files(
            self,
            code_blocks: List[CodeBlock],
            subtask_id: str,
            context: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Map code blocks to filesystem paths.

        Looks for comments like:
        # → src/module/file.py
        """
        files = {}

        for block in code_blocks:
            # Try to extract filepath from first line
            lines = block.content.split("\n")
            filepath = None

            # Look for → comment in first 3 lines
            for line in lines[:3]:
                if "→" in line and ".py" in line:
                    # Extract: # → src/challenges/models.py
                    match = re.search(r"→\s+(.+\.py)", line)
                    if match:
                        filepath = match.group(1)
                        break

            # Fallback: auto-generate path based on subtask
            if not filepath:
                # GG-001 → src/challenges/models.py
                module_map = {
                    "GG-001": "challenges",
                    "GG-002": "blockchain",
                    "GG-003": "evidence",
                    "GG-004": "dashboard",
                    "GG-005": "nft",
                    "GG-006": "token",
                    "GG-007": "anti_cheat",
                    "PR-002": "code_generator",  # This agent itself
                }
                module = module_map.get(subtask_id, "core")
                filepath = f"src/{module}/{subtask_id.lower()}.py"

            # Normalize path
            filepath = str(Path(filepath))

            # Remove filepath comment from content for clean output
            clean_content = block.content
            for line in lines[:3]:
                if "→" in line:
                    clean_content = clean_content.replace(line + "\n", "")

            files[filepath] = clean_content

        return files

    def _estimate_effort(self, num_blocks: int, num_files: int) -> float:
        """Rough estimate of implementation effort in hours."""
        # Base estimate: 1h per file + 0.5h per block for integration
        return num_files * 1.0 + num_blocks * 0.5

    async def save_artifacts(
            self,
            artifacts: List[CodeArtifact],
            output_dir: Path,
            dry_run: bool = False,
    ) -> Dict[str, Path]:
        """
        Save generated code artifacts to filesystem.

        Args:
            artifacts: List of CodeArtifact objects
            output_dir: Root output directory
            dry_run: If True, only log without writing

        Returns:
            Mapping of filepath → Path (actual written files)
        """
        written_files = {}

        for artifact in artifacts:
            self.log(f"[{artifact.subtask_id}] Saving {len(artifact.files_to_create)} files...")

            for filepath, content in artifact.files_to_create.items():
                full_path = output_dir / filepath

                # Create parent directories
                full_path.parent.mkdir(parents=True, exist_ok=True)

                if dry_run:
                    self.log(f"  [DRY RUN] Would write: {full_path}")
                else:
                    full_path.write_text(content, encoding="utf-8")
                    self.log(f"  ✓ Wrote: {full_path}")
                    written_files[filepath] = full_path

        return written_files


# ============================================================================
# USAGE EXAMPLE (in orchestrator or execution_engine)
# ============================================================================

async def example_usage():
    """
    Example: How to use CodeGeneratorAgent in orchestrator.

    Call this from ExecutionEngine.execute() after AgentCoordinator
    distributes tasks.
    """
    # Initialize (injected from main)
    from deepseek_client import DeepseekClient
    from agent_coordinator import AgentCoordinator

    deepseek = DeepseekClient(api_key="sk-...")
    coordinator = AgentCoordinator()
    project_dir = Path("./hunt-satoshi")

    # Create agent
    code_gen = CodeGeneratorAgent(
        deepseek_client=deepseek,
        agent_coordinator=coordinator,
        project_dir=project_dir,
    )

    # Generate code for GG-001
    artifact = await code_gen.generate(
        subtask_id="GG-001",
        subtask_title="Gamified Challenge System",
        subtask_description="Design a quest/challenge system...",
        acceptance_criteria=[
            "50+ unique challenges in database",
            "Challenge difficulty tiers (Easy → Legendary)",
            "Point system: 10-100 points per challenge",
        ],
    )

    # Check result
    if artifact.is_valid():
        print(f"✓ Generated {len(artifact.code_blocks)} code blocks")

        # Save to filesystem
        written = await code_gen.save_artifacts([artifact], project_dir / "output")
        print(f"✓ Wrote {len(written)} files")
    else:
        print(f"✗ Validation errors: {artifact.validation_errors}")
