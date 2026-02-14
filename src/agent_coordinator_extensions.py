# -*- coding: utf-8 -*-
"""
agent_coordinator_extensions.py - PRD Agent Assignment Extensions

ITEM-004: Agent Assignment Protocol

Extensions to AgentCoordinator for assigning PRD items to agents.
"""

import asyncio
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from prd_model import PRDItem

logger = logging.getLogger(__name__)


class PRDAgentAssignmentMixin:
    """Mixin for AgentCoordinator to add PRD item assignment.
    
    Usage:
        class AgentCoordinator(PRDAgentAssignmentMixin, ...):
            ...
    
    This mixin provides assign_prd_item() method that:
    1. Takes a PRD item
    2. Generates prompt for agent
    3. Calls DeepSeek to generate code
    4. Extracts code blocks from response
    5. Saves files to project directory
    6. Returns files created
    """
    
    async def assign_prd_item(
        self,
        item: PRDItem,
        agent_id: str,
        project_dir: Path,
        deepseek_client: Any,
    ) -> Dict[str, Any]:
        """Assign PRD item to agent for implementation.
        
        Args:
            item: PRD item to implement
            agent_id: ID of agent (e.g., "agent_1")
            project_dir: Project directory path
            deepseek_client: DeepSeek client instance
        
        Returns:
            Dict with:
            - status: "success" | "error"
            - files_created: List of file paths created
            - response: Raw LLM response
            - error: Error message if failed
        """
        logger.info(f"[{agent_id}] Assigning {item.item_id}: {item.title}")
        
        try:
            # Generate prompt
            prompt = self._generate_prd_item_prompt(item)
            
            # Call DeepSeek
            if not deepseek_client:
                return {
                    "status": "error",
                    "error": "DeepSeek client not initialized",
                    "files_created": [],
                }
            
            logger.info(f"[{agent_id}] Calling DeepSeek...")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: deepseek_client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior software engineer implementing PRD items. "
                            "Generate complete, production-ready code with tests. "
                            "Include file paths as comments: # File: path/to/file.py"
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=4000,
                ),
            )
            
            response_text = response.get("content", "")
            logger.info(f"[{agent_id}] Received response ({len(response_text)} chars)")
            
            # Extract and save code blocks
            files_created = await self._extract_and_save_code(
                response_text=response_text,
                project_dir=project_dir,
                item_id=item.item_id,
            )
            
            logger.info(f"[{agent_id}] Created {len(files_created)} file(s)")
            
            return {
                "status": "success",
                "files_created": files_created,
                "response": response_text,
            }
        
        except Exception as e:
            error = f"Agent assignment error: {str(e)}"
            logger.exception(f"[{agent_id}] Assignment failed")
            return {
                "status": "error",
                "error": error,
                "files_created": [],
            }
    
    def _generate_prd_item_prompt(self, item: PRDItem) -> str:
        """Generate prompt for PRD item implementation.
        
        Args:
            item: PRD item to implement
        
        Returns:
            Formatted prompt string
        """
        acceptance = "\n".join([f"  - {c}" for c in item.acceptance_criteria])
        files = ", ".join(item.files_touched) if item.files_touched else "(to be determined)"
        
        return f"""Implement the following PRD item:

**Item ID**: {item.item_id}
**Title**: {item.title}
**Priority**: {item.priority}

**Acceptance Criteria**:
{acceptance}

**Verification Command**: {item.verification_command}

**Files to Create/Modify**: {files}

**Requirements**:
1. Generate complete, working code for this item
2. Include all necessary imports, type hints, and docstrings
3. Add error handling and logging where appropriate
4. Ensure code is production-ready and follows best practices
5. Write unit tests that will pass the verification command
6. At the top of each code block, include: # File: path/to/file.py

**Output Format**:
- Use markdown code blocks with language tags
- Start each block with file path comment
- Generate all necessary files (source + tests)
"""
    
    async def _extract_and_save_code(
        self,
        response_text: str,
        project_dir: Path,
        item_id: str,
    ) -> List[str]:
        """Extract code blocks from LLM response and save to files.
        
        Args:
            response_text: Raw LLM response
            project_dir: Project directory
            item_id: PRD item ID (for fallback naming)
        
        Returns:
            List of file paths created
        """
        files_created = []
        
        # Extract code blocks: ```language\ncode\n```
        code_blocks = re.findall(
            r"```(\w+)?\n(.*?)\n```",
            response_text,
            re.DOTALL
        )
        
        if not code_blocks:
            logger.warning(f"No code blocks found in response for {item_id}")
            # Save raw response as fallback
            fallback_file = project_dir / "workspace" / "output" / "generated_code" / f"{item_id}_raw.txt"
            fallback_file.parent.mkdir(parents=True, exist_ok=True)
            fallback_file.write_text(response_text, encoding="utf-8")
            files_created.append(str(fallback_file))
            return files_created
        
        # Process each code block
        for block_idx, (lang, code) in enumerate(code_blocks, 1):
            # Extract file path from comment
            file_path = self._extract_file_path(code)
            
            if not file_path:
                # Infer filename from code or use fallback
                file_path = self._infer_filename(code, lang, item_id, block_idx)
            
            # Create full path
            target_file = project_dir / file_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            try:
                target_file.write_text(code, encoding="utf-8")
                files_created.append(str(target_file))
                logger.info(f"   💾 Saved: {file_path}")
            except Exception as e:
                logger.error(f"   ❌ Failed to save {file_path}: {e}")
        
        return files_created
    
    def _extract_file_path(self, code: str) -> Optional[str]:
        """Extract file path from code comment.
        
        Looks for patterns like:
        - # File: src/example.py
        - # file: tests/test_example.py
        - // File: src/example.js
        
        Args:
            code: Code block text
        
        Returns:
            File path if found, None otherwise
        """
        match = re.search(r"^[#/]+\s*[Ff]ile:\s*(.+?)\s*$", code, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
    
    def _infer_filename(
        self,
        code: str,
        lang: str,
        item_id: str,
        block_idx: int,
    ) -> str:
        """Infer filename from code content.
        
        Args:
            code: Code block text
            lang: Language tag (python, javascript, etc.)
            item_id: PRD item ID
            block_idx: Block index in response
        
        Returns:
            Inferred file path
        """
        # Try to extract class name
        match = re.search(r"^class\s+(\w+)", code, re.MULTILINE)
        if match:
            class_name = match.group(1)
            # Convert CamelCase to snake_case
            snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
            return f"src/{snake_case}.py"
        
        # Try to extract function name
        match = re.search(r"^def\s+(\w+)", code, re.MULTILINE)
        if match:
            func_name = match.group(1)
            return f"src/{func_name}.py"
        
        # Check if it's a test file
        if "import pytest" in code or "import unittest" in code or "def test_" in code:
            return f"tests/test_{item_id}_block{block_idx}.py"
        
        # Language-specific extensions
        lang_ext = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rust": "rs",
            "sql": "sql",
            "bash": "sh",
            "shell": "sh",
        }
        ext = lang_ext.get(lang or "", "txt")
        
        # Fallback to generic name
        return f"src/{item_id}_block{block_idx}.{ext}"
