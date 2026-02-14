# -*- coding: utf-8 -*-
"""
test_agent_coordinator_extensions.py - Tests for ITEM-004

Tests for Agent Assignment Protocol
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.prd_model import PRDItem
from src.agent_coordinator_extensions import PRDAgentAssignmentMixin


class MockAgentCoordinator(PRDAgentAssignmentMixin):
    """Mock coordinator for testing mixin."""
    pass


class TestPRDAgentAssignmentMixin:
    """Tests for agent assignment protocol."""
    
    @pytest.mark.asyncio
    async def test_assign_prd_item_success(self, tmp_path):
        """Test successful PRD item assignment."""
        coordinator = MockAgentCoordinator()
        
        item = PRDItem(
            item_id="ITEM-001",
            title="Test Feature",
            priority=1,
            acceptance_criteria=["Must work", "Must have tests"],
            verification_command="pytest tests/",
            files_touched=["src/feature.py", "tests/test_feature.py"],
        )
        
        # Mock DeepSeek client
        mock_client = Mock()
        mock_client.chat = Mock(return_value={
            "content": """```python
# File: src/feature.py
class Feature:
    def work(self):
        return True
```

```python
# File: tests/test_feature.py
import pytest
from src.feature import Feature

def test_work():
    f = Feature()
    assert f.work() is True
```"""
        })
        
        # Execute assignment
        result = await coordinator.assign_prd_item(
            item=item,
            agent_id="agent_1",
            project_dir=tmp_path,
            deepseek_client=mock_client,
        )
        
        assert result["status"] == "success"
        assert len(result["files_created"]) == 2
        
        # Check files were created
        assert (tmp_path / "src" / "feature.py").exists()
        assert (tmp_path / "tests" / "test_feature.py").exists()
    
    @pytest.mark.asyncio
    async def test_assign_prd_item_no_client(self, tmp_path):
        """Test assignment without DeepSeek client."""
        coordinator = MockAgentCoordinator()
        
        item = PRDItem(
            item_id="ITEM-001",
            title="Test",
            priority=1,
        )
        
        result = await coordinator.assign_prd_item(
            item=item,
            agent_id="agent_1",
            project_dir=tmp_path,
            deepseek_client=None,
        )
        
        assert result["status"] == "error"
        assert "not initialized" in result["error"]
    
    @pytest.mark.asyncio
    async def test_assign_prd_item_client_error(self, tmp_path):
        """Test assignment with DeepSeek client error."""
        coordinator = MockAgentCoordinator()
        
        item = PRDItem(
            item_id="ITEM-001",
            title="Test",
            priority=1,
        )
        
        # Mock DeepSeek client that raises exception
        mock_client = Mock()
        mock_client.chat = Mock(side_effect=Exception("API error"))
        
        result = await coordinator.assign_prd_item(
            item=item,
            agent_id="agent_1",
            project_dir=tmp_path,
            deepseek_client=mock_client,
        )
        
        assert result["status"] == "error"
        assert "error" in result["error"].lower()
    
    def test_generate_prd_item_prompt(self):
        """Test prompt generation."""
        coordinator = MockAgentCoordinator()
        
        item = PRDItem(
            item_id="ITEM-001",
            title="Test Feature",
            priority=1,
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            verification_command="pytest tests/",
            files_touched=["src/test.py"],
        )
        
        prompt = coordinator._generate_prd_item_prompt(item)
        
        assert "ITEM-001" in prompt
        assert "Test Feature" in prompt
        assert "Criterion 1" in prompt
        assert "pytest tests/" in prompt
        assert "src/test.py" in prompt
    
    def test_extract_file_path(self):
        """Test file path extraction from code."""
        coordinator = MockAgentCoordinator()
        
        code_with_path = """# File: src/example.py
class Example:
    pass
"""
        
        path = coordinator._extract_file_path(code_with_path)
        assert path == "src/example.py"
    
    def test_extract_file_path_not_found(self):
        """Test file path extraction when not present."""
        coordinator = MockAgentCoordinator()
        
        code_without_path = """class Example:
    pass
"""
        
        path = coordinator._extract_file_path(code_without_path)
        assert path is None
    
    def test_infer_filename_from_class(self):
        """Test filename inference from class name."""
        coordinator = MockAgentCoordinator()
        
        code = """class MyExampleClass:
    pass
"""
        
        filename = coordinator._infer_filename(code, "python", "ITEM-001", 1)
        assert filename == "src/my_example_class.py"
    
    def test_infer_filename_from_function(self):
        """Test filename inference from function name."""
        coordinator = MockAgentCoordinator()
        
        code = """def my_function():
    pass
"""
        
        filename = coordinator._infer_filename(code, "python", "ITEM-001", 1)
        assert filename == "src/my_function.py"
    
    def test_infer_filename_test_file(self):
        """Test filename inference for test files."""
        coordinator = MockAgentCoordinator()
        
        code = """import pytest

def test_example():
    assert True
"""
        
        filename = coordinator._infer_filename(code, "python", "ITEM-001", 1)
        assert filename.startswith("tests/test_")
        assert "ITEM-001" in filename
    
    def test_infer_filename_fallback(self):
        """Test filename inference fallback."""
        coordinator = MockAgentCoordinator()
        
        code = """// Some code
console.log('hello');
"""
        
        filename = coordinator._infer_filename(code, "javascript", "ITEM-001", 1)
        assert filename == "src/ITEM-001_block1.js"
    
    @pytest.mark.asyncio
    async def test_extract_and_save_code_no_blocks(self, tmp_path):
        """Test code extraction with no code blocks."""
        coordinator = MockAgentCoordinator()
        
        response = "Just plain text response, no code blocks."
        
        files = await coordinator._extract_and_save_code(
            response_text=response,
            project_dir=tmp_path,
            item_id="ITEM-001",
        )
        
        # Should create fallback file
        assert len(files) == 1
        assert "_raw.txt" in files[0]
