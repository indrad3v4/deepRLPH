# -*- coding: utf-8 -*-
"""
test_orchestrator_extensions.py - Tests for ITEM-002

Tests for Orchestrator PRD Execution Loop Integration
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.prd_model import PRDItem, PRDBacklog, PRDItemStatus
from src.orchestrator_extensions import PRDExecutionMixin


class MockOrchestrator(PRDExecutionMixin):
    """Mock orchestrator for testing mixin."""
    
    def __init__(self):
        self.execution_engine = None
        self.agent_coordinator = None
        self.deepseek_client = None


class TestPRDExecutionMixin:
    """Tests for PRD execution loop."""
    
    @pytest.mark.asyncio
    async def test_execute_prd_backlog_success(self, tmp_path):
        """Test successful backlog execution."""
        # Create backlog
        backlog = PRDBacklog(project_id="test")
        item1 = PRDItem(
            item_id="ITEM-001",
            title="Test Item 1",
            priority=1,
            verification_command="echo 'pass'",
        )
        backlog.add_item(item1)
        
        # Mock orchestrator
        orchestrator = MockOrchestrator()
        
        # Mock execution engine
        mock_engine = Mock()
        mock_engine.run_verification_for_prd = AsyncMock(return_value={
            "success": True,
            "output": "Tests passed",
        })
        orchestrator.execution_engine = mock_engine
        
        # Mock agent coordinator
        mock_coordinator = Mock()
        mock_coordinator.assign_prd_item = AsyncMock(return_value={
            "status": "success",
            "files_created": ["src/test.py"],
        })
        orchestrator.agent_coordinator = mock_coordinator
        
        # Execute
        result = await orchestrator.execute_prd_backlog(
            backlog=backlog,
            project_dir=tmp_path,
        )
        
        assert result["status"] == "success"
        assert result["completed_count"] == 1
        assert result["failed_count"] == 0
        assert item1.status == PRDItemStatus.PASS
    
    @pytest.mark.asyncio
    async def test_execute_prd_backlog_failure(self, tmp_path):
        """Test backlog execution with failures."""
        # Create backlog
        backlog = PRDBacklog(project_id="test")
        item1 = PRDItem(
            item_id="ITEM-001",
            title="Test Item 1",
            priority=1,
            verification_command="exit 1",
        )
        backlog.add_item(item1)
        
        # Mock orchestrator
        orchestrator = MockOrchestrator()
        
        # Mock execution engine with failure
        mock_engine = Mock()
        mock_engine.run_verification_for_prd = AsyncMock(return_value={
            "success": False,
            "error": "Tests failed",
            "output": "Error output",
        })
        orchestrator.execution_engine = mock_engine
        
        # Mock agent coordinator
        mock_coordinator = Mock()
        mock_coordinator.assign_prd_item = AsyncMock(return_value={
            "status": "success",
            "files_created": ["src/test.py"],
        })
        orchestrator.agent_coordinator = mock_coordinator
        
        # Execute
        result = await orchestrator.execute_prd_backlog(
            backlog=backlog,
            project_dir=tmp_path,
        )
        
        assert result["status"] == "failed"
        assert result["completed_count"] == 0
        assert result["failed_count"] == 1
        assert item1.status == PRDItemStatus.FAIL
    
    @pytest.mark.asyncio
    async def test_execute_prd_backlog_no_verification(self, tmp_path):
        """Test item without verification command."""
        # Create backlog
        backlog = PRDBacklog(project_id="test")
        item1 = PRDItem(
            item_id="ITEM-001",
            title="Test Item 1",
            priority=1,
            verification_command="",  # No verification
        )
        backlog.add_item(item1)
        
        # Mock orchestrator
        orchestrator = MockOrchestrator()
        
        # Mock agent coordinator
        mock_coordinator = Mock()
        mock_coordinator.assign_prd_item = AsyncMock(return_value={
            "status": "success",
            "files_created": ["src/test.py"],
        })
        orchestrator.agent_coordinator = mock_coordinator
        
        # Execute (no engine needed since no verification)
        result = await orchestrator.execute_prd_backlog(
            backlog=backlog,
            project_dir=tmp_path,
        )
        
        assert result["status"] == "success"
        assert result["completed_count"] == 1
        assert item1.status == PRDItemStatus.PASS
    
    @pytest.mark.asyncio
    async def test_execute_prd_backlog_multiple_items(self, tmp_path):
        """Test backlog with multiple items."""
        # Create backlog with 3 items
        backlog = PRDBacklog(project_id="test")
        for i in range(3):
            item = PRDItem(
                item_id=f"ITEM-{i:03d}",
                title=f"Test Item {i}",
                priority=i + 1,
                verification_command="echo 'pass'",
            )
            backlog.add_item(item)
        
        # Mock orchestrator
        orchestrator = MockOrchestrator()
        
        # Mock execution engine
        mock_engine = Mock()
        mock_engine.run_verification_for_prd = AsyncMock(return_value={
            "success": True,
            "output": "Tests passed",
        })
        orchestrator.execution_engine = mock_engine
        
        # Mock agent coordinator
        mock_coordinator = Mock()
        mock_coordinator.assign_prd_item = AsyncMock(return_value={
            "status": "success",
            "files_created": ["src/test.py"],
        })
        orchestrator.agent_coordinator = mock_coordinator
        
        # Execute
        result = await orchestrator.execute_prd_backlog(
            backlog=backlog,
            project_dir=tmp_path,
        )
        
        assert result["status"] == "success"
        assert result["completed_count"] == 3
        assert result["failed_count"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_prd_backlog_partial_success(self, tmp_path):
        """Test backlog with some failures."""
        # Create backlog
        backlog = PRDBacklog(project_id="test")
        for i in range(3):
            item = PRDItem(
                item_id=f"ITEM-{i:03d}",
                title=f"Test Item {i}",
                priority=i + 1,
                verification_command="test",
            )
            backlog.add_item(item)
        
        # Mock orchestrator
        orchestrator = MockOrchestrator()
        
        # Mock execution engine with alternating success/failure
        call_count = 0
        async def mock_verify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return {"success": False, "error": "Failed", "output": ""}
            return {"success": True, "output": "Passed"}
        
        mock_engine = Mock()
        mock_engine.run_verification_for_prd = mock_verify
        orchestrator.execution_engine = mock_engine
        
        # Mock agent coordinator
        mock_coordinator = Mock()
        mock_coordinator.assign_prd_item = AsyncMock(return_value={
            "status": "success",
            "files_created": ["src/test.py"],
        })
        orchestrator.agent_coordinator = mock_coordinator
        
        # Execute
        result = await orchestrator.execute_prd_backlog(
            backlog=backlog,
            project_dir=tmp_path,
        )
        
        assert result["status"] == "partial"
        assert result["completed_count"] > 0
        assert result["failed_count"] > 0
    
    @pytest.mark.asyncio
    async def test_execute_prd_item_creates_engines(self, tmp_path):
        """Test that _execute_prd_item creates engines if needed."""
        backlog = PRDBacklog(project_id="test")
        item = PRDItem(
            item_id="ITEM-001",
            title="Test",
            priority=1,
            verification_command="",
        )
        
        orchestrator = MockOrchestrator()
        # Don't set execution_engine or agent_coordinator
        
        with patch('src.orchestrator_extensions.ExecutionEngine'), \
             patch('src.orchestrator_extensions.AgentCoordinator') as mock_coordinator_class:
            
            mock_coordinator = Mock()
            mock_coordinator.assign_prd_item = AsyncMock(return_value={
                "status": "success",
                "files_created": [],
            })
            mock_coordinator_class.return_value = mock_coordinator
            
            result = await orchestrator._execute_prd_item(
                backlog=backlog,
                item=item,
                project_dir=tmp_path,
            )
            
            assert orchestrator.agent_coordinator is not None
