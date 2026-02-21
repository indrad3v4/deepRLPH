# -*- coding: utf-8 -*-
"""
test_execution_engine_extensions.py - Tests for ITEM-003

Tests for Verification Command Integration
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import subprocess

from src.execution_engine_extensions import PRDVerificationMixin


class MockExecutionEngine(PRDVerificationMixin):
    """Mock execution engine for testing mixin."""
    pass


class TestPRDVerificationMixin:
    """Tests for verification command execution."""
    
    @pytest.mark.asyncio
    async def test_run_verification_success(self, tmp_path):
        """Test successful verification command."""
        engine = MockExecutionEngine()
        
        # Mock subprocess.run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "All tests passed\n"
        mock_result.stderr = ""
        
        with patch('subprocess.run', return_value=mock_result):
            result = await engine.run_verification_for_prd(
                command="pytest tests/",
                project_dir=tmp_path,
            )
        
        assert result["success"] is True
        assert "All tests passed" in result["output"]
        assert result["exit_code"] == 0
    
    @pytest.mark.asyncio
    async def test_run_verification_failure(self, tmp_path):
        """Test failed verification command."""
        engine = MockExecutionEngine()
        
        # Mock subprocess.run with failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "Tests failed\n"
        mock_result.stderr = "Error: assertion failed\n"
        
        with patch('subprocess.run', return_value=mock_result):
            result = await engine.run_verification_for_prd(
                command="pytest tests/",
                project_dir=tmp_path,
            )
        
        assert result["success"] is False
        assert "Tests failed" in result["output"]
        assert "assertion failed" in result["output"]
        assert result["exit_code"] == 1
    
    @pytest.mark.asyncio
    async def test_run_verification_timeout(self, tmp_path):
        """Test verification command timeout."""
        engine = MockExecutionEngine()
        
        # Mock subprocess.run to raise TimeoutExpired
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("cmd", 10)):
            result = await engine.run_verification_for_prd(
                command="sleep 1000",
                project_dir=tmp_path,
                timeout=10,
            )
        
        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert result["exit_code"] == -1
    
    @pytest.mark.asyncio
    async def test_run_verification_exception(self, tmp_path):
        """Test verification command with exception."""
        engine = MockExecutionEngine()
        
        # Mock subprocess.run to raise exception
        with patch('subprocess.run', side_effect=Exception("Command not found")):
            result = await engine.run_verification_for_prd(
                command="nonexistent_command",
                project_dir=tmp_path,
            )
        
        assert result["success"] is False
        assert "error" in result["error"].lower()
        assert result["exit_code"] == -1
    
    @pytest.mark.asyncio
    async def test_run_verification_with_stderr_only(self, tmp_path):
        """Test verification with stderr output only."""
        engine = MockExecutionEngine()
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = "Warning: deprecated function\n"
        
        with patch('subprocess.run', return_value=mock_result):
            result = await engine.run_verification_for_prd(
                command="python script.py",
                project_dir=tmp_path,
            )
        
        assert result["success"] is True
        assert "Warning" in result["output"]
    
    @pytest.mark.asyncio
    async def test_run_verification_custom_timeout(self, tmp_path):
        """Test verification with custom timeout."""
        engine = MockExecutionEngine()
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Done\n"
        mock_result.stderr = ""
        
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            await engine.run_verification_for_prd(
                command="pytest",
                project_dir=tmp_path,
                timeout=300,
            )
            
            # Check timeout was passed to subprocess.run
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['timeout'] == 300
