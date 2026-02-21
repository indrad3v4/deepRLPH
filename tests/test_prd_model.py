# -*- coding: utf-8 -*-
"""
test_prd_model.py - Tests for PRD Model

Tests for ITEM-001: PRD Item Status Tracking Model
"""

import pytest
import json
import threading
import time
from pathlib import Path
from src.prd_model import PRDItem, PRDBacklog, PRDItemStatus


class TestPRDItem:
    """Tests for PRDItem dataclass."""
    
    def test_create_item(self):
        """Test creating a PRD item."""
        item = PRDItem(
            item_id="ITEM-001",
            title="Test Item",
            priority=1,
            acceptance_criteria=["Criteria 1", "Criteria 2"],
            verification_command="pytest tests/",
            files_touched=["src/test.py"],
        )
        
        assert item.item_id == "ITEM-001"
        assert item.status == PRDItemStatus.PENDING
        assert item.attempt_count == 0
        assert len(item.acceptance_criteria) == 2
    
    def test_status_transition_pending_to_in_progress(self):
        """Test PENDING → IN_PROGRESS transition."""
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        item.mark_in_progress("agent_1")
        
        assert item.status == PRDItemStatus.IN_PROGRESS
        assert item.agent_id == "agent_1"
        assert item.start_time is not None
        assert item.attempt_count == 1
    
    def test_status_transition_in_progress_to_testing(self):
        """Test IN_PROGRESS → TESTING transition."""
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        item.mark_in_progress("agent_1")
        item.mark_testing()
        
        assert item.status == PRDItemStatus.TESTING
    
    def test_status_transition_testing_to_pass(self):
        """Test TESTING → PASS transition."""
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        item.mark_in_progress("agent_1")
        item.mark_testing()
        item.mark_pass()
        
        assert item.status == PRDItemStatus.PASS
        assert item.end_time is not None
    
    def test_status_transition_in_progress_to_fail(self):
        """Test IN_PROGRESS → FAIL transition."""
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        item.mark_in_progress("agent_1")
        item.mark_fail("Test error")
        
        assert item.status == PRDItemStatus.FAIL
        assert item.end_time is not None
        assert len(item.error_log) == 1
        assert "Test error" in item.error_log[0]
    
    def test_invalid_status_transition(self):
        """Test that invalid status transitions raise ValueError."""
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        
        with pytest.raises(ValueError, match="Cannot transition"):
            item.mark_testing()  # Can't go from PENDING to TESTING
    
    def test_reset_for_retry(self):
        """Test resetting failed item for retry."""
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        item.mark_in_progress("agent_1")
        item.mark_fail("Test error")
        item.reset_for_retry()
        
        assert item.status == PRDItemStatus.PENDING
        assert item.start_time is None
        assert item.end_time is None
        assert item.attempt_count == 1  # Preserved
        assert len(item.error_log) == 1  # Preserved
    
    def test_get_duration(self):
        """Test getting execution duration."""
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        item.mark_in_progress("agent_1")
        time.sleep(0.1)
        item.mark_pass()
        
        duration = item.get_duration()
        assert duration >= 0.1
        assert duration < 1.0
    
    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        item = PRDItem(
            item_id="ITEM-001",
            title="Test Item",
            priority=1,
            acceptance_criteria=["Criteria 1"],
            verification_command="pytest",
            files_touched=["test.py"],
        )
        item.mark_in_progress("agent_1")
        
        data = item.to_dict()
        restored = PRDItem.from_dict(data)
        
        assert restored.item_id == item.item_id
        assert restored.status == item.status
        assert restored.agent_id == item.agent_id
        assert restored.start_time == item.start_time


class TestPRDBacklog:
    """Tests for PRDBacklog manager."""
    
    def test_create_backlog(self):
        """Test creating backlog."""
        backlog = PRDBacklog(project_id="test_project")
        
        assert backlog.project_id == "test_project"
        assert len(backlog.items) == 0
        assert backlog.created_at is not None
    
    def test_add_item(self):
        """Test adding items to backlog."""
        backlog = PRDBacklog(project_id="test_project")
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        
        backlog.add_item(item)
        assert len(backlog.items) == 1
    
    def test_get_next_pending(self):
        """Test getting next pending item by priority."""
        backlog = PRDBacklog(project_id="test_project")
        item1 = PRDItem(item_id="ITEM-001", title="Low Priority", priority=3)
        item2 = PRDItem(item_id="ITEM-002", title="High Priority", priority=1)
        item3 = PRDItem(item_id="ITEM-003", title="Med Priority", priority=2)
        
        backlog.add_item(item1)
        backlog.add_item(item2)
        backlog.add_item(item3)
        
        next_item = backlog.get_next_pending()
        assert next_item.item_id == "ITEM-002"  # Priority 1 first
    
    def test_get_next_pending_empty(self):
        """Test getting next pending when all items done."""
        backlog = PRDBacklog(project_id="test_project")
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        item.mark_in_progress("agent_1")
        backlog.add_item(item)
        
        next_item = backlog.get_next_pending()
        assert next_item is None
    
    def test_mark_in_progress(self):
        """Test marking item in progress."""
        backlog = PRDBacklog(project_id="test_project")
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        backlog.add_item(item)
        
        success = backlog.mark_in_progress("ITEM-001", "agent_1")
        assert success
        assert item.status == PRDItemStatus.IN_PROGRESS
    
    def test_mark_complete_success(self):
        """Test marking item as passed."""
        backlog = PRDBacklog(project_id="test_project")
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        backlog.add_item(item)
        
        backlog.mark_in_progress("ITEM-001", "agent_1")
        success = backlog.mark_complete("ITEM-001", success=True)
        
        assert success
        assert item.status == PRDItemStatus.PASS
    
    def test_mark_complete_failure(self):
        """Test marking item as failed."""
        backlog = PRDBacklog(project_id="test_project")
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        backlog.add_item(item)
        
        backlog.mark_in_progress("ITEM-001", "agent_1")
        success = backlog.mark_complete("ITEM-001", success=False, error="Test failed")
        
        assert success
        assert item.status == PRDItemStatus.FAIL
        assert len(item.error_log) > 0
    
    def test_get_statistics(self):
        """Test getting backlog statistics."""
        backlog = PRDBacklog(project_id="test_project")
        
        item1 = PRDItem(item_id="ITEM-001", title="Done", priority=1)
        item1.mark_in_progress("agent_1")
        item1.mark_pass()
        
        item2 = PRDItem(item_id="ITEM-002", title="Failed", priority=1)
        item2.mark_in_progress("agent_1")
        item2.mark_fail("Error")
        
        item3 = PRDItem(item_id="ITEM-003", title="Pending", priority=1)
        
        backlog.add_item(item1)
        backlog.add_item(item2)
        backlog.add_item(item3)
        
        stats = backlog.get_statistics()
        
        assert stats["total"] == 3
        assert stats["pass"] == 1
        assert stats["fail"] == 1
        assert stats["pending"] == 1
        assert stats["progress_pct"] == pytest.approx(33.33, rel=0.1)
    
    def test_has_pending_items(self):
        """Test checking for pending items."""
        backlog = PRDBacklog(project_id="test_project")
        
        assert not backlog.has_pending_items()
        
        item = PRDItem(item_id="ITEM-001", title="Test", priority=1)
        backlog.add_item(item)
        
        assert backlog.has_pending_items()
        
        item.mark_in_progress("agent_1")
        assert not backlog.has_pending_items()
    
    def test_get_failed_items(self):
        """Test getting failed items."""
        backlog = PRDBacklog(project_id="test_project")
        
        item1 = PRDItem(item_id="ITEM-001", title="Pass", priority=1)
        item1.mark_in_progress("agent_1")
        item1.mark_pass()
        
        item2 = PRDItem(item_id="ITEM-002", title="Fail", priority=1)
        item2.mark_in_progress("agent_1")
        item2.mark_fail("Error")
        
        backlog.add_item(item1)
        backlog.add_item(item2)
        
        failed = backlog.get_failed_items()
        assert len(failed) == 1
        assert failed[0].item_id == "ITEM-002"
    
    def test_thread_safety(self):
        """Test concurrent access is thread-safe."""
        backlog = PRDBacklog(project_id="test_project")
        
        for i in range(100):
            item = PRDItem(item_id=f"ITEM-{i:03d}", title=f"Item {i}", priority=1)
            backlog.add_item(item)
        
        def worker():
            while True:
                item = backlog.get_next_pending()
                if not item:
                    break
                backlog.mark_in_progress(item.item_id, threading.current_thread().name)
                time.sleep(0.001)
                backlog.mark_complete(item.item_id, success=True)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        stats = backlog.get_statistics()
        assert stats["pass"] == 100
        assert stats["pending"] == 0
    
    def test_save_load_file(self, tmp_path):
        """Test saving and loading backlog state."""
        backlog = PRDBacklog(project_id="test_project")
        
        item1 = PRDItem(item_id="ITEM-001", title="Test 1", priority=1)
        item1.mark_in_progress("agent_1")
        item1.mark_pass()
        
        item2 = PRDItem(item_id="ITEM-002", title="Test 2", priority=2)
        
        backlog.add_item(item1)
        backlog.add_item(item2)
        
        # Save
        path = tmp_path / "prd_state.json"
        backlog.save_to_file(path)
        
        assert path.exists()
        
        # Load
        loaded = PRDBacklog.load_from_file(path)
        
        assert loaded.project_id == backlog.project_id
        assert len(loaded.items) == 2
        assert loaded.items[0].status == PRDItemStatus.PASS
        assert loaded.items[1].status == PRDItemStatus.PENDING
