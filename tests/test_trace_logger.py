# -*- coding: utf-8 -*-
"""
test_trace_logger.py - Tests for JSONL Trace Logger (BE-008)
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.trace_logger import TraceLogger


class TestTraceLogger:
    """Test suite for TraceLogger"""
    
    @pytest.fixture
    def temp_trace_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            yield Path(f.name)
            Path(f.name).unlink(missing_ok=True)
    
    def test_initialization(self, temp_trace_file):
        """Test logger initialization"""
        logger = TraceLogger(temp_trace_file, "test_exec_001")
        
        assert temp_trace_file.exists()
        
        # Check first line is execution_start
        with open(temp_trace_file, 'r') as f:
            first_line = f.readline()
            event = json.loads(first_line)
            
            assert event['event'] == 'execution_start'
            assert event['execution_id'] == 'test_exec_001'
            assert 'timestamp' in event
    
    def test_log_event(self, temp_trace_file):
        """Test basic event logging"""
        logger = TraceLogger(temp_trace_file, "test_exec_002")
        
        logger.log_event('custom_event', {
            'agent_id': 'agent_1',
            'message': 'Test message'
        })
        
        # Read events
        events = []
        with open(temp_trace_file, 'r') as f:
            for line in f:
                events.append(json.loads(line))
        
        assert len(events) >= 2  # start + custom
        custom_event = events[-1]
        assert custom_event['event'] == 'custom_event'
        assert custom_event['agent_id'] == 'agent_1'
    
    def test_agent_lifecycle(self, temp_trace_file):
        """Test agent lifecycle events"""
        logger = TraceLogger(temp_trace_file, "test_exec_003")
        
        logger.agent_start('agent_1', assigned_items=5)
        logger.item_start('agent_1', 'ITEM-001', 'Test item')
        logger.item_complete('agent_1', 'ITEM-001', duration_seconds=10.5, files_created=['test.py'])
        logger.agent_finish('agent_1', completed_items=1, failed_items=0, duration_seconds=15.0)
        
        # Parse all events
        with open(temp_trace_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        # Check event sequence
        event_types = [e['event'] for e in events]
        assert 'agent_start' in event_types
        assert 'item_start' in event_types
        assert 'item_complete' in event_types
        assert 'agent_finish' in event_types
    
    def test_item_failure(self, temp_trace_file):
        """Test failure logging"""
        logger = TraceLogger(temp_trace_file, "test_exec_004")
        
        logger.item_fail(
            'agent_2',
            'ITEM-002',
            error_message='Test error',
            attempt=1
        )
        
        with open(temp_trace_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        fail_event = next(e for e in events if e['event'] == 'item_fail')
        assert fail_event['error_message'] == 'Test error'
        assert fail_event['attempt'] == 1
    
    def test_metric_logging(self, temp_trace_file):
        """Test metric update logging"""
        logger = TraceLogger(temp_trace_file, "test_exec_005")
        
        logger.metric_update(
            agent_id='agent_1',
            metric_name='weighted_pearson',
            metric_value=0.42,
            target=0.35
        )
        
        with open(temp_trace_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        metric_event = next(e for e in events if e['event'] == 'metric_update')
        assert metric_event['metric_name'] == 'weighted_pearson'
        assert metric_event['metric_value'] == 0.42
        assert metric_event['target'] == 0.35
    
    def test_execution_end(self, temp_trace_file):
        """Test execution end event"""
        logger = TraceLogger(temp_trace_file, "test_exec_006")
        
        logger.execution_end(status='success', total_duration=120.5)
        
        with open(temp_trace_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        end_event = events[-1]
        assert end_event['event'] == 'execution_end'
        assert end_event['status'] == 'success'
        assert end_event['total_duration_seconds'] == 120.5
    
    def test_concurrent_writes(self, temp_trace_file):
        """Test thread safety with concurrent writes"""
        import threading
        
        logger = TraceLogger(temp_trace_file, "test_exec_007")
        
        def write_events(agent_id, count):
            for i in range(count):
                logger.log_event('test_event', {
                    'agent_id': agent_id,
                    'counter': i
                })
        
        # Spawn multiple threads
        threads = [
            threading.Thread(target=write_events, args=(f'agent_{i}', 10))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all events written
        with open(temp_trace_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        test_events = [e for e in events if e['event'] == 'test_event']
        assert len(test_events) == 50  # 5 agents * 10 events
