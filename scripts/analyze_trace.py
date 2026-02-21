#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_trace.py - BE-008.3: JSONL Trace Analyzer

Parses execution trace JSONL files and generates summary reports:
- Per-agent statistics
- Timeline of events
- Failed items with errors
- Performance metrics
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("TraceAnalyzer")


class TraceAnalyzer:
    """Analyze JSONL execution trace files"""
    
    def __init__(self, trace_file: Path):
        self.trace_file = Path(trace_file)
        self.events: List[Dict[str, Any]] = []
        self.agent_stats = defaultdict(lambda: {
            "started_items": 0,
            "completed_items": 0,
            "failed_items": 0,
            "total_duration": 0.0,
            "failures": [],
        })
    
    def load_trace(self) -> None:
        """Load JSONL file"""
        if not self.trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_file}")
        
        with open(self.trace_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        self.events.append(event)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipped malformed line: {e}")
        
        logger.info(f"Loaded {len(self.events)} events")
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze trace and generate report"""
        if not self.events:
            self.load_trace()
        
        # Process events
        for event in self.events:
            event_type = event.get('event')
            agent_id = event.get('agent_id')
            
            if event_type == 'item_start' and agent_id:
                self.agent_stats[agent_id]['started_items'] += 1
            
            elif event_type == 'item_complete' and agent_id:
                self.agent_stats[agent_id]['completed_items'] += 1
                self.agent_stats[agent_id]['total_duration'] += event.get('duration_seconds', 0)
            
            elif event_type == 'item_fail' and agent_id:
                self.agent_stats[agent_id]['failed_items'] += 1
                self.agent_stats[agent_id]['failures'].append({
                    "item_id": event.get('item_id'),
                    "error": event.get('error_message'),
                    "attempt": event.get('attempt'),
                })
        
        # Generate summary
        summary = {
            "total_events": len(self.events),
            "agents": dict(self.agent_stats),
            "execution_id": self.events[0].get('execution_id') if self.events else None,
        }
        
        # Extract timing info
        start_event = next((e for e in self.events if e['event'] == 'execution_start'), None)
        end_event = next((e for e in self.events if e['event'] == 'execution_end'), None)
        
        if start_event and end_event:
            start_time = datetime.fromisoformat(start_event['timestamp'])
            end_time = datetime.fromisoformat(end_event['timestamp'])
            summary['total_duration_seconds'] = (end_time - start_time).total_seconds()
        
        return summary
    
    def print_report(self, summary: Dict[str, Any]) -> None:
        """Print human-readable report"""
        print("\n" + "="*60)
        print("📊 EXECUTION TRACE ANALYSIS")
        print("="*60)
        print(f"\nExecution ID: {summary.get('execution_id', 'N/A')}")
        print(f"Total Events: {summary['total_events']}")
        
        if 'total_duration_seconds' in summary:
            print(f"Total Duration: {summary['total_duration_seconds']:.1f}s")
        
        print("\n" + "-"*60)
        print("AGENT STATISTICS")
        print("-"*60)
        
        agents = summary.get('agents', {})
        if not agents:
            print("No agent data found")
            return
        
        # Table header
        print(f"{'Agent':<15} {'Started':<10} {'Completed':<12} {'Failed':<10} {'Avg Time (s)':<15}")
        print("-" * 60)
        
        for agent_id, stats in sorted(agents.items()):
            completed = stats['completed_items']
            failed = stats['failed_items']
            avg_time = stats['total_duration'] / completed if completed > 0 else 0
            
            print(f"{agent_id:<15} {stats['started_items']:<10} {completed:<12} {failed:<10} {avg_time:<15.2f}")
        
        # Failed items detail
        print("\n" + "-"*60)
        print("FAILED ITEMS")
        print("-"*60)
        
        has_failures = False
        for agent_id, stats in agents.items():
            if stats['failures']:
                has_failures = True
                print(f"\n{agent_id}:")
                for failure in stats['failures']:
                    print(f"  - {failure['item_id']}: {failure['error'][:80]}...")
        
        if not has_failures:
            print("No failures ✓")
        
        print("\n" + "="*60)
    
    def export_json(self, output_file: Path, summary: Dict[str, Any]) -> None:
        """Export summary as JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary exported: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trace.py <trace_file.jsonl> [--json output.json]")
        print("Example: python scripts/analyze_trace.py logs/execution_xxx.jsonl")
        print("Example: python scripts/analyze_trace.py logs/execution_xxx.jsonl --json summary.json")
        sys.exit(1)
    
    trace_file = Path(sys.argv[1])
    
    if not trace_file.exists():
        logger.error(f"Trace file not found: {trace_file}")
        sys.exit(1)
    
    analyzer = TraceAnalyzer(trace_file)
    summary = analyzer.analyze()
    analyzer.print_report(summary)
    
    # Check for --json flag
    if "--json" in sys.argv:
        json_index = sys.argv.index("--json")
        if json_index + 1 < len(sys.argv):
            output_file = Path(sys.argv[json_index + 1])
            analyzer.export_json(output_file, summary)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
