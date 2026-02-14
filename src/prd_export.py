# -*- coding: utf-8 -*-
"""
prd_export.py - Markdown PRD Export

ITEM-013: Export PRD backlog to markdown format for human review.
"""

from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

from src.prd_model import PRDBacklog, PRDItem, PRDItemStatus

logger = logging.getLogger(__name__)


class PRDMarkdownExporter:
    """Export PRD backlog to markdown format."""
    
    def __init__(self, backlog: PRDBacklog):
        self.backlog = backlog
    
    def export_to_markdown(self, output_path: Path) -> None:
        """Export backlog to markdown file."""
        md_content = self._generate_markdown()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Exported PRD backlog to {output_path}")
    
    def _generate_markdown(self) -> str:
        """Generate markdown content from backlog."""
        stats = self.backlog.get_statistics()
        
        md_lines = [
            f"# PRD Backlog: {self.backlog.project_id}",
            "",
            f"**Created:** {self.backlog.created_at}",
            f"**Generated:** {datetime.now().isoformat()}",
            "",
            "## Statistics",
            "",
            f"- Total Items: {stats['total']}",
            f"- Pending: {stats['pending']}",
            f"- In Progress: {stats['in_progress']}",
            f"- Testing: {stats['testing']}",
            f"- Passed: {stats['pass']} ✅",
            f"- Failed: {stats['fail']} ❌",
            f"- Progress: {stats['progress_pct']:.1f}%",
            "",
            "## Items",
            "",
        ]
        
        # Group by status
        for status in PRDItemStatus:
            items = [i for i in self.backlog.items if i.status == status]
            if items:
                md_lines.append(f"### {status.value}")
                md_lines.append("")
                for item in sorted(items, key=lambda x: x.priority):
                    md_lines.extend(self._format_item(item))
                    md_lines.append("")
        
        return "\n".join(md_lines)
    
    def _format_item(self, item: PRDItem) -> list[str]:
        """Format single PRD item as markdown."""
        lines = [
            f"#### {item.item_id}: {item.title}",
            "",
            f"**Priority:** {item.priority}",
            f"**Status:** {item.status.value}",
        ]
        
        if item.agent_id:
            lines.append(f"**Agent:** {item.agent_id}")
        
        if item.start_time:
            lines.append(f"**Started:** {item.start_time}")
        
        if item.end_time:
            lines.append(f"**Completed:** {item.end_time}")
            lines.append(f"**Duration:** {item.get_duration():.1f}s")
        
        if item.attempt_count > 0:
            lines.append(f"**Attempts:** {item.attempt_count}")
        
        lines.append("")
        lines.append("**Acceptance Criteria:**")
        for criterion in item.acceptance_criteria:
            lines.append(f"- {criterion}")
        
        if item.verification_command:
            lines.append("")
            lines.append(f"**Verification:** `{item.verification_command}`")
        
        if item.files_touched:
            lines.append("")
            lines.append("**Files:**")
            for file_path in item.files_touched:
                lines.append(f"- `{file_path}`")
        
        if item.error_log:
            lines.append("")
            lines.append("**Errors:**")
            for error in item.error_log:
                lines.append(f"- {error}")
        
        return lines


def export_prd_to_markdown(backlog: PRDBacklog, output_path: Path) -> None:
    """Convenience function to export PRD backlog to markdown."""
    exporter = PRDMarkdownExporter(backlog)
    exporter.export_to_markdown(output_path)
