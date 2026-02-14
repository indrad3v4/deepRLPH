# -*- coding: utf-8 -*-
"""
context_ui_formatter.py - FE-002: Context Usage Panel Data Formatter

Prepares ingested project context for UI display.
Provides structured data for a frontend "Context Usage" panel.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import logging

logger = logging.getLogger("ContextUIFormatter")


class ContextUIFormatter:
    """Format project context for UI consumption."""

    @staticmethod
    def format_for_ui(project_dir: Path) -> Dict[str, Any]:
        """Generate UI-ready context summary.

        Returns dict with:
        - summary: High-level stats
        - categories: Detailed breakdown by type
        - files: Individual file listings
        """
        config_file = project_dir / "config.json"
        if not config_file.exists():
            return {
                "status": "no_config",
                "message": "No project config found",
            }

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.error("Error loading config: %s", e)
            return {"status": "error", "message": str(e)}

        metadata = config.get("metadata", {})
        context = metadata.get("ingested_context", {})

        if not context:
            return {
                "status": "no_context",
                "message": "No context has been ingested yet",
                "summary": {
                    "total_files": 0,
                    "total_size_mb": 0,
                    "categories": {},
                },
            }

        # Build summary stats
        docs = context.get("docs", [])
        datasets = context.get("datasets", [])
        code = context.get("code", [])
        models = context.get("models", [])

        total_files = len(docs) + len(datasets) + len(code) + len(models)
        total_size = sum(
            item.get("size_mb", 0)
            for category in [docs, datasets, code, models]
            for item in category
        )

        # Format categories
        categories = {
            "documentation": {
                "count": len(docs),
                "size_mb": round(
                    sum(d.get("size_mb", 0) for d in docs), 2
                ),
                "types": list(set(d.get("type", "unknown") for d in docs)),
                "files": ContextUIFormatter._format_file_list(docs),
            },
            "datasets": {
                "count": len(datasets),
                "size_mb": round(
                    sum(d.get("size_mb", 0) for d in datasets), 2
                ),
                "formats": list(set(d.get("format", "unknown") for d in datasets)),
                "total_rows": sum(d.get("rows", 0) for d in datasets),
                "files": ContextUIFormatter._format_file_list(datasets),
            },
            "code": {
                "count": len(code),
                "size_mb": round(
                    sum(c.get("size_mb", 0) for c in code), 2
                ),
                "languages": list(set(c.get("language", "unknown") for c in code)),
                "files": ContextUIFormatter._format_file_list(code),
            },
            "models": {
                "count": len(models),
                "size_mb": round(
                    sum(m.get("size_mb", 0) for m in models), 2
                ),
                "formats": list(set(m.get("format", "unknown") for m in models)),
                "files": ContextUIFormatter._format_file_list(models),
            },
        }

        # Add dataset schemas if available
        if "dataset_schemas" in metadata:
            categories["datasets"]["schemas"] = ContextUIFormatter._format_schemas(
                metadata["dataset_schemas"]
            )

        # Add model inspection if available
        if "onnx_models" in metadata:
            categories["models"]["inspections"] = ContextUIFormatter._format_model_inspections(
                metadata["onnx_models"]
            )

        return {
            "status": "success",
            "summary": {
                "total_files": total_files,
                "total_size_mb": round(total_size, 2),
                "categories": {
                    "documentation": len(docs),
                    "datasets": len(datasets),
                    "code": len(code),
                    "models": len(models),
                },
            },
            "categories": categories,
            "metadata": {
                "project_id": config.get("name", "Unknown"),
                "project_type": metadata.get("project_type", "unknown"),
                "ingestion_complete": True,
            },
        }

    @staticmethod
    def _format_file_list(items: List[Dict]) -> List[Dict[str, Any]]:
        """Format file list for UI display."""
        formatted = []
        for item in items[:10]:  # Limit to first 10 for UI
            formatted.append({
                "name": item.get("path", item.get("file", "Unknown")),
                "size_mb": round(item.get("size_mb", 0), 3),
                "type": item.get("type", item.get("format", "unknown")),
                "metadata": {
                    k: v for k, v in item.items()
                    if k not in ["path", "file", "size_mb", "content"]
                },
            })
        
        if len(items) > 10:
            formatted.append({
                "name": f"... and {len(items) - 10} more",
                "size_mb": 0,
                "type": "summary",
            })
        
        return formatted

    @staticmethod
    def _format_schemas(schemas: List[Dict]) -> List[Dict[str, Any]]:
        """Format dataset schemas for UI."""
        return [
            {
                "dataset": schema.get("file", "Unknown"),
                "features": len(schema.get("columns", [])),
                "rows": schema.get("shape", [0, 0])[0],
                "columns": schema.get("columns", []),
                "dtypes": schema.get("dtypes", {}),
            }
            for schema in schemas
        ]

    @staticmethod
    def _format_model_inspections(models: List[Dict]) -> List[Dict[str, Any]]:
        """Format ONNX model inspections for UI."""
        return [
            {
                "model": model.get("file", "Unknown"),
                "size_mb": model.get("size_mb", 0),
                "inputs": model.get("inputs", []),
                "outputs": model.get("outputs", []),
                "parameters": model.get("parameters", {}).get("total", 0),
                "operators": len(model.get("operators", {})),
            }
            for model in models
            if "error" not in model
        ]


def get_context_panel_data(project_dir: Path) -> Dict[str, Any]:
    """Convenience function to get context panel data.

    Args:
        project_dir: Project root directory

    Returns:
        UI-ready context data dict
    """
    formatter = ContextUIFormatter()
    return formatter.format_for_ui(project_dir)


# Example Flask/FastAPI endpoint (for reference)
"""
Example integration:

@app.get("/api/projects/{project_id}/context")
def get_project_context(project_id: str):
    project_dir = get_project_dir(project_id)
    return get_context_panel_data(project_dir)

Frontend usage:
- Fetch GET /api/projects/{project_id}/context
- Display in a collapsible panel with tabs:
  - Summary (pie chart of file types + size)
  - Documentation (list with icons)
  - Datasets (table with schemas)
  - Code (tree view)
  - Models (cards with input/output info)
"""
