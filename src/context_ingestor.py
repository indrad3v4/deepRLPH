# -*- coding: utf-8 -*-
"""context_ingestor.py - Grounded Project context ingestion (TRANSACTIONAL ASYNC EDITION).

Refactored to support pure async pipelines and deeper grounding for BE-002/BE-005.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import asyncio

# Internal components for binary grounding
from dataset_inspector import summarize_dataset
from onnx_inspector import summarize_onnx

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Enhanced container for ingested project artefacts with deep grounding metadata."""

    root: Path
    docs: List[Path] = field(default_factory=list)
    datasets: List[Path] = field(default_factory=list)
    code: List[Path] = field(default_factory=list)
    models: List[Path] = field(default_factory=list)

    # Binary summaries keyed by relative path (BE-002 / BE-005 support)
    binary_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "docs": [p.relative_to(self.root).as_posix() for p in self.docs],
            "datasets": [p.relative_to(self.root).as_posix() for p in self.datasets],
            "code": [p.relative_to(self.root).as_posix() for p in self.code],
            "models": [p.relative_to(self.root).as_posix() for p in self.models],
            "binary_summaries": self.binary_summaries,
        }

    def to_text_summary(self) -> str:
        """Deeply grounded text summary safe to embed into AI prompts."""
        lines: List[str] = []
        rel = lambda p: p.relative_to(self.root).as_posix()

        if self.docs:
            lines.append("# Documentation")
            for p in sorted(self.docs):
                lines.append(f"- {rel(p)}")

        if self.datasets:
            lines.append("\n# Data Schemas (BE-002 Grounding)")
            for p in sorted(self.datasets):
                key = rel(p)
                ds = self.binary_summaries.get(key) or {}
                # Extract deeper metadata for ML model grounding
                schema_info = f"rows={ds.get('row_count')}, cols={ds.get('column_count')}"
                if ds.get('columns'):
                    schema_info += f", schema={list(ds['columns'].keys())[:5]}..."
                lines.append(f"- {key} ({schema_info})")

        if self.models:
            lines.append("\n# Model Architectures")
            for p in sorted(self.models):
                key = rel(p)
                ms = self.binary_summaries.get(key) or {}
                meta = ms.get("model_type_hint", "unknown")
                lines.append(f"- {key} (type={meta}, inputs={ms.get('input_shapes', 'n/a')})")

        if self.code:
            lines.append("\n# Codebase (Relative paths)")
            for p in sorted(self.code)[:20]:
                lines.append(f"- {rel(p)}")

        return "\n".join(lines)


async def ingest_project_context(project_dir: Path) -> ProjectContext:
    """Pure async project directory ingestion.

    Walks the project root and collects summaries for:
    - Documentation: *.md, *.txt
    - Datasets: data/**/*.csv|parquet (via dataset_inspector)
    - Models: models/**/*.onnx (via onnx_inspector)
    - Code: src/**/*.py
    """
    root = Path(project_dir).resolve()
    ctx = ProjectContext(root=root)
    loop = asyncio.get_running_loop()

    # Define blocking scan to run in executor
    def _scan_disk():
        results = {"docs": [], "datasets": [], "models": [], "code": []}
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            rel = path.relative_to(root)
            suffix = path.suffix.lower()

            if suffix in {".md", ".txt"}:
                results["docs"].append(path)
            elif suffix in {".csv", ".parquet"} and "data" in rel.parts:
                results["datasets"].append(path)
            elif suffix == ".onnx" and "models" in rel.parts:
                results["models"].append(path)
            elif suffix == ".py" and "src" in rel.parts:
                results["code"].append(path)
        return results

    # Execute file walk in a thread pool to prevent event loop block
    scan_data = await loop.run_in_executor(None, _scan_disk)
    ctx.docs = scan_data["docs"]
    ctx.datasets = scan_data["datasets"]
    ctx.models = scan_data["models"]
    ctx.code = scan_data["code"]

    # Concurrently attach binary summaries
    tasks = []
    for ds_path in ctx.datasets:
        tasks.append(_attach_dataset_summary(ctx, ds_path))
    for ml_path in ctx.models:
        tasks.append(_attach_model_summary(ctx, ml_path))

    if tasks:
        await asyncio.gather(*tasks)

    return ctx


async def _attach_dataset_summary(ctx: ProjectContext, path: Path) -> None:
    """Async wrapper for binary dataset inspection."""
    rel_key = path.relative_to(ctx.root).as_posix()
    loop = asyncio.get_running_loop()
    try:
        # Run binary inspection in executor
        summary = await loop.run_in_executor(None, lambda: summarize_dataset(path))
        ctx.binary_summaries[rel_key] = summary
    except Exception as exc:
        logger.warning("Failed to summarise dataset %s: %s", path, exc)


async def _attach_model_summary(ctx: ProjectContext, path: Path) -> None:
    """Async wrapper for ONNX model inspection."""
    rel_key = path.relative_to(ctx.root).as_posix()
    loop = asyncio.get_running_loop()
    try:
        # Run binary model inspection in executor
        summary = await loop.run_in_executor(None, lambda: summarize_onnx(path))
        ctx.binary_summaries[rel_key] = summary
    except Exception as exc:
        logger.warning("Failed to summarise ONNX model %s: %s", path, exc)