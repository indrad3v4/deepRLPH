# -*- coding: utf-8 -*-
"""context_ingestor.py - Project context ingestion with binary inspectors.

Used by the orchestrator before task clarification / PRD generation to
provide DeepSeek with a compact summary of local project files
(including `.parquet` datasets and `.onnx` models) (ITEM-004).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List
import logging

from dataset_inspector import summarize_dataset
from onnx_inspector import summarize_onnx

logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Lightweight container for ingested project artefacts."""

    root: Path
    docs: List[Path] = field(default_factory=list)
    datasets: List[Path] = field(default_factory=list)
    code: List[Path] = field(default_factory=list)
    models: List[Path] = field(default_factory=list)

    # Binary summaries keyed by relative path
    binary_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "docs": [str(p) for p in self.docs],
            "datasets": [str(p) for p in self.datasets],
            "code": [str(p) for p in self.code],
            "models": [str(p) for p in self.models],
            "binary_summaries": self.binary_summaries,
        }

    def to_text_summary(self) -> str:
        """Compact text summary safe to embed into prompts."""

        lines: List[str] = []
        rel = lambda p: p.relative_to(self.root).as_posix()

        if self.docs:
            lines.append("# Documentation files")
            for p in sorted(self.docs):
                lines.append(f"- {rel(p)}")

        if self.datasets:
            lines.append("\n# Datasets")
            for p in sorted(self.datasets):
                key = rel(p)
                ds = self.binary_summaries.get(key) or {}
                meta = f"rows={ds.get('row_count')}, cols={ds.get('column_count')}" if ds else "(no summary)"
                lines.append(f"- {key} ({meta})")

        if self.models:
            lines.append("\n# Models")
            for p in sorted(self.models):
                key = rel(p)
                ms = self.binary_summaries.get(key) or {}
                meta = ms.get("model_type_hint", "unknown") if ms else "unknown"
                lines.append(f"- {key} (type={meta})")

        if self.code:
            lines.append("\n# Code files (truncated list)")
            for p in sorted(self.code)[:20]:
                lines.append(f"- {rel(p)}")

        text = "\n".join(lines)

        # Rough token estimate
        approx_tokens = max(1, len(text) // 4)
        if approx_tokens > 4000:
            logger.warning("Project context summary is large (~%d tokens)", approx_tokens)

        return text


def ingest_project_context(project_dir: Path) -> ProjectContext:
    """Walk the project directory and collect context.

    - Documentation: *.md, *.txt under root
    - Datasets: data/**/*.csv|parquet (with intelligent parquet summary)
    - Models: models/**/*.onnx (summarised via onnx_inspector)
    - Code: src/**/*.py
    """

    root = Path(project_dir).resolve()
    ctx = ProjectContext(root=root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        suffix = path.suffix.lower()

        if suffix in {".md", ".txt"}:
            ctx.docs.append(path)
        elif suffix in {".csv", ".parquet"} and "data" in rel.parts:
            ctx.datasets.append(path)
            _attach_dataset_summary(ctx, path)
        elif suffix == ".onnx" and "models" in rel.parts:
            ctx.models.append(path)
            _attach_model_summary(ctx, path)
        elif suffix == ".py" and "src" in rel.parts:
            ctx.code.append(path)

    return ctx


def _attach_dataset_summary(ctx: ProjectContext, path: Path) -> None:
    rel_key = path.relative_to(ctx.root).as_posix()
    try:
        summary = summarize_dataset(path)
        ctx.binary_summaries[rel_key] = summary
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to summarise dataset %s: %s", path, exc)


def _attach_model_summary(ctx: ProjectContext, path: Path) -> None:
    rel_key = path.relative_to(ctx.root).as_posix()
    try:
        summary = summarize_onnx(path)
        ctx.binary_summaries[rel_key] = summary
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to summarise ONNX model %s: %s", path, exc)
