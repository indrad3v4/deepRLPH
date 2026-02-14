# -*- coding: utf-8 -*-
"""context_ingestor.py - Lightweight project context ingestion (PRD-01)

Summarizes local project files so DeepSeek can see real docs, datasets,
baseline code, and model artifacts without reading entire large files.

Scope (per project root):
- Docs: README*.md, *.md, *.txt under root, workspace/, docs/ (≤5 files total).
- Datasets: *.csv, *.parquet, *.json under data/ and workspace/input/.
  For CSV we include header line as a simple "schema".
- Baseline code: *.py and *.ipynb under src/ and models/.
- Baseline models: common binary formats under models/ (e.g. .pt, .onnx, .bin,
  .pkl, .safetensors).

Output is a small dict that can be attached to project_metadata and also
flattened into a text summary for prompts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("ContextIngestor")


MAX_DOC_FILES = 5
MAX_DOC_CHARS = 3000
MAX_SCHEMA_LINES = 3


@dataclass
class DocSnippet:
    path: str
    kind: str
    excerpt: str


@dataclass
class DatasetSummary:
    path: str
    kind: str
    schema: str


@dataclass
class CodeSummary:
    path: str
    kind: str
    excerpt: str


@dataclass
class ModelArtifact:
    path: str
    kind: str


@dataclass
class ProjectContext:
    docs: List[DocSnippet]
    datasets: List[DatasetSummary]
    code: List[CodeSummary]
    models: List[ModelArtifact]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "docs": [asdict(d) for d in self.docs],
            "datasets": [asdict(d) for d in self.datasets],
            "code": [asdict(c) for c in self.code],
            "models": [asdict(m) for m in self.models],
        }

    def to_text_summary(self) -> str:
        """Flatten context into a compact human-readable summary."""
        parts: List[str] = []

        if self.docs:
            doc_lines = []
            for d in self.docs:
                excerpt_clean = d.excerpt[:200].replace('\n', ' ')
                doc_lines.append(f"- {d.path}: {excerpt_clean}")
            parts.append("DOCUMENTATION SNIPPETS:\n" + "\n".join(doc_lines))

        if self.datasets:
            parts.append("DATASETS:\n" + "\n".join(
                f"- {d.path} ({d.kind}) schema: {d.schema}" for d in self.datasets
            ))

        if self.code:
            code_lines = []
            for c in self.code:
                excerpt_clean = c.excerpt[:200].replace('\n', ' ')
                code_lines.append(f"- {c.path}: {excerpt_clean}")
            parts.append("BASELINE CODE SNIPPETS:\n" + "\n".join(code_lines))

        if self.models:
            parts.append("MODEL ARTIFACTS:\n" + "\n".join(
                f"- {m.path} ({m.kind})" for m in self.models
            ))

        return "\n\n".join(parts)


def _read_text_safely(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > max_chars:
            return text[:max_chars] + "... [truncated]"
        return text
    except Exception as e:
        logger.warning("⚠️ Failed to read %s: %s", path, e)
        return "[unreadable]"


def _summarize_csv_schema(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for _ in range(MAX_SCHEMA_LINES):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
        return " | ".join(lines)
    except Exception as e:
        logger.warning("⚠️ Failed to read CSV header %s: %s", path, e)
        return "[schema unavailable]"


def ingest_project_context(project_root: Path) -> ProjectContext:
    """Scan project tree and build lightweight context summary.

    project_root is the per-project directory created by WorkspaceManager
    (the folder that contains config.json, README.md, src/, data/, etc.).
    """
    docs: List[DocSnippet] = []
    datasets: List[DatasetSummary] = []
    code: List[CodeSummary] = []
    models: List[ModelArtifact] = []

    root = project_root

    # ---------------------- Docs ----------------------
    doc_dirs = [
        root,
        root / "workspace",
        root / "docs",
    ]
    doc_globs = ["README*.md", "*.md", "*.txt"]

    for base in doc_dirs:
        if not base.exists():
            continue
        for pattern in doc_globs:
            for path in sorted(base.rglob(pattern)):
                if len(docs) >= MAX_DOC_FILES:
                    break
                if path.is_file():
                    rel = path.relative_to(root).as_posix()
                    excerpt = _read_text_safely(path, MAX_DOC_CHARS)
                    docs.append(DocSnippet(path=rel, kind="doc", excerpt=excerpt))
            if len(docs) >= MAX_DOC_FILES:
                break

    # ---------------------- Datasets ----------------------
    data_dirs = [root / "data", root / "workspace" / "input"]
    data_globs = ["*.csv", "*.parquet", "*.json"]

    for base in data_dirs:
        if not base.exists():
            continue
        for pattern in data_globs:
            for path in sorted(base.rglob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                suffix = path.suffix.lower()
                kind = {
                    ".csv": "csv",
                    ".parquet": "parquet",
                    ".json": "json",
                }.get(suffix, suffix.lstrip("."))

                schema = ""
                if suffix == ".csv":
                    schema = _summarize_csv_schema(path)
                else:
                    schema = kind

                datasets.append(DatasetSummary(path=rel, kind=kind, schema=schema))

    # ---------------------- Baseline code ----------------------
    code_dirs = [root / "src", root / "models"]
    code_globs = ["*.py", "*.ipynb"]

    for base in code_dirs:
        if not base.exists():
            continue
        for pattern in code_globs:
            for path in sorted(base.rglob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                excerpt = _read_text_safely(path, MAX_DOC_CHARS // 2)
                kind = "notebook" if path.suffix == ".ipynb" else "python"
                code.append(CodeSummary(path=rel, kind=kind, excerpt=excerpt))

    # ---------------------- Model artifacts ----------------------
    model_dir = root / "models"
    model_globs = ["*.pt", "*.onnx", "*.bin", "*.pkl", "*.safetensors"]

    if model_dir.exists():
        for pattern in model_globs:
            for path in sorted(model_dir.rglob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                models.append(ModelArtifact(path=rel, kind=path.suffix.lstrip(".")))

    ctx = ProjectContext(docs=docs, datasets=datasets, code=code, models=models)

    logger.info(
        "📚 Ingested project context: %d docs, %d datasets, %d code files, %d model artifacts",
        len(docs),
        len(datasets),
        len(code),
        len(models),
    )

    return ctx
