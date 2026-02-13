"""Project context ingestion utilities (PRD-01).

These helpers scan a RALPH project directory for:
- documentation files (README, .md, .txt)
- dataset files (under data/ and workspace/input)
- baseline code / model files (src/, models/)

They return a small, prompt-friendly summary dictionary that can be
attached to ProjectConfig.metadata so DeepSeek can see real project
artifacts during clarification/PRD generation.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

import json


# -------------------------- helpers ---------------------------------


def _safe_read_text(path: Path, max_chars: int = 4000) -> str:
    """Read at most max_chars of a text file, best-effort.

    Binary or unreadable files return an empty string.
    """

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [truncated]"
    return text


def _find_files(root: Path, patterns: List[str]) -> List[Path]:
    """Glob-like search for multiple patterns under root (non-recursive safe variant)."""

    results: List[Path] = []
    for pattern in patterns:
        results.extend(root.rglob(pattern))
    return results


# ------------------------ public API --------------------------------


def build_project_context(project_dir: Path) -> Dict[str, Any]:
    """Build a lightweight context summary for a project.

    Returns a dict with keys:
        - doc_summaries: {relative_path: excerpt}
        - dataset_files: [relative_path, ...]
        - dataset_schemas: {relative_path: header/first-line sample}
        - baseline_code_files: [relative_path, ...]
        - summary: short human-readable bullet text
    """

    project_dir = project_dir.resolve()

    # --- docs ---
    doc_files: List[Path] = []
    for sub in [project_dir, project_dir / "workspace", project_dir / "docs"]:
        if sub.exists():
            doc_files.extend(_find_files(sub, ["README*.md", "*.md", "*.txt"]))

    doc_summaries: Dict[str, str] = {}
    for p in sorted(set(doc_files))[:5]:  # cap number of docs for prompts
        rel = str(p.relative_to(project_dir))
        snippet = _safe_read_text(p, max_chars=1500)
        if snippet.strip():
            doc_summaries[rel] = snippet

    # --- datasets ---
    dataset_roots = [project_dir / "data", project_dir / "workspace" / "input"]
    dataset_files: List[str] = []
    dataset_schemas: Dict[str, str] = {}

    for root in dataset_roots:
        if not root.exists():
            continue
        for p in _find_files(root, ["*.csv", "*.parquet", "*.json"]):
            rel = str(p.relative_to(project_dir))
            if rel not in dataset_files:
                dataset_files.append(rel)
            # very cheap header/schema hint
            if p.suffix.lower() == ".csv":
                try:
                    with p.open("r", encoding="utf-8", errors="ignore") as f:
                        header = f.readline().strip()
                    if header:
                        dataset_schemas[rel] = header[:500]
                except Exception:
                    continue

    # --- baseline code/models ---
    code_roots = [project_dir / "src", project_dir / "models"]
    baseline_code_files: List[str] = []
    for root in code_roots:
        if not root.exists():
            continue
        for p in _find_files(root, ["*.py", "*.ipynb"]):
            rel = str(p.relative_to(project_dir))
            if rel not in baseline_code_files:
                baseline_code_files.append(rel)

    # --- human-readable rollup ---
    summary_lines = [
        f"Docs: {len(doc_summaries)} files summarised",
        f"Datasets: {len(dataset_files)} files detected",
        f"Schemas: {len(dataset_schemas)} CSV headers captured",
        f"Baseline code: {len(baseline_code_files)} files (src/models)",
    ]

    return {
        "doc_summaries": doc_summaries,
        "dataset_files": dataset_files,
        "dataset_schemas": dataset_schemas,
        "baseline_code_files": baseline_code_files,
        "summary": "; ".join(summary_lines),
    }
