# -*- coding: utf-8 -*-
"""dataset_inspector.py - Intelligent summaries for competition datasets.

Provides `summarize_dataset(path)` and `DatasetInspector` class for AI context
and schema integration (ITEM-002 from Issue #7).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Simple in-process cache so repeated calls do not re-scan huge files
_DATASET_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass
class DatasetSchema:
    """Schema representation for a dataset file."""
    
    name: str
    path: str
    format: str
    num_rows: int
    num_columns: int
    columns: List[str]
    dtypes: Dict[str, str]
    num_features: Optional[int] = None
    sequence_length: Optional[int] = None
    num_samples: Optional[int] = None
    detected_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetInspector:
    """Inspect datasets and extract schema information."""
    
    def __init__(self):
        self._cache: Dict[str, DatasetSchema] = {}
    
    def inspect(self, path: Path) -> DatasetSchema:
        """Inspect a single dataset file and return its schema."""
        
        path = Path(path)
        cache_key = str(path.resolve())
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        summary = summarize_dataset(path)
        
        # Infer ML-specific metadata from summary
        num_features = summary.get('column_count', 0)
        num_samples = summary.get('row_count')
        detected_type = self._detect_type(summary)
        
        schema = DatasetSchema(
            name=path.name,
            path=str(path),
            format=summary.get('format', path.suffix.lstrip('.')),
            num_rows=summary.get('row_count', 0) or 0,
            num_columns=summary.get('column_count', 0),
            columns=summary.get('columns', []),
            dtypes=summary.get('dtypes', {}),
            num_features=num_features,
            num_samples=num_samples,
            detected_type=detected_type,
        )
        
        self._cache[cache_key] = schema
        return schema
    
    def inspect_directory(self, data_dir: Path) -> Dict[str, DatasetSchema]:
        """Inspect all dataset files in a directory."""
        
        schemas = {}
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return schemas
        
        # Supported extensions
        extensions = {'.parquet', '.csv', '.json', '.feather'}
        
        for file_path in data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    schema = self.inspect(file_path)
                    schemas[file_path.name] = schema
                except Exception as e:
                    logger.error(f"Failed to inspect {file_path}: {e}")
        
        return schemas
    
    def _detect_type(self, summary: Dict[str, Any]) -> Optional[str]:
        """Heuristic to detect dataset type for ML."""
        
        columns = [c.lower() for c in summary.get('columns', [])]
        dtypes = summary.get('dtypes', {})
        
        # Time series detection
        time_cols = ['timestamp', 'datetime', 'date', 'time']
        if any(col in columns for col in time_cols):
            return 'time_series'
        
        # Check for target column patterns
        target_patterns = ['target', 'label', 'y', 'prediction', 'value']
        if any(any(p in col for p in target_patterns) for col in columns):
            return 'supervised_learning'
        
        # Default based on size
        num_cols = summary.get('column_count', 0)
        if num_cols > 10:
            return 'tabular'
        
        return 'generic'


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def summarize_dataset(path: str | Path, max_numeric_columns: int = 3, sample_rows: int = 5) -> Dict[str, Any]:
    """Summarize a dataset file (optimised for `.parquet`).

    The summary is intentionally compact so it can be embedded into
    AI prompts without blowing up token count. For large parquet files
    we rely on `pyarrow` metadata and a single row group for statistics.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    cache_key = str(p.resolve())
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        summary = _summarize_parquet(p, max_numeric_columns, sample_rows)
    else:
        # Fallback: treat as small CSV-like text file
        summary = _summarize_tabular_fallback(p, sample_rows)

    _DATASET_CACHE[cache_key] = summary
    return summary


def _summarize_parquet(path: Path, max_numeric_columns: int, sample_rows: int) -> Dict[str, Any]:
    """Summarise a parquet dataset using pyarrow metadata plus a small sample."""

    pf = pq.ParquetFile(path)
    meta = pf.metadata

    row_count = meta.num_rows
    column_count = meta.num_columns
    columns: List[str] = list(pf.schema.names)

    dtypes: Dict[str, str] = {}
    for field in pf.schema:
        dtypes[str(field.name)] = str(field.physical_type)

    # Read only the first row group into pandas for lightweight stats/sample
    table = pf.read_row_group(0)
    df = table.to_pandas()

    # Approximate memory usage per row for rough scaling intuition
    approx_mem = int(df.memory_usage(deep=True).sum() / max(len(df), 1))

    # Numeric stats for first N numeric columns
    num_cols = [c for c in df.select_dtypes("number").columns][:max_numeric_columns]
    numeric_stats: Dict[str, Dict[str, float]] = {}
    for col in num_cols:
        s = df[col]
        numeric_stats[col] = {
            "min": _safe_float(s.min()),
            "max": _safe_float(s.max()),
            "mean": _safe_float(s.mean()),
            "std": _safe_float(s.std(ddof=0)),
        }

    sample_records = df.head(sample_rows).to_dict(orient="records")

    summary: Dict[str, Any] = {
        "path": str(path),
        "format": "parquet",
        "row_count": int(row_count),
        "column_count": int(column_count),
        "columns": columns,
        "dtypes": dtypes,
        "approx_memory_per_row_bytes": approx_mem,
        "numeric_stats": numeric_stats,
        "sample": sample_records,
    }

    # Very rough token estimate: assume ~1 token per 4 characters
    import json

    token_estimate = len(json.dumps(summary)) / 4
    if token_estimate > 500:
        logger.warning("Dataset summary for %s is quite large (~%d tokens)", path, token_estimate)

    return summary


def _summarize_tabular_fallback(path: Path, sample_rows: int) -> Dict[str, Any]:
    """Fallback summariser for small CSV-like files.

    Used mainly for tests and non-parquet datasets.
    """

    df = pd.read_csv(path, nrows=max(sample_rows, 10))

    summary: Dict[str, Any] = {
        "path": str(path),
        "format": path.suffix.lstrip("."),
        "row_count": None,
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "sample": df.head(sample_rows).to_dict(orient="records"),
    }

    return summary
