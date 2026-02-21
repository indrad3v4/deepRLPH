# -*- coding: utf-8 -*-
"""onnx_inspector.py - Light-weight summaries for ONNX models (ITEM-003).

Provides `summarize_onnx(path)` that extracts graph-level information:
- input / output tensors
- layer op types
- total parameter count
- a coarse model type hint (RNN / Transformer / CNN / generic)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set
import logging

import numpy as np
import onnx
from onnx import numpy_helper

logger = logging.getLogger(__name__)


def summarize_onnx(path: str | Path) -> Dict[str, Any]:
    """Summarise an ONNX model into a compact JSON-serialisable dict.

    Errors are caught and returned in the summary so that callers and
    tests can handle corrupted models gracefully instead of crashing.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX model not found: {p}")

    try:
        model = onnx.load(str(p))
        onnx.checker.check_model(model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load/validate ONNX model %s: %s", p, exc)
        return {
            "path": str(p),
            "valid": False,
            "error": str(exc),
        }

    graph = model.graph

    inputs = [_tensor_info(t) for t in graph.input]
    outputs = [_tensor_info(t) for t in graph.output]

    op_types: Set[str] = {node.op_type for node in graph.node}

    total_params = 0
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        total_params += int(np.prod(arr.shape))

    model_type = _infer_model_type(op_types)

    summary: Dict[str, Any] = {
        "path": str(p),
        "valid": True,
        "inputs": inputs,
        "outputs": outputs,
        "num_nodes": len(graph.node),
        "op_types": sorted(op_types),
        "total_parameters": int(total_params),
        "model_type_hint": model_type,
    }

    return summary


def _tensor_info(value_info: Any) -> Dict[str, Any]:
    """Extract name, shape and dtype from an ONNX ValueInfoProto."""

    t = value_info.type.tensor_type
    shape: List[int] = [int(d.dim_value or 0) for d in t.shape.dim]
    return {
        "name": value_info.name,
        "shape": shape,
        "dtype": int(t.elem_type),
    }


def _infer_model_type(op_types: Set[str]) -> str:
    ops = {op.upper() for op in op_types}

    if {"LSTM", "GRU", "RNN"} & ops:
        return "rnn_seq"
    if {"ATTENTION", "MULTIHEADATTENTION", "TRANSFORMER"} & ops:
        return "transformer_like"
    if {"CONV", "CONV1D", "CONV2D", "CONV3D"} & ops:
        return "cnn_like"
    return "generic"
