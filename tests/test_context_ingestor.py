from pathlib import Path

from context_ingestor import ingest_project_context
from dataset_inspector import summarize_dataset
from onnx_inspector import summarize_onnx

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import onnx
from onnx import helper, TensorProto


def _build_tiny_project(root: Path) -> None:
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)

    # Docs
    (root / "README.md").write_text("WunderNN project docs")

    # Dataset
    df = pd.DataFrame({"id": [1, 2, 3], "value": [0.1, 0.5, 0.9]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, root / "data" / "raw" / "train.parquet")

    # Minimal ONNX model
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])
    W_init = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[4, 4],
        vals=np.ones((4, 4), dtype=np.float32).flatten().tolist(),
    )
    node = helper.make_node("MatMul", inputs=["input", "W"], outputs=["output"])
    graph = helper.make_graph(
        nodes=[node], name="TinyGraph", inputs=[X], outputs=[Y], initializer=[W_init]
    )
    model = helper.make_model(graph)
    onnx.save(model, root / "models" / "baseline.onnx")

    # Simple code file
    (root / "src" / "main.py").write_text("print('hello')\n")


def test_context_ingestor_with_binary_summaries(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    _build_tiny_project(project_root)

    ctx = ingest_project_context(project_root)
    summary = ctx.to_dict()

    # Basic structure
    assert summary["root"] == str(project_root.resolve())
    assert any(path.endswith("README.md") for path in summary["docs"])
    assert any("train.parquet" in path for path in summary["datasets"])
    assert any("baseline.onnx" in path for path in summary["models"])

    # Binary summaries must include both dataset and model
    keys = set(summary["binary_summaries"].keys())
    assert any("train.parquet" in k for k in keys)
    assert any("baseline.onnx" in k for k in keys)

    # Text summary should mention datasets and models sections
    text = ctx.to_text_summary()
    assert "# Datasets" in text
    assert "# Models" in text
