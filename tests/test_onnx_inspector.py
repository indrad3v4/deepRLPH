from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

from onnx_inspector import summarize_onnx


def _build_tiny_onnx(path: Path) -> None:
    # Simple model: y = x + b
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4])

    W_init = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[4, 4],
        vals=np.ones((4, 4), dtype=np.float32).flatten().tolist(),
    )

    node = helper.make_node(
        "MatMul",
        inputs=["input", "W"],
        outputs=["output"],
    )

    graph = helper.make_graph(
        nodes=[node],
        name="TinyGraph",
        inputs=[X],
        outputs=[Y],
        initializer=[W_init],
    )

    model = helper.make_model(graph)
    onnx.save(model, path)


def test_onnx_summarizer(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    _build_tiny_onnx(path)

    summary = summarize_onnx(path)

    assert summary["valid"] is True
    assert summary["total_parameters"] > 0
    assert any(inp["name"] == "input" for inp in summary["inputs"])
    assert any(out["name"] == "output" for out in summary["outputs"])
