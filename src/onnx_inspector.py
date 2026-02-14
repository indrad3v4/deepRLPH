# -*- coding: utf-8 -*-
"""
onnx_inspector.py - BE-003: ONNX Model Inspection

Extracts metadata from uploaded ONNX models:
- Input/output tensor shapes and types
- Model operators (layers)
- Parameter count and file size
- ONNX opset version
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ONNXInspector")


class ONNXInspector:
    """Inspect ONNX model files and extract metadata."""

    def __init__(self):
        self.onnx_available = self._check_onnx()

    def _check_onnx(self) -> bool:
        """Check if onnx library is available."""
        try:
            import onnx
            return True
        except ImportError:
            logger.warning(
                "⚠️  onnx library not installed. Install with: pip install onnx"
            )
            return False

    def inspect_model(self, model_path: Path) -> Dict[str, Any]:
        """Inspect single ONNX model file.

        Args:
            model_path: Path to .onnx file

        Returns:
            Dict with model metadata
        """
        if not self.onnx_available:
            return {
                "error": "onnx library not installed",
                "file": str(model_path),
                "size_mb": model_path.stat().st_size / (1024 * 1024),
            }

        try:
            import onnx
            from onnx import numpy_helper

            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)

            # Extract basic info
            graph = model.graph
            metadata = {
                "file": model_path.name,
                "path": str(model_path),
                "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
                "opset_version": model.opset_import[0].version if model.opset_import else None,
                "producer": model.producer_name or "Unknown",
                "inputs": self._extract_io_info(graph.input, graph.value_info),
                "outputs": self._extract_io_info(graph.output, graph.value_info),
                "operators": self._extract_operators(graph.node),
                "parameters": self._count_parameters(graph.initializer),
            }

            logger.info("✅ Inspected ONNX model: %s", model_path.name)
            return metadata

        except Exception as e:
            logger.error("❌ Error inspecting ONNX model %s: %s", model_path.name, e)
            return {
                "error": str(e),
                "file": str(model_path),
                "size_mb": model_path.stat().st_size / (1024 * 1024),
            }

    def _extract_io_info(
        self, io_list, value_info_list
    ) -> List[Dict[str, Any]]:
        """Extract input/output tensor information."""
        result = []
        for tensor in io_list:
            info = {
                "name": tensor.name,
                "type": self._get_tensor_type(tensor.type),
                "shape": self._get_tensor_shape(tensor.type),
            }
            result.append(info)
        return result

    def _get_tensor_type(self, tensor_type) -> str:
        """Get tensor data type as string."""
        try:
            from onnx import TensorProto
            
            type_map = {
                TensorProto.FLOAT: "float32",
                TensorProto.UINT8: "uint8",
                TensorProto.INT8: "int8",
                TensorProto.UINT16: "uint16",
                TensorProto.INT16: "int16",
                TensorProto.INT32: "int32",
                TensorProto.INT64: "int64",
                TensorProto.STRING: "string",
                TensorProto.BOOL: "bool",
                TensorProto.FLOAT16: "float16",
                TensorProto.DOUBLE: "float64",
            }
            
            if tensor_type.HasField("tensor_type"):
                elem_type = tensor_type.tensor_type.elem_type
                return type_map.get(elem_type, f"unknown_{elem_type}")
            return "unknown"
        except Exception:
            return "unknown"

    def _get_tensor_shape(self, tensor_type) -> List[Any]:
        """Extract tensor shape, handling dynamic dimensions."""
        try:
            if not tensor_type.HasField("tensor_type"):
                return []
            
            shape = []
            for dim in tensor_type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)  # Dynamic dimension like "batch"
                else:
                    shape.append("?")  # Unknown dimension
            return shape
        except Exception:
            return []

    def _extract_operators(self, nodes) -> Dict[str, int]:
        """Count operators by type."""
        op_counts = {}
        for node in nodes:
            op_type = node.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        return dict(sorted(op_counts.items(), key=lambda x: x[1], reverse=True))

    def _count_parameters(self, initializers) -> Dict[str, Any]:
        """Count total parameters in model."""
        try:
            from onnx import numpy_helper
            
            total_params = 0
            param_sizes = []
            
            for init in initializers:
                tensor = numpy_helper.to_array(init)
                param_count = tensor.size
                total_params += param_count
                param_sizes.append({
                    "name": init.name,
                    "shape": list(tensor.shape),
                    "count": param_count,
                })
            
            return {
                "total": total_params,
                "total_mb": round(total_params * 4 / (1024 * 1024), 2),  # Assume float32
                "layers": len(param_sizes),
            }
        except Exception as e:
            logger.warning("Could not count parameters: %s", e)
            return {"total": 0, "layers": 0}

    def inspect_and_save(
        self,
        project_dir: Path,
        model_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Inspect ONNX models in project and save metadata.

        Args:
            project_dir: Project root directory
            model_path: Optional specific model path. If None, scan models/ directory

        Returns:
            Dict with status and inspected models
        """
        models_dir = project_dir / "models"
        output_file = models_dir / "model_inspection.json"

        # Find models
        if model_path:
            model_files = [model_path] if model_path.suffix == ".onnx" else []
        else:
            model_files = list(models_dir.rglob("*.onnx")) if models_dir.exists() else []

        if not model_files:
            logger.info("No ONNX models found for inspection")
            return {
                "status": "no_models",
                "models": [],
                "message": "No .onnx files found",
            }

        # Inspect each model
        inspected = []
        for model_file in model_files:
            metadata = self.inspect_model(model_file)
            inspected.append(metadata)

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "models": inspected,
                    "total_models": len(inspected),
                    "inspection_date": None,  # Will be set by caller
                },
                f,
                indent=2,
            )

        logger.info("📊 Saved model inspection to %s", output_file)

        return {
            "status": "success",
            "models": inspected,
            "total_models": len(inspected),
            "output_file": str(output_file),
        }

    def attach_to_metadata(
        self,
        metadata: Dict[str, Any],
        models: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Attach model inspection results to project metadata.

        Args:
            metadata: Existing project metadata dict
            models: List of inspected model dicts

        Returns:
            Updated metadata dict
        """
        if not models:
            return metadata

        meta = dict(metadata or {})
        meta["onnx_models"] = models
        meta["total_onnx_models"] = len(models)

        # Extract summary info from first model
        if models and "error" not in models[0]:
            first_model = models[0]
            if first_model.get("inputs"):
                meta["model_input_shape"] = first_model["inputs"][0].get("shape", [])
                meta["model_input_type"] = first_model["inputs"][0].get("type", "unknown")
            if first_model.get("outputs"):
                meta["model_output_shape"] = first_model["outputs"][0].get("shape", [])
            meta["model_parameters"] = first_model.get("parameters", {}).get("total", 0)

        return meta


def inspect_onnx_models(project_dir: Path, model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience function to inspect ONNX models.

    Args:
        project_dir: Project directory
        model_path: Optional specific model file path

    Returns:
        Inspection results dict
    """
    inspector = ONNXInspector()
    return inspector.inspect_and_save(project_dir, model_path)
