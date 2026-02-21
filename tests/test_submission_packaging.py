# -*- coding: utf-8 -*-
"""Submission packager tests for ML competition projects."""

from pathlib import Path

from scripts.make_submission import SubmissionPackager


def test_submission_packager_creates_zip(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()

    # Minimal required structure
    models_final = project_root / "models" / "final"
    models_final.mkdir(parents=True)

    model_path = models_final / "model.onnx"
    model_path.write_bytes(b"ONNX")

    solution_path = project_root / "solution.py"
    solution_path.write_text(
        """\
class PredictionModel:
    def predict(self, df):
        return [0.0] * len(df)
""",
        encoding="utf-8",
    )

    packager = SubmissionPackager(project_root)
    zip_path_str = packager.create_submission("test_submission.zip")

    zip_path = Path(zip_path_str)
    assert zip_path.exists()

    # Basic sanity: zip should contain solution.py and model.onnx
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "solution.py" in names
    assert "model.onnx" in names
