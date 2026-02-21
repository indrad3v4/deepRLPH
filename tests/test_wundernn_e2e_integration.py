# -*- coding: utf-8 -*-
"""End-to-end WunderNN integration test.

This test uses the existing create_dummy_wundernn_project helper to:
- create a temporary project
- run PRD generation for ml_competition domain
- ensure competition rules / KPI / dataset summaries are reflected in PRD
- run submission packaging on the dummy project
"""

from __future__ import annotations

from pathlib import Path

from context_ingestor import ingest_project_context
from src.competition_rules import get_competition_rules
from src.prd_generator import generate_prd
from scripts.make_submission import SubmissionPackager

from .test_wundernn_e2e import create_dummy_wundernn_project  # reuse helper


def test_wundernn_e2e_prd_and_submission(tmp_path: Path) -> None:
    orchestrator = create_dummy_wundernn_project(tmp_path)

    # Load project metadata from orchestrator
    projects = orchestrator.list_projects()
    assert projects, "No projects created by helper"
    project = projects[0]

    # list_projects returns dicts, not objects
    project_dir = Path(project["path"])

    # Ingest context so dataset / ONNX summaries are available
    ctx = ingest_project_context(project_dir)
    ctx_dict = ctx.to_dict()

    # Prepare metadata for PRD generation
    competition_rules = get_competition_rules("wundernn")

    project_meta = {
        "project_type": "ml_competition",
        "competition": "wundernn",
        "kpi_metric": competition_rules["metric_key"],
        "kpi_target": 0.1,
        "ingested_context": ctx_dict,
    }

    technical_brief = {"clarified_task": "Build WunderNN forecaster", "key_requirements": []}

    prd = generate_prd(technical_brief, domain="ml_competition", project_metadata=project_meta)

    # Basic sanity on PRD
    assert prd["domain"] == "ml_competition"
    stories = prd["user_stories"]
    meta = prd["project_metadata"]

    # KPI normalization
    assert meta["kpi_metric"] == competition_rules["metric_key"]
    assert meta["kpi_target"] == 0.1

    # Evaluation story should mention metric and target
    eval_story = next(s for s in stories if s["id"] == "ML-007")
    eval_text = " ".join(eval_story["acceptance_criteria"]).lower()
    assert competition_rules["metric_key"] in eval_text
    assert "0.10" in eval_text or "0.1" in eval_text

    # Submission story should mention solution.py and PredictionModel
    submit_story = next(s for s in stories if s["id"] == "ML-009")
    submit_text = " ".join(submit_story["acceptance_criteria"]).lower()
    assert "solution.py" in submit_text
    assert "predictionmodel" in submit_text

    # Dataset story should reflect that a parquet file was seen in context
    data_story = next(s for s in stories if s["id"] == "ML-002")
    data_text = " ".join(data_story["acceptance_criteria"]).lower()
    assert "train.parquet" in data_text

    # Finally, ensure submission packaging works on the dummy project
    packager = SubmissionPackager(project_dir)
    zip_path_str = packager.create_submission("wundernn_dummy_submission.zip")
    zip_path = project_dir / zip_path_str.split("/")[-1]
    assert zip_path.exists()
