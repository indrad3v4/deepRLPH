from src.prd_generator import generate_prd


def test_ml_competition_injects_kpi_and_context():
    technical_brief = {"clarified_task": "Build WunderNN forecaster", "key_requirements": []}

    ingested_context = {
        "binary_summaries": {
            "data/raw/train.parquet": {
                "format": "parquet",
                "row_count": 10721,
                "column_count": 42,
            },
            "models/baseline.onnx": {
                "valid": True,
                "total_parameters": 123456,
                "model_type_hint": "transformer_like",
            },
        }
    }

    project_metadata = {
        "competition": "wundernn",
        "kpi_metric": "weighted_pearson",
        "kpi_target": 0.45,
        "ingested_context": ingested_context,
    }

    prd = generate_prd(technical_brief, domain="ml_competition", project_metadata=project_metadata)

    meta = prd["project_metadata"]
    assert meta["kpi_metric"] == "weighted_pearson"
    assert meta["kpi_target"] == 0.45

    stories = prd["user_stories"]

    # Evaluation story must mention metric + target
    eval_story = next(s for s in stories if s["id"] == "ML-007")
    eval_text = " ".join(eval_story["acceptance_criteria"]).lower()
    assert "weighted_pearson" in eval_text
    assert "0.45" in eval_text

    # Data story must mention dataset row count from ingested context
    data_story = next(s for s in stories if s["id"] == "ML-002")
    data_text = " ".join(data_story["acceptance_criteria"])
    assert "10721" in data_text
