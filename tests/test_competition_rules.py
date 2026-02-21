import pytest

from src.competition_rules import (
    CompetitionRules,
    UnknownCompetitionError,
    get_competition_rules,
    list_competitions,
)


def test_wundernn_rules_basic_shape() -> None:
    rules = get_competition_rules("wundernn")

    assert rules["metric_key"] == "weighted_pearson"
    assert "submission_format" in rules
    assert "resources" in rules
    assert "environment" in rules

    sub = rules["submission_format"]
    assert "solution.py" in sub["root_files"]
    assert sub["required_class"] == "PredictionModel"
    assert sub["required_method"] == "predict"

    resources = rules["resources"]
    assert resources["time_limit_minutes"] == 60
    assert resources["cpu_cores"] == 1
    assert resources["ram_gb"] == 16
    assert resources["offline"] is True


def test_generic_kaggle_placeholder_registered() -> None:
    rules = get_competition_rules("GENERIC_KAGGLE_TS")  # case-insensitive

    assert rules["metric_key"] == "rmse"
    assert "main.py" in rules["submission_format"]["root_files"]


def test_list_competitions_contains_both() -> None:
    all_rules = list_competitions()

    assert "wundernn" in all_rules
    assert "generic_kaggle_ts" in all_rules


def test_unknown_competition_raises() -> None:
    with pytest.raises(UnknownCompetitionError):
        get_competition_rules("nonexistent_competition")
