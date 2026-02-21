# -*- coding: utf-8 -*-
"""competition_rules.py - Registry of ML competition rules.

MVP: supports WunderNN Predictorium and a placeholder second competition
schema to prove extensibility.

The registry is intentionally simple and pure-data so orchestrator and
PRD generator can query it without importing heavy libs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class CompetitionRules:
    """Normalized structure for a competition's hard constraints and format."""

    name: str
    metric_name: str
    metric_key: str
    submission_format: Dict[str, Any]
    resources: Dict[str, Any]
    environment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - tiny wrapper
        return asdict(self)


_RULES: Dict[str, CompetitionRules] = {}


def _register(rules: CompetitionRules) -> None:
    key = rules.name.lower()
    if key in _RULES:
        raise ValueError(f"Competition rules already registered for {rules.name!r}")
    _RULES[key] = rules


# WunderNN Predictorium (time-series forecaster)
_register(
    CompetitionRules(
        name="wundernn",
        metric_name="Weighted Pearson Correlation",
        metric_key="weighted_pearson",
        submission_format={
            "root_files": ["solution.py"],
            "required_class": "PredictionModel",
            "required_method": "predict",
            "predict_signature": "predict(self, data_point) -> np.ndarray | None",
            "data_point_doc": "utils.DataPoint with fields seq_ix, step_in_seq, need_prediction, state",
        },
        resources={
            "time_limit_minutes": 60,
            "cpu_cores": 1,
            "ram_gb": 16,
            "gpu": False,
            "offline": True,
        },
        environment={
            "docker_base_image": "python:3.11-slim-bookworm",
            "notes": "No internet, CPU-only, local SSD; requirements installed from competition image.",
        },
    )
)


# Placeholder second competition to prove extensibility
_register(
    CompetitionRules(
        name="generic_kaggle_ts",
        metric_name="Root Mean Squared Error",
        metric_key="rmse",
        submission_format={
            "root_files": ["main.py", "model.pkl"],
            "required_class": "Model",
            "required_method": "predict",
            "predict_signature": "predict(self, X) -> np.ndarray",
            "data_point_doc": "Tabular features matrix X with shape (n_samples, n_features)",
        },
        resources={
            "time_limit_minutes": 120,
            "cpu_cores": 2,
            "ram_gb": 32,
            "gpu": False,
            "offline": True,
        },
        environment={
            "docker_base_image": "python:3.10-slim",
            "notes": "Typical Kaggle CPU environment (placeholder, non-authoritative).",
        },
    )
)


class UnknownCompetitionError(KeyError):
    """Raised when a competition id is not registered."""


def get_competition_rules(name: str) -> Dict[str, Any]:
    """Return competition rules as a plain dict.

    Args:
        name: Competition identifier, case-insensitive (e.g. "wundernn").

    Raises:
        UnknownCompetitionError: if no rules are registered under this name.
    """

    key = name.lower().strip()
    try:
        return _RULES[key].to_dict()
    except KeyError as exc:  # pragma: no cover - simple branch
        raise UnknownCompetitionError(name) from exc


def list_competitions() -> Dict[str, Dict[str, Any]]:
    """Return all registered competitions as plain dicts."""

    return {name: rules.to_dict() for name, rules in _RULES.items()}
