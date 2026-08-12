from __future__ import annotations

from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


_EPSILON = 1e-12


def _serialize_json_value(value: Any) -> Any:
    """Recursively convert values into JSON-serializable equivalents."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_json_value(item) for item in value]
    return value


def _normalize_metric_value(value: Any, *, field_name: str, experiment_id: str, metric_name: str) -> float | int | None:
    """Validate metric values and normalize them to JSON-friendly numeric scalars."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Metric '{metric_name}' in experiment '{experiment_id}' has non-numeric "
            f"'{field_name}' value: {value!r}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _calculate_delta(original: float | int | None, reproduced: float | int | None) -> float | None:
    """Compute the normalized absolute difference requested by the report format."""
    if original is None or reproduced is None:
        return None
    denominator = max(abs(reproduced), abs(original), _EPSILON)
    return abs(reproduced - original) / denominator


def calculate_cv_star(scores: list[float | int] | tuple[float | int, ...]) -> float | None:
    """Compute the small-sample bias-corrected coefficient of variation."""
    if len(scores) < 2:
        return None

    normalized_scores = [float(score) for score in scores]
    mean = sum(normalized_scores) / len(normalized_scores)
    if abs(mean) <= _EPSILON:
        return None

    squared_deviations = [(score - mean) ** 2 for score in normalized_scores]
    sample_size = len(normalized_scores)
    unbiased_std = math.sqrt(sum(squared_deviations) / (sample_size - 1))
    bias_correction = 1.0 + (1.0 / (4.0 * sample_size))
    return bias_correction * (unbiased_std / abs(mean))


def _calculate_metric_cv_star(
    original: float | int | None,
    reproduced: float | int | None,
) -> float | None:
    if original is None or reproduced is None:
        return None
    return calculate_cv_star([original, reproduced])


def generate_reproduction_report(
    paper_id: str,
    reproduction_metadata: dict[str, Any],
    experiments_data: dict[str, dict[str, Any]],
) -> str:
    """
    Build a standardized JSON report from reproduction experiment results.

    Expected ``experiments_data`` shape:
    {
        "experiment_id": {
            "configuration": {...},
            "metrics": {
                "metric_name": {
                    "original": 0.95,
                    "reproduced": 0.94,
                    "delta": 0.0106,
                    "cv_star": 0.0079
                }
            }
        }
    }
    """
    serialized_metadata = _serialize_json_value(reproduction_metadata)
    formatted_experiments: dict[str, dict[str, Any]] = {}

    for experiment_id, experiment_payload in experiments_data.items():
        if not isinstance(experiment_payload, dict):
            raise TypeError(
                f"Experiment '{experiment_id}' must map to a dictionary, got {type(experiment_payload).__name__}."
            )

        configuration = experiment_payload.get("configuration", {})
        metrics = experiment_payload.get("metrics")

        if not isinstance(configuration, dict):
            raise TypeError(
                f"Experiment '{experiment_id}' configuration must be a dictionary, "
                f"got {type(configuration).__name__}."
            )
        if not isinstance(metrics, dict):
            raise TypeError(
                f"Experiment '{experiment_id}' metrics must be a dictionary, got {type(metrics).__name__}."
            )

        formatted_metrics: dict[str, dict[str, float | int | None]] = {}
        for metric_name, metric_payload in metrics.items():
            if not isinstance(metric_payload, dict):
                raise TypeError(
                    f"Metric '{metric_name}' in experiment '{experiment_id}' must be a dictionary, "
                    f"got {type(metric_payload).__name__}."
                )

            original = _normalize_metric_value(
                metric_payload.get("original"),
                field_name="original",
                experiment_id=experiment_id,
                metric_name=metric_name,
            )
            reproduced = _normalize_metric_value(
                metric_payload.get("reproduced"),
                field_name="reproduced",
                experiment_id=experiment_id,
                metric_name=metric_name,
            )

            formatted_metrics[metric_name] = {
                "original": original,
                "reproduced": reproduced,
                "delta": _calculate_delta(original, reproduced),
                "cv_star": _calculate_metric_cv_star(original, reproduced),
            }

        formatted_experiments[experiment_id] = {
            "configuration": _serialize_json_value(configuration),
            "metrics": formatted_metrics,
        }

    report = {
        "paper_id": paper_id,
        "reproduction_metadata": serialized_metadata,
        "experiments": formatted_experiments,
    }
    return json.dumps(report, indent=2)


def write_reproduction_report(
    output_path: str | Path,
    paper_id: str,
    reproduction_metadata: dict[str, Any],
    experiments_data: dict[str, dict[str, Any]],
) -> Path:
    """Generate a report and persist it to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        generate_reproduction_report(
            paper_id=paper_id,
            reproduction_metadata=reproduction_metadata,
            experiments_data=experiments_data,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
