from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ThresholdSummary:
    threshold: float
    passed: int
    total: int

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100.0


@dataclass(frozen=True)
class MethodResult:
    method: str
    report_path: Path | None
    source_script: str | None
    experiment_count: int
    comparable_metrics: int
    threshold_summaries: list[ThresholdSummary]
    status: str
    reason: str | None = None


def discover_reproduce_methods() -> list[str]:
    methods: list[str] = []
    for path in EXPERIMENT_DIR.glob("*/test_*_reproduce.py"):
        method = path.parent.name
        if path.name == f"test_{method}_reproduce.py":
            methods.append(method)
    return sorted(set(methods))


def validate_methods(selected_methods: list[str], available_methods: list[str]) -> list[str]:
    invalid_methods = sorted(set(selected_methods) - set(available_methods))
    if invalid_methods:
        available = ", ".join(available_methods)
        invalid = ", ".join(invalid_methods)
        raise SystemExit(
            f"Unknown reproduce method(s): {invalid}\nAvailable methods: {available}"
        )
    return selected_methods


def get_report_path(method: str) -> Path:
    return EXPERIMENT_DIR / method / "reproduction_report.json"


def parse_threshold_string(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise SystemExit(
            "Invalid --thresholds value. Use a comma-separated list like 0.05,0.1,0.2."
        )

    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(f"Invalid --thresholds value: {value}") from exc


def parse_thresholds(single_values: list[float] | None, grouped_values: list[str] | None) -> list[float]:
    values = list(single_values or [])
    for group in grouped_values or []:
        values.extend(parse_threshold_string(group))

    if not values:
        raise SystemExit("At least one --threshold or --thresholds value is required.")

    thresholds = sorted(set(values))
    for threshold in thresholds:
        if threshold < 0:
            raise SystemExit(f"Threshold values must be non-negative. Got: {threshold}")
    return thresholds


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_report(report_path: Path) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON object must be a dictionary")
    return data


def extract_standard_comparable_deltas(report: dict[str, Any]) -> tuple[list[float], int]:
    experiments = report.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError("missing or invalid 'experiments' object")

    comparable_deltas: list[float] = []

    for experiment_payload in experiments.values():
        if not isinstance(experiment_payload, dict):
            continue
        metrics = experiment_payload.get("metrics")
        if not isinstance(metrics, dict):
            continue

        for metric_payload in metrics.values():
            if not isinstance(metric_payload, dict):
                continue

            original = metric_payload.get("original")
            delta = metric_payload.get("delta")

            if original is None or delta is None:
                continue

            if not is_number(original) or not is_number(delta):
                continue

            comparable_deltas.append(float(delta))

    return comparable_deltas, len(experiments)


def extract_cogs_comparable_deltas(report: dict[str, Any]) -> tuple[list[float], int]:
    reproduced = report.get("reproduced")
    if not isinstance(reproduced, dict):
        raise ValueError("missing or invalid 'reproduced' object")

    comparable_deltas: list[float] = []
    experiment_count = 0

    for table_payload in reproduced.values():
        if not isinstance(table_payload, dict):
            continue
        for model_payload in table_payload.values():
            if not isinstance(model_payload, dict):
                continue
            for metric_payload in model_payload.values():
                if not isinstance(metric_payload, dict):
                    continue
                for threshold_payload in metric_payload.values():
                    if not isinstance(threshold_payload, dict):
                        continue
                    for dataset_payload in threshold_payload.values():
                        if not isinstance(dataset_payload, dict):
                            continue
                        delta_payload = dataset_payload.get("delta")
                        if not isinstance(delta_payload, dict):
                            continue

                        experiment_count += 1
                        for delta_value in delta_payload.values():
                            if is_number(delta_value):
                                comparable_deltas.append(float(delta_value))

    if experiment_count == 0:
        raise ValueError("no comparable delta entries found in 'reproduced' object")

    return comparable_deltas, experiment_count


def extract_comparable_deltas(report: dict[str, Any]) -> tuple[list[float], int]:
    if isinstance(report.get("experiments"), dict):
        return extract_standard_comparable_deltas(report)

    if isinstance(report.get("reproduced"), dict):
        return extract_cogs_comparable_deltas(report)

    raise ValueError("report did not match a supported reproduction report schema")


def summarize_method(method: str, thresholds: list[float]) -> MethodResult:
    report_path = get_report_path(method)
    source_script = f"test_{method}_reproduce.py"

    if not report_path.exists():
        return MethodResult(
            method=method,
            report_path=None,
            source_script=source_script,
            experiment_count=0,
            comparable_metrics=0,
            threshold_summaries=[],
            status="missing_report",
            reason="reproduction_report.json not found",
        )

    try:
        report = load_report(report_path)
        comparable_deltas, experiment_count = extract_comparable_deltas(report)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        metadata = {}
        if "report" in locals():
            raw_metadata = report.get("reproduction_metadata")
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata
        return MethodResult(
            method=method,
            report_path=report_path,
            source_script=metadata.get("source_script", source_script),
            experiment_count=0,
            comparable_metrics=0,
            threshold_summaries=[],
            status="invalid_report",
            reason=str(exc),
        )

    metadata = report.get("reproduction_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    if not comparable_deltas:
        return MethodResult(
            method=method,
            report_path=report_path,
            source_script=metadata.get("source_script", source_script),
            experiment_count=experiment_count,
            comparable_metrics=0,
            threshold_summaries=[
                ThresholdSummary(threshold=value, passed=0, total=0) for value in thresholds
            ],
            status="no_comparable_metrics",
            reason="no metrics had both non-null original and delta values",
        )

    threshold_summaries = [
        ThresholdSummary(
            threshold=threshold,
            passed=sum(delta <= threshold for delta in comparable_deltas),
            total=len(comparable_deltas),
        )
        for threshold in thresholds
    ]

    return MethodResult(
        method=method,
        report_path=report_path,
        source_script=metadata.get("source_script", source_script),
        experiment_count=experiment_count,
        comparable_metrics=len(comparable_deltas),
        threshold_summaries=threshold_summaries,
        status="ok",
    )


def print_summary(results: list[MethodResult], thresholds: list[float]) -> None:
    header = ["Method", "Comparable", "Experiments", *[f"<={threshold:g}" for threshold in thresholds]]
    rows: list[list[str]] = []

    for result in results:
        if result.status == "ok":
            row = [
                result.method,
                str(result.comparable_metrics),
                str(result.experiment_count),
            ]
            for summary in result.threshold_summaries:
                row.append(f"{summary.passed}/{summary.total} {summary.percentage:.1f}%")
            rows.append(row)

    if rows:
        widths = [
            max(len(header[index]), *(len(row[index]) for row in rows))
            for index in range(len(header))
        ]
        print(" ".join(header[index].ljust(widths[index]) for index in range(len(header))))
        print(" ".join("-" * widths[index] for index in range(len(header))))
        for row in rows:
            print(" ".join(row[index].ljust(widths[index]) for index in range(len(row))))
    else:
        print("No valid reports with comparable metrics were found.")

    skipped = [result for result in results if result.status != "ok"]
    if skipped:
        print("\nOther statuses")
        for result in skipped:
            reason = f" ({result.reason})" if result.reason else ""
            print(f"- {result.method}: {result.status}{reason}")


def build_json_payload(results: list[MethodResult], thresholds: list[float]) -> dict[str, Any]:
    return {
        "thresholds": thresholds,
        "methods": [
            {
                "method": result.method,
                "status": result.status,
                "reason": result.reason,
                "report_path": str(result.report_path) if result.report_path else None,
                "source_script": result.source_script,
                "experiment_count": result.experiment_count,
                "comparable_metrics": result.comparable_metrics,
                "thresholds": [
                    {
                        "threshold": summary.threshold,
                        "passed": summary.passed,
                        "total": summary.total,
                        "percentage": summary.percentage,
                    }
                    for summary in result.threshold_summaries
                ],
            }
            for result in results
        ],
    }


def write_json_output(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    available_methods = discover_reproduce_methods()

    parser = argparse.ArgumentParser(
        description="Summarize reproduction reports by delta threshold pass rates."
    )
    parser.add_argument(
        "methods",
        nargs="*",
        help="Method names to analyze. Defaults to all discovered reproduction methods.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=None,
        help="Acceptable delta threshold. Repeat to evaluate multiple thresholds.",
    )
    parser.add_argument(
        "--thresholds",
        action="append",
        default=None,
        help="Comma-separated threshold list, for example 0.05,0.1,0.2.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the sweep summary as JSON.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print the discovered reproduction methods and exit.",
    )
    args = parser.parse_args()

    if args.list_methods:
        for method in available_methods:
            print(method)
        return 0

    if not available_methods:
        print("No reproduction methods were discovered.")
        return 1

    thresholds = parse_thresholds(args.threshold, args.thresholds)
    selected_methods = (
        validate_methods(args.methods, available_methods)
        if args.methods
        else available_methods
    )

    results = [summarize_method(method, thresholds) for method in selected_methods]
    print_summary(results, thresholds)

    if args.json_out:
        payload = build_json_payload(results, thresholds)
        write_json_output(Path(args.json_out), payload)

    invalid_results = any(result.status == "invalid_report" for result in results)
    return 1 if invalid_results else 0


if __name__ == "__main__":
    raise SystemExit(main())
