from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_PATH = Path("benchmark/results/smoke_multiseed_results.csv")
DISPLAY_METRICS = [
    ("validity", "Validity"),
    ("distance_l0", "L0"),
    ("distance_l2", "L2"),
    ("knn_5", "yNN"),
    ("run_duration_seconds", "Runtime (s)"),
]


def _format_mean_variance(row: pd.Series, metric: str) -> str:
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    if mean_key not in row or pd.isna(row[mean_key]):
        return "-"

    mean = float(row[mean_key])
    std = float(row.get(std_key, float("nan")))
    variance = std * std if pd.notna(std) else float("nan")
    return f"{mean:.3f}, {variance:.3f}" if pd.notna(variance) else f"{mean:.3f}"


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=str(DEFAULT_RESULTS_PATH),
        help="Path to a multiseed benchmark CSV",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Missing results CSV: {path}")

    df = pd.read_csv(path)
    if df.empty:
        print(f"No rows found in {path}")
        return

    aggregate = df[
        (df["row_type"].astype(str) == "summary")
        & (df["summary_scope"].astype(str) == "aggregate")
        & (df["model_name"].astype(str) == "mlp")
    ].copy()

    print(f"Multiseed Benchmark Report: {path}")
    print("Model filter: mlp")
    print(f"Aggregate rows: {len(aggregate)}")

    if aggregate.empty:
        return

    _print_section("MLP Stability Table")
    table_rows: list[dict[str, object]] = []
    ordered = aggregate.sort_values(by=["dataset_name", "method_name"])
    for _, row in ordered.iterrows():
        table_row = {
            "Dataset": row["dataset_name"],
            "Method": row["method_name"],
            "Seeds": int(row.get("completed_sample_seed_count", 0)),
        }
        for metric, label in DISPLAY_METRICS:
            table_row[f"{label} (mean, var)"] = _format_mean_variance(row, metric)
        table_rows.append(table_row)

    print(pd.DataFrame(table_rows).to_string(index=False))


if __name__ == "__main__":
    main()
