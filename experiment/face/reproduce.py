from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiment import Experiment

SEED = 482
DESIRED_CLASS = 1
CACHE_DIR = PROJECT_ROOT / "cache" / "face"
SUMMARY_CSV_PATH = CACHE_DIR / "synthetic_face_summary.csv"
SUMMARY_JSON_PATH = CACHE_DIR / "synthetic_face_summary.json"
# Approximate original-space query point (x1 ~= 0, x2 ~= 7) after min-max normalization.
TRIPTYCH_QUERY_POINT = (0.1465, 0.7269)

KDE_RUNS = [
    {"label": "kde_eps_0.25", "mode": "kde", "epsilon": 0.25, "td": 0.001},
    {"label": "kde_eps_0.50", "mode": "kde", "epsilon": 0.50, "td": 0.001},
    {"label": "kde_eps_2.00", "mode": "kde", "epsilon": 2.00, "td": 0.001},
]
EPSILON_RUNS = [
    {"label": "epsilon_eps_0.25", "mode": "epsilon", "epsilon": 0.25, "td": None},
    {"label": "epsilon_eps_0.50", "mode": "epsilon", "epsilon": 0.50, "td": None},
    {"label": "epsilon_eps_1.00", "mode": "epsilon", "epsilon": 1.00, "td": None},
]
KNN_RUNS = [
    {"label": "knn_k_2_eps_0.25", "mode": "knn", "k": 2, "epsilon": 0.25, "td": None},
    {"label": "knn_k_4_eps_0.35", "mode": "knn", "k": 4, "epsilon": 0.35, "td": None},
    {"label": "knn_k_10_eps_0.80", "mode": "knn", "k": 10, "epsilon": 0.80, "td": None},
]
RUN_GROUPS = {
    "kde": KDE_RUNS,
    "epsilon": EPSILON_RUNS,
    "knn": KNN_RUNS,
}


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_config(run_cfg: dict) -> dict:
    method_cfg = {
        "name": "face",
        "seed": SEED,
        "device": _resolve_device(),
        "desired_class": DESIRED_CLASS,
        "mode": run_cfg["mode"],
        "epsilon": float(run_cfg["epsilon"]),
        "prediction_threshold": 0.75,
        "density_threshold": run_cfg.get("td"),
        "weight_function": "negative_log",
        "kde_bandwidth": 0.5,
        "kde_kernel": "tophat",
        "store_top_k_paths": 5,
    }
    if run_cfg["mode"] == "knn":
        method_cfg["k_neighbors"] = int(run_cfg["k"])

    return {
        "name": f"synthetic_face_{run_cfg['label']}",
        "logger": {
            "level": "INFO",
            "path": f"./logs/synthetic_face_{run_cfg['label']}.log",
        },
        "caching": {"path": "./cache/"},
        "dataset": {"name": "synthetic_face"},
        "preprocess": [
            {
                "name": "scale",
                "seed": SEED,
                "scaling": "normalize",
                "range": True,
            }
        ],
        "model": {
            "name": "mlp",
            "seed": SEED,
            "device": _resolve_device(),
            "epochs": 320,
            "learning_rate": 0.01,
            "batch_size": 16,
            "layers": [10, 10],
            "optimizer": "adam",
            "criterion": "cross_entropy",
            "output_activation": "softmax",
            "save_name": None,
        },
        "method": method_cfg,
        "evaluation": [
            {"name": "validity"},
            {"name": "distance", "metrics": ["l2"]},
        ],
    }


def _materialize_datasets(experiment: Experiment) -> tuple[object, object]:
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets
    return experiment._resolve_train_test(datasets)


def _build_filtered_dataset(dataset, keep_mask: pd.Series, marker: str):
    filtered = dataset.clone()
    target_column = dataset.target_column
    combined = pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    filtered_df = combined.loc[keep_mask].copy(deep=True)
    filtered_df = filtered_df.loc[
        :, [*dataset.get(target=False).columns, target_column]
    ]
    filtered.update(marker, True, df=filtered_df)
    filtered.freeze()
    return filtered


def _select_negative_factuals(model, dataset):
    predictions = model.predict(dataset).argmax(dim=1).detach().cpu().numpy()
    keep_mask = pd.Series(
        predictions != model.get_class_to_index()[DESIRED_CLASS],
        index=dataset.get(target=False).index,
        dtype=bool,
    )
    return _build_filtered_dataset(dataset, keep_mask, "testset")


def _compute_model_accuracy(model, dataset) -> float:
    prediction = model.predict(dataset).argmax(dim=1).detach().cpu().numpy()
    target = dataset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    class_to_index = model.get_class_to_index()
    encoded_target = [class_to_index[int(value)] for value in target.tolist()]
    return float((prediction == encoded_target).mean())


def _compute_face_metrics(counterfactuals) -> dict[str, float]:
    evaluation_mask = pd.Series(True, index=counterfactuals.get(target=False).index)
    if hasattr(counterfactuals, "evaluation_filter"):
        evaluation_mask = (
            counterfactuals.attr("evaluation_filter").iloc[:, 0].astype(bool)
        )

    path_found = counterfactuals.attr("face_path_found").iloc[:, 0].astype(bool)
    path_cost = counterfactuals.attr("face_path_cost").iloc[:, 0].astype(float)
    path_hops = counterfactuals.attr("face_path_hops").iloc[:, 0].astype(float)
    endpoint_density = (
        counterfactuals.attr("face_endpoint_density").iloc[:, 0].astype(float)
    )
    endpoint_confidence = (
        counterfactuals.attr("face_endpoint_confidence").iloc[:, 0].astype(float)
    )

    denominator = int(evaluation_mask.sum())
    selected = evaluation_mask & path_found
    if denominator == 0:
        reachability = float("nan")
    else:
        reachability = float(
            path_found.loc[evaluation_mask.index][evaluation_mask].mean()
        )

    metrics = {
        "reachability": reachability,
        "path_cost_mean": (
            float(path_cost.loc[selected].mean())
            if bool(selected.any())
            else float("nan")
        ),
        "path_hops_mean": (
            float(path_hops.loc[selected].mean())
            if bool(selected.any())
            else float("nan")
        ),
        "endpoint_density_mean": (
            float(endpoint_density.loc[selected].mean())
            if bool(selected.any())
            else float("nan")
        ),
        "endpoint_confidence_mean": (
            float(endpoint_confidence.loc[selected].mean())
            if bool(selected.any())
            else float("nan")
        ),
    }
    return metrics


def _build_query_dataset(dataset, x1: float, x2: float, y: int = 0):
    query_dataset = dataset.clone()
    query_df = pd.DataFrame(
        [{"x1": float(x1), "x2": float(x2), dataset.target_column: int(y)}]
    )
    query_dataset.update("testset", True, df=query_df)
    query_dataset.freeze()
    return query_dataset


def _build_triptych_query(
    run_group_results: list[dict],
) -> tuple[tuple[float, float], list[dict]]:
    template_dataset = run_group_results[0]["testset"]
    model = run_group_results[0]["experiment"]._target_model
    desired_index = model.get_class_to_index()[DESIRED_CLASS]
    query_dataset = _build_query_dataset(
        template_dataset,
        x1=float(TRIPTYCH_QUERY_POINT[0]),
        x2=float(TRIPTYCH_QUERY_POINT[1]),
    )
    predicted_index = int(model.predict(query_dataset).argmax(dim=1).detach().cpu().numpy()[0])
    if predicted_index == desired_index:
        raise ValueError(
            "Fixed FACE triptych query point is already predicted as the desired class"
        )

    query_results: list[dict] = []
    for result in run_group_results:
        counterfactuals = result["experiment"]._method.predict(
            query_dataset,
            batch_size=1,
        )
        query_results.append(
            {
                "counterfactuals": counterfactuals,
                "top_paths": counterfactuals.attr("face_top_paths"),
            }
        )
    return TRIPTYCH_QUERY_POINT, query_results


def _plot_face_triptych(
    *,
    background_df: pd.DataFrame,
    query_point: tuple[float, float],
    query_results: list[dict],
    captions: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        len(query_results),
        figsize=(5 * len(query_results), 5),
        sharex=True,
        sharey=True,
    )
    if len(query_results) == 1:
        axes = [axes]

    negatives = background_df.loc[background_df["y"] == 0]
    positives = background_df.loc[background_df["y"] == 1]

    for axis, result, caption in zip(axes, query_results, captions):
        axis.scatter(negatives["x1"], negatives["x2"], c="#d95f02", s=18, alpha=0.65)
        axis.scatter(positives["x1"], positives["x2"], c="#1f77b4", s=18, alpha=0.65)

        top_paths = result["top_paths"].iloc[0]
        if top_paths:
            for rank, path in enumerate(top_paths):
                xs = [step["x1"] for step in path["trace"]]
                ys = [step["x2"] for step in path["trace"]]
                axis.plot(
                    xs,
                    ys,
                    color="#2e8b57",
                    linewidth=2.6 if rank == 0 else 1.5,
                    alpha=max(0.28, 0.9 - 0.15 * rank),
                )
            endpoint = top_paths[0]["trace"][-1]
            axis.scatter(
                [endpoint["x1"]],
                [endpoint["x2"]],
                c="#c1121f",
                s=70,
                marker="o",
            )
        else:
            axis.text(
                0.03,
                0.96,
                "no counterfactual",
                transform=axis.transAxes,
                va="top",
                fontsize=10,
            )

        axis.scatter([query_point[0]], [query_point[1]], c="#c1121f", s=70, marker="o")
        axis.set_title(caption)
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_single_experiment(run_cfg: dict) -> dict:
    experiment = Experiment(_build_config(run_cfg))
    trainset, testset = _materialize_datasets(experiment)

    experiment._target_model.fit(trainset)
    model_accuracy = _compute_model_accuracy(experiment._target_model, testset)

    experiment._method.fit(trainset)
    factuals = _select_negative_factuals(experiment._target_model, testset)
    counterfactuals = experiment._method.predict(factuals)

    evaluation_results = [
        evaluation_step.evaluate(factuals, counterfactuals)
        for evaluation_step in experiment._evaluation
    ]
    metrics = pd.concat(evaluation_results, axis=1)
    face_metrics = _compute_face_metrics(counterfactuals)

    summary_row = {
        "label": run_cfg["label"],
        "mode": run_cfg["mode"],
        "epsilon": float(run_cfg["epsilon"]),
        "k_neighbors": int(run_cfg["k"]) if "k" in run_cfg else float("nan"),
        "prediction_threshold": 0.75,
        "density_threshold": (
            float(run_cfg["td"]) if run_cfg.get("td") is not None else float("nan")
        ),
        "train_accuracy": model_accuracy,
        "validity": float(metrics.loc[0, "validity"]),
        "distance_l2": float(metrics.loc[0, "distance_l2"]),
        **face_metrics,
    }
    return {
        "summary": summary_row,
        "experiment": experiment,
        "trainset": trainset,
        "testset": testset,
        "factuals": factuals,
        "counterfactuals": counterfactuals,
        "metrics": metrics,
        "path_found": counterfactuals.attr("face_path_found").iloc[:, 0].astype(bool),
        "top_paths": counterfactuals.attr("face_top_paths"),
    }


def _assert_results(
    summary_df: pd.DataFrame,
    grouped_results: dict[str, list[dict]],
    triptych_queries: dict[str, tuple[tuple[float, float], list[dict]]],
) -> None:
    assert bool((summary_df["train_accuracy"] >= 0.95).all())
    assert (
        float(
            summary_df.loc[summary_df["label"] == "kde_eps_0.50", "reachability"].iloc[
                0
            ]
        )
        > 0.0
    )
    assert (
        float(
            summary_df.loc[
                summary_df["label"] == "epsilon_eps_0.50", "reachability"
            ].iloc[0]
        )
        > 0.0
    )
    assert (
        float(
            summary_df.loc[
                summary_df["label"] == "epsilon_eps_1.00", "reachability"
            ].iloc[0]
        )
        > 0.0
    )
    assert (
        float(
            summary_df.loc[
                summary_df["label"] == "knn_k_4_eps_0.35", "reachability"
            ].iloc[0]
        )
        > 0.0
    )
    knn_rows = summary_df.loc[summary_df["mode"] == "knn"].set_index("k_neighbors")
    assert knn_rows.loc[2, "reachability"] <= knn_rows.loc[4, "reachability"]
    assert knn_rows.loc[4, "reachability"] <= knn_rows.loc[10, "reachability"]

    for mode, results in grouped_results.items():
        _, query_results = triptych_queries[mode]
        found_sequence = [
            bool(result["counterfactuals"].attr("face_path_found").iloc[0, 0])
            for result in query_results
        ]
        assert any(found_sequence[1:])


def run_reproduction() -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    grouped_results: dict[str, list[dict]] = {}
    summary_rows: list[dict] = []
    for mode, run_group in RUN_GROUPS.items():
        grouped_results[mode] = []
        for run_cfg in run_group:
            result = _run_single_experiment(run_cfg)
            grouped_results[mode].append(result)
            summary_rows.append(result["summary"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["mode", "epsilon", "k_neighbors"], na_position="last"
    ).reset_index(drop=True)

    background_dataset = grouped_results["kde"][0]["testset"]
    background_df = pd.concat(
        [background_dataset.get(target=False), background_dataset.get(target=True)],
        axis=1,
    )

    triptych_queries: dict[str, tuple[tuple[float, float], list[dict]]] = {}
    for mode, results in grouped_results.items():
        triptych_queries[mode] = _build_triptych_query(results)
        query_point, query_results = triptych_queries[mode]
        captions = []
        if mode == "kde":
            captions = [
                "epsilon = 0.25",
                "epsilon = 0.50",
                "epsilon = 2.00",
            ]
        elif mode == "epsilon":
            captions = [
                "epsilon = 0.25",
                "epsilon = 0.50",
                "epsilon = 1.00",
            ]
        else:
            captions = [
                "k = 2, epsilon = 0.25",
                "k = 4, epsilon = 0.35",
                "k = 10, epsilon = 0.80",
            ]
        _plot_face_triptych(
            background_df=background_df,
            query_point=query_point,
            query_results=query_results,
            captions=captions,
            output_path=CACHE_DIR / f"{mode}_triptych.png",
        )

    _assert_results(summary_df, grouped_results, triptych_queries)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    SUMMARY_JSON_PATH.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return summary_df


def main() -> None:
    summary_df = run_reproduction()
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
