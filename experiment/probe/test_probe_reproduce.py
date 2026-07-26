from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from dataset.gmc_carla.gmc_carla import GmcCarlaDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment.utils import write_reproduction_report
from experiments import Experiment
from method.probe.utils import compute_invalidation_rate, reparametrization_trick
from utils.registry import get_registry, register

if "credit_carla" not in get_registry("Dataset"):
    # The reference PROBE script uses `credit_carla`, while the repo registers
    # the matching Give Me Some Credit dataset as `gmc_carla`.
    @register("credit_carla")
    class CreditCarlaDataset(GmcCarlaDataset):
        pass

SEED = 42
TRAIN_SPLIT = 0.8
INVALIDATION_TARGET = 0.35
NOISE_VARIANCE = 0.01
INVARIANCE_EPS = 0.01
METHOD_LR = 0.001
METHOD_LAMBDA = 0.01
METHOD_Y_TARGET = [0.45, 0.55]
METHOD_N_ITER = 200
METHOD_T_MAX_MIN = 0.5
BATCH_SIZE = 20
MC_SAMPLES = 10000
SMOKE_MC_SAMPLES = 256
DEFAULT_N_CFS = 500
SMOKE_N_CFS = 32
REPORT_PATH = PROJECT_ROOT / "experiment" / "probe" / "reproduction_report.json"

PAPER_PROBE_RESULTS = {
    ("compas_carla", "linear"): {
        "ra": 1.00,
        "air_mean": 0.28,
        "air_std": 0.02,
        "ac_mean": 0.63,
        "ac_std": 0.39,
        "paper_accuracy": 0.84,
    },
    ("compas_carla", "mlp"): {
        "ra": 1.00,
        "air_mean": 0.33,
        "air_std": 0.02,
        "ac_mean": 0.80,
        "ac_std": 0.34,
        "paper_accuracy": 0.85,
    },
    ("credit_carla", "linear"): {
        "ra": 1.00,
        "air_mean": 0.24,
        "air_std": 0.01,
        "ac_mean": 0.60,
        "ac_std": 0.56,
        "paper_accuracy": 0.92,
    },
    ("credit_carla", "mlp"): {
        "ra": 1.00,
        "air_mean": 0.25,
        "air_std": 0.03,
        "ac_mean": 0.47,
        "ac_std": 0.21,
        "paper_accuracy": 0.93,
    },
    ("adult_carla", "linear"): {
        "ra": 1.00,
        "air_mean": 0.34,
        "air_std": 0.02,
        "ac_mean": 1.56,
        "ac_std": 0.92,
        "paper_accuracy": 0.83,
    },
    ("adult_carla", "mlp"): {
        "ra": 0.99,
        "air_mean": 0.35,
        "air_std": 0.01,
        "ac_mean": 1.43,
        "ac_std": 0.49,
        "paper_accuracy": 0.85,
    },
}
REFERENCE_TRAINING = {
    ("compas_carla", "linear"): {"epochs": 25, "learning_rate": 0.002, "batch_size": 128},
    ("compas_carla", "mlp"): {"epochs": 25, "learning_rate": 0.002, "batch_size": 25, "layers": [50]},
    ("credit_carla", "linear"): {"epochs": 50, "learning_rate": 0.002, "batch_size": 2048},
    ("credit_carla", "mlp"): {"epochs": 50, "learning_rate": 0.002, "batch_size": 2048, "layers": [50]},
    ("adult_carla", "linear"): {"epochs": 100, "learning_rate": 0.002, "batch_size": 1024},
    ("adult_carla", "mlp"): {"epochs": 30, "learning_rate": 0.002, "batch_size": 1024, "layers": [50]},
}
REFERENCE_FEATURE_ORDER = {
    ("compas_carla", "linear"): [
        "age",
        "two_year_recid",
        "priors_count",
        "length_of_stay",
        "c_charge_degree_cat_F",
        "c_charge_degree_cat_M",
        "race_cat_African-American",
        "race_cat_Other",
        "sex_cat_Female",
        "sex_cat_Male",
    ],
    ("compas_carla", "mlp"): [
        "age",
        "two_year_recid",
        "priors_count",
        "length_of_stay",
        "c_charge_degree_cat_F",
        "c_charge_degree_cat_M",
        "race_cat_African-American",
        "race_cat_Other",
        "sex_cat_Female",
        "sex_cat_Male",
    ],
    ("credit_carla", "linear"): [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ],
    ("credit_carla", "mlp"): [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
    ],
    ("adult_carla", "linear"): [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_cat_Non-Private",
        "workclass_cat_Private",
        "marital-status_cat_Married",
        "marital-status_cat_Non-Married",
        "occupation_cat_Managerial-Specialist",
        "occupation_cat_Other",
        "relationship_cat_Husband",
        "relationship_cat_Non-Husband",
        "race_cat_Non-White",
        "race_cat_White",
        "sex_cat_Female",
        "sex_cat_Male",
        "native-country_cat_Non-US",
        "native-country_cat_US",
    ],
    ("adult_carla", "mlp"): [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_cat_Non-Private",
        "workclass_cat_Private",
        "marital-status_cat_Married",
        "marital-status_cat_Non-Married",
        "occupation_cat_Managerial-Specialist",
        "occupation_cat_Other",
        "relationship_cat_Husband",
        "relationship_cat_Non-Husband",
        "race_cat_Non-White",
        "race_cat_White",
        "sex_cat_Female",
        "sex_cat_Male",
        "native-country_cat_Non-US",
        "native-country_cat_US",
    ],
}

@dataclass(frozen=True)
class ReproductionCase:
    dataset_name: str
    model_name: str

    @property
    def slug(self) -> str:
        return f"{self.dataset_name}_{self.model_name}"


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_target_index(class_to_index: dict[int | str, int], value: Any) -> int:
    if value in class_to_index:
        return int(class_to_index[value])
    if isinstance(value, float) and float(value).is_integer():
        return int(class_to_index[int(value)])
    if isinstance(value, np.floating) and float(value).is_integer():
        return int(class_to_index[int(value)])
    return int(class_to_index[str(value)])


def _reference_style_split(dataset, split_preprocess) -> tuple[object, object]:
    df = dataset.snapshot()
    split = split_preprocess._split
    sample = split_preprocess._sample
    seed = split_preprocess._seed

    if isinstance(split, float):
        train_df, test_df = train_test_split(
            df,
            train_size=1.0 - split,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=split,
            random_state=seed,
            shuffle=True,
        )

    if sample is not None:
        test_df = test_df.sample(n=sample, random_state=seed).copy(deep=True)
    else:
        test_df = test_df.copy(deep=True)
    train_df = train_df.copy(deep=True)

    trainset = dataset
    testset = dataset.clone()
    trainset.update("trainset", True, df=train_df)
    testset.update("testset", True, df=test_df)
    return trainset, testset


def _materialize_datasets(experiment: Experiment) -> tuple[object, object]:
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets = []
        for current_dataset in datasets:
            if preprocess_step.__class__.__name__ == "SplitPreProcess":
                transformed = _reference_style_split(current_dataset, preprocess_step)
            else:
                transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets
    return experiment._resolve_train_test(datasets)


def _compute_model_metrics(model, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu()
    prediction = probabilities.argmax(dim=1)

    y = testset.get(target=True).iloc[:, 0]
    class_to_index = model.get_class_to_index()
    encoded_target = torch.tensor(
        [_resolve_target_index(class_to_index, value) for value in y.tolist()],
        dtype=torch.long,
    )

    accuracy = float((prediction == encoded_target).to(dtype=torch.float32).mean())
    positive_index = class_to_index.get(1, max(class_to_index.values()))
    if len(set(encoded_target.tolist())) < 2:
        auc = float("nan")
    else:
        auc = float(
            roc_auc_score(
                encoded_target.numpy(),
                probabilities[:, positive_index].numpy(),
            )
        )
    return {"test_accuracy": accuracy, "test_auc": auc}


def _build_frozen_dataset(template, df: pd.DataFrame, marker: str):
    dataset = template.clone()
    dataset.update(marker, True, df=df.copy(deep=True))
    dataset.freeze()
    return dataset


def _build_experiment_config(case: ReproductionCase, device: str) -> dict[str, Any]:
    training = REFERENCE_TRAINING[(case.dataset_name, case.model_name)]
    preprocess = [
        {
            "name": "scale",
            "seed": SEED,
            "scaling": "normalize",
            "range": True,
        },
        {
            "name": "encode",
            "seed": SEED,
            "encoding": "onehot",
        },
        {
            "name": "reorder",
            "seed": SEED,
            "order": list(REFERENCE_FEATURE_ORDER[(case.dataset_name, case.model_name)]),
        },
        {
            "name": "split",
            "seed": SEED,
            "split": 1.0 - TRAIN_SPLIT,
        },
    ]
    model_cfg: dict[str, Any] = {
        "name": case.model_name,
        "seed": SEED,
        "device": device,
        "epochs": int(training["epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "batch_size": int(training["batch_size"]),
        "optimizer": "rms",
        "criterion": "cross_entropy",
        "output_activation": "softmax",
        "save_name": f"probe_reference_{case.slug}",
    }
    if case.model_name == "mlp":
        model_cfg["layers"] = list(training["layers"])

    return {
        "name": f"probe_reference_{case.slug}",
        "logger": {
            "level": "INFO",
            "path": f"./logs/probe_reference_{case.slug}.log",
        },
        "caching": {"path": "./cache/"},
        "dataset": {"name": case.dataset_name},
        "preprocess": preprocess,
        "model": model_cfg,
        "method": {
            "name": "probe",
            "seed": SEED,
            "device": device,
            "lr": METHOD_LR,
            "lambda_": METHOD_LAMBDA,
            "y_target": list(METHOD_Y_TARGET),
            "n_iter": METHOD_N_ITER,
            "t_max_min": METHOD_T_MAX_MIN,
            "norm": 1,
            "clamp": True,
            "loss_type": "BCE",
            "binary_cat_features": True,
            "noise_variance": NOISE_VARIANCE,
            "invalidation_target": INVALIDATION_TARGET,
            "inval_target_eps": INVARIANCE_EPS,
        },
        "evaluation": [],
    }


def _select_negative_factuals(model, combined_dataset, n_cfs: int | None):
    features = combined_dataset.get(target=False)
    predictions = model.get_prediction(features, proba=False).argmax(dim=1).cpu().numpy()
    class_to_index = model.get_class_to_index()
    negative_index = class_to_index.get(0, min(class_to_index.values()))
    negative_mask = predictions == negative_index
    negative_indices = features.index[negative_mask]

    if n_cfs is not None:
        negative_indices = negative_indices[:n_cfs]

    combined_df = pd.concat(
        [combined_dataset.get(target=False), combined_dataset.get(target=True)],
        axis=1,
    )
    factual_df = combined_df.loc[negative_indices].copy(deep=True)
    return _build_frozen_dataset(combined_dataset, factual_df, "selected_for_probe")


def _estimate_air(experiment: Experiment, counterfactual_rows: pd.DataFrame, monte_carlo_samples: int) -> list[float]:
    method = experiment._method
    air_values: list[float] = []
    for _, row in counterfactual_rows.iterrows():
        candidate = torch.tensor(
            row.to_numpy(dtype="float32"),
            dtype=torch.float32,
            device=method._device,
        ).reshape(1, -1)
        random_samples = reparametrization_trick(
            candidate,
            torch.tensor(
                float(method._noise_variance),
                dtype=torch.float32,
                device=method._device,
            ),
            monte_carlo_samples,
        )
        air = compute_invalidation_rate(
            lambda batch: method._predict_probabilities_tensor(batch, target_index=1),
            random_samples,
        )
        air_values.append(float(air.detach().cpu().item()))
    return air_values


def _aggregate_probe_metrics(
    experiment: Experiment,
    factuals,
    counterfactuals,
    monte_carlo_samples: int,
) -> dict[str, float]:
    factual_features, counterfactual_features, evaluation_mask, success_mask = resolve_evaluation_inputs(
        factuals,
        counterfactuals,
    )
    selected_mask = evaluation_mask & success_mask
    num_factuals = int(evaluation_mask.sum())
    num_successful = int(selected_mask.sum())
    ra = float(num_successful / num_factuals) if num_factuals > 0 else float("nan")

    if num_successful == 0:
        return {
            "ra": ra,
            "air_mean": float("nan"),
            "air_std": float("nan"),
            "ac_mean": float("nan"),
            "ac_std": float("nan"),
            "num_factuals": num_factuals,
            "num_successful": num_successful,
        }

    factual_success = factual_features.loc[selected_mask.to_numpy()]
    counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]

    air_values = _estimate_air(
        experiment=experiment,
        counterfactual_rows=counterfactual_success,
        monte_carlo_samples=monte_carlo_samples,
    )
    deltas = np.abs(
        counterfactual_success.to_numpy(dtype=np.float32)
        - factual_success.to_numpy(dtype=np.float32)
    )
    l1_values = deltas.sum(axis=1).astype(np.float64)

    return {
        "ra": ra,
        "air_mean": float(np.mean(air_values)),
        "air_std": float(np.std(air_values, ddof=0)),
        "ac_mean": float(np.mean(l1_values)),
        "ac_std": float(np.std(l1_values, ddof=0)),
        "num_factuals": num_factuals,
        "num_successful": num_successful,
    }


def _build_result_row(case: ReproductionCase, model_metrics: dict[str, float], probe_metrics: dict[str, float]) -> dict[str, Any]:
    paper = PAPER_PROBE_RESULTS[(case.dataset_name, case.model_name)]
    dataset_label = {
        "adult_carla": "adult",
        "compas_carla": "compas",
        "credit_carla": "gmc",
    }[case.dataset_name]
    return {
        "dataset": dataset_label,
        "dataset_internal": case.dataset_name,
        "model": case.model_name,
        "test_accuracy": float(model_metrics["test_accuracy"]),
        "test_auc": float(model_metrics["test_auc"]),
        "paper_accuracy": float(paper["paper_accuracy"]),
        "accuracy_delta": float(model_metrics["test_accuracy"] - paper["paper_accuracy"]),
        "ra": float(probe_metrics["ra"]),
        "paper_ra": float(paper["ra"]),
        "ra_delta": float(probe_metrics["ra"] - paper["ra"]),
        "air_mean": float(probe_metrics["air_mean"]),
        "paper_air_mean": float(paper["air_mean"]),
        "air_mean_delta": float(probe_metrics["air_mean"] - paper["air_mean"]),
        "air_std": float(probe_metrics["air_std"]),
        "paper_air_std": float(paper["air_std"]),
        "air_std_delta": float(probe_metrics["air_std"] - paper["air_std"]),
        "ac_mean": float(probe_metrics["ac_mean"]),
        "paper_ac_mean": float(paper["ac_mean"]),
        "ac_mean_delta": float(probe_metrics["ac_mean"] - paper["ac_mean"]),
        "ac_std": float(probe_metrics["ac_std"]),
        "paper_ac_std": float(paper["ac_std"]),
        "ac_std_delta": float(probe_metrics["ac_std"] - paper["ac_std"]),
        "num_factuals": int(probe_metrics["num_factuals"]),
        "num_successful": int(probe_metrics["num_successful"]),
    }


def _run_case(case: ReproductionCase, device: str, n_cfs: int | None, monte_carlo_samples: int) -> dict[str, Any]:
    config = _build_experiment_config(case, device)
    experiment = Experiment(config)
    logger = experiment._logger

    trainset, testset = _materialize_datasets(experiment)
    logger.info("Training target model for %s", case.slug)
    experiment._target_model.fit(trainset)
    model_metrics = _compute_model_metrics(experiment._target_model, testset)

    logger.info("Training PROBE for %s", case.slug)
    experiment._method.fit(trainset)

    combined_df = pd.concat(
        [
            pd.concat([trainset.get(target=False), trainset.get(target=True)], axis=1),
            pd.concat([testset.get(target=False), testset.get(target=True)], axis=1),
        ],
        axis=0,
    )
    combined = _build_frozen_dataset(trainset, combined_df, "reference_source")
    factuals = _select_negative_factuals(experiment._target_model, combined, n_cfs)
    logger.info("Selected %d negative factuals for %s", len(factuals), case.slug)

    counterfactuals = experiment._method.predict(factuals, batch_size=BATCH_SIZE)
    probe_metrics = _aggregate_probe_metrics(
        experiment=experiment,
        factuals=factuals,
        counterfactuals=counterfactuals,
        monte_carlo_samples=monte_carlo_samples,
    )
    return {
        "case": case,
        "model_metrics": model_metrics,
        "probe_metrics": probe_metrics,
    }


def _write_reproduction_report(
    output_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    return write_reproduction_report(
        output_path=output_path,
        paper_id="probe_probabilistic_recourse",
        reproduction_metadata={
            "timestamp": datetime.now(timezone.utc),
            "framework_version": "1.0.0",
            "source_script": Path(__file__).name,
            "device": summary["device"],
            "datasets": summary["datasets"],
            "models": summary["models"],
            "n_cfs": summary["n_cfs"],
            "monte_carlo_samples": summary["monte_carlo_samples"],
        },
        experiments_data={
            f"{row['dataset']}_{row['model']}": {
                "configuration": {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "n_cfs": summary["n_cfs"],
                    "monte_carlo_samples": summary["monte_carlo_samples"],
                },
                "metrics": {
                    "test_accuracy": {
                        "original": row["paper_accuracy"],
                        "reproduced": row["test_accuracy"],
                    },
                    "test_auc": {
                        "original": None,
                        "reproduced": row["test_auc"],
                    },
                    "ra": {
                        "original": row["paper_ra"],
                        "reproduced": row["ra"],
                    },
                    "air_mean": {
                        "original": row["paper_air_mean"],
                        "reproduced": row["air_mean"],
                    },
                    "air_std": {
                        "original": row["paper_air_std"],
                        "reproduced": row["air_std"],
                    },
                    "ac_mean": {
                        "original": row["paper_ac_mean"],
                        "reproduced": row["ac_mean"],
                    },
                    "ac_std": {
                        "original": row["paper_ac_std"],
                        "reproduced": row["ac_std"],
                    },
                },
            }
            for row in rows
        },
    )


def run_reproduction(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    n_cfs: int | None = DEFAULT_N_CFS,
    monte_carlo_samples: int = MC_SAMPLES,
    device: str | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    selected_datasets = datasets or ["adult_carla", "compas_carla", "credit_carla"]
    selected_models = models or ["linear", "mlp"]
    cases = [
        ReproductionCase(dataset_name=dataset_name, model_name=model_name)
        for dataset_name in selected_datasets
        for model_name in selected_models
    ]

    rows: list[dict[str, Any]] = []
    for case in cases:
        result = _run_case(
            case=case,
            device=resolved_device,
            n_cfs=n_cfs,
            monte_carlo_samples=monte_carlo_samples,
        )
        rows.append(
            _build_result_row(
                case=case,
                model_metrics=result["model_metrics"],
                probe_metrics=result["probe_metrics"],
            )
        )

    frame = pd.DataFrame(rows)
    summary = {
        "device": resolved_device,
        "datasets": selected_datasets,
        "models": selected_models,
        "n_cfs": n_cfs,
        "monte_carlo_samples": monte_carlo_samples,
        "notes": [
            "This reproduction now covers adult_carla, compas_carla, and credit_carla.",
            "Training defaults follow the original reference experiment flow for wachter_rip/PROBE, not the hardcoded Table 3 values.",
        ],
    }
    return {"rows": rows, "frame": frame, "summary": summary}

@pytest.mark.slow
def test_reproduce() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["adult_carla", "compas_carla", "credit_carla"],
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["linear", "mlp"],
        default=None,
    )
    parser.add_argument(
        "--n-cfs",
        type=int,
        default=DEFAULT_N_CFS,
        help="Number of negative factuals to evaluate from the full predicted-negative pool.",
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=MC_SAMPLES,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
    )
    args = parser.parse_args()

    n_cfs = args.n_cfs
    mc_samples = args.monte_carlo_samples
    if args.smoke:
        n_cfs = min(int(n_cfs), SMOKE_N_CFS)
        mc_samples = min(int(mc_samples), SMOKE_MC_SAMPLES)

    output = run_reproduction(
        datasets=args.datasets,
        models=args.models,
        n_cfs=n_cfs,
        monte_carlo_samples=mc_samples,
        device=args.device,
    )
    report_path = _write_reproduction_report(
        REPORT_PATH,
        output["rows"],
        output["summary"],
    )
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(
        output["frame"].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"reproduction_report_path: {report_path}")


if __name__ == "__main__":
    test_reproduce()
