from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from model.model_object import ModelObject, process_nan
from model.model_utils import resolve_device
from utils.registry import register

GLOBAL_SEED = 10_000_000
SAMPLE_SIZES = [1, 2, 3]
SAMPLING_TRIALS = 10
MAD_FEATURE_WEIGHTS = {"age": 10.0, "hours_per_week": 3.0}
TARGET_MODEL_FILENAME = "adult-target-model.pth"

METHOD_GROUPS = {
    "adult-age": {
        "constraint": "age",
        "methods": {
            "BaseCVAE": "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth",
            "BaseVAE": "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth",
            "ModelApprox": "adult-margin-0.764-constraint-reg-192.0-validity_reg-29.0-epoch-25-unary-gen.pth",
            "ExampleBased": "adult-eval-case-0-supervision-limit-100-const-case-0-margin-0.084-oracle_reg-5999.0-validity_reg-159.0-epoch-50-oracle-gen.pth",
        },
    },
    "adult-age-ed": {
        "constraint": "age-ed",
        "methods": {
            "BaseCVAE": "adult-margin-0.165-validity_reg-42.0-epoch-25-base-gen.pth",
            "BaseVAE": "adult-margin-0.369-validity_reg-73.0-ae_reg-2.0-epoch-25-ae-gen.pth",
            "ModelApprox": "adult-margin-0.344-constraint-reg-87.0-validity_reg-76.0-epoch-25-unary-ed-gen.pth",
            "ExampleBased": "adult-eval-case-0-supervision-limit-100-const-case-1-margin-0.117-oracle_reg-3807.0-validity_reg-175.0-epoch-50-oracle-gen.pth",
        },
    },
}

REFERENCE_RESULTS = {
    "adult-age": {
        "BaseCVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([56.82554814, 56.93040991, 56.9399428]),
            "cont_proximity": ([-2.24059021, -2.254801, -2.24498223]),
            "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
        },
        "BaseVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([42.85033365, 43.11248808, 42.99014935]),
            "cont_proximity": ([-2.66647616, -2.66112855, -2.66558379]),
            "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
        },
        "ModelApprox": {
            "target_class_validity": ([99.5900858, 99.5900858, 99.55513187]),
            "constraint_feasibility_score": ([84.26668079, 83.60122942, 83.58960991]),
            "cont_proximity": ([-2.73294234, -2.73510212, -2.73455902]),
            "cat_proximity": ([-3.26167779, -3.26172545, -3.26151891]),
        },
        "ExampleBased": {
            "target_class_validity": ([99.52335558, 99.52812202, 99.46298062]),
            "constraint_feasibility_score": ([74.03769743, 74.18632186, 74.21309514]),
            "cont_proximity": ([-6.80110826, -6.7996033, -6.80052672]),
            "cat_proximity": ([-3.72411821, -3.7242612, -3.72443597]),
        },
    },
    "adult-age-ed": {
        "BaseCVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([57.01620591, 57.20209724, 56.80012711]),
            "cont_proximity": ([-2.25337728, -2.24797755, -2.24245525]),
            "cat_proximity": ([-3.26024786, -3.26024786, -3.26024786]),
        },
        "BaseVAE": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([42.59294566, 43.06482364, 43.02192564]),
            "cont_proximity": ([-2.6661057, -2.66556556, -2.6650458]),
            "cat_proximity": ([-3.12011439, -3.12011439, -3.12011439]),
        },
        "ModelApprox": {
            "target_class_validity": ([100.0, 100.0, 100.0]),
            "constraint_feasibility_score": ([79.5042898, 79.32793136, 79.30092151]),
            "cont_proximity": ([-2.90320554, -2.90264891, -2.90204051]),
            "cat_proximity": ([-3.22097235, -3.22054337, -3.21916111]),
        },
        "ExampleBased": {
            "target_class_validity": ([99.93326978, 99.89513823, 99.92691452]),
            "constraint_feasibility_score": ([66.35181132, 66.50929669, 66.46958292]),
            "cont_proximity": ([-3.20324281, -3.2077721, -3.20357766]),
            "cat_proximity": ([-3.5914204, -3.59394662, -3.59634573]),
        },
    },
}


class _AdultCfvaeBlackBoxNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.predict_net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.Linear(10, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_net(x)


def _load_blackbox_network(pretrained_path: str, device: str) -> nn.Module:
    checkpoint_path = Path(pretrained_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing CFVAE target model checkpoint: {checkpoint_path}"
        )

    model = _AdultCfvaeBlackBoxNetwork(input_dim=29).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@register("cfvae_adult_blackbox")
class CfvaeAdultBlackBoxModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        pretrained_path: str | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._class_to_index = {0: 0, 1: 1}
        if pretrained_path is None:
            raise ValueError("CfvaeAdultBlackBoxModel requires pretrained_path")
        self._pretrained_path = str(pretrained_path)
        self._model = _load_blackbox_network(self._pretrained_path, self._device)
        self._is_trained = True

    def fit(self, trainset):
        if trainset is None:
            raise ValueError("trainset is required for CfvaeAdultBlackBoxModel.fit()")
        feature_count = trainset.get(target=False).shape[1]
        if feature_count != 29:
            raise ValueError(
                "CfvaeAdultBlackBoxModel expects 29 encoded features, "
                f"received {feature_count}"
            )
        self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        X_tensor = torch.tensor(
            X.to_numpy(dtype="float32"), dtype=torch.float32, device=self._device
        )
        with torch.no_grad():
            probabilities = self._model(X_tensor)
        if proba:
            return probabilities.detach().cpu()
        indices = probabilities.argmax(dim=1)
        return (
            torch.nn.functional.one_hot(indices, num_classes=2)
            .to(dtype=torch.float32)
            .detach()
            .cpu()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        return self._model(X.to(self._device).float())


@register("cfvae_paper_metrics")
class CfvaePaperMetricsEvaluation(EvaluationObject):
    _CONTINUOUS_FEATURES = ("age", "hours_per_week")
    _EDUCATION_SCORE = {
        "HS-grad": 0,
        "School": 0,
        "Bachelors": 1,
        "Assoc": 1,
        "Some-college": 1,
        "Masters": 2,
        "Prof-school": 2,
        "Doctorate": 3,
    }

    def __init__(
        self,
        target_model_pretrained_path: str,
        constraint: str,
        mad_feature_weights: dict[str, float] | None = None,
        sample_sizes: list[int] | None = None,
        sampling_trials: int = SAMPLING_TRIALS,
        device: str = "cpu",
        **kwargs,
    ):
        self._device = resolve_device(device)
        self._constraint = str(constraint)
        self._sample_sizes = [int(value) for value in (sample_sizes or SAMPLE_SIZES)]
        self._sampling_trials = int(sampling_trials)
        self._mad_feature_weights = {
            str(key): float(value)
            for key, value in (mad_feature_weights or MAD_FEATURE_WEIGHTS).items()
        }
        self._target_model = _load_blackbox_network(
            str(target_model_pretrained_path),
            self._device,
        )

    def _predict_labels(self, values: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(values, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            probabilities = self._target_model(tensor)
        return probabilities.argmax(dim=1).detach().cpu().numpy()

    def _decode_features(
        self,
        dataset,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        encoding = dataset.attr("encoding")
        bounds = dataset.attr("cfvae_continuous_bounds")
        raw_feature_order = [
            column
            for column in dataset.attr("feature_order")
            if column != dataset.target_column
        ]

        decoded = pd.DataFrame(index=features.index)
        consumed_columns: set[str] = set()
        for raw_feature in raw_feature_order:
            if raw_feature in encoding:
                columns = [
                    column
                    for column in encoding[raw_feature]
                    if column in features.columns
                ]
                if not columns:
                    raise ValueError(
                        f"Missing encoded columns for feature '{raw_feature}'"
                    )
                values = features.loc[:, columns].to_numpy(dtype=np.float32)
                indices = values.argmax(axis=1)
                prefix = f"{raw_feature}_cat_"
                categories = [column[len(prefix) :] for column in columns]
                decoded[raw_feature] = [categories[index] for index in indices]
                consumed_columns.update(columns)
            else:
                lower, upper = bounds[raw_feature]
                numeric = features.loc[:, raw_feature].to_numpy(dtype=np.float32)
                decoded[raw_feature] = numeric * float(upper - lower) + float(lower)
                consumed_columns.add(raw_feature)

        unexpected = [
            column for column in features.columns if column not in consumed_columns
        ]
        if unexpected:
            raise ValueError(
                f"Unexpected encoded columns during CFVAE decode: {unexpected}"
            )
        return decoded

    def _subset_sampled_features(
        self,
        sampled_features: np.ndarray,
        columns: list[str],
        row_positions: np.ndarray,
        sample_index: int,
    ) -> pd.DataFrame:
        subset = sampled_features[sample_index, row_positions, :]
        return pd.DataFrame(subset, columns=columns)

    def _target_class_validity(
        self,
        sampled_features: np.ndarray,
        factual_predictions: np.ndarray,
        row_positions: np.ndarray,
    ) -> float:
        valid_cf_count = 0.0
        for sample_index in range(sampled_features.shape[0]):
            cf_predictions = self._predict_labels(
                sampled_features[sample_index, row_positions, :]
            )
            valid_cf_count += float(np.sum(factual_predictions != cf_predictions))
        valid_cf_count = valid_cf_count / float(sampled_features.shape[0])
        return 100.0 * valid_cf_count / float(len(row_positions))

    def _constraint_feasibility_score(
        self,
        dataset,
        sampled_features: np.ndarray,
        factual_raw: pd.DataFrame,
        factual_predictions: np.ndarray,
        columns: list[str],
        row_positions: np.ndarray,
    ) -> float:
        valid_change = 0.0
        invalid_change = 0.0
        dataset_size = 0.0

        for sample_index in range(sampled_features.shape[0]):
            cf_predictions = self._predict_labels(
                sampled_features[sample_index, row_positions, :]
            )
            decoded_cf = self._decode_features(
                dataset,
                self._subset_sampled_features(
                    sampled_features, columns, row_positions, sample_index
                ),
            )
            for row_index in range(factual_raw.shape[0]):
                if int(cf_predictions[row_index]) == int(
                    factual_predictions[row_index]
                ):
                    continue
                dataset_size += 1.0
                if self._constraint == "age":
                    if (
                        decoded_cf.iloc[row_index]["age"]
                        >= factual_raw.iloc[row_index]["age"]
                    ):
                        valid_change += 1.0
                    else:
                        invalid_change += 1.0
                    continue

                if self._constraint != "age-ed":
                    raise ValueError(
                        f"Unsupported CFVAE constraint: {self._constraint}"
                    )

                cf_education = str(decoded_cf.iloc[row_index]["education"])
                factual_education = str(factual_raw.iloc[row_index]["education"])
                cf_score = self._EDUCATION_SCORE[cf_education]
                factual_score = self._EDUCATION_SCORE[factual_education]
                cf_age = float(decoded_cf.iloc[row_index]["age"])
                factual_age = float(factual_raw.iloc[row_index]["age"])

                if cf_score < factual_score:
                    invalid_change += 1.0
                elif cf_score == factual_score:
                    if cf_age >= factual_age:
                        valid_change += 1.0
                    else:
                        invalid_change += 1.0
                else:
                    if cf_age > factual_age:
                        valid_change += 1.0
                    else:
                        invalid_change += 1.0

        valid_change = valid_change / float(sampled_features.shape[0])
        invalid_change = invalid_change / float(sampled_features.shape[0])
        dataset_size = dataset_size / float(sampled_features.shape[0])
        return (
            100.0 * valid_change / float(valid_change + invalid_change)
            if dataset_size > 0
            else float("nan")
        )

    def _cont_proximity(
        self,
        dataset,
        sampled_features: np.ndarray,
        factual_raw: pd.DataFrame,
        columns: list[str],
        row_positions: np.ndarray,
    ) -> float:
        diff_amount = 0.0
        for sample_index in range(sampled_features.shape[0]):
            decoded_cf = self._decode_features(
                dataset,
                self._subset_sampled_features(
                    sampled_features, columns, row_positions, sample_index
                ),
            )
            for column in self._CONTINUOUS_FEATURES:
                diff_amount += float(
                    np.sum(
                        np.abs(
                            factual_raw.loc[:, column].to_numpy(dtype=np.float64)
                            - decoded_cf.loc[:, column].to_numpy(dtype=np.float64)
                        )
                    )
                    / self._mad_feature_weights[column]
                )
        diff_amount = diff_amount / float(sampled_features.shape[0])
        return -1.0 * diff_amount / float(len(row_positions))

    def _cat_proximity(
        self,
        dataset,
        sampled_features: np.ndarray,
        factual_raw: pd.DataFrame,
        columns: list[str],
        row_positions: np.ndarray,
    ) -> float:
        categorical_columns = [
            column
            for column in factual_raw.columns
            if column not in self._CONTINUOUS_FEATURES
        ]
        diff_count = 0.0
        for sample_index in range(sampled_features.shape[0]):
            decoded_cf = self._decode_features(
                dataset,
                self._subset_sampled_features(
                    sampled_features, columns, row_positions, sample_index
                ),
            )
            for column in categorical_columns:
                diff_count += float(
                    np.sum(
                        factual_raw.loc[:, column].astype(str).to_numpy()
                        != decoded_cf.loc[:, column].astype(str).to_numpy()
                    )
                )
        diff_count = diff_count / float(sampled_features.shape[0])
        return -1.0 * diff_count / float(len(row_positions))

    def evaluate(self, factuals, counterfactuals) -> pd.DataFrame:
        factual_features, _, evaluation_mask, _ = resolve_evaluation_inputs(
            factuals, counterfactuals
        )
        target_series = factuals.get(target=True).iloc[:, 0].astype(int)
        label_zero_mask = target_series == 0
        base_mask = evaluation_mask & label_zero_mask
        base_positions = np.flatnonzero(base_mask.to_numpy())
        if base_positions.size == 0:
            raise ValueError("CFVAE reproduce requires at least one label-0 factual")

        base_features = factual_features.iloc[base_positions]
        predicted_labels = self._predict_labels(
            base_features.to_numpy(dtype=np.float32)
        )
        eligible_mask = predicted_labels == 0
        row_positions = base_positions[eligible_mask]
        if row_positions.size == 0:
            raise ValueError(
                "CFVAE reproduce requires at least one factual predicted as the low-income class"
            )

        selected_features = factual_features.iloc[row_positions]
        selected_columns = list(selected_features.columns)
        factual_raw = self._decode_features(factuals, selected_features)
        factual_predictions = self._predict_labels(
            selected_features.to_numpy(dtype=np.float32)
        )

        artifacts = counterfactuals.attr("cfvae_sampling_artifacts")
        result_row: dict[str, list[float]] = {
            "target_class_validity": [],
            "constraint_feasibility_score": [],
            "cont_proximity": [],
            "cat_proximity": [],
        }

        for sample_size in self._sample_sizes:
            target_trials = []
            constraint_trials = []
            cont_trials = []
            cat_trials = []

            for trial_index in range(self._sampling_trials):
                target_samples = artifacts["target_class_validity"][sample_size][
                    trial_index
                ]
                constraint_samples = artifacts["constraint_feasibility_score"][
                    sample_size
                ][trial_index]
                cont_samples = artifacts["cont_proximity"][sample_size][trial_index]
                cat_samples = artifacts["cat_proximity"][sample_size][trial_index]

                target_trials.append(
                    self._target_class_validity(
                        sampled_features=target_samples,
                        factual_predictions=factual_predictions,
                        row_positions=row_positions,
                    )
                )
                constraint_trials.append(
                    self._constraint_feasibility_score(
                        dataset=factuals,
                        sampled_features=constraint_samples,
                        factual_raw=factual_raw,
                        factual_predictions=factual_predictions,
                        columns=selected_columns,
                        row_positions=row_positions,
                    )
                )
                cont_trials.append(
                    self._cont_proximity(
                        dataset=factuals,
                        sampled_features=cont_samples,
                        factual_raw=factual_raw,
                        columns=selected_columns,
                        row_positions=row_positions,
                    )
                )
                cat_trials.append(
                    self._cat_proximity(
                        dataset=factuals,
                        sampled_features=cat_samples,
                        factual_raw=factual_raw,
                        columns=selected_columns,
                        row_positions=row_positions,
                    )
                )

            result_row["target_class_validity"].append(float(np.mean(target_trials)))
            result_row["constraint_feasibility_score"].append(
                float(np.mean(constraint_trials))
            )
            result_row["cont_proximity"].append(float(np.mean(cont_trials)))
            result_row["cat_proximity"].append(float(np.mean(cat_trials)))

        return pd.DataFrame([result_row])


def compare_results(
    results: Dict, ref: Dict, tolerance: float = 1.0, raise_on_fail: bool = False
) -> bool:
    print(f"Comparing results to reference with tolerance ±{tolerance}")
    success = True
    failure_messages: List[str] = []

    for dataset_name, ref_methods in ref.items():
        dataset_results = results.get(dataset_name)
        if dataset_results is None:
            message = f"[MISSING] Dataset `{dataset_name}` not found in results."
            print(message)
            failure_messages.append(message)
            success = False
            continue
        for method_name, ref_metrics in ref_methods.items():
            method_results = dataset_results.get(method_name)
            if method_results is None:
                message = f"[MISSING] Method `{dataset_name}/{method_name}` not found in results."
                print(message)
                failure_messages.append(message)
                success = False
                continue
            for metric_name, ref_value in ref_metrics.items():
                result_value = method_results.get(metric_name)
                if result_value is None:
                    message = f"[MISSING] Metric `{dataset_name}/{method_name}/{metric_name}` not found in results."
                    print(message)
                    failure_messages.append(message)
                    success = False
                    continue
                result_array = np.array(result_value, dtype=float)
                ref_array = np.array(ref_value, dtype=float)
                if result_array.shape != ref_array.shape:
                    message = f"[SHAPE DIFF] `{dataset_name}/{method_name}/{metric_name}` result shape {result_array.shape} != reference shape {ref_array.shape}."
                    print(message)
                    failure_messages.append(message)
                    success = False
                    continue
                diff = np.abs(result_array - ref_array)
                max_diff = float(np.max(diff)) if diff.size else 0.0
                if np.all(diff <= tolerance):
                    print(
                        f"[OK] `{dataset_name}/{method_name}/{metric_name}` max diff {max_diff:.6f}"
                    )
                else:
                    message = f"[DIFF] `{dataset_name}/{method_name}/{metric_name}` max diff {max_diff:.6f} exceeds tolerance"
                    print(message)
                    failure_messages.append(message)
                    success = False

    if raise_on_fail and failure_messages:
        raise AssertionError("\n".join(failure_messages))

    return success


def _resolve_weights_dir(weights_dir: str | Path) -> Path:
    path = Path(weights_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"weights_dir does not exist: {path}")
    return path


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_experiment_config(
    *,
    weights_dir: Path,
    dataset_name: str,
    method_name: str,
    method_checkpoint_name: str,
    constraint: str,
) -> dict:
    device = _resolve_device()
    target_model_path = (weights_dir / TARGET_MODEL_FILENAME).as_posix()
    method_checkpoint_path = (weights_dir / method_checkpoint_name).as_posix()
    experiment_name = f"cfvae_{dataset_name}_{method_name.lower()}"

    return {
        "name": experiment_name,
        "logger": {
            "level": "INFO",
            "path": f"./logs/{experiment_name}.log",
        },
        "caching": {"path": "./cache/"},
        "dataset": {"name": "adult_cfvae"},
        "preprocess": [
            {"name": "encode", "seed": None, "encoding": "onehot"},
            {"name": "scale", "seed": None, "scaling": "normalize", "range": True},
        ],
        "model": {
            "name": "cfvae_adult_blackbox",
            "seed": None,
            "device": device,
            "pretrained_path": target_model_path,
        },
        "method": {
            "name": "cfvae",
            "seed": None,
            "device": device,
            "desired_class": 1,
            "encoded_size": 10,
            "train": False,
            "pretrained_path": method_checkpoint_path,
            "store_sampling_artifacts": True,
            "sampling_sample_sizes": SAMPLE_SIZES,
            "sampling_trials": SAMPLING_TRIALS,
        },
        "evaluation": [
            {
                "name": "cfvae_paper_metrics",
                "target_model_pretrained_path": target_model_path,
                "constraint": constraint,
                "mad_feature_weights": MAD_FEATURE_WEIGHTS,
                "sample_sizes": SAMPLE_SIZES,
                "sampling_trials": SAMPLING_TRIALS,
                "device": device,
            }
        ],
    }


def _normalize_metric_row(row: dict) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key, value in row.items():
        result[key] = np.asarray(value, dtype=float).tolist()
    return result


def run_cfvae_reproduce(
    weights_dir: str | Path,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    weights_path = _resolve_weights_dir(weights_dir)
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    results: dict[str, dict[str, dict[str, list[float]]]] = {}
    for dataset_name, dataset_cfg in METHOD_GROUPS.items():
        results[dataset_name] = {}
        for method_name, checkpoint_name in dataset_cfg["methods"].items():
            config = _build_experiment_config(
                weights_dir=weights_path,
                dataset_name=dataset_name,
                method_name=method_name,
                method_checkpoint_name=checkpoint_name,
                constraint=dataset_cfg["constraint"],
            )
            experiment = Experiment(config)
            metrics = experiment.run()
            results[dataset_name][method_name] = _normalize_metric_row(
                metrics.iloc[0].to_dict()
            )
    return results


def _assert_cfvae_reproduce(weights_dir: Path, tolerance: float) -> None:
    compare_results(
        run_cfvae_reproduce(weights_dir=weights_dir),
        REFERENCE_RESULTS,
        tolerance=tolerance,
        raise_on_fail=True,
    )


@pytest.mark.parametrize("tolerance", [1.0])
def test_cfvae_reproduce(tolerance):
    weights_dir = os.environ.get("CFVAE_WEIGHTS_DIR")
    if weights_dir is None:
        raise RuntimeError(
            "CFVAE_WEIGHTS_DIR must be set when running test_cfvae_reproduce"
        )
    _assert_cfvae_reproduce(Path(weights_dir), tolerance=float(tolerance))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Directory containing the temporary CFVAE reproduction weights.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Allowed absolute difference when comparing against reference.",
    )
    args = parser.parse_args()
    _assert_cfvae_reproduce(
        weights_dir=Path(args.weights_dir),
        tolerance=float(args.tolerance),
    )


if __name__ == "__main__":
    main()
