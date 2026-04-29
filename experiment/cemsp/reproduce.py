from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject, process_nan
from model.model_utils import build_optimizer, logits_to_prediction, resolve_device
from utils.registry import get_registry
from utils.seed import seed_context


class _CemspReferenceMlp(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        layers: list[int] | None = None,
        optimizer: str = "adam",
        criterion: str = "cross_entropy",
        output_activation: str = "softmax",
        pretrained_path: str | None = None,
        save_name: str | None = None,
        **kwargs,
    ):
        self._model: torch.nn.Module
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._batch_size = int(batch_size)
        self._layers = list(layers or [30, 10])
        self._optimizer_name = str(optimizer)
        self._criterion_name = str(criterion).lower()
        self._output_activation_name = str(output_activation).lower()
        self._pretrained_path = pretrained_path
        self._save_name = save_name
        self._output_dim: int | None = None

        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._criterion_name not in {"cross_entropy", "bce"}:
            raise ValueError(
                "_CemspReferenceMlp supports criterion='cross_entropy' or 'bce' only"
            )
        if self._output_activation_name not in {"softmax", "sigmoid"}:
            raise ValueError(
                "_CemspReferenceMlp output_activation must be 'softmax' or 'sigmoid'"
            )
        if self._output_activation_name == "sigmoid" and self._criterion_name != "bce":
            raise ValueError(
                "_CemspReferenceMlp output_activation='sigmoid' requires criterion='bce'"
            )
        if self._output_activation_name == "softmax" and self._criterion_name == "bce":
            raise ValueError(
                "_CemspReferenceMlp output_activation='softmax' is incompatible with criterion='bce'"
            )

    def _build_network(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        blocks: list[torch.nn.Module] = []
        current_dim = input_dim
        for hidden_dim in self._layers:
            blocks.append(torch.nn.Linear(current_dim, hidden_dim))
            blocks.append(torch.nn.ReLU())
            current_dim = hidden_dim
        blocks.append(torch.nn.Linear(current_dim, output_dim))
        return torch.nn.Sequential(*blocks)

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for _CemspReferenceMlp.fit()")

        with seed_context(self._seed):
            X, labels, output_dim = self.extract_training_data(trainset)
            if self._output_activation_name == "sigmoid":
                if len(self.get_class_to_index()) != 2:
                    raise ValueError(
                        "_CemspReferenceMlp output_activation='sigmoid' supports binary classification only"
                    )
                network_output_dim = 1
            else:
                network_output_dim = output_dim

            input_dim = X.shape[1]
            self._output_dim = network_output_dim
            self._model = self._build_network(input_dim, network_output_dim).to(
                self._device
            )

            if self._pretrained_path is not None and Path(self._pretrained_path).exists():
                state_dict = torch.load(
                    self._pretrained_path,
                    map_location=self._device,
                )
                self._model.load_state_dict(state_dict)
                self._model.eval()
                self._is_trained = True
                return

            optimizer = build_optimizer(
                self._optimizer_name,
                self._model.parameters(),
                self._learning_rate,
            )
            X_np = X.to_numpy(dtype="float32")
            y_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)

            train_dataset = TensorDataset(
                torch.from_numpy(X_np),
                torch.from_numpy(y_np),
            )
            class_counts = np.bincount(y_np)
            if np.any(class_counts == 0):
                raise ValueError(
                    "_CemspReferenceMlp weighted sampling requires every class to appear at least once"
                )
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_np]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.float32),
                num_samples=X_np.shape[0],
                replacement=True,
            )
            train_loader = DataLoader(
                train_dataset,
                sampler=sampler,
                shuffle=False,
                batch_size=self._batch_size,
            )

            if self._criterion_name == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.BCELoss()
            criterion = criterion.to(self._device)

            self._model.train()
            for _ in tqdm(range(self._epochs), desc="cemsp-reference-mlp-fit", leave=False):
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self._device)
                    if self._criterion_name == "cross_entropy":
                        target = batch_y.to(self._device)
                    else:
                        target = batch_y.to(self._device).to(dtype=torch.float32).unsqueeze(1)

                    optimizer.zero_grad()
                    logits = self._model(batch_X)
                    loss_input = (
                        torch.sigmoid(logits)
                        if self._criterion_name == "bce"
                        else logits
                    )
                    loss = criterion(loss_input, target)
                    loss.backward()
                    optimizer.step()

            self._model.eval()
            self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"),
                dtype=torch.float32,
                device=self._device,
            )
            with torch.no_grad():
                logits = self._model(X_tensor)
                prediction = logits_to_prediction(
                    logits,
                    proba=proba,
                    output_activation=self._output_activation_name,
                )
            return prediction.detach().cpu()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            logits = self._model(X.to(self._device))
            if self._output_activation_name == "sigmoid":
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                return torch.cat([-logits, logits], dim=1)
            return logits


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _build_dataset(config: dict):
    dataset_cfg = deepcopy(config["dataset"])
    dataset_name = dataset_cfg.pop("name")
    dataset_class = get_registry("dataset")[dataset_name]
    return dataset_class(**dataset_cfg)


def _build_model(config: dict, seed: int):
    model_cfg = deepcopy(config["model"])
    model_name = model_cfg.pop("name")
    if str(model_name).lower() != "mlp":
        raise ValueError(
            "CEMSP reproduction currently expects model.name='mlp' and uses a local reference MLP implementation"
        )
    model_cfg["seed"] = int(seed)
    model_cfg.pop("sampling", None)
    return _CemspReferenceMlp(**model_cfg)


def _build_method(
    config: dict,
    target_model,
    explicit_lower_bounds: Sequence[float] | None = None,
    explicit_upper_bounds: Sequence[float] | None = None,
):
    method_cfg = deepcopy(config["method"])
    method_name = method_cfg.pop("name")
    if explicit_lower_bounds is not None:
        method_cfg["explicit_lower_bounds"] = list(explicit_lower_bounds)
    if explicit_upper_bounds is not None:
        method_cfg["explicit_upper_bounds"] = list(explicit_upper_bounds)
    method_class = get_registry("method")[method_name]
    return method_class(target_model=target_model, **method_cfg)


def _build_dataset_clone(dataset, features: pd.DataFrame, target: pd.DataFrame, flag: str):
    clone = dataset.clone()
    target_column = dataset.target_column
    combined = pd.concat([features.copy(deep=True), target.copy(deep=True)], axis=1)
    combined = combined.loc[:, [*features.columns, target_column]]
    clone.update(flag, True, df=combined)
    clone.freeze()
    return clone


def _materialize_reference_split(raw_dataset, split_seed: int):
    df = raw_dataset.snapshot()
    target_column = raw_dataset.target_column
    target = df[target_column]
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=split_seed,
        stratify=target,
    )
    train_df = train_df.copy(deep=True)
    test_df = test_df.copy(deep=True)

    trainset_raw = raw_dataset
    testset_raw = raw_dataset.clone()
    trainset_raw.update("trainset", True, df=train_df)
    testset_raw.update("testset", True, df=test_df)
    trainset_raw.freeze()
    testset_raw.freeze()
    return trainset_raw, testset_raw


def _scale_reference_split(trainset_raw, testset_raw):
    scaler = StandardScaler()
    train_x_raw = trainset_raw.get(target=False)
    test_x_raw = testset_raw.get(target=False)
    train_x = pd.DataFrame(
        scaler.fit_transform(train_x_raw),
        index=train_x_raw.index,
        columns=train_x_raw.columns,
    )
    test_x = pd.DataFrame(
        scaler.transform(test_x_raw),
        index=test_x_raw.index,
        columns=test_x_raw.columns,
    )
    train_y = trainset_raw.get(target=True)
    test_y = testset_raw.get(target=True)
    trainset = _build_dataset_clone(trainset_raw, train_x, train_y, "trainset")
    testset = _build_dataset_clone(testset_raw, test_x, test_y, "testset")
    return trainset, testset, scaler


def _compute_model_metrics(model, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu()
    predictions = probabilities.argmax(dim=1)
    target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    accuracy = float(accuracy_score(target, predictions.numpy()))
    f1 = float(f1_score(target, predictions.numpy()))
    return {"test_accuracy": accuracy, "test_f1": f1}


def _hausdorff_score(cfs1: np.ndarray, cfs2: np.ndarray) -> float:
    cfs1 = np.asarray(cfs1, dtype=np.float64)
    cfs2 = np.asarray(cfs2, dtype=np.float64)
    pairwise_distance = cdist(cfs1, cfs2)
    h_a_b = pairwise_distance.min(1).mean()
    h_b_a = pairwise_distance.min(0).mean()
    return float(max(h_a_b, h_b_a))


def _percent_range(values, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return array
    start = int(lower * array.shape[0])
    end = int(upper * array.shape[0])
    sliced = array[start:end]
    if sliced.size == 0:
        return array
    return sliced


class _ReferenceEvaluator:
    def __init__(self, dataset: np.ndarray, tp_set: np.ndarray):
        self.dataset = np.asarray(dataset, dtype=np.float64)
        self.true_positive = np.asarray(tp_set, dtype=np.float64)
        self._mads: np.ndarray | None = None

    def sparsity(self, input_x: np.ndarray, cfs: np.ndarray, precision: int = 3) -> float:
        lens = len(cfs)
        sparsity = 0.0
        for index in range(lens):
            tmp = (input_x - cfs[index]).round(precision)
            sparsity += float((tmp == 0).sum())
        return float(sparsity / (lens * len(cfs[0])))

    def average_percentile_shift(self, input_x: np.ndarray, cfs: np.ndarray) -> float:
        lens = len(cfs)
        shift = np.zeros(input_x.shape[1], dtype=np.float64)
        for index in range(lens):
            for feature_index in range(input_x.shape[1]):
                src_percentile = stats.percentileofscore(
                    self.dataset[:, feature_index],
                    float(input_x[0, feature_index]),
                )
                tgt_percentile = stats.percentileofscore(
                    self.dataset[:, feature_index],
                    float(cfs[index, feature_index]),
                )
                shift[feature_index] += abs(src_percentile - tgt_percentile)
        return float(shift.sum() / (100.0 * input_x.shape[1]))

    def proximity(self, cfs: np.ndarray) -> float:
        lens = len(cfs)
        proximity = 0.0
        for index in range(lens):
            cf = cfs[index : index + 1]
            distance = cdist(cf, self.true_positive).squeeze()
            nearest = int(np.argmin(distance))
            pivot = self.true_positive[nearest]
            pivot_distance = cdist(pivot[np.newaxis, :], self.true_positive).squeeze()
            pivot_distance[nearest] = float("inf")
            pivot_nearest = self.true_positive[int(np.argmin(pivot_distance))]
            proximity += euclidean(cf.squeeze(), pivot) / (
                euclidean(pivot, pivot_nearest) + 1e-6
            )
        return float(proximity / lens)

    def diversity(self, cfs: np.ndarray) -> float:
        k, _ = cfs.shape
        if k <= 1:
            return -1.0
        diversity = 0.0
        for left in range(k - 1):
            for right in range(left + 1, k):
                diversity += self.compute_dist(cfs[left], cfs[right])
        return float(diversity * 2.0 / (k * (k - 1)))

    def count_diversity(self, cfs: np.ndarray, precision: int = 3) -> float:
        k, d = cfs.shape
        if k <= 1:
            return -1.0
        diversity = 0.0
        for left in range(k - 1):
            for right in range(left + 1, k):
                tmp = (cfs[left] - cfs[right]).round(precision)
                diversity += float((tmp != 0).sum())
        return float(diversity * 2.0 / (k * (k - 1) * d))

    def get_mads(self) -> np.ndarray:
        if self._mads is None:
            self._mads = np.median(
                np.abs(self.dataset - np.median(self.dataset, axis=0)),
                axis=0,
            )
        return self._mads

    def compute_dist(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sum(np.multiply(np.abs(left - right), self.get_mads()), axis=0))


def _collect_reference_sets(model, trainset, testset):
    train_x = trainset.get(target=False)
    test_x = testset.get(target=False)
    train_y = trainset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    test_y = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()

    pred_test = (
        model.get_prediction(test_x, proba=False).argmax(dim=1).detach().cpu().numpy()
    )
    pred_train = (
        model.get_prediction(train_x, proba=False).argmax(dim=1).detach().cpu().numpy()
    )

    tn_idx = sorted(
        set(np.where(test_y == 0)[0]).intersection(np.where(pred_test == 0)[0])
    )
    tp_idx = sorted(
        set(np.where(train_y == 1)[0]).intersection(np.where(pred_train == 1)[0])
    )

    abnormal_test = test_x.iloc[tn_idx].to_numpy(dtype=np.float64)
    normal_train = train_x.iloc[tp_idx].to_numpy(dtype=np.float64)
    return abnormal_test, normal_train


def _run_cemsp_for_model(
    method,
    evaluator: _ReferenceEvaluator,
    abnormal_test: np.ndarray,
    desired_class: int | str,
    collect_metrics: bool,
):
    counts: list[int] = []
    cf_sets: list[np.ndarray] = []
    cf_records: list[list[dict[str, object]]] = []
    diversity_values: list[float] = []
    count_diversity_values: list[float] = []

    for input_x in tqdm(abnormal_test, desc="cemsp-reproduce", leave=False):
        input_row = input_x.reshape(1, -1)
        cfs = method._generate_counterfactuals_for_factual(input_x, desired_class)
        if cfs:
            cf_array = np.asarray(cfs, dtype=np.float64).reshape((-1, input_row.shape[1]))
        else:
            cf_array = np.empty((0, input_row.shape[1]), dtype=np.float64)

        counts.append(int(cf_array.shape[0]))
        cf_sets.append(cf_array)
        diversity_values.append(evaluator.diversity(cf_array))
        count_diversity_values.append(evaluator.count_diversity(cf_array))

        records: list[dict[str, object]] = []
        if collect_metrics:
            for cf in cf_array:
                cf_2d = cf.reshape(1, -1)
                records.append(
                    {
                        "cf": cf_2d,
                        "mask": (~np.isclose(cf, input_x, atol=1e-8, rtol=1e-8)).astype(
                            np.float64
                        ),
                        "sparsity": evaluator.sparsity(input_row, cf_2d),
                        "aps": evaluator.average_percentile_shift(input_row, cf_2d),
                        "proximity": evaluator.proximity(cf_2d),
                    }
                )
        cf_records.append(records)

    return {
        "num": counts,
        "cf_sets": cf_sets,
        "cf_records": cf_records,
        "diversity": diversity_values,
        "count_diversity": count_diversity_values,
    }


def _aggregate_figure4_metrics(primary: dict, secondary: dict) -> dict[str, float]:
    distance: list[float] = []
    proximity: list[float] = []
    sparsity: list[float] = []
    aps: list[float] = []

    for cf_records, cf_a, cf_b in zip(
        primary["cf_records"],
        primary["cf_sets"],
        secondary["cf_sets"],
        strict=False,
    ):
        for record in cf_records:
            proximity.append(float(record["proximity"]))
            sparsity.append(float(record["sparsity"]))
            aps.append(float(record["aps"]))
        if cf_a.shape[0] > 0 and cf_b.shape[0] > 0:
            distance.append(_hausdorff_score(cf_a, cf_b))

    diversity = np.asarray(primary["diversity"], dtype=np.float64)
    diversity2 = np.asarray(secondary["diversity"], dtype=np.float64)
    count_diversity = np.asarray(primary["count_diversity"], dtype=np.float64)
    count_diversity2 = np.asarray(secondary["count_diversity"], dtype=np.float64)

    diversity = diversity[diversity != -1.0]
    diversity2 = diversity2[diversity2 != -1.0]
    count_diversity = count_diversity[count_diversity != -1.0]
    count_diversity2 = count_diversity2[count_diversity2 != -1.0]

    return {
        "num_factuals": float(len(primary["num"])),
        "num_with_cf": float(sum(count > 0 for count in primary["num"])),
        "mean_num_cf": float(np.mean(primary["num"])) if primary["num"] else float("nan"),
        "median_num_cf": float(np.median(primary["num"])) if primary["num"] else float("nan"),
        "mean_sparsity": float(np.mean(_percent_range(sparsity)))
        if sparsity
        else float("nan"),
        "mean_aps": float(np.mean(_percent_range(aps))) if aps else float("nan"),
        "mean_proximity": float(np.mean(_percent_range(proximity)))
        if proximity
        else float("nan"),
        "mean_diversity": float(np.mean(_percent_range(diversity)))
        if diversity.size > 0
        else float("nan"),
        "mean_diversity2": float(np.mean(_percent_range(diversity2)))
        if diversity2.size > 0
        else float("nan"),
        "mean_inconsistency": float(np.mean(_percent_range(distance)))
        if distance
        else float("nan"),
        "mean_count_diversity": float(np.mean(_percent_range(count_diversity)))
        if count_diversity.size > 0
        else float("nan"),
        "mean_count_diversity2": float(np.mean(_percent_range(count_diversity2)))
        if count_diversity2.size > 0
        else float("nan"),
    }


def _print_metric(name: str, value: float, reference: float | None = None) -> None:
    if reference is None or not np.isfinite(reference):
        print(f"{name}: {value:.6f}")
        return
    delta = value - reference
    print(f"{name}: {value:.6f} | reference: {reference:.6f} | delta: {delta:+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./experiment/cemsp/config.yml")
    parser.add_argument("--max-factuals", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _apply_device(_load_config(config_path), device)

    reproduction_cfg = deepcopy(config["reproduction"])
    desired_class = reproduction_cfg.get("desired_class", config["method"]["desired_class"])
    model_seeds = list(reproduction_cfg.get("model_seeds", [config["model"]["seed"], int(config["model"]["seed"]) + 1]))
    if len(model_seeds) != 2:
        raise ValueError("reproduction.model_seeds must contain exactly two seeds")

    raw_dataset = _build_dataset(config)
    split_seed = int(reproduction_cfg["split_seed"])
    trainset_raw, testset_raw = _materialize_reference_split(raw_dataset, split_seed)
    trainset, testset, scaler = _scale_reference_split(trainset_raw, testset_raw)

    model_a = _build_model(config, seed=int(model_seeds[0]))
    model_a.fit(trainset)
    metrics_a = _compute_model_metrics(model_a, testset)

    model_b = _build_model(config, seed=int(model_seeds[1]))
    model_b.fit(trainset)
    metrics_b = _compute_model_metrics(model_b, testset)

    abnormal_test, normal_train = _collect_reference_sets(model_a, trainset, testset)
    if abnormal_test.shape[0] == 0:
        raise RuntimeError("No true-negative Hepatitis test instances were found")
    if normal_train.shape[0] == 0:
        raise RuntimeError("No true-positive Hepatitis train instances were found")
    if args.max_factuals is not None:
        abnormal_test = abnormal_test[: int(args.max_factuals)]

    evaluator = _ReferenceEvaluator(
        trainset.get(target=False).to_numpy(dtype=np.float64),
        normal_train.astype(np.float64),
    )

    feature_columns = trainset.get(target=False).columns
    normal_range = pd.DataFrame(
        [
            reproduction_cfg["normal_range"]["lower"],
            reproduction_cfg["normal_range"]["upper"],
        ],
        columns=feature_columns,
        dtype=np.float64,
    )
    normal_range = scaler.transform(normal_range).astype(np.float64)
    normal_range *= float(reproduction_cfg.get("normal_range_scale", 1.0))
    lower_bounds = normal_range[0]
    upper_bounds = normal_range[1]

    method_a = _build_method(
        config,
        model_a,
        explicit_lower_bounds=lower_bounds,
        explicit_upper_bounds=upper_bounds,
    )
    method_a.fit(trainset)
    primary_result = _run_cemsp_for_model(
        method_a,
        evaluator,
        abnormal_test,
        desired_class=desired_class,
        collect_metrics=True,
    )

    method_b = _build_method(
        config,
        model_b,
        explicit_lower_bounds=lower_bounds,
        explicit_upper_bounds=upper_bounds,
    )
    method_b.fit(trainset)
    secondary_result = _run_cemsp_for_model(
        method_b,
        evaluator,
        abnormal_test,
        desired_class=desired_class,
        collect_metrics=False,
    )

    aggregated = _aggregate_figure4_metrics(primary_result, secondary_result)
    reference_model = reproduction_cfg.get("reference_model", {})
    reference_metrics = reproduction_cfg.get("reference_metrics", {})

    print("CEMSP Hepatitis Figure 4 Reproduction")
    print(f"device: {device}")
    print(f"split_seed: {split_seed}")
    print(f"model_seeds: {model_seeds}")
    if "reference_source" in reproduction_cfg:
        print(f"reference_source: {reproduction_cfg['reference_source']}")
    print(f"num_abnormal_factuals: {int(aggregated['num_factuals'])}")
    if args.max_factuals is not None:
        print("subset_run: true")
        print("subset_note: reference comparisons below are for the full original setting")

    print()
    print("Model Metrics")
    _print_metric(
        "model_a_test_accuracy",
        metrics_a["test_accuracy"],
        float(reference_model.get("test_accuracy", float("nan"))),
    )
    _print_metric(
        "model_a_test_f1",
        metrics_a["test_f1"],
        float(reference_model.get("test_f1", float("nan"))),
    )
    _print_metric(
        "model_b_test_accuracy",
        metrics_b["test_accuracy"],
        float(reference_model.get("test_accuracy", float("nan"))),
    )
    _print_metric(
        "model_b_test_f1",
        metrics_b["test_f1"],
        float(reference_model.get("test_f1", float("nan"))),
    )

    print()
    print("Figure 4 HCV Metrics")
    for key in [
        "num_with_cf",
        "mean_num_cf",
        "median_num_cf",
        "mean_sparsity",
        "mean_aps",
        "mean_proximity",
        "mean_diversity",
        "mean_diversity2",
        "mean_inconsistency",
        "mean_count_diversity",
        "mean_count_diversity2",
    ]:
        _print_metric(
            key,
            float(aggregated[key]),
            float(reference_metrics.get(key, float("nan"))),
        )


if __name__ == "__main__":
    main()
