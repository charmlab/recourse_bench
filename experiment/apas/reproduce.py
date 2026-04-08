from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from gurobipy import GRB, GurobiError, Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier

from dataset.diabetes.diabetes import DiabetesDataset
from method.apas.apas import ApasMethod
from method.apas.support import BinaryNetwork, extract_binary_target_networks
from model.mlp.mlp import MlpModel

warnings.filterwarnings("ignore")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_device(configured_device: str) -> str:
    configured_device = str(configured_device).lower()
    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Configured device 'cuda' is unavailable")
    if configured_device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {configured_device}")
    return configured_device


def _make_frozen_dataset(
    template: DiabetesDataset,
    df: pd.DataFrame,
    marker: str,
) -> DiabetesDataset:
    dataset = template.clone()
    dataset.update(marker, True, df=df.copy(deep=True))
    dataset.freeze()
    return dataset


def _load_scaled_diabetes(config: dict) -> tuple[DiabetesDataset, pd.DataFrame, list[str], str]:
    dataset_cfg = config["dataset"]
    dataset = DiabetesDataset(path=dataset_cfg["path"])
    df = dataset.snapshot().dropna().reset_index(drop=True)
    target_column = dataset.target_column
    feature_columns = [column for column in df.columns if column != target_column]

    min_vals = df[feature_columns].min(axis=0)
    max_vals = df[feature_columns].max(axis=0)
    scaled_df = df.copy(deep=True)
    scaled_df.loc[:, feature_columns] = (
        scaled_df.loc[:, feature_columns] - min_vals
    ) / (max_vals - min_vals)

    dataset.update("scaled", True, df=scaled_df)
    return dataset, scaled_df, feature_columns, target_column


def _split_reference_d1_d2(
    scaled_df: pd.DataFrame,
    seed_numpy: int,
    d1_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed_numpy)
    d1_indices = np.sort(np.random.choice(range(len(scaled_df)), d1_size))
    d2_indices = np.array([index for index in range(len(scaled_df)) if index not in d1_indices])

    d1_df = pd.DataFrame(scaled_df.values[d1_indices], columns=scaled_df.columns)
    d2_df = pd.DataFrame(scaled_df.values[d2_indices], columns=scaled_df.columns)
    return d1_df, d2_df


def _build_reference_classifier(config: dict) -> MLPClassifier:
    model_cfg = config["model"]
    hidden_layers = model_cfg["hidden_layer_sizes"]
    if len(hidden_layers) == 1:
        hidden_layer_sizes = int(hidden_layers[0])
    else:
        hidden_layer_sizes = tuple(int(value) for value in hidden_layers)

    return MLPClassifier(
        learning_rate=model_cfg["learning_rate"],
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=float(model_cfg["learning_rate_init"]),
        batch_size=int(model_cfg["batch_size"]),
        max_iter=int(model_cfg["max_iter"]),
        random_state=int(config["reproduction"]["seed_model"]),
        activation=str(model_cfg["activation"]),
        solver=str(model_cfg["solver"]),
    )


def _flatten_weights_and_biases(
    clf: MLPClassifier,
    include_weights: bool = True,
    include_biases: bool = True,
) -> np.ndarray:
    values: list[np.ndarray] = []
    if include_weights:
        values.extend(np.asarray(weight).reshape(-1) for weight in clf.coefs_)
    if include_biases:
        values.extend(np.asarray(bias).reshape(-1) for bias in clf.intercepts_)
    if not values:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(values).astype(np.float64, copy=False)


def _inf_norm(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(x - y)))


def _partial_fit_clone(
    clf: MLPClassifier,
    X_update: pd.DataFrame,
    y_update: pd.Series,
) -> MLPClassifier:
    clone = copy.deepcopy(clf)
    clone.partial_fit(X_update.to_numpy(dtype=np.float64), y_update.to_numpy(dtype=np.int64))
    return clone


def _compute_delta_min(
    clf: MLPClassifier,
    X2: pd.DataFrame,
    y2: pd.Series,
    gap: float,
) -> float:
    reference = _flatten_weights_and_biases(clf)
    delta_min = -1.0
    subset_size = int(gap * len(X2))
    for seed in range(5):
        np.random.seed(seed)
        indices = np.random.choice(range(len(X2)), subset_size)
        shifted = _partial_fit_clone(clf, X2.iloc[indices], y2.iloc[indices])
        delta_min = max(
            delta_min,
            _inf_norm(reference, _flatten_weights_and_biases(shifted)),
        )
    return float(delta_min)


def _compute_delta_max_reference(
    clf: MLPClassifier,
    X2: pd.DataFrame,
    y2: pd.Series,
) -> float:
    reference = _flatten_weights_and_biases(clf)
    delta_max = -1.0
    subset_size = int(0.99 * len(X2))
    for seed in range(5):
        np.random.seed(seed)
        indices = np.random.choice(range(len(X2)), subset_size)
        shifted = _partial_fit_clone(clf, X2.iloc[indices], y2.iloc[indices])
        delta_max = max(
            delta_max,
            _inf_norm(reference, _flatten_weights_and_biases(shifted)),
        )
    return float(delta_max)


def _compute_delta_e_notebook_style(
    clf: MLPClassifier,
    retrained_clf: MLPClassifier,
) -> float:
    difference = np.abs(
        np.asarray(clf.coefs_[0], dtype=np.float64)
        - np.asarray(retrained_clf.coefs_[0], dtype=np.float64)
    )
    return float(difference.max())


def _sklearn_to_benchmark_mlp(
    clf: MLPClassifier,
    device: str,
    seed: int,
    learning_rate_init: float,
    batch_size: int,
) -> MlpModel:
    hidden_sizes = clf.hidden_layer_sizes
    if isinstance(hidden_sizes, int):
        layers = [int(hidden_sizes)]
    else:
        layers = [int(size) for size in hidden_sizes]

    model = MlpModel(
        seed=seed,
        device=device,
        epochs=1,
        learning_rate=learning_rate_init,
        batch_size=batch_size,
        layers=layers,
        optimizer="adam",
        criterion="bce",
        output_activation="sigmoid",
        save_name=None,
    )

    model._class_to_index = {
        int(class_value): index for index, class_value in enumerate(clf.classes_.tolist())
    }
    model._output_dim = 1
    model._model = model._build_model(int(clf.n_features_in_), 1).to(model._device)

    linear_layers = [
        layer for layer in model._model.modules() if isinstance(layer, torch.nn.Linear)
    ]
    for linear_layer, weights, bias in zip(linear_layers, clf.coefs_, clf.intercepts_):
        linear_layer.weight.data = torch.tensor(
            np.asarray(weights, dtype=np.float32).T,
            dtype=torch.float32,
            device=model._device,
        )
        linear_layer.bias.data = torch.tensor(
            np.asarray(bias, dtype=np.float32),
            dtype=torch.float32,
            device=model._device,
        )

    model._model.eval()
    model._is_trained = True
    return model


def _predict_label_indices(model: MlpModel, X: pd.DataFrame) -> np.ndarray:
    prediction = model.get_prediction(X, proba=True)
    return prediction.detach().cpu().numpy().argmax(axis=1)


def _linear_expr(coefficients: np.ndarray, terms: list[float | object]):
    expression = 0.0
    for coefficient, term in zip(coefficients, terms):
        value = float(coefficient)
        if isinstance(term, (int, float, np.floating)):
            expression += value * float(term)
        else:
            expression += value * term
    return expression


def _compute_interval_lower_bound(
    network: BinaryNetwork,
    point: np.ndarray,
    delta: float,
    big_m: float,
    use_biases: bool,
    seed: int | None = None,
) -> float | None:
    bias_delta = float(delta) if use_biases else 0.0
    try:
        model = Model("apas_interval_lower")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        if seed is not None:
            model.Params.Seed = int(seed)

        previous_layer: list[float | object] = [float(value) for value in point.reshape(-1)]

        for layer_index, (weights, bias) in enumerate(
            zip(network.hidden_weights, network.hidden_biases)
        ):
            current_layer: list[object] = []
            for node_index in range(weights.shape[0]):
                node = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"n_{layer_index}_{node_index}",
                )
                active = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"a_{layer_index}_{node_index}",
                )
                ub_expr = _linear_expr(
                    weights[node_index] + float(delta),
                    previous_layer,
                ) + float(bias[node_index]) + bias_delta
                lb_expr = _linear_expr(
                    weights[node_index] - float(delta),
                    previous_layer,
                ) + float(bias[node_index]) - bias_delta

                model.addConstr(
                    node <= float(big_m) * (1 - active),
                    name=f"relu_upper_{layer_index}_{node_index}",
                )
                model.addConstr(
                    ub_expr + float(big_m) * active >= node,
                    name=f"relu_active_{layer_index}_{node_index}",
                )
                model.addConstr(
                    lb_expr <= node,
                    name=f"relu_lower_{layer_index}_{node_index}",
                )
                current_layer.append(node)
            previous_layer = current_layer

        output_score = model.addVar(
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="output_score",
        )
        output_weights = network.output_weight[0]
        output_ub_expr = _linear_expr(
            output_weights + float(delta),
            previous_layer,
        ) + float(network.output_bias) + bias_delta
        output_lb_expr = _linear_expr(
            output_weights - float(delta),
            previous_layer,
        ) + float(network.output_bias) - bias_delta

        model.addConstr(output_score <= output_ub_expr, name="output_upper")
        model.addConstr(output_score >= output_lb_expr, name="output_lower")
        model.setObjective(output_score, GRB.MINIMIZE)
        model.optimize()
    except GurobiError:
        return None

    if getattr(model, "SolCount", 0) < 1:
        return None
    return float(output_score.X)


def _is_interval_robust(
    network: BinaryNetwork,
    point: np.ndarray,
    delta: float,
    big_m: float,
    use_biases: bool,
    seed: int | None = None,
) -> bool:
    lower_bound = _compute_interval_lower_bound(
        network=network,
        point=point,
        delta=delta,
        big_m=big_m,
        use_biases=use_biases,
        seed=seed,
    )
    return lower_bound is not None and lower_bound >= 0.0


def _select_reference_test_instances(
    base_model: MlpModel,
    X1: pd.DataFrame,
    desired_class: int,
    seed_numpy: int,
    num_test_instances: int,
) -> pd.DataFrame:
    np.random.seed(seed_numpy)
    predictions = _predict_label_indices(base_model, X1)
    candidate_indices = np.where(predictions == (1 - desired_class))[0]
    sampled_indices = np.random.choice(
        candidate_indices,
        min(num_test_instances, len(candidate_indices)),
    )
    return pd.DataFrame(X1.values[sampled_indices], columns=X1.columns)


def _verify_soundness(
    base_model: MlpModel,
    candidate_factuals: pd.DataFrame,
    delta_min: float,
    big_m: float,
    use_biases: bool,
) -> tuple[float, list[int]]:
    target_networks = extract_binary_target_networks(base_model)
    predictions = _predict_label_indices(base_model, candidate_factuals)

    valid_positions: list[int] = []
    for position, (_, row) in enumerate(candidate_factuals.iterrows()):
        predicted_class = int(predictions[position])
        network = target_networks[predicted_class]
        if _is_interval_robust(
            network=network,
            point=row.to_numpy(dtype=np.float64),
            delta=delta_min,
            big_m=big_m,
            use_biases=use_biases,
            seed=position,
        ):
            valid_positions.append(position)

    sound_fraction = len(valid_positions) / float(max(len(candidate_factuals), 1))
    return sound_fraction, valid_positions


def _normalised_l1_all(counterfactual: np.ndarray, factual: np.ndarray) -> float:
    return float(np.sum(np.abs(counterfactual - factual)) / counterfactual.shape[0])


def _normalised_l0(counterfactual: np.ndarray, factual: np.ndarray) -> float:
    return float(np.mean(np.abs(counterfactual - factual) >= 1e-4))


def _evaluate_counterfactuals(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    base_model: MlpModel,
    retrained_model: MlpModel,
    delta_min: float,
    desired_class: int,
    big_m: float,
    use_biases: bool,
    lof_model: LocalOutlierFactor,
) -> dict[str, float]:
    target_networks = extract_binary_target_networks(base_model)
    original_predictions = _predict_label_indices(base_model, factuals)

    found = 0
    vm1 = 0
    delta_validity = 0
    vm2 = 0
    l1_sum = 0.0
    l0_sum = 0.0
    lof_sum = 0.0

    for row_index in range(factuals.shape[0]):
        factual = factuals.iloc[row_index].to_numpy(dtype=np.float64)
        counterfactual = counterfactuals.iloc[row_index]

        if counterfactual.isna().any():
            continue

        found += 1
        counterfactual_array = counterfactual.to_numpy(dtype=np.float64)
        counterfactual_df = pd.DataFrame(
            [counterfactual_array], columns=factuals.columns
        )
        cf_prediction = int(_predict_label_indices(base_model, counterfactual_df)[0])
        original_prediction = int(original_predictions[row_index])

        if _is_interval_robust(
            network=target_networks[desired_class],
            point=counterfactual_array,
            delta=delta_min,
            big_m=big_m,
            use_biases=use_biases,
            seed=10_000 + row_index,
        ):
            delta_validity += 1

        if cf_prediction != original_prediction:
            vm1 += 1
            l1_sum += _normalised_l1_all(counterfactual_array, factual)
            l0_sum += _normalised_l0(counterfactual_array, factual)
            lof_sum += float(lof_model.predict(counterfactual_array.reshape(1, -1))[0])

            retrained_prediction = int(_predict_label_indices(retrained_model, counterfactual_df)[0])
            if retrained_prediction != original_prediction:
                vm2 += 1

    denominator = float(max(factuals.shape[0], 1))
    vm1_denominator = float(max(vm1, 1))
    return {
        "found": found / denominator,
        "vm1": vm1 / denominator,
        "vm2": vm2 / denominator,
        "delta_validity": delta_validity / denominator,
        "l1": l1_sum / vm1_denominator if vm1 > 0 else float("nan"),
        "l0": l0_sum / vm1_denominator if vm1 > 0 else float("nan"),
        "lof": lof_sum / vm1_denominator if vm1 > 0 else float("nan"),
        "num_found": float(found),
        "num_valid": float(vm1),
    }


def _compute_reference_datasets(config: dict) -> dict[str, object]:
    reproduction_cfg = config["reproduction"]
    dataset_template, scaled_df, feature_columns, target_column = _load_scaled_diabetes(config)
    d1_df, d2_df = _split_reference_d1_d2(
        scaled_df=scaled_df,
        seed_numpy=int(reproduction_cfg["seed_numpy"]),
        d1_size=int(reproduction_cfg["d1_size"]),
    )

    X1 = d1_df.loc[:, feature_columns].copy(deep=True)
    y1 = d1_df.loc[:, target_column].copy(deep=True)
    X2 = d2_df.loc[:, feature_columns].copy(deep=True)
    y2 = d2_df.loc[:, target_column].copy(deep=True)

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1,
        y1,
        stratify=y1,
        test_size=float(reproduction_cfg["d1_train_test_split"]),
        shuffle=True,
        random_state=int(reproduction_cfg["seed_model"]),
    )

    full_dataset = _make_frozen_dataset(dataset_template, scaled_df, "fullset")
    d1_train_df = pd.concat([X1_train, y1_train], axis=1).loc[:, scaled_df.columns]
    d1_train_dataset = _make_frozen_dataset(dataset_template, d1_train_df, "trainset")

    return {
        "dataset_template": dataset_template,
        "scaled_df": scaled_df,
        "full_dataset": full_dataset,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "X1": X1.reset_index(drop=True),
        "y1": y1.reset_index(drop=True),
        "X2": X2.reset_index(drop=True),
        "y2": y2.reset_index(drop=True),
        "X1_train": X1_train.reset_index(drop=True),
        "y1_train": y1_train.reset_index(drop=True),
        "X1_test": X1_test.reset_index(drop=True),
        "y1_test": y1_test.reset_index(drop=True),
        "d1_train_dataset": d1_train_dataset,
    }


def _compare_against_targets(results: dict[str, float], targets: dict[str, float]) -> dict[str, float]:
    comparison: dict[str, float] = {}
    for key, target_value in targets.items():
        if key not in results:
            continue
        comparison[f"{key}_abs_diff"] = float(abs(results[key] - float(target_value)))
    return comparison


def run_reproduction(config: dict) -> dict[str, object]:
    device = _resolve_device(config["model"]["device"])
    datasets = _compute_reference_datasets(config)

    sklearn_base = _build_reference_classifier(config)
    sklearn_base.fit(
        datasets["X1_train"].to_numpy(dtype=np.float64),
        datasets["y1_train"].to_numpy(dtype=np.int64),
    )

    delta_min = _compute_delta_min(
        clf=sklearn_base,
        X2=datasets["X2"],
        y2=datasets["y2"],
        gap=float(config["reproduction"]["gap"]),
    )
    delta_max_reference = _compute_delta_max_reference(
        clf=sklearn_base,
        X2=datasets["X2"],
        y2=datasets["y2"],
    )
    sklearn_retrained = _partial_fit_clone(
        sklearn_base,
        datasets["X2"],
        datasets["y2"],
    )
    delta_e = _compute_delta_e_notebook_style(sklearn_base, sklearn_retrained)

    base_model = _sklearn_to_benchmark_mlp(
        clf=sklearn_base,
        device=device,
        seed=int(config["reproduction"]["seed_model"]),
        learning_rate_init=float(config["model"]["learning_rate_init"]),
        batch_size=int(config["model"]["batch_size"]),
    )
    retrained_model = _sklearn_to_benchmark_mlp(
        clf=sklearn_retrained,
        device=device,
        seed=int(config["reproduction"]["seed_model"]),
        learning_rate_init=float(config["model"]["learning_rate_init"]),
        batch_size=int(config["model"]["batch_size"]),
    )

    desired_class = int(config["apas"]["desired_class"])
    candidate_factuals = _select_reference_test_instances(
        base_model=base_model,
        X1=datasets["X1"],
        desired_class=desired_class,
        seed_numpy=int(config["reproduction"]["seed_numpy"]),
        num_test_instances=int(config["reproduction"]["num_test_instances"]),
    )
    sound_fraction, sound_positions = _verify_soundness(
        base_model=base_model,
        candidate_factuals=candidate_factuals,
        delta_min=delta_min,
        big_m=float(config["apas"]["big_m"]),
        use_biases=bool(config["apas"]["use_biases"]),
    )
    num_sound_instances = int(config["reproduction"]["num_sound_instances"])
    final_positions = sound_positions[:num_sound_instances]
    if len(final_positions) < num_sound_instances:
        raise ValueError(
            f"Expected at least {num_sound_instances} sound factuals, found {len(final_positions)}"
        )
    factuals = candidate_factuals.iloc[final_positions].reset_index(drop=True)

    apas_delta = delta_min if str(config["apas"]["delta_source"]).lower() == "delta_min" else float(config["apas"]["delta"])
    apas_method = ApasMethod(
        target_model=base_model,
        seed=int(config["reproduction"]["seed_model"]),
        device=device,
        desired_class=desired_class,
        delta=apas_delta,
        alpha=float(config["apas"]["alpha"]),
        r=float(config["apas"]["r"]),
        delta_init=float(config["apas"]["delta_init"]),
        num_concretizations=int(config["apas"]["num_concretizations"]),
        eps_init=float(config["apas"]["eps_init"]),
        eps_step=float(config["apas"]["eps_step"]),
        max_iter=int(config["apas"]["max_iter"]),
        big_m=float(config["apas"]["big_m"]),
        use_biases=bool(config["apas"]["use_biases"]),
        rounding=config["apas"]["rounding"],
    )
    apas_method.fit(datasets["full_dataset"])
    counterfactuals = apas_method.get_counterfactuals(factuals)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof_model.fit(datasets["X1"].to_numpy(dtype=np.float64))

    metric_results = _evaluate_counterfactuals(
        factuals=factuals,
        counterfactuals=counterfactuals,
        base_model=base_model,
        retrained_model=retrained_model,
        delta_min=delta_min,
        desired_class=desired_class,
        big_m=float(config["apas"]["big_m"]),
        use_biases=bool(config["apas"]["use_biases"]),
        lof_model=lof_model,
    )

    results = {
        "device": device,
        "d1_size": int(datasets["X1"].shape[0]),
        "d2_size": int(datasets["X2"].shape[0]),
        "candidate_factuals": int(candidate_factuals.shape[0]),
        "sound_fraction": float(sound_fraction),
        "sound_count": int(len(sound_positions)),
        "evaluated_factuals": int(factuals.shape[0]),
        "delta": float(delta_min),
        "delta_min": float(delta_min),
        "delta_max": float(delta_max_reference),
        "delta_max_reference": float(delta_max_reference),
        "delta_e": float(delta_e),
        **metric_results,
    }

    comparison = {
        "notebook": _compare_against_targets(results, config["targets"]["notebook"]),
        "paper": _compare_against_targets(results, config["targets"]["paper"]),
    }
    return {
        "results": results,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./experiment/apas/config.yml",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _load_config(config_path)
    output = run_reproduction(config)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
