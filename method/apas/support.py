from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from gurobipy import Env, GRB, GurobiError, Model, max_, quicksum

from dataset.dataset_object import DatasetObject
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from preprocess.preprocess_utils import dataset_has_attr, resolve_feature_metadata


class FeatureKind(str, Enum):
    DISCRETE = "discrete"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"
    BINARY = "binary"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    variable_indices: tuple[int, ...]
    mutable: bool
    actionability: str
    lower_bound: float
    upper_bound: float
    distance_scale: float


@dataclass(frozen=True)
class FeatureSchema:
    feature_names: tuple[str, ...]
    feature_specs: tuple[FeatureSpec, ...]


@dataclass(frozen=True)
class BinaryNetwork:
    input_dim: int
    hidden_weights: tuple[np.ndarray, ...]
    hidden_biases: tuple[np.ndarray, ...]
    output_weight: np.ndarray
    output_bias: float


@dataclass(frozen=True)
class ApasContext:
    feature_schema: FeatureSchema
    class_to_index: dict[int | str, int]
    target_networks: dict[int, BinaryNetwork]


@dataclass(frozen=True)
class CertificationResult:
    is_robust: bool
    robust_fraction: float
    num_concretizations: int


global_gurobi_env: Env | None = None


def get_silent_gurobi_env() -> Env:
    global global_gurobi_env
    if global_gurobi_env is not None:
        return global_gurobi_env

    env = Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()
    global_gurobi_env = env
    return env


def create_silent_gurobi_model(name: str, seed: int | None = None) -> Model:
    model = Model(name, env=get_silent_gurobi_env())
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    if seed is not None:
        model.Params.Seed = int(seed)
    return model


def ensure_supported_target_model(
    target_model: ModelObject,
    supported_types: Sequence[type[ModelObject]],
    method_name: str,
) -> None:
    if isinstance(target_model, tuple(supported_types)):
        return

    supported_names = ", ".join(cls.__name__ for cls in supported_types)
    raise TypeError(
        f"{method_name} supports target models [{supported_names}] only, "
        f"received {target_model.__class__.__name__}"
    )


def ensure_binary_mlp_target_model(target_model: ModelObject, method_name: str) -> None:
    ensure_supported_target_model(target_model, (MlpModel,), method_name)

    if not target_model._is_trained:
        raise RuntimeError(f"{method_name} requires a trained target model")

    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError(f"{method_name} supports binary classification only")

    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model has not been initialized")

    linear_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    if not linear_layers:
        raise ValueError(f"{method_name} could not find any linear layers in the target model")

    output_activation = str(
        getattr(target_model, "_output_activation_name", "softmax")
    ).lower()
    if output_activation not in {"softmax", "sigmoid"}:
        raise ValueError(
            f"{method_name} supports only softmax/sigmoid binary MLP outputs"
        )

    final_layer = linear_layers[-1]
    if output_activation == "softmax" and int(final_layer.out_features) != 2:
        raise ValueError("Softmax target models must expose exactly two output logits")
    if output_activation == "sigmoid" and int(final_layer.out_features) != 1:
        raise ValueError("Sigmoid target models must expose exactly one output logit")


def _resolve_group_kind(
    source_feature: str,
    group_columns: list[str],
    raw_feature_type: dict[str, str],
    encoded_feature_type: dict[str, str],
) -> FeatureKind:
    raw_kind = str(raw_feature_type.get(source_feature, "")).lower()

    if any("_therm_" in column for column in group_columns):
        return FeatureKind.ORDINAL
    if any("_cat_" in column for column in group_columns):
        return FeatureKind.DISCRETE

    if raw_kind == "numerical":
        return FeatureKind.CONTINUOUS
    if raw_kind == "binary":
        return FeatureKind.BINARY
    if raw_kind == "categorical":
        if len(group_columns) > 1:
            return FeatureKind.DISCRETE
        encoded_kind = str(encoded_feature_type.get(group_columns[0], raw_kind)).lower()
        if encoded_kind == "binary":
            return FeatureKind.BINARY
        raise ValueError(
            f"APAS requires categorical feature '{source_feature}' to be encoded"
        )

    encoded_kind = str(encoded_feature_type.get(group_columns[0], "")).lower()
    if encoded_kind == "numerical":
        return FeatureKind.CONTINUOUS
    if encoded_kind == "binary":
        return FeatureKind.BINARY if len(group_columns) == 1 else FeatureKind.DISCRETE
    raise ValueError(
        f"Could not resolve APAS feature kind for feature '{source_feature}'"
    )


def build_feature_schema(trainset: DatasetObject) -> FeatureSchema:
    feature_df = trainset.get(target=False)
    try:
        feature_df = feature_df.astype("float32")
    except ValueError as error:
        raise ValueError("APAS requires fully numeric input features") from error

    feature_names = list(feature_df.columns)
    encoded_feature_type, encoded_feature_mutability, encoded_feature_actionability = (
        resolve_feature_metadata(trainset)
    )
    raw_feature_type = (
        trainset.attr("raw_feature_type")
        if dataset_has_attr(trainset, "raw_feature_type")
        else encoded_feature_type
    )
    raw_feature_mutability = (
        trainset.attr("raw_feature_mutability")
        if dataset_has_attr(trainset, "raw_feature_mutability")
        else encoded_feature_mutability
    )
    raw_feature_actionability = (
        trainset.attr("raw_feature_actionability")
        if dataset_has_attr(trainset, "raw_feature_actionability")
        else encoded_feature_actionability
    )
    encoding_map = trainset.attr("encoding") if dataset_has_attr(trainset, "encoding") else {}

    processed_to_source: dict[str, str] = {}
    for source_feature, encoded_columns in encoding_map.items():
        for column in encoded_columns:
            processed_to_source[str(column)] = str(source_feature)

    grouped_columns: dict[str, list[str]] = {}
    ordered_sources: list[str] = []
    for column in feature_names:
        source_feature = processed_to_source.get(column, column)
        if source_feature not in grouped_columns:
            grouped_columns[source_feature] = []
            ordered_sources.append(source_feature)
        grouped_columns[source_feature].append(column)

    feature_specs: list[FeatureSpec] = []
    for source_feature in ordered_sources:
        group_columns = grouped_columns[source_feature]
        variable_indices = tuple(feature_names.index(column) for column in group_columns)
        kind = _resolve_group_kind(
            source_feature=source_feature,
            group_columns=group_columns,
            raw_feature_type=raw_feature_type,
            encoded_feature_type=encoded_feature_type,
        )

        mutable = bool(
            raw_feature_mutability.get(
                source_feature,
                encoded_feature_mutability[group_columns[0]],
            )
        )
        actionability = str(
            raw_feature_actionability.get(
                source_feature,
                encoded_feature_actionability[group_columns[0]],
            )
        ).lower()

        if kind == FeatureKind.CONTINUOUS:
            column = group_columns[0]
            lower_bound = float(feature_df[column].min())
            upper_bound = float(feature_df[column].max())
            distance_scale = max(upper_bound - lower_bound, 1e-6)
        elif kind == FeatureKind.ORDINAL:
            lower_bound = 0.0
            upper_bound = 1.0
            distance_scale = float(max(len(group_columns) - 1, 1))
        else:
            lower_bound = 0.0
            upper_bound = 1.0
            distance_scale = 1.0

        if actionability in {"same-or-increase", "same-or-decrease"} and kind == FeatureKind.DISCRETE:
            raise ValueError(
                f"APAS requires ordered categorical feature '{source_feature}' "
                f"to use thermometer encoding for monotone actionability"
            )

        feature_specs.append(
            FeatureSpec(
                name=str(source_feature),
                kind=kind,
                variable_indices=variable_indices,
                mutable=mutable,
                actionability=actionability,
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                distance_scale=float(distance_scale),
            )
        )

    return FeatureSchema(
        feature_names=tuple(feature_names),
        feature_specs=tuple(feature_specs),
    )


def _extract_linear_layers(target_model: MlpModel) -> list[torch.nn.Linear]:
    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model has not been initialized")

    if isinstance(model, torch.nn.Sequential):
        layers = list(model.children())
    else:
        layers = list(model.modules())[1:]

    linear_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
    if not linear_layers:
        raise RuntimeError("Could not extract any linear layers from the target model")
    return linear_layers


def extract_binary_target_networks(target_model: MlpModel) -> dict[int, BinaryNetwork]:
    linear_layers = _extract_linear_layers(target_model)
    class_to_index = target_model.get_class_to_index()
    output_activation = str(
        getattr(target_model, "_output_activation_name", "softmax")
    ).lower()

    hidden_weights: list[np.ndarray] = []
    hidden_biases: list[np.ndarray] = []
    for layer in linear_layers[:-1]:
        hidden_weights.append(
            layer.weight.detach().cpu().numpy().astype(np.float64, copy=True)
        )
        hidden_biases.append(
            layer.bias.detach().cpu().numpy().astype(np.float64, copy=True)
        )

    final_layer = linear_layers[-1]
    final_weight = final_layer.weight.detach().cpu().numpy().astype(np.float64, copy=True)
    final_bias = final_layer.bias.detach().cpu().numpy().astype(np.float64, copy=True)

    if output_activation == "softmax":
        if final_weight.shape[0] != 2:
            raise ValueError("Binary softmax target models must expose two output logits")

        target_networks: dict[int, BinaryNetwork] = {}
        for target_index in sorted(class_to_index.values()):
            other_index = 1 - int(target_index)
            target_networks[int(target_index)] = BinaryNetwork(
                input_dim=int(hidden_weights[0].shape[1] if hidden_weights else final_weight.shape[1]),
                hidden_weights=tuple(hidden_weights),
                hidden_biases=tuple(hidden_biases),
                output_weight=(final_weight[target_index] - final_weight[other_index]).reshape(1, -1),
                output_bias=float(final_bias[target_index] - final_bias[other_index]),
            )
        return target_networks

    if final_weight.shape[0] != 1:
        raise ValueError("Binary sigmoid target models must expose one output logit")

    target_networks = {}
    base_weight = final_weight[0].reshape(1, -1)
    base_bias = float(final_bias[0])
    for target_index in sorted(class_to_index.values()):
        sign = 1.0 if int(target_index) == 1 else -1.0
        target_networks[int(target_index)] = BinaryNetwork(
            input_dim=int(hidden_weights[0].shape[1] if hidden_weights else base_weight.shape[1]),
            hidden_weights=tuple(hidden_weights),
            hidden_biases=tuple(hidden_biases),
            output_weight=sign * base_weight,
            output_bias=sign * base_bias,
        )
    return target_networks


def prepare_apas_context(target_model: ModelObject, trainset: DatasetObject) -> ApasContext:
    ensure_binary_mlp_target_model(target_model, "APAS")

    feature_schema = build_feature_schema(trainset)
    target_networks = extract_binary_target_networks(target_model)

    input_dim = len(feature_schema.feature_names)
    for network in target_networks.values():
        if network.input_dim != input_dim:
            raise ValueError(
                "APAS feature dimension does not match the trained target model input dimension"
            )

    return ApasContext(
        feature_schema=feature_schema,
        class_to_index=target_model.get_class_to_index(),
        target_networks=target_networks,
    )


def _predict_label_indices(
    target_model: ModelObject,
    X: pd.DataFrame,
) -> np.ndarray:
    probabilities = target_model.get_prediction(X, proba=True)
    if isinstance(probabilities, torch.Tensor):
        return probabilities.detach().cpu().numpy().argmax(axis=1)
    return np.asarray(probabilities).argmax(axis=1)


def resolve_target_indices(
    target_model: ModelObject,
    original_prediction: np.ndarray,
    desired_class: int | str | None,
) -> np.ndarray:
    class_to_index = target_model.get_class_to_index()
    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return np.full(
            shape=original_prediction.shape,
            fill_value=int(class_to_index[desired_class]),
            dtype=np.int64,
        )

    if len(class_to_index) != 2:
        raise ValueError("desired_class=None is supported for binary classification only")
    return 1 - original_prediction.astype(np.int64, copy=False)


def validate_counterfactuals(
    target_model: ModelObject,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    desired_class: int | str | None = None,
) -> pd.DataFrame:
    if list(candidates.columns) != list(factuals.columns):
        candidates = candidates.reindex(columns=factuals.columns)
    candidates = candidates.copy(deep=True)

    if candidates.shape[0] != factuals.shape[0]:
        raise ValueError("Candidates must preserve the number of factual rows")

    valid_rows = ~candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return candidates

    original_prediction = _predict_label_indices(target_model, factuals)
    target_prediction = resolve_target_indices(
        target_model=target_model,
        original_prediction=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = _predict_label_indices(
        target_model,
        candidates.loc[valid_rows].astype("float32"),
    )
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_prediction[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates


def resolve_target_index(
    class_to_index: dict[int | str, int],
    original_prediction: int,
    desired_class: int | str | None,
) -> int:
    if desired_class is not None:
        return int(class_to_index[desired_class])
    if len(class_to_index) != 2:
        raise ValueError("desired_class=None is supported for binary classification only")
    return 1 - int(original_prediction)


def _coerce_binary_value(value: float) -> float:
    return float(1.0 if value >= 0.5 else 0.0)


def _add_input_constraints(
    model: Model,
    schema: FeatureSchema,
    factual: np.ndarray,
) -> list:
    ordered_vars: list = [None] * len(schema.feature_names)

    for feature_index, spec in enumerate(schema.feature_specs):
        variable_names = [schema.feature_names[index] for index in spec.variable_indices]
        original_values = factual[list(spec.variable_indices)]
        actionability = spec.actionability
        is_immutable = (not spec.mutable) or actionability in {"none", "same"}

        if spec.kind == FeatureKind.DISCRETE:
            group_vars = []
            for local_index, column_index in enumerate(spec.variable_indices):
                var = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_disc_{feature_index}_{local_index}",
                )
                ordered_vars[column_index] = var
                group_vars.append(var)
            model.addConstr(quicksum(group_vars) == 1.0, name=f"disc_sum_{feature_index}")
            if is_immutable:
                for var, value, column_name in zip(group_vars, original_values, variable_names):
                    model.addConstr(
                        var == _coerce_binary_value(float(value)),
                        name=f"disc_fix_{feature_index}_{column_name}",
                    )
            continue

        if spec.kind == FeatureKind.ORDINAL:
            group_vars = []
            for local_index, column_index in enumerate(spec.variable_indices):
                var = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_ord_{feature_index}_{local_index}",
                )
                ordered_vars[column_index] = var
                group_vars.append(var)

            for local_index in range(1, len(group_vars)):
                model.addConstr(
                    group_vars[local_index - 1] >= group_vars[local_index],
                    name=f"ord_monotone_{feature_index}_{local_index}",
                )
            model.addConstr(
                quicksum(group_vars) >= 1.0,
                name=f"ord_nonzero_{feature_index}",
            )
            if is_immutable:
                for var, value, column_name in zip(group_vars, original_values, variable_names):
                    model.addConstr(
                        var == _coerce_binary_value(float(value)),
                        name=f"ord_fix_{feature_index}_{column_name}",
                    )
            elif actionability == "same-or-increase":
                model.addConstr(
                    quicksum(group_vars) >= float(np.round(original_values.sum())),
                    name=f"ord_increase_{feature_index}",
                )
            elif actionability == "same-or-decrease":
                model.addConstr(
                    quicksum(group_vars) <= float(np.round(original_values.sum())),
                    name=f"ord_decrease_{feature_index}",
                )
            continue

        if spec.kind == FeatureKind.BINARY:
            column_index = spec.variable_indices[0]
            var = model.addVar(vtype=GRB.BINARY, name=f"x_bin_{feature_index}")
            ordered_vars[column_index] = var
            original_value = _coerce_binary_value(float(original_values[0]))
            if is_immutable:
                model.addConstr(var == original_value, name=f"bin_fix_{feature_index}")
            elif actionability == "same-or-increase":
                model.addConstr(var >= original_value, name=f"bin_increase_{feature_index}")
            elif actionability == "same-or-decrease":
                model.addConstr(var <= original_value, name=f"bin_decrease_{feature_index}")
            continue

        column_index = spec.variable_indices[0]
        original_value = float(original_values[0])
        lower_bound = float(min(spec.lower_bound, original_value))
        upper_bound = float(max(spec.upper_bound, original_value))
        var = model.addVar(
            lb=lower_bound,
            ub=upper_bound,
            vtype=GRB.CONTINUOUS,
            name=f"x_cont_{feature_index}",
        )
        ordered_vars[column_index] = var
        if is_immutable:
            model.addConstr(var == original_value, name=f"cont_fix_{feature_index}")
        elif actionability == "same-or-increase":
            model.addConstr(var >= original_value, name=f"cont_increase_{feature_index}")
        elif actionability == "same-or-decrease":
            model.addConstr(var <= original_value, name=f"cont_decrease_{feature_index}")

    if any(variable is None for variable in ordered_vars):
        raise RuntimeError("APAS could not create all input variables")
    return ordered_vars


def _add_hidden_constraints(
    model: Model,
    network: BinaryNetwork,
    input_vars: list,
    big_m: float,
) -> list:
    previous_layer = list(input_vars)
    for layer_index, (weight, bias) in enumerate(
        zip(network.hidden_weights, network.hidden_biases)
    ):
        current_layer = []
        for node_index in range(weight.shape[0]):
            node = model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"n_{layer_index}_{node_index}",
            )
            active = model.addVar(
                vtype=GRB.BINARY,
                name=f"a_{layer_index}_{node_index}",
            )
            expression = quicksum(
                float(weight[node_index, input_index]) * previous_layer[input_index]
                for input_index in range(weight.shape[1])
            ) + float(bias[node_index])

            model.addConstr(
                node <= big_m * (1 - active),
                name=f"relu_upper_{layer_index}_{node_index}",
            )
            model.addConstr(
                expression + big_m * active >= node,
                name=f"relu_active_{layer_index}_{node_index}",
            )
            model.addConstr(
                expression <= node,
                name=f"relu_exact_{layer_index}_{node_index}",
            )
            current_layer.append(node)
        previous_layer = current_layer
    return previous_layer


def _add_output_constraint(
    model: Model,
    network: BinaryNetwork,
    previous_layer: list,
    eps: float,
):
    score = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="output_score")
    expression = quicksum(
        float(network.output_weight[0, input_index]) * previous_layer[input_index]
        for input_index in range(network.output_weight.shape[1])
    ) + float(network.output_bias)
    model.addConstr(score == expression, name="output_exact")
    model.addConstr(score >= float(eps), name="output_margin")
    return score


def _set_l1_objective(
    model: Model,
    schema: FeatureSchema,
    input_vars: list,
    factual: np.ndarray,
) -> None:
    feature_costs = []
    for feature_index, spec in enumerate(schema.feature_specs):
        feature_cost = model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name=f"cost_{feature_index}",
        )

        if spec.kind == FeatureKind.DISCRETE:
            abs_differences = []
            for local_index, column_index in enumerate(spec.variable_indices):
                aux = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"cost_disc_abs_{feature_index}_{local_index}",
                )
                original_value = _coerce_binary_value(float(factual[column_index]))
                model.addConstr(
                    aux >= input_vars[column_index] - original_value,
                    name=f"cost_disc_pos_{feature_index}_{local_index}",
                )
                model.addConstr(
                    aux >= original_value - input_vars[column_index],
                    name=f"cost_disc_neg_{feature_index}_{local_index}",
                )
                abs_differences.append(aux)
            model.addConstr(feature_cost == max_(abs_differences), name=f"cost_disc_{feature_index}")
            feature_costs.append(feature_cost)
            continue

        if spec.kind == FeatureKind.ORDINAL:
            current_sum = quicksum(input_vars[index] for index in spec.variable_indices)
            original_sum = float(np.round(factual[list(spec.variable_indices)].sum()))
            scale = float(max(spec.distance_scale, 1.0))
            model.addConstr(
                feature_cost >= (current_sum - original_sum) / scale,
                name=f"cost_ord_pos_{feature_index}",
            )
            model.addConstr(
                feature_cost >= (original_sum - current_sum) / scale,
                name=f"cost_ord_neg_{feature_index}",
            )
            feature_costs.append(feature_cost)
            continue

        column_index = spec.variable_indices[0]
        original_value = float(factual[column_index])
        scale = float(max(spec.distance_scale, 1e-6))
        model.addConstr(
            feature_cost >= (input_vars[column_index] - original_value) / scale,
            name=f"cost_scalar_pos_{feature_index}",
        )
        model.addConstr(
            feature_cost >= (original_value - input_vars[column_index]) / scale,
            name=f"cost_scalar_neg_{feature_index}",
        )
        feature_costs.append(feature_cost)

    divisor = float(max(len(feature_costs), 1))
    model.setObjective(quicksum(feature_costs) / divisor, GRB.MINIMIZE)


def solve_counterfactual(
    schema: FeatureSchema,
    network: BinaryNetwork,
    factual: np.ndarray,
    eps: float,
    big_m: float,
    seed: int | None = None,
) -> np.ndarray | None:
    try:
        model = create_silent_gurobi_model("apas_counterfactual", seed=seed)

        input_vars = _add_input_constraints(model, schema, factual)
        hidden_layer = _add_hidden_constraints(
            model=model,
            network=network,
            input_vars=input_vars,
            big_m=float(big_m),
        )
        _add_output_constraint(
            model=model,
            network=network,
            previous_layer=hidden_layer if hidden_layer else input_vars,
            eps=float(eps),
        )
        _set_l1_objective(model=model, schema=schema, input_vars=input_vars, factual=factual)
        model.optimize()
    except GurobiError:
        return None

    if getattr(model, "SolCount", 0) < 1:
        return None

    candidate = np.asarray([float(var.X) for var in input_vars], dtype=np.float64)
    return candidate


def forward_binary_network(network: BinaryNetwork, X: np.ndarray) -> np.ndarray:
    activations = np.asarray(X, dtype=np.float64)
    if activations.ndim == 1:
        activations = activations.reshape(1, -1)

    for weight, bias in zip(network.hidden_weights, network.hidden_biases):
        activations = np.maximum(0.0, activations @ weight.T + bias.reshape(1, -1))

    scores = activations @ network.output_weight.T + float(network.output_bias)
    return scores.reshape(-1)


def compute_num_concretizations(alpha: float, r: float) -> int:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not (0.0 < r < 1.0):
        raise ValueError("r must be in (0, 1)")
    return int(math.ceil(math.log(1.0 - alpha) / math.log(r)))


def resolve_num_concretizations(
    alpha: float,
    r: float,
    num_concretizations: int | None = None,
) -> int:
    if num_concretizations is None:
        return compute_num_concretizations(alpha=alpha, r=r)
    if int(num_concretizations) < 1:
        raise ValueError("num_concretizations must be >= 1 when provided")
    return int(num_concretizations)


def certify_candidate(
    network: BinaryNetwork,
    candidate: np.ndarray,
    delta: float,
    alpha: float,
    r: float,
    num_concretizations: int | None = None,
    use_biases: bool = True,
    seed: int | None = None,
) -> CertificationResult:
    num_concretizations = resolve_num_concretizations(
        alpha=alpha,
        r=r,
        num_concretizations=num_concretizations,
    )
    if delta < 0:
        raise ValueError("delta must be >= 0")

    rng = np.random.default_rng(seed if seed is not None else 1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(1, -1)

    robust_count = 0
    for _ in range(num_concretizations):
        activations = candidate
        for weight, bias in zip(network.hidden_weights, network.hidden_biases):
            perturbed_weight = weight + rng.uniform(-delta, delta, size=weight.shape)
            if use_biases:
                perturbed_bias = bias + rng.uniform(-delta, delta, size=bias.shape)
            else:
                perturbed_bias = bias
            activations = np.maximum(
                0.0,
                activations @ perturbed_weight.T + perturbed_bias.reshape(1, -1),
            )

        perturbed_output_weight = network.output_weight + rng.uniform(
            -delta,
            delta,
            size=network.output_weight.shape,
        )
        perturbed_output_bias = float(network.output_bias)
        if use_biases:
            perturbed_output_bias += float(rng.uniform(-delta, delta))

        score = activations @ perturbed_output_weight.T + perturbed_output_bias
        if float(score.reshape(-1)[0]) < 0.0:
            robust_fraction = robust_count / float(num_concretizations)
            return CertificationResult(
                is_robust=False,
                robust_fraction=robust_fraction,
                num_concretizations=num_concretizations,
            )
        robust_count += 1

    return CertificationResult(
        is_robust=True,
        robust_fraction=1.0,
        num_concretizations=num_concretizations,
    )


def compute_delta_max(
    network: BinaryNetwork,
    candidate: np.ndarray,
    alpha: float,
    r: float,
    delta_init: float = 1e-4,
    num_concretizations: int | None = None,
    use_biases: bool = True,
    seed: int | None = None,
    max_expansions: int = 64,
) -> float:
    if delta_init <= 0:
        raise ValueError("delta_init must be > 0")

    certification = certify_candidate(
        network=network,
        candidate=candidate,
        delta=delta_init,
        alpha=alpha,
        r=r,
        num_concretizations=num_concretizations,
        use_biases=use_biases,
        seed=seed,
    )
    if not certification.is_robust:
        return 0.0

    lower = float(delta_init)
    upper = float(delta_init)
    expansion_seed = seed if seed is not None else 1

    for expansion in range(max_expansions):
        if expansion > 0:
            lower = upper
        upper = upper * 2.0
        certification = certify_candidate(
            network=network,
            candidate=candidate,
            delta=upper,
            alpha=alpha,
            r=r,
            num_concretizations=num_concretizations,
            use_biases=use_biases,
            seed=expansion_seed + expansion + 1,
        )
        if not certification.is_robust:
            break
    else:
        return lower

    while abs(upper - lower) > delta_init:
        midpoint = (lower + upper) / 2.0
        certification = certify_candidate(
            network=network,
            candidate=candidate,
            delta=midpoint,
            alpha=alpha,
            r=r,
            num_concretizations=num_concretizations,
            use_biases=use_biases,
            seed=expansion_seed + 10_000 + int(midpoint / delta_init),
        )
        if certification.is_robust:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def generate_apas_counterfactual(
    schema: FeatureSchema,
    network: BinaryNetwork,
    factual: np.ndarray,
    delta: float | None,
    alpha: float,
    r: float,
    delta_init: float,
    num_concretizations: int | None,
    eps_init: float,
    eps_step: float,
    max_iter: int,
    big_m: float,
    use_biases: bool,
    seed: int | None = None,
) -> np.ndarray | None:
    if eps_init <= 0:
        raise ValueError("eps_init must be > 0")
    if eps_step < 0:
        raise ValueError("eps_step must be >= 0")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    if delta is None:
        return solve_counterfactual(
            schema=schema,
            network=network,
            factual=factual,
            eps=eps_init,
            big_m=big_m,
            seed=seed,
        )

    eps = float(eps_init)
    for attempt in range(max_iter):
        candidate = solve_counterfactual(
            schema=schema,
            network=network,
            factual=factual,
            eps=eps,
            big_m=big_m,
            seed=None if seed is None else seed + attempt,
        )
        if candidate is None:
            return None

        certification = certify_candidate(
            network=network,
            candidate=candidate,
            delta=float(delta),
            alpha=alpha,
            r=r,
            num_concretizations=num_concretizations,
            use_biases=use_biases,
            seed=None if seed is None else seed + 10_000 + attempt,
        )
        if certification.is_robust:
            return candidate
        eps += float(eps_step)

    return None
