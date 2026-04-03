from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import gurobipy as gp
import numpy as np
import pandas as pd
import torch
from gurobipy import GRB
from scipy.spatial import ConvexHull, QhullError

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject

TorchModelTypes = (LinearModel, MlpModel)


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


def dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    return True


def resolve_feature_metadata(
    dataset: DatasetObject,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    if dataset_has_attr(dataset, "encoded_feature_type"):
        feature_type = dataset.attr("encoded_feature_type")
        feature_mutability = dataset.attr("encoded_feature_mutability")
        feature_actionability = dataset.attr("encoded_feature_actionability")
    else:
        feature_type = dataset.attr("raw_feature_type")
        feature_mutability = dataset.attr("raw_feature_mutability")
        feature_actionability = dataset.attr("raw_feature_actionability")
    return feature_type, feature_mutability, feature_actionability


def to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


def predict_label_indices(
    target_model: ModelObject,
    X: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> np.ndarray:
    features = to_feature_dataframe(X, feature_names)
    prediction = target_model.get_prediction(features, proba=False)
    if isinstance(prediction, torch.Tensor):
        return prediction.detach().cpu().numpy().argmax(axis=1)
    return np.asarray(prediction).argmax(axis=1)


def resolve_target_index(
    target_model: ModelObject,
    original_prediction: int,
    desired_class: int | str | None,
) -> int:
    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError("ProplaceMethod supports binary classification only")

    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return int(class_to_index[desired_class])

    return 1 - int(original_prediction)


class DataType(Enum):
    DISCRETE = 0
    ORDINAL = 1
    CONTINUOUS_REAL = 2


@dataclass(frozen=True)
class ProplaceDataset:
    feature_names: list[str]
    feature_types: dict[int, DataType]
    feat_var_map: dict[int, list[int]]
    continuous_bounds: dict[int, tuple[float, float]]

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    @property
    def num_variables(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True)
class ScalarNetwork:
    layer_weights: tuple[np.ndarray, ...]
    layer_biases: tuple[np.ndarray, ...]

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        sizes = [int(self.layer_weights[0].shape[0])]
        sizes.extend(int(weight.shape[1]) for weight in self.layer_weights)
        return tuple(sizes)


@dataclass(frozen=True)
class SolverConfig:
    output: bool = False
    time_limit: float | None = None
    threads: int | None = None
    seed: int | None = None


class Interval:
    def __init__(self, value: float, lb: float | None = None, ub: float | None = None):
        self.value = float(value)
        self.lb = self.value if lb is None else float(lb)
        self.ub = self.value if ub is None else float(ub)


class Node:
    def __init__(self, layer: int, index: int):
        self.layer = int(layer)
        self.index = int(index)
        self.loc = (self.layer, self.index)

    def __str__(self) -> str:
        return str(self.loc)


class Inn:
    def __init__(
        self,
        num_layers: int,
        delta: float,
        nodes: dict[int, list[Node]],
        weights: dict[tuple[Node, Node], Interval],
        biases: dict[Node, Interval],
    ):
        self.num_layers = int(num_layers)
        self.delta = float(delta)
        self.nodes = nodes
        self.weights = weights
        self.biases = biases


def build_proplace_dataset(trainset: DatasetObject) -> tuple[ProplaceDataset, np.ndarray]:
    feature_df = trainset.get(target=False)
    try:
        feature_array = feature_df.to_numpy(dtype=np.float64)
    except ValueError as error:
        raise ValueError(
            "ProplaceMethod requires fully numeric input features"
        ) from error

    if np.isnan(feature_array).any():
        raise ValueError("ProplaceMethod does not support NaN values")

    feature_names = list(feature_df.columns)
    feature_type, _, _ = resolve_feature_metadata(trainset)
    feature_types: dict[int, DataType] = {}
    feat_var_map: dict[int, list[int]] = {}
    continuous_bounds: dict[int, tuple[float, float]] = {}

    for feature_index, feature_name in enumerate(feature_names):
        feature_kind = str(feature_type[feature_name]).lower()
        feat_var_map[feature_index] = [feature_index]

        if feature_kind == "numerical":
            column = feature_array[:, feature_index]
            feature_types[feature_index] = DataType.CONTINUOUS_REAL
            continuous_bounds[feature_index] = (
                float(np.min(column)),
                float(np.max(column)),
            )
            continue

        if feature_kind == "binary":
            column = feature_array[:, feature_index]
            is_zero = np.isclose(column, 0.0, atol=1e-6)
            is_one = np.isclose(column, 1.0, atol=1e-6)
            if not np.all(is_zero | is_one):
                raise ValueError(
                    "ProplaceMethod requires binary features to be encoded as 0/1: "
                    f"{feature_name}"
                )
            feature_types[feature_index] = DataType.DISCRETE
            continue

        raise ValueError(
            "ProplaceMethod requires categorical features to be encoded before use; "
            f"unsupported feature '{feature_name}' has type '{feature_type[feature_name]}'"
        )

    return (
        ProplaceDataset(
            feature_names=feature_names,
            feature_types=feature_types,
            feat_var_map=feat_var_map,
            continuous_bounds=continuous_bounds,
        ),
        feature_array,
    )


def _collect_linear_layers(target_model: ModelObject) -> list[torch.nn.Linear]:
    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model has not been initialized")

    if isinstance(model, torch.nn.Linear):
        return [model]

    if not isinstance(model, torch.nn.Sequential):
        raise TypeError(
            "ProplaceMethod requires a torch.nn.Linear or torch.nn.Sequential model"
        )

    modules = list(model.children())
    if not modules or not isinstance(modules[-1], torch.nn.Linear):
        raise TypeError("ProplaceMethod requires a final Linear output layer")

    expected_linear = True
    linear_layers: list[torch.nn.Linear] = []
    for index, module in enumerate(modules):
        is_last = index == len(modules) - 1
        if expected_linear:
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(
                    "ProplaceMethod requires an MLP made of Linear/ReLU blocks only"
                )
            linear_layers.append(module)
            expected_linear = False
            continue

        if is_last:
            raise TypeError("ProplaceMethod requires a final Linear output layer")
        if not isinstance(module, torch.nn.ReLU):
            raise TypeError(
                "ProplaceMethod requires an MLP made of Linear/ReLU blocks only"
            )
        expected_linear = True

    return linear_layers


def extract_scalar_network(
    target_model: ModelObject,
    desired_index: int,
) -> ScalarNetwork:
    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError("ProplaceMethod supports binary classification only")
    if desired_index not in {0, 1}:
        raise ValueError("desired_index must be either 0 or 1")

    linear_layers = _collect_linear_layers(target_model)
    layer_weights: list[np.ndarray] = []
    layer_biases: list[np.ndarray] = []

    for layer in linear_layers[:-1]:
        layer_weights.append(
            layer.weight.detach().cpu().numpy().astype(np.float64, copy=False).T
        )
        layer_biases.append(
            layer.bias.detach().cpu().numpy().astype(np.float64, copy=False)
        )

    output_layer = linear_layers[-1]
    output_weight = output_layer.weight.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    output_bias = output_layer.bias.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    output_activation = str(
        getattr(target_model, "_output_activation_name", "softmax")
    ).lower()

    if output_activation == "sigmoid":
        if output_weight.shape[0] != 1:
            raise TypeError(
                "ProplaceMethod requires a single-logit output layer for sigmoid models"
            )
        sign = 1.0 if desired_index == 1 else -1.0
        collapsed_weight = sign * output_weight
        collapsed_bias = sign * output_bias
    elif output_activation == "softmax":
        if output_weight.shape[0] != 2:
            raise TypeError(
                "ProplaceMethod requires exactly two logits for softmax models"
            )
        other_index = 1 - desired_index
        collapsed_weight = (
            output_weight[desired_index] - output_weight[other_index]
        ).reshape(1, -1)
        collapsed_bias = np.array(
            [output_bias[desired_index] - output_bias[other_index]],
            dtype=np.float64,
        )
    else:
        raise TypeError(
            "ProplaceMethod supports target models with softmax or sigmoid outputs only"
        )

    layer_weights.append(collapsed_weight.T.astype(np.float64, copy=False))
    layer_biases.append(collapsed_bias.astype(np.float64, copy=False))
    return ScalarNetwork(
        layer_weights=tuple(layer_weights),
        layer_biases=tuple(layer_biases),
    )


def build_inn_nodes(layer_sizes: Sequence[int]) -> dict[int, list[Node]]:
    nodes: dict[int, list[Node]] = {}
    for layer_index, layer_size in enumerate(layer_sizes):
        nodes[layer_index] = [Node(layer_index, node_index) for node_index in range(int(layer_size))]
    return nodes


def build_inn(scalar_network: ScalarNetwork, delta: float) -> Inn:
    layer_sizes = scalar_network.layer_sizes
    nodes = build_inn_nodes(layer_sizes)
    weights: dict[tuple[Node, Node], Interval] = {}
    biases: dict[Node, Interval] = {}

    for layer_index, (layer_weight, layer_bias) in enumerate(
        zip(scalar_network.layer_weights, scalar_network.layer_biases)
    ):
        for node_from in nodes[layer_index]:
            for node_to in nodes[layer_index + 1]:
                weight_value = round(float(layer_weight[node_from.index, node_to.index]), 8)
                weights[(node_from, node_to)] = Interval(
                    weight_value,
                    weight_value - float(delta),
                    weight_value + float(delta),
                )

        for node_to in nodes[layer_index + 1]:
            bias_value = round(float(layer_bias[node_to.index]), 8)
            biases[node_to] = Interval(
                bias_value,
                bias_value - float(delta),
                bias_value + float(delta),
            )

    return Inn(
        num_layers=len(layer_sizes),
        delta=delta,
        nodes=nodes,
        weights=weights,
        biases=biases,
    )


def _configure_model(model: gp.Model, solver_config: SolverConfig, nonconvex: bool = False) -> None:
    model.Params.OutputFlag = 1 if solver_config.output else 0
    if solver_config.time_limit is not None:
        model.Params.TimeLimit = float(solver_config.time_limit)
    if solver_config.threads is not None:
        model.Params.Threads = int(solver_config.threads)
    if solver_config.seed is not None:
        model.Params.Seed = int(solver_config.seed)
    if nonconvex:
        model.Params.NonConvex = 2


def _has_solution(model: gp.Model) -> bool:
    return model.SolCount > 0 and model.Status in {
        GRB.OPTIMAL,
        GRB.SUBOPTIMAL,
        GRB.TIME_LIMIT,
        GRB.ITERATION_LIMIT,
        GRB.NODE_LIMIT,
        GRB.SOLUTION_LIMIT,
    }


def build_vertex_hull(points: np.ndarray, prune: bool) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")

    rounded = np.round(points.astype(np.float64, copy=False), decimals=10)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    ordered_indices = np.sort(unique_indices)
    points = points[ordered_indices]
    if (not prune) or points.shape[0] <= 2:
        return points

    if points.shape[0] <= points.shape[1]:
        return points

    try:
        hull = ConvexHull(points, qhull_options="QJ")
    except QhullError:
        return points
    return points[np.unique(hull.vertices)]


class OptSolver:
    def __init__(
        self,
        dataset: ProplaceDataset,
        inn: Inn,
        y_prime: int,
        x: np.ndarray,
        mode: int = 0,
        eps: float = 1e-4,
        big_m: float = 1000.0,
        x_prime: np.ndarray | None = None,
        solver_config: SolverConfig | None = None,
    ):
        self.mode = int(mode)
        self.dataset = dataset
        self.inn = inn
        self.y_prime = int(y_prime)
        self.x = np.asarray(x, dtype=np.float64)
        self.x_prime = None if x_prime is None else np.asarray(x_prime, dtype=np.float64)
        self.eps = float(eps)
        self.big_m = float(big_m)
        self.solver_config = solver_config or SolverConfig()
        self.model = gp.Model()
        self.output_node_name: str | None = None

    def add_input_variable_constraints(self) -> dict[int, gp.Var]:
        node_var: dict[int, gp.Var] = {}
        for feat_idx in range(self.dataset.num_features):
            feature_type = self.dataset.feature_types[feat_idx]
            var_idxs = self.dataset.feat_var_map[feat_idx]

            if feature_type == DataType.DISCRETE:
                for var_idx in var_idxs:
                    if self.mode == 1:
                        variable = self.model.addVar(
                            lb=-GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=f"x_disc_0_{var_idx}",
                        )
                        self.model.addConstr(variable == float(self.x_prime[var_idx]))
                    else:
                        variable = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_disc_0_{var_idx}",
                        )
                    node_var[var_idx] = variable
                continue

            if feature_type == DataType.ORDINAL:
                prev_var: gp.Var | None = None
                ordinal_vars: list[gp.Var] = []
                for index, var_idx in enumerate(var_idxs):
                    if self.mode == 1:
                        variable = self.model.addVar(
                            lb=-GRB.INFINITY,
                            vtype=GRB.CONTINUOUS,
                            name=f"x_ord_0_{var_idx}",
                        )
                        self.model.addConstr(variable == float(self.x_prime[var_idx]))
                    else:
                        variable = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_ord_0_{var_idx}",
                        )
                        if index != 0 and prev_var is not None:
                            self.model.addConstr(prev_var >= variable)
                    node_var[var_idx] = variable
                    ordinal_vars.append(variable)
                    prev_var = variable
                if self.mode == 0 and ordinal_vars:
                    self.model.addConstr(gp.quicksum(ordinal_vars) >= 1)
                continue

            if feature_type == DataType.CONTINUOUS_REAL:
                var_idx = var_idxs[0]
                if self.mode == 1:
                    lower_bound = upper_bound = float(self.x_prime[var_idx])
                else:
                    lower_bound, upper_bound = self.dataset.continuous_bounds[feat_idx]
                node_var[var_idx] = self.model.addVar(
                    lb=lower_bound,
                    ub=upper_bound,
                    vtype=GRB.CONTINUOUS,
                    name=f"x_cont_0_{var_idx}",
                )
                continue

            raise TypeError(f"Unsupported data type: {feature_type}")
        return node_var

    def add_node_variables_constraints(
        self,
        node_vars: dict[int, dict[int, gp.Var]],
        aux_vars: dict[int, dict[int, gp.Var]],
    ) -> tuple[dict[int, dict[int, gp.Var]], dict[int, dict[int, gp.Var]]]:
        for layer_index in range(1, self.inn.num_layers):
            node_var: dict[int, gp.Var] = {}
            aux_var: dict[int, gp.Var] = {}
            is_output_layer = layer_index == (self.inn.num_layers - 1)

            for node in self.inn.nodes[layer_index]:
                if not is_output_layer:
                    current = self.model.addVar(
                        lb=-GRB.INFINITY,
                        vtype=GRB.CONTINUOUS,
                        name=f"n_{node}",
                    )
                    aux = self.model.addVar(
                        vtype=GRB.BINARY,
                        name=f"a_{node}",
                    )
                    node_var[node.index] = current
                    aux_var[node.index] = aux
                    self.model.addConstr(current >= 0.0)
                    self.model.addConstr(current <= self.big_m * (1.0 - aux))
                    self.model.addConstr(
                        current
                        <= gp.quicksum(
                            self.inn.weights[(prev_node, node)].ub
                            * node_vars[layer_index - 1][prev_node.index]
                            for prev_node in self.inn.nodes[layer_index - 1]
                        )
                        + self.inn.biases[node].ub
                        + self.big_m * aux
                    )
                    self.model.addConstr(
                        current
                        >= gp.quicksum(
                            self.inn.weights[(prev_node, node)].lb
                            * node_vars[layer_index - 1][prev_node.index]
                            for prev_node in self.inn.nodes[layer_index - 1]
                        )
                        + self.inn.biases[node].lb
                    )
                    continue

                current = self.model.addVar(
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"n_{node}",
                )
                self.output_node_name = f"n_{node}"
                node_var[node.index] = current
                self.model.addConstr(
                    current
                    <= gp.quicksum(
                        self.inn.weights[(prev_node, node)].ub
                        * node_vars[layer_index - 1][prev_node.index]
                        for prev_node in self.inn.nodes[layer_index - 1]
                    )
                    + self.inn.biases[node].ub
                )
                self.model.addConstr(
                    current
                    >= gp.quicksum(
                        self.inn.weights[(prev_node, node)].lb
                        * node_vars[layer_index - 1][prev_node.index]
                        for prev_node in self.inn.nodes[layer_index - 1]
                    )
                    + self.inn.biases[node].lb
                )
                if self.mode == 0:
                    if self.y_prime == 1:
                        self.model.addConstr(current >= self.eps)
                    else:
                        self.model.addConstr(current <= -self.eps)

            node_vars[layer_index] = node_var
            if not is_output_layer:
                aux_vars[layer_index] = aux_var
        return node_vars, aux_vars

    def create_constraints(
        self,
    ) -> tuple[dict[int, dict[int, gp.Var]], dict[int, dict[int, gp.Var]]]:
        node_vars: dict[int, dict[int, gp.Var]] = {}
        aux_vars: dict[int, dict[int, gp.Var]] = {}
        node_vars[0] = self.add_input_variable_constraints()
        return self.add_node_variables_constraints(node_vars, aux_vars)

    def set_objective_l1(self, node_vars: dict[int, dict[int, gp.Var]]) -> None:
        obj_vars_l1: list[gp.Var] = []
        for feat_idx in range(self.dataset.num_features):
            this_obj_var = self.model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"objl1_feat_{feat_idx}",
            )
            var_idxs = self.dataset.feat_var_map[feat_idx]
            feature_type = self.dataset.feature_types[feat_idx]

            if feature_type == DataType.DISCRETE:
                abs_vars: list[gp.Var] = []
                for var_idx in var_idxs:
                    abs_var = self.model.addVar(
                        lb=0.0,
                        vtype=GRB.CONTINUOUS,
                        name=f"objl1_feat_disc_{var_idx}",
                    )
                    self.model.addConstr(abs_var >= node_vars[0][var_idx] - float(self.x[var_idx]))
                    self.model.addConstr(abs_var >= float(self.x[var_idx]) - node_vars[0][var_idx])
                    abs_vars.append(abs_var)
                if len(abs_vars) == 1:
                    self.model.addConstr(this_obj_var == abs_vars[0])
                else:
                    self.model.addConstr(this_obj_var == gp.max_(abs_vars))
            elif feature_type == DataType.ORDINAL:
                denominator = max(len(var_idxs) - 1, 1)
                current_value = float(np.sum(self.x[var_idxs]))
                self.model.addConstr(
                    this_obj_var
                    >= (
                        gp.quicksum(node_vars[0][idx] for idx in var_idxs) - current_value
                    )
                    / denominator
                )
                self.model.addConstr(
                    this_obj_var
                    >= (
                        current_value - gp.quicksum(node_vars[0][idx] for idx in var_idxs)
                    )
                    / denominator
                )
            elif feature_type == DataType.CONTINUOUS_REAL:
                current_value = float(np.sum(self.x[var_idxs]))
                self.model.addConstr(
                    this_obj_var
                    >= gp.quicksum(node_vars[0][idx] for idx in var_idxs) - current_value
                )
                self.model.addConstr(
                    this_obj_var
                    >= current_value - gp.quicksum(node_vars[0][idx] for idx in var_idxs)
                )
            else:
                raise TypeError(f"Unsupported data type: {feature_type}")

            obj_vars_l1.append(this_obj_var)

        self.model.setObjective(
            gp.quicksum(obj_vars_l1) / max(self.dataset.num_features, 1),
            GRB.MINIMIZE,
        )

    def set_objective_output_node(
        self,
        node_vars: dict[int, dict[int, gp.Var]],
    ) -> None:
        output_var = node_vars[self.inn.num_layers - 1][0]
        if self.y_prime == 1:
            self.model.setObjective(output_var, GRB.MINIMIZE)
        else:
            self.model.setObjective(output_var, GRB.MAXIMIZE)

    def _extract_solution(
        self,
        input_node_vars: dict[int, gp.Var],
    ) -> np.ndarray | None:
        if not _has_solution(self.model):
            return None
        return np.array(
            [float(input_node_vars[index].X) for index in range(self.dataset.num_variables)],
            dtype=np.float64,
        )

    def compute_counterfactual(self) -> np.ndarray | None:
        node_vars, _ = self.create_constraints()
        self.set_objective_l1(node_vars)
        _configure_model(self.model, self.solver_config, nonconvex=False)
        self.model.optimize()
        return self._extract_solution(node_vars[0])

    def compute_inn_bounds(self) -> tuple[int, float | None]:
        node_vars, _ = self.create_constraints()
        self.set_objective_output_node(node_vars)
        _configure_model(self.model, self.solver_config, nonconvex=False)
        self.model.optimize()
        if not _has_solution(self.model):
            return -1, None

        bound = float(node_vars[self.inn.num_layers - 1][0].X)
        result = 0
        if self.y_prime == 1 and bound >= 0.0:
            result = 1
        if self.y_prime == 0 and bound < 0.0:
            result = 1
        return result, bound


class OptSolverRC4:
    def __init__(
        self,
        dataset: ProplaceDataset,
        inn: Inn,
        y_prime: int,
        x: np.ndarray,
        eps: float = 0.01,
        big_m: float = 1000.0,
        delta: float = 0.03,
        nns: np.ndarray | None = None,
        solver_config: SolverConfig | None = None,
        max_iterations: int = 100,
    ):
        self.dataset = dataset
        self.inn = inn
        self.y_prime = int(y_prime)
        self.x = np.asarray(x, dtype=np.float64)
        self.eps = float(eps)
        self.big_m = float(big_m)
        self.delta = float(delta)
        self.nns = None if nns is None else np.asarray(nns, dtype=np.float64)
        self.solver_config = solver_config or SolverConfig()
        self.max_iterations = int(max_iterations)

        self.x_prime_current: np.ndarray | None = None
        self._achieved = False
        self._weight_bias_names, initial_weights = self._initialise_weight_bias_names()
        self._worst_case_perturbations: list[dict[str, float]] = [initial_weights]

    def _initialise_weight_bias_names(self) -> tuple[list[str], dict[str, float]]:
        names: list[str] = []
        initial_weights: dict[str, float] = {}
        for layer_index in range(1, self.inn.num_layers):
            for node in self.inn.nodes[layer_index]:
                bias_name = f"bb{node}"
                names.append(bias_name)
                initial_weights[bias_name] = self.inn.biases[node].value
                for prev_node in self.inn.nodes[layer_index - 1]:
                    weight_name = f"ww{prev_node}{node}"
                    names.append(weight_name)
                    initial_weights[weight_name] = self.inn.weights[(prev_node, node)].value
        return names, initial_weights

    def run(self) -> np.ndarray | None:
        for _ in range(self.max_iterations):
            self.x_prime_current = self.master_prob()
            if self.x_prime_current is None:
                return None
            if not self.adv_prob():
                return None
            if self._achieved:
                return self.x_prime_current
        return None

    def master_prob(self) -> np.ndarray | None:
        model = gp.Model()
        input_node_vars = self.mst_add_input_variable_constraints(model)
        self.mst_add_plausibility_constraints(model, input_node_vars)
        for perturbation in self._worst_case_perturbations:
            self.mst_add_node_variables_constraints_one_perturbation(
                model,
                perturbation,
                input_node_vars,
            )
        self.mst_set_objective_l1(model, input_node_vars)
        _configure_model(model, self.solver_config, nonconvex=False)
        model.optimize()
        if not _has_solution(model):
            return None
        return np.array(
            [float(input_node_vars[index].X) for index in range(self.dataset.num_variables)],
            dtype=np.float64,
        )

    def mst_add_input_variable_constraints(self, model: gp.Model) -> dict[int, gp.Var]:
        node_var: dict[int, gp.Var] = {}
        for feat_idx in range(self.dataset.num_features):
            feature_type = self.dataset.feature_types[feat_idx]
            var_idxs = self.dataset.feat_var_map[feat_idx]

            if feature_type == DataType.DISCRETE:
                for var_idx in var_idxs:
                    node_var[var_idx] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_disc_0_{var_idx}",
                    )
                continue

            if feature_type == DataType.CONTINUOUS_REAL:
                var_idx = var_idxs[0]
                lower_bound, upper_bound = self.dataset.continuous_bounds[feat_idx]
                node_var[var_idx] = model.addVar(
                    lb=lower_bound,
                    ub=upper_bound,
                    vtype=GRB.CONTINUOUS,
                    name=f"x_cont_0_{var_idx}",
                )
                continue

            raise TypeError(
                "ProplaceMethod currently supports discrete and continuous features only"
            )
        return node_var

    def mst_add_plausibility_constraints(
        self,
        model: gp.Model,
        input_node_vars: dict[int, gp.Var],
    ) -> None:
        if self.nns is None or self.nns.shape[0] == 0:
            return

        lambda_vars = [
            model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"lambda_{index}")
            for index in range(self.nns.shape[0])
        ]
        model.addConstr(gp.quicksum(lambda_vars) == 1.0)
        for feat_idx in range(self.dataset.num_features):
            model.addConstr(
                input_node_vars[feat_idx]
                == gp.quicksum(
                    float(self.nns[row_index, feat_idx]) * lambda_vars[row_index]
                    for row_index in range(self.nns.shape[0])
                )
            )

    def mst_add_node_variables_constraints_one_perturbation(
        self,
        model: gp.Model,
        perturbation: dict[str, float],
        input_node_var: dict[int, gp.Var],
    ) -> None:
        node_vars: dict[int, dict[int, gp.Var]] = {0: input_node_var}
        for layer_index in range(1, self.inn.num_layers):
            node_var: dict[int, gp.Var] = {}
            is_output_layer = layer_index == (self.inn.num_layers - 1)
            for node in self.inn.nodes[layer_index]:
                if not is_output_layer:
                    current = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
                    aux = model.addVar(vtype=GRB.BINARY)
                    node_var[node.index] = current
                    model.addConstr(current >= 0.0)
                    model.addConstr(current <= self.big_m * (1.0 - aux))
                    affine = gp.quicksum(
                        perturbation[f"ww{prev_node}{node}"]
                        * node_vars[layer_index - 1][prev_node.index]
                        for prev_node in self.inn.nodes[layer_index - 1]
                    ) + perturbation[f"bb{node}"]
                    model.addConstr(current <= affine + self.big_m * aux)
                    model.addConstr(current >= affine)
                    continue

                current = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
                node_var[node.index] = current
                affine = gp.quicksum(
                    perturbation[f"ww{prev_node}{node}"]
                    * node_vars[layer_index - 1][prev_node.index]
                    for prev_node in self.inn.nodes[layer_index - 1]
                ) + perturbation[f"bb{node}"]
                model.addConstr(current == affine)
                if self.y_prime == 1:
                    model.addConstr(current >= self.eps)
                else:
                    model.addConstr(current <= -self.eps)
            node_vars[layer_index] = node_var

    def mst_set_objective_l1(
        self,
        model: gp.Model,
        input_node_vars: dict[int, gp.Var],
    ) -> None:
        obj_vars_l1: list[gp.Var] = []
        for feat_idx in range(self.dataset.num_features):
            this_obj_var = model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"objl1_feat_{feat_idx}",
            )
            var_idxs = self.dataset.feat_var_map[feat_idx]
            feature_type = self.dataset.feature_types[feat_idx]

            if feature_type == DataType.DISCRETE:
                abs_vars: list[gp.Var] = []
                for var_idx in var_idxs:
                    abs_var = model.addVar(
                        lb=0.0,
                        vtype=GRB.CONTINUOUS,
                        name=f"objl1_feat_disc_{var_idx}",
                    )
                    model.addConstr(abs_var >= input_node_vars[var_idx] - float(self.x[var_idx]))
                    model.addConstr(abs_var >= float(self.x[var_idx]) - input_node_vars[var_idx])
                    abs_vars.append(abs_var)
                if len(abs_vars) == 1:
                    model.addConstr(this_obj_var == abs_vars[0])
                else:
                    model.addConstr(this_obj_var == gp.max_(abs_vars))
            elif feature_type == DataType.CONTINUOUS_REAL:
                current_value = float(np.sum(self.x[var_idxs]))
                model.addConstr(
                    this_obj_var
                    >= gp.quicksum(input_node_vars[idx] for idx in var_idxs) - current_value
                )
                model.addConstr(
                    this_obj_var
                    >= current_value - gp.quicksum(input_node_vars[idx] for idx in var_idxs)
                )
            else:
                raise TypeError(f"Unsupported data type: {feature_type}")

            obj_vars_l1.append(this_obj_var)

        model.setObjective(
            gp.quicksum(obj_vars_l1) / max(self.dataset.num_features, 1),
            GRB.MINIMIZE,
        )

    def adv_prob(self) -> bool:
        if self.x_prime_current is None:
            return False

        model = gp.Model()
        node_vars: dict[int, dict[int, gp.Var]] = {}
        aux_vars: dict[int, dict[int, gp.Var]] = {}
        node_vars[0] = self.adv_add_input_variable_constraints(model)
        node_vars, aux_vars = self.adv_add_node_variables_constraints(
            model,
            node_vars,
            aux_vars,
        )
        model.setObjective(node_vars[self.inn.num_layers - 1][0], GRB.MINIMIZE)
        _configure_model(model, self.solver_config, nonconvex=True)
        model.optimize()
        if not _has_solution(model):
            return False

        bound = float(node_vars[self.inn.num_layers - 1][0].X)
        if bound >= 0.0:
            self._achieved = True
            return True

        perturbation: dict[str, float] = {}
        for name in self._weight_bias_names:
            variable = model.getVarByName(name)
            if variable is None:
                perturbation[name] = self._default_output_bound_value(name)
            else:
                perturbation[name] = float(variable.X)

        if any(
            np.allclose(
                [perturbation[name] for name in self._weight_bias_names],
                [existing[name] for name in self._weight_bias_names],
                atol=1e-7,
                rtol=1e-7,
            )
            for existing in self._worst_case_perturbations
        ):
            return False

        self._worst_case_perturbations.append(perturbation)
        return True

    def _default_output_bound_value(self, name: str) -> float:
        for layer_index in range(1, self.inn.num_layers):
            output_layer = layer_index == (self.inn.num_layers - 1)
            if not output_layer:
                continue
            for node in self.inn.nodes[layer_index]:
                if name == f"bb{node}":
                    return self.inn.biases[node].lb
                for prev_node in self.inn.nodes[layer_index - 1]:
                    if name == f"ww{prev_node}{node}":
                        return self.inn.weights[(prev_node, node)].lb
        raise KeyError(f"Unknown weight or bias name: {name}")

    def adv_add_input_variable_constraints(self, model: gp.Model) -> dict[int, gp.Var]:
        node_var: dict[int, gp.Var] = {}
        for feat_idx in range(self.dataset.num_features):
            feature_type = self.dataset.feature_types[feat_idx]
            var_idxs = self.dataset.feat_var_map[feat_idx]
            if feature_type in {DataType.DISCRETE, DataType.CONTINUOUS_REAL}:
                for var_idx in var_idxs:
                    variable = model.addVar(
                        lb=-GRB.INFINITY,
                        vtype=GRB.CONTINUOUS,
                        name=f"x_0_{var_idx}",
                    )
                    model.addConstr(variable == float(self.x_prime_current[var_idx]))
                    node_var[var_idx] = variable
                continue
            raise TypeError(
                "ProplaceMethod currently supports discrete and continuous features only"
            )
        return node_var

    def adv_add_node_variables_constraints(
        self,
        model: gp.Model,
        node_vars: dict[int, dict[int, gp.Var]],
        aux_vars: dict[int, dict[int, gp.Var]],
    ) -> tuple[dict[int, dict[int, gp.Var]], dict[int, dict[int, gp.Var]]]:
        for layer_index in range(1, self.inn.num_layers):
            node_var: dict[int, gp.Var] = {}
            aux_var: dict[int, gp.Var] = {}
            is_output_layer = layer_index == (self.inn.num_layers - 1)

            for node in self.inn.nodes[layer_index]:
                weight_vars: dict[tuple[Node, Node], gp.Var] = {}
                for prev_node in self.inn.nodes[layer_index - 1]:
                    weight_interval = self.inn.weights[(prev_node, node)]
                    weight_vars[(prev_node, node)] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=weight_interval.lb,
                        ub=weight_interval.ub,
                        name=f"ww{prev_node}{node}",
                    )
                bias_interval = self.inn.biases[node]
                bias_var = model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=bias_interval.lb,
                    ub=bias_interval.ub,
                    name=f"bb{node}",
                )

                if not is_output_layer:
                    current = model.addVar(
                        lb=-GRB.INFINITY,
                        vtype=GRB.CONTINUOUS,
                        name=f"n_{node}",
                    )
                    aux = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"a_{node}",
                    )
                    node_var[node.index] = current
                    aux_var[node.index] = aux
                    affine = gp.quicksum(
                        weight_vars[(prev_node, node)] * node_vars[layer_index - 1][prev_node.index]
                        for prev_node in self.inn.nodes[layer_index - 1]
                    ) + bias_var
                    model.addConstr(current >= 0.0)
                    model.addConstr(current <= self.big_m * (1.0 - aux))
                    model.addConstr(current <= affine + self.big_m * aux)
                    model.addConstr(current >= affine)
                    continue

                current = model.addVar(
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"n_{node}",
                )
                node_var[node.index] = current
                affine = gp.quicksum(
                    weight_vars[(prev_node, node)] * node_vars[layer_index - 1][prev_node.index]
                    for prev_node in self.inn.nodes[layer_index - 1]
                ) + bias_var
                model.addConstr(current == affine)

            node_vars[layer_index] = node_var
            if not is_output_layer:
                aux_vars[layer_index] = aux_var
        return node_vars, aux_vars


def validate_counterfactual_array(
    target_model: ModelObject,
    feature_names: Sequence[str],
    candidate: np.ndarray,
    target_index: int,
) -> bool:
    if candidate is None or np.isnan(candidate).any():
        return False
    feature_df = pd.DataFrame(
        np.asarray(candidate, dtype=np.float64).reshape(1, -1),
        columns=list(feature_names),
    )
    prediction = predict_label_indices(target_model, feature_df, feature_names)
    return bool(prediction[0] == int(target_index))
