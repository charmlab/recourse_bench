from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import pdist

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from preprocess.preprocess_utils import resolve_feature_metadata

TorchModelTypes = (LinearModel, MlpModel)
BlackBoxModelTypes = (LinearModel, MlpModel, RandomForestModel)
VALID_CVXPY_STATUSES = {"optimal", "optimal_inaccurate"}


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


def to_feature_dataframe(
    values: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


@dataclass(frozen=True)
class ProjectionFeatures:
    feature_names: list[str]
    boolean_feature_indices: list[int]


@dataclass(frozen=True)
class LinearSurrogate:
    coef: np.ndarray
    intercept: float


def resolve_projection_features(dataset: DatasetObject) -> ProjectionFeatures:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, _, _ = resolve_feature_metadata(dataset)

    unsupported = [
        feature_name
        for feature_name in feature_names
        if str(feature_type[feature_name]).lower() not in {"numerical", "binary"}
    ]
    if unsupported:
        raise ValueError(
            "CvasProjMethod requires numeric/binary input features only; "
            f"unsupported categorical features: {unsupported}"
        )

    boolean_feature_indices = [
        index
        for index, feature_name in enumerate(feature_names)
        if str(feature_type[feature_name]).lower() == "binary"
    ]
    return ProjectionFeatures(
        feature_names=feature_names,
        boolean_feature_indices=boolean_feature_indices,
    )


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)

    def get_ordered_features(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=True)
        return prediction.detach().cpu().numpy().astype(np.float64, copy=False)

    def predict_label_indices(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return probabilities.argmax(axis=1).astype(np.int64, copy=False)


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
        raise ValueError(
            "desired_class=None is supported for binary classification only"
        )
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

    adapter = RecourseModelAdapter(target_model, factuals.columns)
    original_prediction = adapter.predict_label_indices(factuals)
    target_prediction = resolve_target_indices(
        target_model=target_model,
        original_prediction=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = adapter.predict_label_indices(candidates.loc[valid_rows])
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_prediction[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates


def derive_row_seed(seed: int | None, row_index: object) -> int | None:
    if seed is None:
        return None

    row_hash = pd.util.hash_pandas_object(
        pd.Index([row_index]),
        index=False,
    ).to_numpy(dtype="uint64")[0]
    modulus = np.uint64(2**32 - 1)
    return int((np.uint64(seed) + row_hash) % modulus)


def compute_max_distance(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if X.shape[0] < 2:
        return 1.0

    distances = pdist(X, metric="euclidean")
    if distances.size == 0:
        return 1.0
    max_distance = float(np.max(distances))
    return max_distance if max_distance > 0 else 1.0


def sample_uniform_ball(
    center: np.ndarray,
    radius: float,
    num_samples: int,
    random_state: int | None,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if radius <= 0:
        return np.repeat(center.reshape(1, -1), num_samples, axis=0)

    rng = np.random.default_rng(random_state)
    directions = rng.normal(size=(num_samples, center.shape[0]))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    directions = directions / norms
    scales = rng.random(num_samples).reshape(-1, 1) ** (1.0 / center.shape[0])
    return center.reshape(1, -1) + directions * scales * float(radius)


def select_nearest_training_points(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    x0: np.ndarray,
    target_label: int,
    num_neighbors: int,
) -> np.ndarray:
    mask = train_labels.astype(np.int64, copy=False) == int(target_label)
    candidates = train_data[mask]
    if candidates.shape[0] == 0:
        return candidates

    distances = np.linalg.norm(candidates - x0.reshape(1, -1), ord=1, axis=1)
    order = np.argsort(distances)
    return candidates[order[:num_neighbors]]


def find_boundary_point(
    x0: np.ndarray,
    target_label: int,
    prototypes: np.ndarray,
    adapter: RecourseModelAdapter,
    line_search_steps: int,
) -> np.ndarray | None:
    if prototypes.shape[0] == 0:
        return None

    lambdas = np.linspace(0.0, 1.0, line_search_steps, dtype=np.float64)
    best_point: np.ndarray | None = None
    best_distance = float("inf")

    for prototype in prototypes:
        segment = (
            (1.0 - lambdas).reshape(-1, 1) * x0.reshape(1, -1)
            + lambdas.reshape(-1, 1) * prototype.reshape(1, -1)
        )
        labels = adapter.predict_label_indices(segment)
        hit_indices = np.flatnonzero(labels == int(target_label))
        if hit_indices.size == 0:
            continue

        candidate = segment[int(hit_indices[0])]
        distance = float(np.linalg.norm(candidate - x0, ord=1))
        if distance < best_distance:
            best_distance = distance
            best_point = candidate.copy()

    return best_point


def _ensure_covariance_shape(covariance: np.ndarray, dimension: int) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    covariance = covariance.reshape(dimension, dimension)
    return 0.5 * (covariance + covariance.T)


def compute_sample_covariance(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] <= 1:
        raise ValueError("At least two samples are required to estimate covariance")
    covariance = np.cov(samples, rowvar=False)
    return _ensure_covariance_shape(covariance, samples.shape[1])


def sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = eigh(_ensure_covariance_shape(matrix, matrix.shape[0]))
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def solve_problem(problem: cp.Problem, solver_names: Sequence[str]) -> bool:
    for solver_name in solver_names:
        solver_name = str(solver_name).upper()
        if solver_name not in {name.upper() for name in cp.installed_solvers()}:
            continue

        try:
            problem.solve(solver=getattr(cp, solver_name), verbose=False)
        except Exception:
            continue

        if str(problem.status).lower() in VALID_CVXPY_STATUSES:
            return True
    return False


def _solve_mpm_problem(
    mu_neg: np.ndarray,
    sigma_neg: np.ndarray,
    mu_pos: np.ndarray,
    sigma_pos: np.ndarray,
    objective_type: str,
    rho_neg: float,
    rho_pos: float,
    solver_name: str,
) -> LinearSurrogate | None:
    dimension = mu_neg.shape[0]
    sigma_neg_sqrt = sqrtm_psd(sigma_neg)
    sigma_pos_sqrt = sqrtm_psd(sigma_pos)

    w = cp.Variable(dimension)
    z = cp.Variable(2, nonneg=True)
    constraints = [
        (-mu_neg + mu_pos) @ w == 1.0,
        cp.SOC(z[0], sigma_neg_sqrt @ w),
        cp.SOC(z[1], sigma_pos_sqrt @ w),
    ]

    if objective_type == "mpm":
        objective = cp.Minimize(cp.sum(z))
    elif objective_type == "bw_rmpm":
        t = cp.Variable(nonneg=True)
        constraints.append(cp.SOC(t, w))
        objective = cp.Minimize(cp.sum(z) + float(rho_neg + rho_pos) * t)
    elif objective_type == "fr_rmpm":
        weights = np.exp(np.array([rho_neg, rho_pos], dtype=np.float64) / 2.0)
        objective = cp.Minimize(weights @ z)
    else:
        raise ValueError(f"Unsupported objective_type: {objective_type}")

    problem = cp.Problem(objective, constraints)
    if not solve_problem(problem, [solver_name, "SCS"]):
        return None
    if w.value is None or problem.value is None:
        return None

    w_opt = np.asarray(w.value, dtype=np.float64).reshape(-1)
    objective_value = float(problem.value)
    if (not np.isfinite(objective_value)) or objective_value <= 0:
        return None

    kappa = 1.0 / objective_value
    sigma_pos_term = float(np.sqrt(max(w_opt @ sigma_pos @ w_opt, 0.0)))

    if objective_type == "mpm":
        tau_pos = sigma_pos_term
    elif objective_type == "bw_rmpm":
        tau_pos = float(rho_pos) * float(np.linalg.norm(w_opt, ord=2)) + sigma_pos_term
    else:
        tau_pos = float(np.exp(rho_pos / 2.0)) * sigma_pos_term

    intercept = float(-(w_opt @ mu_pos - kappa * tau_pos))
    if not np.all(np.isfinite(w_opt)) or not np.isfinite(intercept):
        return None
    return LinearSurrogate(coef=w_opt, intercept=intercept)


def fit_local_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    rho_neg: float,
    rho_pos: float,
    solver_name: str,
) -> LinearSurrogate | None:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    X_neg = X[y == 0]
    X_pos = X[y == 1]
    if X_neg.shape[0] <= 1 or X_pos.shape[0] <= 1:
        return None

    mu_neg = X_neg.mean(axis=0)
    mu_pos = X_pos.mean(axis=0)
    sigma_neg = compute_sample_covariance(X_neg)
    sigma_pos = compute_sample_covariance(X_pos)

    if method == "mpm":
        return _solve_mpm_problem(
            mu_neg=mu_neg,
            sigma_neg=sigma_neg,
            mu_pos=mu_pos,
            sigma_pos=sigma_pos,
            objective_type="mpm",
            rho_neg=rho_neg,
            rho_pos=rho_pos,
            solver_name=solver_name,
        )
    if method == "quad_rmpm":
        dimension = mu_neg.shape[0]
        identity = np.eye(dimension, dtype=np.float64)
        return _solve_mpm_problem(
            mu_neg=mu_neg,
            sigma_neg=sigma_neg + np.sqrt(float(rho_neg)) * identity,
            mu_pos=mu_pos,
            sigma_pos=sigma_pos + np.sqrt(float(rho_pos)) * identity,
            objective_type="mpm",
            rho_neg=rho_neg,
            rho_pos=rho_pos,
            solver_name=solver_name,
        )
    if method == "bw_rmpm":
        return _solve_mpm_problem(
            mu_neg=mu_neg,
            sigma_neg=sigma_neg,
            mu_pos=mu_pos,
            sigma_pos=sigma_pos,
            objective_type="bw_rmpm",
            rho_neg=rho_neg,
            rho_pos=rho_pos,
            solver_name=solver_name,
        )
    if method == "fr_rmpm":
        return _solve_mpm_problem(
            mu_neg=mu_neg,
            sigma_neg=sigma_neg,
            mu_pos=mu_pos,
            sigma_pos=sigma_pos,
            objective_type="fr_rmpm",
            rho_neg=rho_neg,
            rho_pos=rho_pos,
            solver_name=solver_name,
        )
    raise ValueError(f"Unsupported surrogate method: {method}")


def solve_l1_projection(
    x0: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    boolean_feature_indices: Sequence[int],
    epsilon: float,
    solver_name: str,
) -> np.ndarray | None:
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    if x0.shape != coef.shape:
        raise ValueError("x0 and coef must have the same shape")

    bool_index_set = set(int(index) for index in boolean_feature_indices)
    if bool_index_set:
        variables = [
            cp.Variable(boolean=True, name=f"x_{index}")
            if index in bool_index_set
            else cp.Variable(name=f"x_{index}")
            for index in range(x0.shape[0])
        ]
        x = cp.hstack(variables)
        fallback_solvers: list[str] = []
    else:
        x = cp.Variable(x0.shape[0], name="x")
        fallback_solvers = ["CLARABEL", "SCIPY"]

    objective = cp.Minimize(cp.norm1(x - x0))
    constraints = [coef @ x + float(intercept) >= float(epsilon)]
    problem = cp.Problem(objective, constraints)
    if not solve_problem(problem, [solver_name, *fallback_solvers]):
        return None
    if x.value is None:
        return None

    solution = np.asarray(x.value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(solution)):
        return None
    if bool_index_set:
        solution[list(bool_index_set)] = np.clip(
            np.rint(solution[list(bool_index_set)]),
            0.0,
            1.0,
        )
    return solution
