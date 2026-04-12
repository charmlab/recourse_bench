from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree

from method.wachter.support import ensure_supported_target_model
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject

try:  # pragma: no cover - optional dependency
    from alibi.confidence import TrustScore
except Exception:  # pragma: no cover - fallback path
    TrustScore = None


@dataclass
class DiverseDistTrace:
    status: str
    target_class_index: int | None
    original_class_index: int | None
    candidate_indices: list[int]
    candidate_distances: list[float]
    selected_candidate_indices: list[int]
    selected_candidate_distances: list[float]
    counterfactuals: list[np.ndarray]
    chosen_counterfactual: np.ndarray | None


def normalize_candidate_selection(candidate_selection: str) -> str:
    value = str(candidate_selection).lower()
    if value not in {"angle", "distance"}:
        raise ValueError("candidate_selection must be 'angle' or 'distance'")
    return value


def normalize_norm(norm: int | float) -> int:
    resolved = int(norm)
    if resolved not in {1, 2}:
        raise ValueError("norm must be 1 or 2")
    return resolved


def to_numpy_features(values: pd.DataFrame | np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        try:
            return values.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
        except ValueError as exc:
            raise ValueError("DiverseDistMethod requires numeric feature values") from exc

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


class DiverseDistModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        ensure_supported_target_model(
            target_model,
            (MlpModel,),
            "DiverseDistModelAdapter",
        )
        raw_model = getattr(target_model, "_model", None)
        if not isinstance(raw_model, torch.nn.Module):
            raise TypeError("DiverseDistMethod requires a torch-backed target model")

        self._target_model = target_model
        self._raw_model = raw_model
        self._feature_names = list(feature_names)
        self._device = target_model._device
        self._output_activation = str(
            getattr(target_model, "_output_activation_name", "softmax")
        ).lower()

    def predict_label_indices(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        array = np.ascontiguousarray(
            to_numpy_features(X, self._feature_names),
            dtype=np.float32,
        )
        tensor = torch.from_numpy(array).to(self._device)
        self._raw_model.eval()
        with torch.inference_mode():
            logits = self._raw_model(tensor)
            if self._output_activation == "sigmoid":
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                probabilities = torch.cat(
                    [1.0 - torch.sigmoid(logits), torch.sigmoid(logits)],
                    dim=1,
                )
            else:
                probabilities = torch.softmax(logits, dim=1)
        return probabilities.argmax(dim=1).detach().cpu().numpy()


def build_class_kdtrees(
    train_array: np.ndarray,
    predicted_labels: np.ndarray,
    num_classes: int,
) -> tuple[dict[int, KDTree], dict[int, np.ndarray], dict[int, np.ndarray]]:
    kdtrees: dict[int, KDTree] = {}
    class_points: dict[int, np.ndarray] = {}
    class_indices: dict[int, np.ndarray] = {}

    for class_index in range(int(num_classes)):
        indices = np.flatnonzero(predicted_labels == class_index)
        if indices.size == 0:
            continue
        class_points[class_index] = train_array[indices]
        class_indices[class_index] = indices

    if TrustScore is not None and len(class_points) == int(num_classes):
        trustscore = TrustScore()
        trustscore.fit(train_array, predicted_labels, classes=int(num_classes))
        for class_index in class_points:
            kdtrees[class_index] = trustscore.kdtrees[class_index]
        return kdtrees, class_points, class_indices

    for class_index, points in class_points.items():
        kdtrees[class_index] = KDTree(points, metric="euclidean")

    return kdtrees, class_points, class_indices


def query_opposite_class_candidates(
    factual: np.ndarray,
    target_class_index: int,
    alpha: int,
    kdtrees: dict[int, KDTree],
    class_points: dict[int, np.ndarray],
    class_indices: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_class_index not in kdtrees:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0, factual.shape[0]), dtype=np.float32),
        )

    kdtree = kdtrees[target_class_index]
    points = class_points[target_class_index]
    k = min(int(alpha), points.shape[0])
    if k <= 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0, factual.shape[0]), dtype=np.float32),
        )

    distances, local_indices = kdtree.query(factual.reshape(1, -1), k=k)
    local_indices = local_indices.reshape(-1)
    distances = distances.reshape(-1).astype(np.float32, copy=False)

    if hasattr(kdtree, "get_arrays"):
        try:
            tree_data, *_ = kdtree.get_arrays()
            selected_points = np.asarray(tree_data[local_indices], dtype=np.float32)
            selected_indices = local_indices.astype(np.int64, copy=False)
            return selected_indices, distances, selected_points
        except Exception:
            pass

    return (
        class_indices[target_class_index][local_indices].astype(np.int64, copy=False),
        distances,
        points[local_indices].astype(np.float32, copy=False),
    )


def _l1_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.abs(x - y)))


def _distance(x: np.ndarray, y: np.ndarray, norm: int) -> float:
    if norm == 1:
        return _l1_distance(x, y)
    return float(np.linalg.norm(x - y, ord=2))


def select_diverse_candidates_angle(
    factual: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    candidate_points: np.ndarray,
    beta: float,
    max_candidates: int,
) -> tuple[list[int], list[float], list[np.ndarray]]:
    selected_indices: list[int] = []
    selected_distances: list[float] = []
    selected_points: list[np.ndarray] = []
    directions: list[np.ndarray] = []

    for index, distance, point in zip(
        candidate_indices.tolist(),
        candidate_distances.tolist(),
        candidate_points,
    ):
        direction = point - factual
        direction_norm = np.linalg.norm(direction, ord=2)
        if direction_norm <= 1e-12:
            continue
        normalized_direction = direction / direction_norm

        if not directions:
            directions.append(normalized_direction)
            selected_indices.append(int(index))
            selected_distances.append(float(distance))
            selected_points.append(point.astype(np.float32, copy=True))
        else:
            if all(
                abs(float(np.dot(normalized_direction, existing_direction))) <= float(beta)
                for existing_direction in directions
            ):
                directions.append(normalized_direction)
                selected_indices.append(int(index))
                selected_distances.append(float(distance))
                selected_points.append(point.astype(np.float32, copy=True))

        if len(selected_points) >= int(max_candidates):
            break

    return selected_indices, selected_distances, selected_points


def select_diverse_candidates_distance(
    factual: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    candidate_points: np.ndarray,
    beta: float,
    max_candidates: int,
    norm: int,
) -> tuple[list[int], list[float], list[np.ndarray]]:
    selected_indices: list[int] = []
    selected_distances: list[float] = []
    selected_points: list[np.ndarray] = []
    if candidate_points.shape[0] == 0:
        return selected_indices, selected_distances, selected_points

    min_dist = _distance(factual, candidate_points[0], norm)
    for index, distance, point in zip(
        candidate_indices.tolist(),
        candidate_distances.tolist(),
        candidate_points,
    ):
        if not selected_points:
            selected_indices.append(int(index))
            selected_distances.append(float(distance))
            selected_points.append(point.astype(np.float32, copy=True))
        else:
            if all(
                _distance(point, selected_point, norm) >= (1.0 + float(beta)) * min_dist
                for selected_point in selected_points
            ):
                selected_indices.append(int(index))
                selected_distances.append(float(distance))
                selected_points.append(point.astype(np.float32, copy=True))

        if len(selected_points) >= int(max_candidates):
            break

    return selected_indices, selected_distances, selected_points


def resolve_binary_search_iterations(
    dist_max: float,
    gamma: float,
) -> int:
    if dist_max <= 0.0:
        return 0
    return max(
        0,
        int(np.ceil(np.log2(max(float(dist_max), 1e-12) / float(gamma)))),
    )


def compute_counterfactuals_on_segments(
    model_adapter: DiverseDistModelAdapter,
    factual: np.ndarray,
    candidates: Sequence[np.ndarray] | np.ndarray,
    original_class_index: int,
    gamma: float,
    dist_max: float,
    opt: bool,
) -> list[np.ndarray]:
    candidate_array = np.asarray(candidates, dtype=np.float32)
    if candidate_array.size == 0:
        return []
    if candidate_array.ndim == 1:
        candidate_array = candidate_array.reshape(1, -1)

    if not opt:
        return [candidate.copy() for candidate in candidate_array]

    iterations = resolve_binary_search_iterations(dist_max=dist_max, gamma=gamma)
    counterfactuals = candidate_array.copy()
    if iterations == 0:
        return [candidate.copy() for candidate in counterfactuals]

    lower = np.zeros(candidate_array.shape[0], dtype=np.float32)
    upper = np.ones(candidate_array.shape[0], dtype=np.float32)
    factual_array = np.asarray(factual, dtype=np.float32).reshape(1, -1)
    original_class_index = int(original_class_index)

    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        probe = (
            midpoint[:, None] * factual_array
            + (1.0 - midpoint)[:, None] * candidate_array
        )
        prediction = model_adapter.predict_label_indices(probe)
        same_class = prediction == original_class_index
        upper[same_class] = midpoint[same_class]

        flipped = ~same_class
        lower[flipped] = midpoint[flipped]
        if flipped.any():
            counterfactuals[flipped] = probe[flipped].astype(np.float32, copy=False)

    return [candidate.copy() for candidate in counterfactuals]


def choose_best_counterfactual(
    factual: np.ndarray,
    counterfactuals: list[np.ndarray],
    norm: int,
) -> np.ndarray | None:
    if not counterfactuals:
        return None
    distances = [_distance(factual, counterfactual, norm) for counterfactual in counterfactuals]
    return counterfactuals[int(np.argmin(np.asarray(distances)))].astype(
        np.float32,
        copy=True,
    )


def generate_diverse_counterfactuals(
    model_adapter: DiverseDistModelAdapter,
    factual: np.ndarray,
    original_class_index: int,
    target_class_index: int,
    alpha: int,
    total_cfs: int,
    beta: float,
    gamma: float,
    norm: int,
    candidate_selection: str,
    opt: bool,
    kdtrees: dict[int, KDTree],
    class_points: dict[int, np.ndarray],
    class_indices: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], DiverseDistTrace]:
    candidate_indices, candidate_distances, candidate_points = query_opposite_class_candidates(
        factual=factual,
        target_class_index=target_class_index,
        alpha=alpha,
        kdtrees=kdtrees,
        class_points=class_points,
        class_indices=class_indices,
    )

    if candidate_points.shape[0] == 0:
        return [], DiverseDistTrace(
            status="no_opposite_class_candidates",
            target_class_index=int(target_class_index),
            original_class_index=int(original_class_index),
            candidate_indices=[],
            candidate_distances=[],
            selected_candidate_indices=[],
            selected_candidate_distances=[],
            counterfactuals=[],
            chosen_counterfactual=None,
        )

    if candidate_selection == "distance":
        selected_indices, selected_distances, selected_points = select_diverse_candidates_distance(
            factual=factual,
            candidate_indices=candidate_indices,
            candidate_distances=candidate_distances,
            candidate_points=candidate_points,
            beta=beta,
            max_candidates=total_cfs,
            norm=norm,
        )
    else:
        selected_indices, selected_distances, selected_points = select_diverse_candidates_angle(
            factual=factual,
            candidate_indices=candidate_indices,
            candidate_distances=candidate_distances,
            candidate_points=candidate_points,
            beta=beta,
            max_candidates=total_cfs,
        )

    if not selected_points:
        return [], DiverseDistTrace(
            status="no_diverse_candidates",
            target_class_index=int(target_class_index),
            original_class_index=int(original_class_index),
            candidate_indices=candidate_indices.tolist(),
            candidate_distances=candidate_distances.astype(np.float64).tolist(),
            selected_candidate_indices=[],
            selected_candidate_distances=[],
            counterfactuals=[],
            chosen_counterfactual=None,
        )

    dist_max = float(candidate_distances[-1]) if candidate_distances.size else 0.0
    counterfactuals = compute_counterfactuals_on_segments(
        model_adapter=model_adapter,
        factual=factual,
        candidates=selected_points,
        original_class_index=original_class_index,
        gamma=gamma,
        dist_max=dist_max,
        opt=opt,
    )

    chosen_counterfactual = choose_best_counterfactual(
        factual=factual,
        counterfactuals=counterfactuals,
        norm=norm,
    )
    status = "success" if chosen_counterfactual is not None else "no_valid_counterfactuals"
    return counterfactuals, DiverseDistTrace(
        status=status,
        target_class_index=int(target_class_index),
        original_class_index=int(original_class_index),
        candidate_indices=candidate_indices.tolist(),
        candidate_distances=candidate_distances.astype(np.float64).tolist(),
        selected_candidate_indices=selected_indices,
        selected_candidate_distances=selected_distances,
        counterfactuals=[cf.copy() for cf in counterfactuals],
        chosen_counterfactual=None
        if chosen_counterfactual is None
        else chosen_counterfactual.copy(),
    )
