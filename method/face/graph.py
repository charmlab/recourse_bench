from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.csgraph import dijkstra


@dataclass
class FaceSearchResult:
    counterfactual: np.ndarray
    path_found: bool
    path_cost: float
    path_hops: float
    endpoint_density: float
    endpoint_confidence: float
    top_paths: list[dict[str, float | int | list[dict[str, float]]]]


def unit_ball_volume(dim: int) -> float:
    return math.pi ** (dim / 2.0) / math.gamma(dim / 2.0 + 1.0)


def apply_weight_function(
    values: np.ndarray,
    weight_function: str,
    eps: float = 1e-12,
) -> np.ndarray:
    safe_values = np.clip(np.asarray(values, dtype=np.float64), eps, None)
    name = weight_function.lower()
    if name in {"negative_log", "-log", "neglog"}:
        transformed = -np.log(np.minimum(safe_values, 1.0))
        overflow_mask = safe_values > 1.0
        if np.any(overflow_mask):
            # Keep the paper's -log transform on [0, 1] and fall back to the
            # positive reciprocal form used by the reference implementation when
            # density proxies exceed 1.0 after normalization.
            transformed = transformed.copy()
            transformed[overflow_mask] = 1.0 / safe_values[overflow_mask]
        return np.maximum(transformed, eps)
    if name in {"identity", "none"}:
        return safe_values
    raise ValueError(f"Unsupported FACE weight_function: {weight_function}")


def build_connectivity_mask(
    distances: np.ndarray,
    feasible_mask: np.ndarray,
    mode: str,
    epsilon: float,
    k_neighbors: int,
) -> np.ndarray:
    base_mask = feasible_mask & (distances > 0.0) & (distances <= epsilon)
    if mode in {"kde", "epsilon"}:
        return base_mask

    mask = np.zeros_like(base_mask, dtype=bool)
    for row_index in range(base_mask.shape[0]):
        candidate_indices = np.flatnonzero(base_mask[row_index])
        if candidate_indices.size == 0:
            continue
        candidate_distances = distances[row_index, candidate_indices]
        selected_count = min(int(k_neighbors), candidate_indices.size)
        nearest_order = np.argsort(candidate_distances, kind="stable")[:selected_count]
        mask[row_index, candidate_indices[nearest_order]] = True
    return mask


def compute_edge_weights(
    *,
    source_values: np.ndarray,
    destination_values: np.ndarray,
    distances: np.ndarray,
    mode: str,
    epsilon: float,
    k_neighbors: int,
    num_reference_points: int,
    density_fn,
    weight_function: str,
) -> np.ndarray:
    if source_values.shape != destination_values.shape:
        raise ValueError(
            "source_values and destination_values must share the same shape"
        )
    if distances.ndim != 1 or distances.shape[0] != source_values.shape[0]:
        raise ValueError("distances must be a vector aligned with source_values")

    dim = int(source_values.shape[1])
    if mode == "kde":
        midpoints = 0.5 * (source_values + destination_values)
        proxy = density_fn(midpoints)
    elif mode == "knn":
        radius_term = float(k_neighbors) / (
            float(num_reference_points) * unit_ball_volume(dim)
        )
        proxy = radius_term / np.clip(distances, 1e-12, None)
    elif mode == "epsilon":
        proxy = (float(epsilon) ** dim) / np.clip(distances, 1e-12, None)
    else:
        raise ValueError(f"Unsupported FACE mode: {mode}")

    transformed = apply_weight_function(proxy, weight_function)
    return transformed * distances


def build_base_graph(
    *,
    values: np.ndarray,
    feasible_mask: np.ndarray,
    distances: np.ndarray,
    mode: str,
    epsilon: float,
    k_neighbors: int,
    density_fn,
    weight_function: str,
) -> csr_matrix:
    connectivity_mask = build_connectivity_mask(
        distances=distances,
        feasible_mask=feasible_mask,
        mode=mode,
        epsilon=epsilon,
        k_neighbors=k_neighbors,
    )
    graph = np.zeros_like(distances, dtype=np.float64)
    edge_indices = np.where(connectivity_mask)
    if edge_indices[0].size == 0:
        return csr_matrix(graph)

    graph[edge_indices] = compute_edge_weights(
        source_values=values[edge_indices[0]],
        destination_values=values[edge_indices[1]],
        distances=distances[edge_indices],
        mode=mode,
        epsilon=epsilon,
        k_neighbors=k_neighbors,
        num_reference_points=values.shape[0],
        density_fn=density_fn,
        weight_function=weight_function,
    )
    return csr_matrix(graph)


def build_source_edge_weights(
    *,
    source_value: np.ndarray,
    reference_values: np.ndarray,
    feasible_mask: np.ndarray,
    distances: np.ndarray,
    mode: str,
    epsilon: float,
    k_neighbors: int,
    density_fn,
    weight_function: str,
) -> np.ndarray:
    base_mask = feasible_mask & (distances > 0.0) & (distances <= epsilon)
    if mode == "knn":
        candidate_indices = np.flatnonzero(base_mask)
        if candidate_indices.size:
            selected_count = min(int(k_neighbors), candidate_indices.size)
            nearest_order = np.argsort(distances[candidate_indices], kind="stable")[
                :selected_count
            ]
            chosen_indices = candidate_indices[nearest_order]
            base_mask = np.zeros_like(base_mask, dtype=bool)
            base_mask[chosen_indices] = True

    weights = np.zeros(reference_values.shape[0], dtype=np.float64)
    edge_indices = np.flatnonzero(base_mask)
    if edge_indices.size == 0:
        return weights

    repeated_source = np.repeat(
        source_value.reshape(1, -1),
        repeats=edge_indices.size,
        axis=0,
    )
    weights[edge_indices] = compute_edge_weights(
        source_values=repeated_source,
        destination_values=reference_values[edge_indices],
        distances=distances[edge_indices],
        mode=mode,
        epsilon=epsilon,
        k_neighbors=k_neighbors,
        num_reference_points=reference_values.shape[0],
        density_fn=density_fn,
        weight_function=weight_function,
    )
    return weights


def augment_graph(base_graph: csr_matrix, source_weights: np.ndarray) -> csr_matrix:
    row = csr_matrix(source_weights.reshape(1, -1), dtype=np.float64)
    zero_column = csr_matrix((base_graph.shape[0], 1), dtype=np.float64)
    zero_tail = csr_matrix((1, 1), dtype=np.float64)
    top = hstack([base_graph, zero_column], format="csr")
    bottom = hstack([row, zero_tail], format="csr")
    return vstack([top, bottom], format="csr")


def reconstruct_path(
    predecessors: np.ndarray,
    source_index: int,
    target_index: int,
) -> list[int]:
    path = [int(target_index)]
    current = int(target_index)
    while current != source_index:
        previous = int(predecessors[current])
        if previous < 0:
            return []
        path.append(previous)
        current = previous
    path.reverse()
    return path


def search_paths(
    *,
    base_graph: csr_matrix,
    source_weights: np.ndarray,
    candidate_indices: np.ndarray,
    source_value: np.ndarray,
    reference_values: np.ndarray,
    feature_names: list[str],
    endpoint_densities: np.ndarray,
    endpoint_confidences: np.ndarray,
    store_top_k_paths: int,
) -> FaceSearchResult:
    source_index = int(reference_values.shape[0])
    graph = augment_graph(base_graph, source_weights)
    distances, predecessors = dijkstra(
        graph,
        directed=True,
        indices=source_index,
        return_predecessors=True,
    )

    reachable = candidate_indices[np.isfinite(distances[candidate_indices])]
    if reachable.size == 0:
        return FaceSearchResult(
            counterfactual=np.full(reference_values.shape[1], np.nan, dtype=np.float64),
            path_found=False,
            path_cost=float("nan"),
            path_hops=float("nan"),
            endpoint_density=float("nan"),
            endpoint_confidence=float("nan"),
            top_paths=[],
        )

    ordered_candidates = sorted(
        reachable.tolist(),
        key=lambda index: (float(distances[index]), int(index)),
    )
    top_paths: list[dict[str, float | int | list[dict[str, float]]]] = []
    for candidate_index in ordered_candidates[: int(store_top_k_paths)]:
        path_indices = reconstruct_path(
            predecessors=predecessors,
            source_index=source_index,
            target_index=int(candidate_index),
        )
        if not path_indices:
            continue

        trace: list[dict[str, float]] = []
        for path_index in path_indices:
            values = (
                source_value
                if path_index == source_index
                else reference_values[path_index]
            )
            trace.append(
                {
                    feature_name: float(feature_value)
                    for feature_name, feature_value in zip(feature_names, values)
                }
            )

        top_paths.append(
            {
                "cost": float(distances[candidate_index]),
                "hops": int(max(0, len(path_indices) - 1)),
                "endpoint_density": float(endpoint_densities[candidate_index]),
                "endpoint_confidence": float(endpoint_confidences[candidate_index]),
                "trace": trace,
            }
        )

    if not top_paths:
        return FaceSearchResult(
            counterfactual=np.full(reference_values.shape[1], np.nan, dtype=np.float64),
            path_found=False,
            path_cost=float("nan"),
            path_hops=float("nan"),
            endpoint_density=float("nan"),
            endpoint_confidence=float("nan"),
            top_paths=[],
        )

    best_candidate_index = int(ordered_candidates[0])
    best = top_paths[0]
    return FaceSearchResult(
        counterfactual=reference_values[best_candidate_index].astype(
            np.float64, copy=True
        ),
        path_found=True,
        path_cost=float(best["cost"]),
        path_hops=float(best["hops"]),
        endpoint_density=float(best["endpoint_density"]),
        endpoint_confidence=float(best["endpoint_confidence"]),
        top_paths=top_paths,
    )
