from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import linalg as LA


def hyper_sphere_coordinates(n_search_samples, instance, high, low, p_norm=2):
    delta_instance = np.random.randn(n_search_samples, instance.shape[1])
    dist = np.random.rand(n_search_samples) * (high - low) + low
    norm_p = LA.norm(delta_instance, ord=p_norm, axis=1)
    d_norm = np.divide(dist, norm_p).reshape(-1, 1)
    delta_instance = np.multiply(delta_instance, d_norm)
    return instance + delta_instance, dist


def growing_spheres_search(
    instance,
    keys_mutable,
    keys_immutable,
    continuous_cols,
    binary_cols,
    feature_order,
    model,
    target_label=None,
    n_search_samples=1000,
    p_norm=2,
    eta=0.2,
    max_iter=1000,
):
    keys_correct = list(feature_order)
    keys_mutable_continuous = [
        feature for feature in keys_mutable if feature not in set(binary_cols)
    ]
    keys_mutable_binary = [
        feature for feature in keys_mutable if feature not in set(continuous_cols)
    ]

    factual = (
        instance.values if isinstance(instance, pd.Series) else np.asarray(instance)
    )
    factual = factual.astype(float, copy=False)
    if factual.ndim != 1:
        factual = factual.reshape(-1)

    instance_label = int(model.predict_label_indices(factual.reshape(1, -1))[0])
    target_label = instance_label if target_label is None else int(target_label)

    candidate_counterfactual_star = np.full(factual.shape[0], np.nan, dtype=float)
    iteration_count = 0
    current_eta = float(eta)

    while iteration_count < max_iter:
        iteration_count += 1
        initial_ball = _sample_candidates(
            instance=instance,
            keys_immutable=keys_immutable,
            keys_mutable_continuous=keys_mutable_continuous,
            keys_mutable_binary=keys_mutable_binary,
            feature_order=keys_correct,
            n_search_samples=n_search_samples,
            low=0.0,
            high=current_eta,
            p_norm=p_norm,
        )
        initial_enemies = _select_enemies(
            candidates=initial_ball,
            factual=factual,
            model=model,
            instance_label=instance_label,
            target_label=target_label,
            distance_features=keys_mutable_continuous,
            feature_order=keys_correct,
        )
        if initial_enemies[0].shape[0] == 0:
            break
        current_eta /= 2.0
        if current_eta <= np.finfo(float).eps:
            break

    low = current_eta
    high = current_eta * 2.0
    while iteration_count < max_iter:
        iteration_count += 1
        shell_candidates = _sample_candidates(
            instance=instance,
            keys_immutable=keys_immutable,
            keys_mutable_continuous=keys_mutable_continuous,
            keys_mutable_binary=keys_mutable_binary,
            feature_order=keys_correct,
            n_search_samples=n_search_samples,
            low=low,
            high=high,
            p_norm=p_norm,
        )
        enemy_candidates, enemy_distances = _select_enemies(
            candidates=shell_candidates,
            factual=factual,
            model=model,
            instance_label=instance_label,
            target_label=target_label,
            distance_features=keys_mutable_continuous,
            feature_order=keys_correct,
        )
        if enemy_candidates.shape[0] > 0:
            candidate_counterfactual_star = enemy_candidates[
                int(np.argmin(enemy_distances))
            ]
            break
        low = high
        high = high + current_eta

    return feature_selection(
        instance,
        candidate_counterfactual_star,
        model,
        keys_mutable,
        feature_order,
        target_label=target_label,
    )


def _sample_candidates(
    instance,
    keys_immutable,
    keys_mutable_continuous,
    keys_mutable_binary,
    feature_order,
    n_search_samples,
    low,
    high,
    p_norm,
):
    instance_immutable_replicated = np.repeat(
        instance[keys_immutable].values.reshape(1, -1), n_search_samples, axis=0
    )
    if keys_mutable_continuous:
        instance_mutable_replicated_continuous = np.repeat(
            instance[keys_mutable_continuous].values.reshape(1, -1),
            n_search_samples,
            axis=0,
        )
        candidate_counterfactuals_continuous, _ = hyper_sphere_coordinates(
            n_search_samples,
            instance_mutable_replicated_continuous,
            high,
            low,
            p_norm,
        )
    else:
        candidate_counterfactuals_continuous = np.empty((n_search_samples, 0))

    if keys_mutable_binary:
        candidate_counterfactuals_binary = np.random.binomial(
            n=1,
            p=0.5,
            size=n_search_samples * len(keys_mutable_binary),
        ).reshape(n_search_samples, -1)
    else:
        candidate_counterfactuals_binary = np.empty((n_search_samples, 0))

    candidate_counterfactuals = pd.DataFrame(
        np.c_[
            instance_immutable_replicated,
            candidate_counterfactuals_continuous,
            candidate_counterfactuals_binary,
        ]
    )
    candidate_counterfactuals.columns = (
        list(keys_immutable) + list(keys_mutable_continuous) + list(keys_mutable_binary)
    )
    return candidate_counterfactuals.loc[:, list(feature_order)]


def _select_enemies(
    candidates,
    factual,
    model,
    instance_label,
    target_label,
    distance_features,
    feature_order,
):
    if distance_features:
        distance_indices = [feature_order.index(feature) for feature in distance_features]
        distances = LA.norm(
            candidates.values[:, distance_indices]
            - factual.reshape(1, -1)[:, distance_indices],
            ord=2,
            axis=1,
        )
    else:
        distances = np.zeros(candidates.shape[0], dtype=float)
    predictions = model.predict_label_indices(candidates.values)
    enemy_mask = predictions == int(target_label)
    if int(target_label) == int(instance_label):
        enemy_mask = predictions != int(instance_label)
    return candidates.values[enemy_mask], distances[enemy_mask]


def feature_selection(
    instance,
    candidate_counterfactual,
    model,
    keys_mutable,
    feature_order,
    target_label=None,
):
    if np.isnan(candidate_counterfactual).any():
        return candidate_counterfactual

    factual = (
        instance.values if isinstance(instance, pd.Series) else np.asarray(instance)
    )
    factual = factual.astype(float, copy=False)
    if factual.ndim != 1:
        factual = factual.reshape(-1)

    instance_label = int(model.predict_label_indices(factual.reshape(1, -1))[0])
    if target_label is None:
        target_label = instance_label

    mutable_indices = [feature_order.index(feature) for feature in keys_mutable]
    current = candidate_counterfactual.astype(float, copy=True)
    best = current.copy()

    while True:
        current_label = int(model.predict_label_indices(current.reshape(1, -1))[0])
        is_enemy = current_label == int(target_label)
        if int(target_label) == int(instance_label):
            is_enemy = current_label != int(instance_label)
        if not is_enemy:
            break

        best = current.copy()
        changed_indices = [
            idx for idx in mutable_indices if current[idx] != factual[idx]
        ]
        if not changed_indices:
            break
        changed_indices.sort(key=lambda idx: abs(current[idx] - factual[idx]))
        current[changed_indices[0]] = factual[changed_indices[0]]

    return best
