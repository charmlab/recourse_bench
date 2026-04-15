from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from method.dice.support import RecourseModelAdapter, project_discrete_feature_space
from method.probe.utils import CategoricalGroup


@dataclass(frozen=True)
class DiceSearchConfig:
    total_cfs: int
    algorithm: str
    yloss_type: str
    diversity_loss_type: str
    proximity_weight: float
    diversity_weight: float
    categorical_penalty: float
    optimizer: str
    learning_rate: float
    min_iter: int
    max_iter: int
    project_iter: int
    loss_diff_thres: float
    loss_converge_maxiter: int
    init_near_query_instance: bool
    tie_random: bool
    stopping_threshold: float
    posthoc_sparsity_param: float
    posthoc_sparsity_algorithm: str
    respect_mutability: bool
    verbose: bool


@dataclass(frozen=True)
class DiceSearchMetadata:
    feature_names: tuple[str, ...]
    continuous_indices: tuple[int, ...]
    continuous_features: tuple[str, ...]
    categorical_groups: tuple[CategoricalGroup, ...]
    binary_feature_value_map: dict[int, np.ndarray]
    lower_bounds: torch.Tensor
    upper_bounds: torch.Tensor
    feature_weights: torch.Tensor
    mutable_mask: torch.Tensor
    sparsity_thresholds: dict[str, float]
    sparsity_order: tuple[int, ...]
    device: torch.device


def optimize_diverse_counterfactuals(
    query_instance: np.ndarray,
    target_index: int,
    adapter: RecourseModelAdapter,
    metadata: DiceSearchMetadata,
    config: DiceSearchConfig,
) -> pd.DataFrame:
    query_tensor = torch.as_tensor(
        query_instance,
        dtype=torch.float32,
        device=metadata.device,
    ).reshape(-1)

    collected_rows: list[np.ndarray] = []
    if config.algorithm == "randominitcf":
        for _ in range(config.total_cfs):
            run_rows = _run_single_search(
                query_tensor=query_tensor,
                target_index=target_index,
                total_cfs=1,
                adapter=adapter,
                metadata=metadata,
                config=config,
                diversity_weight=0.0,
            )
            if run_rows.size > 0:
                collected_rows.append(run_rows[0])
    else:
        run_rows = _run_single_search(
            query_tensor=query_tensor,
            target_index=target_index,
            total_cfs=config.total_cfs,
            adapter=adapter,
            metadata=metadata,
            config=config,
            diversity_weight=config.diversity_weight,
        )
        if run_rows.size > 0:
            collected_rows.extend(list(run_rows))

    if not collected_rows:
        return pd.DataFrame(columns=list(metadata.feature_names))

    candidates = pd.DataFrame(collected_rows, columns=list(metadata.feature_names))
    numeric_columns = candidates.columns
    candidates.loc[:, numeric_columns] = candidates.loc[:, numeric_columns].round(6)
    candidates = candidates.drop_duplicates(ignore_index=True)
    return candidates


def _run_single_search(
    query_tensor: torch.Tensor,
    target_index: int,
    total_cfs: int,
    adapter: RecourseModelAdapter,
    metadata: DiceSearchMetadata,
    config: DiceSearchConfig,
    diversity_weight: float,
) -> np.ndarray:
    cfs = _initialize_counterfactuals(
        query_tensor=query_tensor,
        total_cfs=total_cfs,
        metadata=metadata,
        config=config,
    )
    optimizer = _build_optimizer(config.optimizer, [cfs], config.learning_rate)

    effective_threshold = _resolve_stopping_threshold(
        target_index=target_index,
        stopping_threshold=config.stopping_threshold,
    )

    best_valid: torch.Tensor | None = None
    best_valid_score = float("inf")
    loss_converge_iter = 0
    previous_loss: float | None = None

    for iteration in range(config.max_iter):
        optimizer.zero_grad()
        loss = _compute_loss(
            cfs=cfs,
            query_tensor=query_tensor,
            target_index=target_index,
            adapter=adapter,
            metadata=metadata,
            config=config,
            diversity_weight=diversity_weight,
        )
        loss.backward()

        if config.respect_mutability:
            cfs.grad[:, ~metadata.mutable_mask] = 0.0

        optimizer.step()

        with torch.no_grad():
            cfs.copy_(
                torch.maximum(
                    torch.minimum(cfs, metadata.upper_bounds.unsqueeze(0)),
                    metadata.lower_bounds.unsqueeze(0),
                )
            )

            if config.project_iter > 0 and (iteration + 1) % config.project_iter == 0:
                cfs.copy_(
                    project_discrete_feature_space(
                        cfs,
                        categorical_groups=metadata.categorical_groups,
                        binary_feature_value_map=metadata.binary_feature_value_map,
                        tie_random=config.tie_random,
                    )
                )

            projected = project_discrete_feature_space(
                cfs,
                categorical_groups=metadata.categorical_groups,
                binary_feature_value_map=metadata.binary_feature_value_map,
                tie_random=config.tie_random,
            )
            probabilities = _positive_probability(projected, adapter)
            valid_mask = _is_valid(
                probabilities=probabilities,
                target_index=target_index,
                stopping_threshold=effective_threshold,
            )

            if bool(valid_mask.all()):
                threshold_distance = torch.abs(
                    probabilities - effective_threshold
                ).mean()
                current_score = float(threshold_distance.item())
                if current_score < best_valid_score:
                    best_valid_score = current_score
                    best_valid = projected.detach().clone()

        loss_value = float(loss.detach().item())
        if config.verbose and (iteration + 1) % 50 == 0:
            print(f"dice-search step={iteration + 1} loss={loss_value:.6f}")

        if iteration + 1 < config.min_iter:
            previous_loss = loss_value
            continue

        if (
            previous_loss is not None
            and abs(loss_value - previous_loss) <= config.loss_diff_thres
        ):
            loss_converge_iter += 1
        else:
            loss_converge_iter = 0
        previous_loss = loss_value

        if (
            loss_converge_iter >= config.loss_converge_maxiter
            and best_valid is not None
        ):
            break

    with torch.no_grad():
        projected_final = project_discrete_feature_space(
            cfs.detach(),
            categorical_groups=metadata.categorical_groups,
            binary_feature_value_map=metadata.binary_feature_value_map,
            tie_random=config.tie_random,
        )
        probabilities = _positive_probability(projected_final, adapter)
        valid_mask = _is_valid(
            probabilities=probabilities,
            target_index=target_index,
            stopping_threshold=effective_threshold,
        )

        if best_valid is not None and not bool(valid_mask.all()):
            chosen = best_valid
            probabilities = _positive_probability(chosen, adapter)
            valid_mask = _is_valid(
                probabilities=probabilities,
                target_index=target_index,
                stopping_threshold=effective_threshold,
            )
        else:
            chosen = projected_final

        if not bool(valid_mask.any()):
            return np.empty((0, query_tensor.numel()), dtype=np.float32)

        valid_candidates = chosen[valid_mask].detach().clone()
        valid_candidates = _apply_posthoc_sparsity(
            candidates=valid_candidates,
            query_tensor=query_tensor,
            target_index=target_index,
            adapter=adapter,
            metadata=metadata,
            config=config,
            stopping_threshold=effective_threshold,
        )
        if valid_candidates.numel() == 0:
            return np.empty((0, query_tensor.numel()), dtype=np.float32)

    return valid_candidates.detach().cpu().numpy()


def _initialize_counterfactuals(
    query_tensor: torch.Tensor,
    total_cfs: int,
    metadata: DiceSearchMetadata,
    config: DiceSearchConfig,
) -> torch.nn.Parameter:
    query_row = query_tensor.reshape(1, -1).repeat(total_cfs, 1)
    random_start = torch.rand(
        (total_cfs, query_tensor.numel()),
        device=metadata.device,
        dtype=torch.float32,
    )
    random_start = metadata.lower_bounds.unsqueeze(0) + random_start * (
        metadata.upper_bounds - metadata.lower_bounds
    ).unsqueeze(0)

    if config.init_near_query_instance:
        initial = query_row.clone()
        for row_index in range(total_cfs):
            initial[row_index] = query_row[row_index] + row_index * 0.01
    else:
        initial = random_start

    if config.respect_mutability:
        initial[:, ~metadata.mutable_mask] = query_row[:, ~metadata.mutable_mask]

    return torch.nn.Parameter(initial)


def _build_optimizer(
    optimizer_name: str,
    parameters: list[torch.nn.Parameter],
    learning_rate: float,
) -> torch.optim.Optimizer:
    resolved_name = optimizer_name.lower()
    if ":" in resolved_name:
        resolved_name = resolved_name.split(":", 1)[1]

    if resolved_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if resolved_name in {"rms", "rmsprop"}:
        return torch.optim.RMSprop(parameters, lr=learning_rate)
    raise ValueError(f"Unsupported Dice optimizer: {optimizer_name}")


def _compute_loss(
    cfs: torch.Tensor,
    query_tensor: torch.Tensor,
    target_index: int,
    adapter: RecourseModelAdapter,
    metadata: DiceSearchMetadata,
    config: DiceSearchConfig,
    diversity_weight: float,
) -> torch.Tensor:
    yloss = _compute_yloss(
        cfs=cfs,
        target_index=target_index,
        adapter=adapter,
        yloss_type=config.yloss_type,
    )
    proximity_loss = (
        _compute_proximity_loss(
            cfs=cfs,
            query_tensor=query_tensor,
            feature_weights=metadata.feature_weights,
        )
        if config.proximity_weight > 0
        else torch.zeros((), device=metadata.device)
    )
    diversity_loss = (
        _compute_diversity_loss(
            cfs=cfs,
            feature_weights=metadata.feature_weights,
            diversity_loss_type=config.diversity_loss_type,
        )
        if diversity_weight > 0 and cfs.shape[0] > 1
        else torch.zeros((), device=metadata.device)
    )
    regularization_loss = _compute_regularization_loss(
        cfs=cfs,
        categorical_groups=metadata.categorical_groups,
    )
    return (
        yloss
        + config.proximity_weight * proximity_loss
        - diversity_weight * diversity_loss
        + config.categorical_penalty * regularization_loss
    )


def _compute_yloss(
    cfs: torch.Tensor,
    target_index: int,
    adapter: RecourseModelAdapter,
    yloss_type: str,
) -> torch.Tensor:
    probabilities = _positive_probability(cfs, adapter).clamp(min=1e-6, max=1.0 - 1e-6)
    target_tensor = torch.full_like(probabilities, float(target_index))
    resolved_type = yloss_type.lower()

    if resolved_type == "l2_loss":
        return torch.mean((probabilities - target_tensor) ** 2)

    logits = torch.log(probabilities / (1.0 - probabilities))
    if resolved_type == "log_loss":
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target_tensor,
        )
    if resolved_type != "hinge_loss":
        raise ValueError(f"Unsupported Dice yloss_type: {yloss_type}")

    labels = torch.where(
        target_tensor > 0.5,
        torch.ones_like(target_tensor),
        -torch.ones_like(target_tensor),
    )
    return torch.relu(1.0 - labels * logits).mean()


def _compute_proximity_loss(
    cfs: torch.Tensor,
    query_tensor: torch.Tensor,
    feature_weights: torch.Tensor,
) -> torch.Tensor:
    weighted_diff = torch.abs(
        cfs - query_tensor.unsqueeze(0)
    ) * feature_weights.unsqueeze(0)
    return weighted_diff.sum() / (cfs.shape[0] * cfs.shape[1])


def _compute_diversity_loss(
    cfs: torch.Tensor,
    feature_weights: torch.Tensor,
    diversity_loss_type: str,
) -> torch.Tensor:
    pairwise_distance = torch.sum(
        torch.abs(cfs.unsqueeze(1) - cfs.unsqueeze(0)) * feature_weights.view(1, 1, -1),
        dim=2,
    )
    resolved_type = diversity_loss_type.lower()

    if resolved_type == "avg_dist":
        upper = torch.triu_indices(cfs.shape[0], cfs.shape[0], offset=1)
        similarities = 1.0 / (1.0 + pairwise_distance[upper[0], upper[1]])
        return 1.0 - similarities.mean()

    if not resolved_type.startswith("dpp_style:"):
        raise ValueError(
            "DiceMethod supports diversity_loss_type='avg_dist' or "
            "'dpp_style:inverse_dist'/'dpp_style:exponential_dist' only"
        )

    submethod = resolved_type.split(":", 1)[1]
    if submethod == "inverse_dist":
        kernel = 1.0 / (1.0 + pairwise_distance)
    elif submethod == "exponential_dist":
        kernel = torch.exp(-pairwise_distance)
    else:
        raise ValueError(f"Unsupported Dice DPP diversity submethod: {submethod}")

    kernel = kernel + torch.eye(cfs.shape[0], device=cfs.device, dtype=cfs.dtype) * 1e-4
    return torch.det(kernel)


def _compute_regularization_loss(
    cfs: torch.Tensor,
    categorical_groups: tuple[CategoricalGroup, ...],
) -> torch.Tensor:
    if not categorical_groups:
        return torch.zeros((), device=cfs.device)

    regularization = torch.zeros((), device=cfs.device)
    for group in categorical_groups:
        group_indices = list(group.indices)
        group_values = cfs[:, group_indices]
        regularization = regularization + torch.sum(
            (torch.sum(group_values, dim=1) - 1.0) ** 2
        )
    return regularization


def _positive_probability(
    candidates: torch.Tensor,
    adapter: RecourseModelAdapter,
) -> torch.Tensor:
    probabilities = adapter.predict_proba(candidates)
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Dice search requires a binary probability model")
    return probabilities[:, 1]


def _resolve_stopping_threshold(
    target_index: int,
    stopping_threshold: float,
) -> float:
    if target_index == 0 and stopping_threshold > 0.5:
        return 0.25
    if target_index == 1 and stopping_threshold < 0.5:
        return 0.75
    return stopping_threshold


def _is_valid(
    probabilities: torch.Tensor,
    target_index: int,
    stopping_threshold: float,
) -> torch.Tensor:
    if target_index == 0:
        return probabilities <= stopping_threshold
    return probabilities >= stopping_threshold


def _apply_posthoc_sparsity(
    candidates: torch.Tensor,
    query_tensor: torch.Tensor,
    target_index: int,
    adapter: RecourseModelAdapter,
    metadata: DiceSearchMetadata,
    config: DiceSearchConfig,
    stopping_threshold: float,
) -> torch.Tensor:
    if (
        config.posthoc_sparsity_param <= 0.0
        or not metadata.continuous_indices
        or not metadata.sparsity_order
    ):
        return candidates

    sparse_candidates = candidates.detach().clone()
    for row_index in range(sparse_candidates.shape[0]):
        current = sparse_candidates[row_index].clone()
        for feature_index in metadata.sparsity_order:
            feature_name = metadata.feature_names[feature_index]
            threshold = metadata.sparsity_thresholds.get(feature_name, 0.0)
            if threshold <= 0.0:
                continue

            diff = float(
                torch.abs(query_tensor[feature_index] - current[feature_index]).item()
            )
            if diff > threshold:
                continue

            trial = current.clone()
            trial[feature_index] = query_tensor[feature_index]
            trial = project_discrete_feature_space(
                trial,
                categorical_groups=metadata.categorical_groups,
                binary_feature_value_map=metadata.binary_feature_value_map,
                tie_random=config.tie_random,
            )
            trial_probability = _positive_probability(
                trial.reshape(1, -1),
                adapter,
            )
            if bool(
                _is_valid(
                    probabilities=trial_probability,
                    target_index=target_index,
                    stopping_threshold=stopping_threshold,
                )[0].item()
            ):
                current = trial
        sparse_candidates[row_index] = current

    return sparse_candidates
