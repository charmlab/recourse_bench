from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.optim as optim
from torch import nn

DECISION_THRESHOLD = 0.5


@dataclass(frozen=True)
class CategoricalGroup:
    indices: tuple[int, ...]
    encoding: str


def infer_categorical_groups(
    feature_names: Sequence[str],
    encoding_map: Mapping[str, Sequence[str]] | None,
) -> list[CategoricalGroup]:
    if not encoding_map:
        return []

    feature_index = {feature_name: index for index, feature_name in enumerate(feature_names)}
    groups: list[CategoricalGroup] = []
    seen_groups: set[tuple[int, ...]] = set()

    for encoded_columns in encoding_map.values():
        resolved_indices = tuple(
            feature_index[column]
            for column in encoded_columns
            if column in feature_index
        )
        if len(resolved_indices) < 2 or resolved_indices in seen_groups:
            continue

        present_columns = [
            column for column in encoded_columns if column in feature_index
        ]
        if present_columns and all("_therm_" in column for column in present_columns):
            encoding = "thermometer"
        else:
            encoding = "onehot"

        seen_groups.add(resolved_indices)
        groups.append(
            CategoricalGroup(indices=resolved_indices, encoding=encoding)
        )

    return groups


def infer_binary_feature_indices(
    feature_names: Sequence[str],
    feature_type: Mapping[str, str],
    categorical_groups: Sequence[CategoricalGroup],
) -> list[int]:
    grouped_indices = {
        index for group in categorical_groups for index in group.indices
    }
    return [
        index
        for index, feature_name in enumerate(feature_names)
        if index not in grouped_indices
        and str(feature_type[feature_name]).lower() == "binary"
    ]


def _ensure_2d(instance: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if instance.ndim == 1:
        return instance.unsqueeze(0), True
    return instance, False


def build_thermometer_patterns(
    num_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    patterns = torch.zeros(
        (num_features, num_features),
        device=device,
        dtype=dtype,
    )
    for index in range(num_features):
        patterns[index, : index + 1] = 1.0
    return patterns


def project_discrete_features(
    instance: torch.Tensor,
    categorical_groups: Sequence[CategoricalGroup] | None,
    binary_feature_indices: Sequence[int] | None,
) -> torch.Tensor:
    instance_2d, squeeze_output = _ensure_2d(instance)
    projected = instance_2d.clone()

    if binary_feature_indices:
        binary_tensor = projected[:, list(binary_feature_indices)]
        projected[:, list(binary_feature_indices)] = (binary_tensor >= 0.5).to(
            dtype=projected.dtype
        )

    for group in categorical_groups or []:
        group_indices = list(group.indices)
        group_values = projected[:, group_indices]

        if group.encoding == "thermometer":
            patterns = build_thermometer_patterns(
                len(group_indices),
                device=projected.device,
                dtype=projected.dtype,
            )
            squared_distance = torch.sum(
                (group_values.unsqueeze(1) - patterns.unsqueeze(0)) ** 2,
                dim=2,
            )
            best_pattern = squared_distance.argmin(dim=1)
            projected[:, group_indices] = patterns.index_select(0, best_pattern)
            continue

        winners = group_values.argmax(dim=1, keepdim=True)
        onehot = torch.zeros_like(group_values)
        onehot.scatter_(1, winners, 1.0)
        projected[:, group_indices] = onehot

    if squeeze_output:
        return projected.squeeze(0)
    return projected


def clamp_to_feature_bounds(
    instance: torch.Tensor,
    lower_bounds: torch.Tensor | None,
    upper_bounds: torch.Tensor | None,
) -> torch.Tensor:
    if lower_bounds is None or upper_bounds is None:
        return instance

    instance_2d, squeeze_output = _ensure_2d(instance)
    clamped = torch.maximum(
        torch.minimum(instance_2d, upper_bounds.reshape(1, -1)),
        lower_bounds.reshape(1, -1),
    )
    if squeeze_output:
        return clamped.squeeze(0)
    return clamped


def gradient(
    y: torch.Tensor,
    x: torch.Tensor,
    grad_outputs: torch.Tensor | None = None,
) -> torch.Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    return torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True,
    )[0]


def compute_jacobian(inputs: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if not inputs.requires_grad:
        raise ValueError("inputs must require gradients")
    return gradient(output, inputs)


def compute_invalidation_rate_closed(
    probability_model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)

    probabilities = probability_model(x)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("PROBE requires a binary probability model output")

    positive_probability = probabilities[0, 1].clamp(min=1e-6, max=1.0 - 1e-6)
    negative_probability = probabilities[0, 0].clamp(min=1e-6, max=1.0 - 1e-6)
    logit_x = torch.log(positive_probability / negative_probability)
    jacobian_x = compute_jacobian(x, logit_x).reshape(-1)
    denom = torch.sqrt(sigma2) * torch.norm(jacobian_x, p=2)

    if torch.isclose(denom, torch.zeros_like(denom)):
        return torch.where(
            logit_x > 0.0,
            torch.zeros((), dtype=torch.float32, device=x.device),
            torch.ones((), dtype=torch.float32, device=x.device),
        )

    normal = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    return 1.0 - normal.cdf(logit_x / denom)


def reparametrization_trick(
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    center = mu.reshape(-1)
    std = torch.sqrt(sigma2).to(mu.device)
    epsilon = torch.randn(n_samples, center.shape[0], device=mu.device)
    return center.unsqueeze(0) + std * epsilon


def compute_invalidation_rate(
    probability_model: Callable[[torch.Tensor], torch.Tensor],
    random_samples: torch.Tensor,
) -> torch.Tensor:
    probabilities = probability_model(random_samples)
    positive_probability = probabilities[:, 1]
    predictions = (positive_probability > DECISION_THRESHOLD).to(dtype=torch.float32)
    return 1.0 - predictions.mean(dim=0)


def compute_cost(
    candidate: torch.Tensor,
    original: torch.Tensor,
    norm: int | float,
    feature_costs: torch.Tensor | None,
) -> torch.Tensor:
    difference = candidate.reshape(-1) - original.reshape(-1)
    if feature_costs is not None:
        difference = difference * feature_costs
    return torch.linalg.vector_norm(difference, ord=norm)


def probe_optimize(
    probability_model: Callable[[torch.Tensor], torch.Tensor],
    x: np.ndarray,
    categorical_groups: Sequence[CategoricalGroup] | None = None,
    binary_feature_indices: Sequence[int] | None = None,
    binary_cat_features: bool = True,
    feature_costs: Sequence[float] | None = None,
    feature_lower_bounds: np.ndarray | Sequence[float] | None = None,
    feature_upper_bounds: np.ndarray | Sequence[float] | None = None,
    lr: float = 0.001,
    lambda_param: float = 0.01,
    y_target: Sequence[float] | np.ndarray | None = None,
    n_iter: int = 1000,
    max_minutes: float = 0.5,
    norm: int | float = 1,
    clamp: bool = True,
    loss_type: str = "MSE",
    invalidation_target: float = 0.45,
    inval_target_eps: float = 0.005,
    noise_variance: float = 0.01,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    del binary_cat_features

    if loss_type.upper() not in {"BCE", "MSE"}:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if max_minutes <= 0:
        raise ValueError("max_minutes must be > 0")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be > 0")

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    original = torch.tensor(x, dtype=torch.float32, device=device)
    if original.ndim == 1:
        original = original.unsqueeze(0)

    target = np.asarray(
        y_target if y_target is not None else [0.0, 1.0],
        dtype=np.float32,
    )
    target_tensor = torch.tensor(
        target,
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    lambda_tensor = torch.tensor(
        float(lambda_param),
        dtype=torch.float32,
        device=device,
    )
    sigma2 = torch.tensor(
        float(noise_variance),
        dtype=torch.float32,
        device=device,
    )

    feature_cost_tensor = None
    if feature_costs is not None:
        feature_cost_tensor = torch.tensor(
            feature_costs,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if feature_cost_tensor.numel() != original.shape[1]:
            raise ValueError("feature_costs length must match the number of features")

    lower_bounds_tensor = None
    upper_bounds_tensor = None
    if feature_lower_bounds is not None and feature_upper_bounds is not None:
        lower_bounds_tensor = torch.tensor(
            feature_lower_bounds,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        upper_bounds_tensor = torch.tensor(
            feature_upper_bounds,
            dtype=torch.float32,
            device=device,
        ).reshape(-1)
        if lower_bounds_tensor.numel() != original.shape[1]:
            raise ValueError(
                "feature_lower_bounds length must match the number of features"
            )
        if upper_bounds_tensor.numel() != original.shape[1]:
            raise ValueError(
                "feature_upper_bounds length must match the number of features"
            )

    candidate = original.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([candidate], lr=lr, amsgrad=True)
    loss_fn: nn.Module = nn.MSELoss() if loss_type.upper() == "MSE" else nn.BCELoss()
    timeout = datetime.timedelta(minutes=float(max_minutes))
    start_time = datetime.datetime.now()
    selected_candidates: list[torch.Tensor] = []
    selected_costs: list[float] = []
    latest_projected_candidate = project_discrete_features(
        candidate.detach(),
        categorical_groups=categorical_groups,
        binary_feature_indices=binary_feature_indices,
    )

    with torch.no_grad():
        current_probability = probability_model(candidate.detach())[0, 1]
        random_samples = reparametrization_trick(
            candidate.detach(),
            sigma2=sigma2,
            n_samples=1000,
        )
        invalidation_rate = compute_invalidation_rate(probability_model, random_samples)

    while bool(
        current_probability <= DECISION_THRESHOLD
        or invalidation_rate > invalidation_target + inval_target_eps
    ):
        for _ in range(n_iter):
            optimizer.zero_grad()

            probabilities = probability_model(candidate)
            current_cost = compute_cost(
                candidate=candidate,
                original=original,
                norm=norm,
                feature_costs=feature_cost_tensor,
            )
            invalidation_rate_closed = compute_invalidation_rate_closed(
                probability_model=probability_model,
                x=candidate,
                sigma2=sigma2,
            )
            loss_invalidation = invalidation_rate_closed - float(invalidation_target)
            loss_invalidation = torch.clamp(loss_invalidation, min=0.0)
            loss = (
                3.0 * loss_invalidation
                + loss_fn(probabilities.squeeze(0), target_tensor.reshape(-1))
                + lambda_tensor * current_cost
            )
            if not torch.isfinite(loss):
                logging.getLogger(__name__).warning(
                    "PROBE optimization encountered a non-finite loss"
                )
                break

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                current_probability = probability_model(candidate)[0, 1]
                random_samples = reparametrization_trick(
                    candidate.detach(),
                    sigma2=sigma2,
                    n_samples=10000,
                )
                invalidation_rate = compute_invalidation_rate(
                    probability_model,
                    random_samples,
                )
                if clamp:
                    candidate.clone().clamp_(0.0, 1.0)
                latest_projected_candidate = project_discrete_features(
                    candidate.detach(),
                    categorical_groups=categorical_groups,
                    binary_feature_indices=binary_feature_indices,
                )

        if bool(
            current_probability > DECISION_THRESHOLD
            and invalidation_rate < invalidation_target + inval_target_eps
        ):
            selected_candidates.append(candidate.detach().clone())
            selected_costs.append(
                float(
                    current_cost.detach().cpu().item()
                )
            )
            break

        if datetime.datetime.now() - start_time > timeout:
            logging.getLogger(__name__).info(
                "PROBE optimization timed out before convergence"
            )
            break

        lambda_tensor = torch.clamp(lambda_tensor - 0.10, min=0.0)

    if not selected_candidates:
        logging.getLogger(__name__).info(
            "No PROBE counterfactual found at the requested invalidation target"
        )
        final_candidate = latest_projected_candidate.detach().clone()
        final_samples = reparametrization_trick(
            final_candidate,
            sigma2=sigma2,
            n_samples=10000,
        )
        final_invalidation_rate = float(
            compute_invalidation_rate(probability_model, final_samples).detach().cpu()
        )
        return final_candidate.cpu().numpy().reshape(-1), final_invalidation_rate

    best_index = int(np.argmin(np.asarray(selected_costs, dtype=np.float32)))
    final_candidate = selected_candidates[best_index].detach().clone()
    final_samples = reparametrization_trick(
        final_candidate,
        sigma2=sigma2,
        n_samples=10000,
    )
    final_invalidation_rate = float(
        compute_invalidation_rate(probability_model, final_samples).detach().cpu()
    )
    return final_candidate.cpu().numpy().reshape(-1), final_invalidation_rate
