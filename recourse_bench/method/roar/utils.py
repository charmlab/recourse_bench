from __future__ import annotations

import datetime
import logging
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim
from scipy.optimize import linprog
from torch import nn
from torch.autograd import Variable, grad


def infer_categorical_groups(feature_names: Iterable[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, feature_name in enumerate(feature_names):
        if "_cat_" not in feature_name:
            continue
        prefix, _ = feature_name.rsplit("_cat_", 1)
        groups.setdefault(prefix, []).append(index)
    return [sorted(indices) for _, indices in sorted(groups.items())]


def reconstruct_encoding_constraints(
    instance: torch.Tensor, cat_feature_indices: list[list[int]]
) -> torch.Tensor:
    if not cat_feature_indices:
        return instance

    squeeze_output = False
    if instance.ndim == 1:
        instance = instance.unsqueeze(0)
        squeeze_output = True

    reconstructed = instance.clone()
    for feature_group in cat_feature_indices:
        if len(feature_group) > 1:
            max_indices = torch.argmax(reconstructed[:, feature_group], dim=1)
            reconstructed[:, feature_group] = 0.0
            row_indices = torch.arange(reconstructed.shape[0], device=instance.device)
            absolute_indices = torch.tensor(
                feature_group, dtype=torch.long, device=instance.device
            )[max_indices]
            reconstructed[row_indices, absolute_indices] = 1.0
        else:
            reconstructed[:, feature_group[0]] = torch.round(
                reconstructed[:, feature_group[0]]
            )

    if squeeze_output:
        return reconstructed.squeeze(0)
    return reconstructed


def calc_max_perturbation(
    recourse: torch.Tensor,
    coeff: torch.Tensor,
    intercept: torch.Tensor,
    delta_max: float,
    target_class: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    weight_vector = torch.cat((coeff, intercept.reshape(1)), dim=0)
    augmented_recourse = torch.cat(
        (recourse.reshape(-1), torch.ones(1, device=recourse.device)), dim=0
    )

    loss_fn = nn.BCELoss()
    weight_vector.requires_grad_(True)
    recourse_probability = torch.sigmoid(torch.dot(weight_vector, augmented_recourse))
    weight_loss = loss_fn(recourse_probability, target_class)
    gradient_w_loss = grad(weight_loss, weight_vector)[0]

    objective = (-gradient_w_loss.detach().cpu().numpy()).tolist()
    bounds = [(-delta_max, delta_max)] * len(objective)
    result = linprog(objective, bounds=bounds, method="highs")

    if result.status != 0 or result.x is None:
        logging.getLogger(__name__).warning(
            "ROAR inner perturbation optimization did not converge; using zero perturbation"
        )
        zeros = np.zeros(len(objective), dtype=np.float32)
        return zeros[:-1], zeros[-1:]

    delta_opt = np.asarray(result.x, dtype=np.float32)
    return delta_opt[:-1], delta_opt[-1:]


def roar_optimize(
    x: np.ndarray,
    coeff: np.ndarray,
    intercept: float,
    cat_feature_indices: list[list[int]] | None = None,
    lr: float = 1e-3,
    lambda_param: float = 0.1,
    delta_max: float = 0.1,
    norm: int | float = 1,
    loss_type: str = "BCE",
    loss_threshold: float = 1e-4,
    max_minutes: float = 0.5,
    enforce_encoding: bool = False,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    if loss_type.upper() not in {"BCE", "MSE"}:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if lambda_param < 0:
        raise ValueError("lambda_param must be >= 0")
    if delta_max < 0:
        raise ValueError("delta_max must be >= 0")
    if max_minutes <= 0:
        raise ValueError("max_minutes must be > 0")

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    groups = list(cat_feature_indices or [])
    coeff_tensor = torch.tensor(coeff, dtype=torch.float32, device=device)
    intercept_tensor = torch.tensor(intercept, dtype=torch.float32, device=device)
    original = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1)
    lambda_tensor = torch.tensor(lambda_param, dtype=torch.float32, device=device)

    candidate = Variable(original.clone(), requires_grad=True)
    optimizer = optim.Adam([candidate], lr=lr)

    loss_type = loss_type.upper()
    if loss_type == "BCE":
        target_probability = torch.tensor(1.0, dtype=torch.float32, device=device)
        loss_fn: nn.Module = nn.BCELoss()
    else:
        target_probability = torch.tensor(1.0, dtype=torch.float32, device=device)
        loss_fn = nn.MSELoss()

    previous_loss = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    loss_difference = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    timeout = datetime.timedelta(minutes=float(max_minutes))
    start_time = datetime.datetime.now()

    while float(loss_difference.item()) > float(loss_threshold):
        delta_w, delta_b = calc_max_perturbation(
            recourse=candidate.detach(),
            coeff=coeff_tensor,
            intercept=intercept_tensor,
            delta_max=delta_max,
            target_class=target_probability,
        )
        delta_w_tensor = torch.tensor(delta_w, dtype=torch.float32, device=device)
        delta_b_tensor = torch.tensor(delta_b, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        probability = torch.sigmoid(
            torch.dot(coeff_tensor + delta_w_tensor, candidate)
            + intercept_tensor
            + delta_b_tensor.squeeze()
        )
        prediction = probability
        if loss_type == "MSE":
            probability = probability.clamp(min=1e-6, max=1 - 1e-6)
            prediction = torch.log(probability / (1 - probability))
        distance = torch.dist(candidate, original, p=float(norm))
        loss = (
            loss_fn(prediction.squeeze(), target_probability) + lambda_tensor * distance
        )
        loss.backward()
        optimizer.step()

        if enforce_encoding and groups:
            with torch.no_grad():
                candidate.copy_(reconstruct_encoding_constraints(candidate, groups))

        loss_difference = torch.dist(previous_loss, loss.detach(), p=2)
        previous_loss = loss.detach()

        if datetime.datetime.now() - start_time > timeout:
            logging.getLogger(__name__).info(
                "ROAR optimization timed out before convergence"
            )
            break

    if enforce_encoding and groups:
        with torch.no_grad():
            candidate.copy_(reconstruct_encoding_constraints(candidate, groups))

    return candidate.detach().cpu().numpy()
