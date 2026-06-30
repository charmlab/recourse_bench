from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np
import torch
import torch.optim as optim

DECISION_THRESHOLD = 0.5
LOGGER = logging.getLogger(__name__)


def _project_onehot_group(group_tensor: torch.Tensor) -> torch.Tensor:
    winner = group_tensor.argmax(dim=1, keepdim=True)
    projected = torch.zeros_like(group_tensor)
    projected.scatter_(1, winner, 1.0)
    return projected


def _project_thermometer_group(group_tensor: torch.Tensor) -> torch.Tensor:
    size = int(group_tensor.shape[1])
    legal_patterns = torch.tril(
        torch.ones((size, size), dtype=group_tensor.dtype, device=group_tensor.device)
    )
    distances = torch.norm(
        legal_patterns.unsqueeze(0) - group_tensor.unsqueeze(1),
        dim=2,
    )
    best_index = distances.argmin(dim=1)
    return legal_patterns[best_index]


def reconstruct_encoding_constraints(
    x: torch.Tensor,
    categorical_groups: Sequence[Sequence[int]],
    thermometer_groups: Sequence[Sequence[int]],
    binary_feature_indices: Sequence[int],
) -> torch.Tensor:
    x_enc = x.clone()

    for group in categorical_groups:
        if len(group) < 2:
            continue
        x_enc[:, list(group)] = _project_onehot_group(x_enc[:, list(group)])

    for group in thermometer_groups:
        if len(group) < 2:
            continue
        x_enc[:, list(group)] = _project_thermometer_group(x_enc[:, list(group)])

    if binary_feature_indices:
        x_enc[:, list(binary_feature_indices)] = torch.clamp(
            torch.round(x_enc[:, list(binary_feature_indices)]),
            0.0,
            1.0,
        )

    return x_enc


def wachter_recourse(
    torch_model,
    x: np.ndarray,
    categorical_groups: Sequence[Sequence[int]],
    thermometer_groups: Sequence[Sequence[int]],
    binary_feature_indices: Sequence[int],
    feature_cost: Optional[Sequence[float]],
    lr: float,
    lambda_param: float,
    y_target: Sequence[float],
    n_iter: int,
    t_max_min: float,
    norm: int,
    clamp: bool,
    loss_type: str,
) -> np.ndarray:
    """
    Generates counterfactual example according to Wachter et.al for input instance x

    Parameters
    ----------
    torch_model:
        black-box-model to discover
    x:
        Factual instance to explain.
    categorical_groups:
        Index groups for one-hot encoded features in x.
    thermometer_groups:
        Index groups for thermometer encoded features in x.
    binary_feature_indices:
        Index positions for scalar binary features in x.
    feature_cost:
        Optional cost weight per feature.
    lr:
        Learning rate for gradient descent.
    lambda_param:
        Weight factor for feature_cost.
    y_target:
        Tuple of class probabilities (BCE loss) or [Float] for logit score (MSE loss).
    n_iter:
        Maximum number of iterations.
    t_max_min:
        Maximum time amount of search.
    norm:
        L-norm to calculate cost.
    clamp:
        If true, feature values will be clamped to intverval [0, 1].
    loss_type:
        String for loss function ("MSE" or "BCE").

    Returns
    -------
    Counterfactual example as np.ndarray
    """
    device = torch.device(str(getattr(torch_model, "device", "cpu")))

    if feature_cost is not None:
        feature_cost = (
            torch.as_tensor(feature_cost, dtype=torch.float32, device=device)
            .reshape(-1)
            .clone()
        )

    x = np.asarray(x, dtype="float32")
    if x.ndim == 1:
        x = x.reshape(1, -1)

    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_target = torch.tensor(y_target, dtype=torch.float32, device=device)
    lamb = torch.tensor(float(lambda_param), dtype=torch.float32, device=device)
    x_new = x.clone().detach().requires_grad_(True)
    x_new_enc = reconstruct_encoding_constraints(
        x_new,
        categorical_groups,
        thermometer_groups,
        binary_feature_indices,
    )

    optimizer = optim.Adam([x_new], lr=float(lr), amsgrad=True)

    if loss_type == "MSE":
        if len(y_target) != 1:
            raise ValueError(f"y_target {y_target} is not a single logit score")

        target_class = int(float(y_target[0].item()) > 0.0)
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "BCE":
        if len(y_target) != 2 or not torch.isclose(
            y_target.sum(),
            torch.tensor(1.0, dtype=torch.float32, device=device),
        ):
            raise ValueError(
                f"y_target {y_target} does not contain 2 valid class probabilities"
            )

        target_class = int(torch.round(y_target[1]).item())
        loss_fn = torch.nn.BCELoss()
    else:
        raise ValueError(f"loss_type {loss_type} not supported")

    f_x_new = torch_model(x_new)[:, target_class]

    start_time = time.monotonic()
    timeout_seconds = float(t_max_min) * 60.0
    while float(f_x_new.detach().item()) <= DECISION_THRESHOLD:
        it = 0
        while float(f_x_new.detach().item()) <= DECISION_THRESHOLD and it < n_iter:
            optimizer.zero_grad()
            x_new_enc = reconstruct_encoding_constraints(
                x_new,
                categorical_groups,
                thermometer_groups,
                binary_feature_indices,
            )
            f_x_new = torch_model(x_new_enc)[:, target_class]

            if loss_type == "MSE":
                clamped_probability = torch.clamp(f_x_new, min=1e-6, max=1 - 1e-6)
                f_x_loss = torch.log(clamped_probability / (1 - clamped_probability))
            elif loss_type == "BCE":
                f_x_loss = torch_model(x_new_enc).squeeze(dim=0)
            else:
                raise ValueError(f"loss_type {loss_type} not supported")

            cost = (
                torch.dist(x_new_enc, x, norm)
                if feature_cost is None
                else torch.norm(feature_cost * (x_new_enc - x), norm)
            )

            loss = loss_fn(f_x_loss, y_target) + lamb * cost
            loss.backward()
            optimizer.step()
            if clamp:
                with torch.no_grad():
                    x_new.clamp_(0.0, 1.0)
            it += 1
        lamb -= 0.05

        if time.monotonic() - start_time > timeout_seconds:
            LOGGER.warning("Timeout - No Counterfactual Explanation Found")
            break
        if float(f_x_new.detach().item()) >= DECISION_THRESHOLD:
            LOGGER.debug("Counterfactual Explanation Found")
    return x_new_enc.cpu().detach().numpy().squeeze(axis=0)
