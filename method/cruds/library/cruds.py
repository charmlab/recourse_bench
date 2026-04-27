from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from method.cchvae.autoencoder.models.csvae import CSVAE
from method.cruds.support import CrudsTargetModelAdapter


def reconstruct_encoding_constraints(
    x: torch.Tensor,
    categorical_groups: Sequence[Sequence[int]],
    binary_feature_indices: Sequence[int],
) -> torch.Tensor:
    x_enc = x.clone()

    for group in categorical_groups:
        if len(group) < 2:
            continue
        group_tensor = x_enc[:, list(group)]
        winner = group_tensor.argmax(dim=1, keepdim=True)
        projected = torch.zeros_like(group_tensor)
        projected.scatter_(1, winner, 1.0)
        x_enc[:, list(group)] = projected

    if binary_feature_indices:
        x_enc[:, list(binary_feature_indices)] = torch.clamp(
            torch.round(x_enc[:, list(binary_feature_indices)]),
            0.0,
            1.0,
        )

    return x_enc


def compute_loss(
    cf_candidate: torch.Tensor,
    query_instance: torch.Tensor,
    target: torch.Tensor,
    lambda_param: float,
    mlmodel: CrudsTargetModelAdapter,
) -> torch.Tensor:
    loss_function = nn.BCELoss()
    output = mlmodel.predict_proba(cf_candidate)
    loss1 = loss_function(output, target)
    loss2 = torch.sum((cf_candidate - query_instance) ** 2)
    return loss1 + lambda_param * loss2


def counterfactual_search(
    mlmodel: CrudsTargetModelAdapter,
    csvae: CSVAE,
    factual: np.ndarray,
    categorical_groups: Sequence[Sequence[int]],
    binary_feature_indices: Sequence[int],
    target_class: int = 1,
    lambda_param: float = 0.001,
    optimizer: str = "rmsprop",
    lr: float = 0.008,
    max_iter: int = 2000,
    device: str = "cpu",
) -> np.ndarray:
    device_obj = torch.device(device)

    x_train = factual[:, :-1].astype("float32", copy=False)
    factual_labels = factual[:, -1].astype("int64", copy=False)
    y_train = np.zeros((factual.shape[0], 2), dtype="float32")
    y_train[np.arange(factual.shape[0]), factual_labels] = 1.0

    _, _, _, _, _, _, _, _, z, _ = csvae.forward(
        torch.from_numpy(x_train).to(device_obj),
        torch.from_numpy(y_train).to(device_obj),
    )

    w = torch.rand(2, requires_grad=True, dtype=torch.float32, device=device_obj)
    target = torch.zeros(2, dtype=torch.float32, device=device_obj)
    target[int(target_class)] = 1.0

    optimizer_name = str(optimizer).lower()
    if optimizer_name == "rmsprop":
        optim = torch.optim.RMSprop([w], lr=lr)
    elif optimizer_name == "adam":
        optim = torch.optim.Adam([w], lr=lr)
    else:
        raise ValueError(f"Unsupported CRUDS optimizer: {optimizer}")

    query_instance = torch.tensor(x_train, dtype=torch.float32, device=device_obj)
    z = torch.cat([z.detach(), query_instance[:, ~csvae.mutable_mask]], dim=-1)

    counterfactuals: list[torch.Tensor] = []
    distances: list[float] = []
    cf = query_instance.clone()
    predicted_index = torch.tensor([-1], dtype=torch.long, device=device_obj)

    for _ in range(max_iter):
        cf_mutable, _ = csvae.p_x(z, w.unsqueeze(0))

        candidate = query_instance.clone()
        candidate[:, csvae.mutable_mask] = cf_mutable
        candidate = reconstruct_encoding_constraints(
            candidate,
            categorical_groups=categorical_groups,
            binary_feature_indices=binary_feature_indices,
        ).to(device_obj)

        output = mlmodel.predict_proba(candidate)
        predicted_index = output.argmax(dim=1)
        if int(predicted_index[0].item()) == int(target_class):
            counterfactuals.append(
                torch.cat(
                    [
                        candidate.detach().clone(),
                        predicted_index.to(dtype=torch.float32).reshape(-1, 1),
                    ],
                    dim=-1,
                )
            )

        loss = compute_loss(
            candidate,
            query_instance,
            target.unsqueeze(0),
            lambda_param,
            mlmodel,
        )
        if int(predicted_index[0].item()) == int(target_class):
            distances.append(float(loss.detach().cpu().item()))

        loss.backward()
        optim.step()
        optim.zero_grad()
        cf = candidate.detach()

    if not counterfactuals:
        return (
            torch.cat(
                [cf, predicted_index.to(dtype=torch.float32).reshape(-1, 1)],
                dim=-1,
            )
            .cpu()
            .numpy()
            .squeeze(axis=0)
        )

    best_index = int(np.argmin(np.asarray(distances)))
    return counterfactuals[best_index].cpu().numpy().squeeze(axis=0)
