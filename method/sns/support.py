from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject

TorchModelTypes = (MlpModel,)


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


@dataclass
class FeatureGroups:
    feature_names: list[str]
    continuous: list[str]
    categorical: list[str]
    immutable: list[str]
    mutable: list[str]


def resolve_feature_groups(dataset: DatasetObject) -> FeatureGroups:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    if hasattr(dataset, "encoded_feature_type"):
        feature_type = dataset.attr("encoded_feature_type")
        feature_mutability = dataset.attr("encoded_feature_mutability")
        feature_actionability = dataset.attr("encoded_feature_actionability")
    else:
        feature_type = dataset.attr("raw_feature_type")
        feature_mutability = dataset.attr("raw_feature_mutability")
        feature_actionability = dataset.attr("raw_feature_actionability")

    continuous: list[str] = []
    categorical: list[str] = []
    immutable: list[str] = []
    mutable: list[str] = []

    for feature_name in feature_names:
        feature_kind = str(feature_type[feature_name]).lower()
        is_mutable = bool(feature_mutability[feature_name])
        actionability = str(feature_actionability[feature_name]).lower()
        if feature_kind == "numerical":
            continuous.append(feature_name)
        else:
            categorical.append(feature_name)
        if (not is_mutable) or actionability in {"none", "same"}:
            immutable.append(feature_name)
        else:
            mutable.append(feature_name)

    return FeatureGroups(
        feature_names=feature_names,
        continuous=continuous,
        categorical=categorical,
        immutable=immutable,
        mutable=mutable,
    )


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict_label_indices(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        features = self.get_ordered_features(X)
        probabilities = self._target_model.get_prediction(features, proba=True)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.detach().cpu().numpy().argmax(axis=1)
        return np.asarray(probabilities).argmax(axis=1)


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


def _get_torch_model(target_model: ModelObject) -> torch.nn.Module:
    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model has not been initialized")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("SNS requires a torch-based target model")
    return model


def differentiable_predict_proba(
    target_model: ModelObject,
    X: torch.Tensor,
) -> torch.Tensor:
    ensure_supported_target_model(target_model, TorchModelTypes, "differentiable_predict_proba")
    model = _get_torch_model(target_model)
    logits = model(X.to(target_model._device))
    output_activation = str(
        getattr(target_model, "_output_activation_name", "softmax")
    ).lower()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if output_activation == "sigmoid":
        positive_probability = torch.sigmoid(logits)
        return torch.cat([1.0 - positive_probability, positive_probability], dim=1)
    return torch.softmax(logits, dim=1)


def project_l2_ball(
    x: torch.Tensor,
    delta: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    diff = delta - x
    flat = diff.reshape(diff.shape[0], -1)
    norm = torch.linalg.norm(flat, ord=2, dim=1, keepdim=True) + 1e-12
    coeff = torch.clamp(torch.tensor(epsilon, device=delta.device) / norm, max=1.0)
    flat = flat * coeff
    projected = flat.reshape_as(delta) + x
    return projected


def clamp_tensor(x: torch.Tensor, clamp: tuple[float, float]) -> torch.Tensor:
    lower, upper = clamp
    return torch.clamp(x, min=float(lower), max=float(upper))


def linear_integral_probability(
    target_model: ModelObject,
    x: torch.Tensor,
    target_indices: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    if steps < 1:
        steps = 1
    baseline = torch.zeros_like(x)
    output_scores = []
    for step in range(1, steps + 1):
        t = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device) * step
        x_in = baseline + (x - baseline) * t / steps
        prob = differentiable_predict_proba(target_model, x_in)
        gathered = prob.gather(1, target_indices.reshape(-1, 1)).squeeze(1)
        output_scores.append(gathered)
    return torch.stack(output_scores, dim=0)


def min_l2_search(
    target_model: ModelObject,
    factual: np.ndarray,
    target_index: int,
    clamp: tuple[float, float],
    steps: int = 1000,
    step_size: float = 1e-2,
    lambda_start: float = 1e-2,
    lambda_growth: float = 2.0,
    lambda_max: float = 1e4,
) -> np.ndarray | None:
    x = torch.tensor(
        np.asarray(factual, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=target_model._device,
    )
    target = torch.tensor([target_index], dtype=torch.long, device=target_model._device)
    model = _get_torch_model(target_model)
    model.eval()

    best_candidate = None
    best_distance = float("inf")
    lam = float(lambda_start)
    output_activation = str(getattr(target_model, "_output_activation_name", "softmax")).lower()

    while lam <= lambda_max:
        candidate = x.clone().detach()
        candidate.requires_grad_(True)
        optimizer = torch.optim.Adam([candidate], lr=step_size)
        for _ in range(int(steps)):
            optimizer.zero_grad()
            logits = model(candidate)
            if output_activation == "sigmoid":
                positive_logit = logits.reshape(-1)
                if target_index == 1:
                    class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        positive_logit, torch.ones_like(positive_logit)
                    )
                else:
                    class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        positive_logit, torch.zeros_like(positive_logit)
                    )
            else:
                class_loss = torch.nn.functional.cross_entropy(logits, target)
            distance_loss = torch.linalg.norm(candidate - x, ord=2, dim=1).mean()
            loss = lam * class_loss + distance_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                candidate.copy_(clamp_tensor(candidate, clamp))
            predicted = differentiable_predict_proba(target_model, candidate).argmax(dim=1)
            if int(predicted.item()) == target_index:
                distance = float(torch.linalg.norm(candidate - x, ord=2).item())
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate.detach().cpu().numpy().reshape(-1)
        lam *= float(lambda_growth)
        if best_candidate is not None:
            break
    return best_candidate


def sns_search(
    target_model: ModelObject,
    counterfactual: np.ndarray,
    target_index: int,
    clamp: tuple[float, float],
    sns_eps: float = 0.1,
    sns_nb_iters: int = 200,
    sns_eps_iter: float = 1e-3,
    n_interpolations: int = 10,
) -> np.ndarray:
    x = torch.tensor(
        np.asarray(counterfactual, dtype=np.float32).reshape(1, -1),
        dtype=torch.float32,
        device=target_model._device,
    )
    delta = x.clone()
    optimal = x.clone()
    target_indices = torch.tensor([target_index], dtype=torch.long, device=target_model._device)

    delta = delta + torch.randn_like(delta) * 1e-3
    for _ in range(int(sns_nb_iters)):
        delta = delta.detach()
        delta.requires_grad_(True)
        output_scores = linear_integral_probability(
            target_model, delta, target_indices, n_interpolations
        )
        objective = output_scores.sum(dim=0).mean()
        objective.backward()
        grad = delta.grad
        with torch.no_grad():
            grad_flat = grad.reshape(grad.shape[0], -1)
            grad_norm = torch.linalg.norm(grad_flat, ord=2, dim=1, keepdim=True) + 1e-12
            normalized_grad = (grad_flat / grad_norm).reshape_as(delta)
            delta = delta + float(sns_eps_iter) * normalized_grad
            delta = project_l2_ball(x, delta, float(sns_eps))
            delta = clamp_tensor(delta, clamp)
            predicted = differentiable_predict_proba(target_model, delta).argmax(dim=1)
            keep = predicted == target_index
            if bool(keep.any()):
                optimal[keep] = delta[keep]
    return optimal.detach().cpu().numpy().reshape(-1)
