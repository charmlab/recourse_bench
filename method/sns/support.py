from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from art.attacks.evasion import ElasticNet
from art.estimators.classification import PyTorchClassifier

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


class _ArtBinaryWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self._base_model = base_model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        logits = self._base_model(X)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        if logits.shape[1] != 1:
            raise ValueError("Binary wrapper expects a single-logit network")
        return torch.cat([-logits, logits], dim=1)


def build_art_classifier(
    target_model: ModelObject,
    input_dim: int,
    clamp: tuple[float, float],
) -> PyTorchClassifier:
    ensure_supported_target_model(target_model, TorchModelTypes, "build_art_classifier")
    model = _get_torch_model(target_model)
    output_activation = str(
        getattr(target_model, "_output_activation_name", "softmax")
    ).lower()

    if output_activation == "sigmoid":
        art_model = _ArtBinaryWrapper(model)
    else:
        art_model = model

    nb_classes = len(target_model.get_class_to_index())
    if nb_classes < 2:
        raise ValueError("SNS requires at least two output classes")

    lower, upper = clamp
    device_type = "gpu" if target_model._device == "cuda" else "cpu"
    return PyTorchClassifier(
        model=art_model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(int(input_dim),),
        nb_classes=nb_classes,
        optimizer=None,
        clip_values=(float(lower), float(upper)),
        device_type=device_type,
    )


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
    original_index: int,
    target_index: int,
    clamp: tuple[float, float],
    steps: int = 100,
    step_size: float = 1e-2,
    confidence: float = 0.5,
    beta: float = 0.0,
    targeted: bool = False,
    art_classifier: PyTorchClassifier | None = None,
    **kwargs,
) -> np.ndarray | None:
    x = np.asarray(factual, dtype=np.float32).reshape(1, -1)
    if art_classifier is None:
        art_classifier = build_art_classifier(
            target_model=target_model,
            input_dim=x.shape[1],
            clamp=clamp,
        )

    y_index = int(target_index if targeted else original_index)
    y = np.eye(len(target_model.get_class_to_index()), dtype=np.float32)[[y_index]]
    attack = ElasticNet(
        classifier=art_classifier,
        confidence=float(confidence),
        targeted=bool(targeted),
        learning_rate=float(step_size),
        max_iter=int(steps),
        beta=float(beta),
        batch_size=1,
        decision_rule="L2",
        verbose=False,
    )
    candidate = attack.generate(x, y=y)
    if not np.isfinite(candidate).all():
        return None

    predicted = art_classifier.predict(candidate).argmax(axis=1)
    if targeted:
        success = int(predicted[0]) == int(target_index)
    else:
        success = int(predicted[0]) != int(original_index)
    if not success:
        return None

    return candidate.reshape(-1).astype(np.float64, copy=False)


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
