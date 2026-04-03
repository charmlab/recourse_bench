from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject

TorchModelTypes = (LinearModel, MlpModel)


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


class TreXLogitModel(nn.Module):
    """
    Wrap repository target models so TreX and ART always see 2-class logits.
    """

    def __init__(self, target_model: ModelObject):
        super().__init__()
        ensure_supported_target_model(target_model, TorchModelTypes, "TreXLogitModel")
        base_model = getattr(target_model, "_model", None)
        if base_model is None:
            raise RuntimeError("Target model has not been initialized")

        self._target_model = target_model
        self.base_model = base_model
        self.output_activation = str(
            getattr(target_model, "_output_activation_name", "softmax")
        ).lower()
        self.device = target_model._device

    def _resolve_model_device(self) -> torch.device:
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.base_model.eval()
        model_device = self._resolve_model_device()
        logits = self.base_model(X.to(model_device))
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if self.output_activation == "sigmoid":
            zero_logits = torch.zeros_like(logits)
            return torch.cat([zero_logits, logits], dim=1)
        return logits


def differentiable_predict_proba(
    target_model: ModelObject,
    X: torch.Tensor,
) -> torch.Tensor:
    wrapper = TreXLogitModel(target_model)
    logits = wrapper(X)
    return torch.softmax(logits, dim=1)


def predict_label_indices(
    target_model: ModelObject,
    feature_names: Sequence[str],
    X: pd.DataFrame | np.ndarray | torch.Tensor,
) -> np.ndarray:
    if isinstance(X, torch.Tensor) and isinstance(target_model, TorchModelTypes):
        probabilities = differentiable_predict_proba(target_model, X)
        return probabilities.argmax(dim=1).detach().cpu().numpy()

    features = to_feature_dataframe(X, feature_names)
    probabilities = target_model.get_prediction(features, proba=True)
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

    original_prediction = predict_label_indices(
        target_model=target_model,
        feature_names=factuals.columns,
        X=factuals,
    )
    target_prediction = resolve_target_indices(
        target_model=target_model,
        original_prediction=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = predict_label_indices(
        target_model=target_model,
        feature_names=factuals.columns,
        X=candidates.loc[valid_rows],
    )
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_prediction[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates


class TreXCounterfactualTorch:
    def __init__(
        self,
        model: nn.Module,
        input_dim: int,
        num_classes: int = 2,
        clamp: tuple[float | np.ndarray, float | np.ndarray] | None = None,
        norm: int = 2,
        cf_steps: int = 60,
        cf_step_size: float = 0.02,
        cf_confidence: float = 0.5,
        tau: float = 0.75,
        k: int = 1000,
        sigma: float = 0.1,
        trex_max_steps: int = 100,
        trex_epsilon: float = 1.0,
        trex_step_size: float = 0.01,
        trex_p: float = 2,
        batch_size: int = 1,
        device: str | None = None,
    ):
        if norm not in (1, 2):
            raise ValueError("norm must be 1 or 2")
        if trex_p not in (1, 2, np.inf):
            raise ValueError("trex_p must be 1, 2, or np.inf")
        if num_classes != 2:
            raise ValueError("TreXCounterfactualTorch currently supports 2 classes")

        self.model = model
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.norm = int(norm)
        self.cf_steps = int(cf_steps)
        self.cf_step_size = float(cf_step_size)
        self.cf_confidence = float(cf_confidence)
        self.tau = float(tau)
        self.k = int(k)
        self.sigma = float(sigma)
        self.trex_max_steps = int(trex_max_steps)
        self.trex_epsilon = float(trex_epsilon)
        self.trex_step_size = float(trex_step_size)
        self.trex_p = trex_p
        self.batch_size = int(batch_size)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)
        self.model.eval()
        self._set_clamp(clamp)

    def _set_clamp(
        self, clamp: tuple[float | np.ndarray, float | np.ndarray] | None
    ) -> None:
        self.has_clamp = clamp is not None
        self.per_feature_clamp = False
        self.clip_values = None

        if clamp is None:
            return

        low, high = clamp
        low_is_scalar = np.isscalar(low)
        high_is_scalar = np.isscalar(high)

        if low_is_scalar and high_is_scalar:
            self.clamp_low = float(low)
            self.clamp_high = float(high)
            self.clip_values = (self.clamp_low, self.clamp_high)
            return

        low_np = (
            np.full((1, self.input_dim), float(low), dtype=np.float32)
            if low_is_scalar
            else np.asarray(low, dtype=np.float32).reshape(1, -1)
        )
        high_np = (
            np.full((1, self.input_dim), float(high), dtype=np.float32)
            if high_is_scalar
            else np.asarray(high, dtype=np.float32).reshape(1, -1)
        )
        if low_np.shape[1] != self.input_dim or high_np.shape[1] != self.input_dim:
            raise ValueError("Per-feature clamp bounds must match input_dim")

        self.per_feature_clamp = True
        self.clamp_low_t = torch.tensor(low_np, dtype=torch.float32, device=self.device)
        self.clamp_high_t = torch.tensor(
            high_np, dtype=torch.float32, device=self.device
        )
        self.clip_values = (float(low_np.min()), float(high_np.max()))

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self._logits(x), dim=1)

    @torch.no_grad()
    def _predict_labels_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self._proba(x).argmax(dim=1)

    def predict_labels(self, x_np: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        return self._predict_labels_torch(x).cpu().numpy()

    def _make_art_classifier(self):
        from art.estimators.classification import PyTorchClassifier

        device_type = "gpu" if self.device.type == "cuda" else "cpu"
        return PyTorchClassifier(
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3),
            input_shape=(self.input_dim,),
            nb_classes=self.num_classes,
            clip_values=self.clip_values,
            device_type=device_type,
        )

    def _initial_counterfactual(
        self,
        x_np: np.ndarray,
        target_np: np.ndarray,
    ) -> np.ndarray:
        from art.attacks.evasion import ElasticNet

        classifier = self._make_art_classifier()
        decision_rule = "L1" if self.norm == 1 else "L2"
        beta = 1.0 if self.norm == 1 else 0.0

        attack = ElasticNet(
            classifier=classifier,
            confidence=self.cf_confidence,
            targeted=True,
            learning_rate=self.cf_step_size,
            beta=beta,
            max_iter=self.cf_steps,
            batch_size=min(self.batch_size, x_np.shape[0]),
            decision_rule=decision_rule,
            verbose=False,
        )

        y_onehot = np.eye(self.num_classes, dtype=np.float32)[target_np.astype(int)]
        x_adv = attack.generate(x=x_np.astype(np.float32), y=y_onehot)
        x_adv_t = torch.as_tensor(x_adv, dtype=torch.float32, device=self.device)
        return self._clip_with_clamp(x_adv_t).detach().cpu().numpy()

    def _clip_with_clamp(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_clamp:
            return x
        if self.per_feature_clamp:
            return torch.max(torch.min(x, self.clamp_high_t), self.clamp_low_t)
        return torch.clamp(x, min=self.clamp_low, max=self.clamp_high)

    @staticmethod
    def _normalize_l2(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        n = v.reshape(v.shape[0], -1).norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        return v / n

    def _update_delta(
        self,
        x0: torch.Tensor,
        delta: torch.Tensor,
        grad: torch.Tensor,
        step_size: float,
        epsilon: float,
        p: float,
    ) -> torch.Tensor:
        if p == np.inf:
            proposal = delta + step_size * grad.sign()
        elif p == 2:
            proposal = delta + step_size * self._normalize_l2(grad)
        elif p == 1:
            proposal = delta + step_size * grad.sign()
        else:
            raise ValueError("Unsupported p-norm")

        perturbation = proposal - x0
        flat = perturbation.reshape(perturbation.shape[0], -1)

        if p == 2:
            norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        elif p == np.inf:
            norm = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        else:
            norm = flat.abs().sum(dim=1, keepdim=True).clamp_min(1e-12)

        coeff = torch.clamp(epsilon / norm, min=0.0, max=1.0)
        projected = x0 + perturbation * coeff
        return self._clip_with_clamp(projected)

    def _gaussian_volume(
        self,
        x: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        bsz, dim = x.shape
        noise = torch.randn(self.k, dim, device=x.device, dtype=x.dtype) * self.sigma

        scores = []
        for index in range(bsz):
            x_i = x[index : index + 1].repeat(self.k, 1)
            x_noisy = x_i + noise
            target_index = int(target_labels[index].item())

            p_noisy = self._proba(x_noisy)[:, target_index]
            p_clean = self._proba(x_i)[:, target_index]
            score = p_noisy.mean() - torch.abs(p_noisy - p_clean).mean()
            scores.append(score)

        return torch.stack(scores, dim=0)

    def _trex_refine_batch(
        self,
        x_batch_np: np.ndarray,
        target_batch_np: np.ndarray,
    ) -> np.ndarray:
        x0 = torch.as_tensor(x_batch_np, dtype=torch.float32, device=self.device)
        target_batch = torch.as_tensor(
            target_batch_np, dtype=torch.long, device=self.device
        ).reshape(-1)

        optimal_adv = x0.clone()
        delta = x0.clone()

        for _ in range(self.trex_max_steps):
            delta = delta.detach().requires_grad_(True)
            scores = self._gaussian_volume(delta, target_batch)

            with torch.no_grad():
                keep_current = target_batch == self._predict_labels_torch(delta)
                optimal_adv[keep_current] = delta[keep_current]

            if torch.all(scores >= self.tau):
                delta = delta.detach()
                break

            grad = torch.autograd.grad(scores.sum(), delta)[0]

            with torch.no_grad():
                delta = self._update_delta(
                    x0=x0,
                    delta=delta,
                    grad=grad,
                    step_size=self.trex_step_size,
                    epsilon=self.trex_epsilon,
                    p=self.trex_p,
                ).detach()

        return optimal_adv.detach().cpu().numpy()

    def _trex_refine(
        self,
        x_adv_np: np.ndarray,
        target_np: np.ndarray,
    ) -> np.ndarray:
        refined = []
        for start in range(0, x_adv_np.shape[0], self.batch_size):
            batch_x = x_adv_np[start : start + self.batch_size]
            batch_target = target_np[start : start + self.batch_size]
            refined.append(self._trex_refine_batch(batch_x, batch_target))
        return np.concatenate(refined, axis=0)

    def generate(
        self,
        x_np: np.ndarray,
        target_labels: np.ndarray,
        apply_trex: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_np = np.asarray(x_np, dtype=np.float32)
        target_labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
        if x_np.shape[0] != target_labels.shape[0]:
            raise ValueError("target_labels must match the number of input rows")

        x_cf = self._initial_counterfactual(x_np, target_labels)
        if apply_trex:
            x_cf = self._trex_refine(x_cf, target_labels)

        pred_cf = self.predict_labels(x_cf)
        is_valid = pred_cf == target_labels
        return x_cf, pred_cf, is_valid
