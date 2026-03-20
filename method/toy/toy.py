from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("toy")
class ToyMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        max_iterations: int = 200,
        step_size: float = 0.05,
        lambda_: float = 0.05,
        clamp: bool = True,
        **kwargs,
    ):
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class
        self._max_iterations: int = int(max_iterations)
        self._step_size: float = float(step_size)
        self._lambda: float = float(lambda_)
        self._clamp: bool = bool(clamp)

        if self._desired_class is not None and not isinstance(
            self._desired_class, (int, str)
        ):
            raise TypeError("desired_class must be int, str, or None")
        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if not self._target_model._need_grad:
            raise ValueError("ToyMethod requires a gradient-enabled target model")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ToyMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            if hasattr(trainset, "encoded_feature_type"):
                feature_type = deepcopy(trainset.attr("encoded_feature_type"))
                feature_mutability = deepcopy(
                    trainset.attr("encoded_feature_mutability")
                )
                feature_actionability = deepcopy(
                    trainset.attr("encoded_feature_actionability")
                )
            else:
                feature_type = deepcopy(trainset.attr("raw_feature_type"))
                feature_mutability = deepcopy(trainset.attr("raw_feature_mutability"))
                feature_actionability = deepcopy(
                    trainset.attr("raw_feature_actionability")
                )

            self._feature_names = list(features.columns)
            self._feature_ranges = {
                column: (float(features[column].min()), float(features[column].max()))
                for column in self._feature_names
            }
            self._feature_actionability = {
                column: feature_actionability[column] for column in self._feature_names
            }
            self._mutable_mask = torch.tensor(
                [
                    feature_type[column].lower() == "numerical"
                    and bool(feature_mutability[column])
                    and feature_actionability[column].lower() != "none"
                    for column in self._feature_names
                ],
                dtype=torch.bool,
                device=self._device,
            )
            if not bool(self._mutable_mask.any()):
                raise ValueError(
                    "ToyMethod could not find any mutable numerical features"
                )

            output = self._target_model.predict(trainset)
            if output.shape[1] < 2:
                raise ValueError(
                    "ToyMethod requires a target model with at least two classes"
                )
            class_to_index = self._target_model.get_class_to_index()
            if (
                self._desired_class is not None
                and self._desired_class not in class_to_index
            ):
                raise ValueError("desired_class is invalid")
            if self._desired_class is not None and output.shape[1] != len(
                class_to_index
            ):
                raise ValueError(
                    "desired_class is incompatible with the trained target model"
                )

            self._is_trained = True

    def _apply_constraints(
        self, candidate: torch.Tensor, original: torch.Tensor
    ) -> None:
        candidate[~self._mutable_mask] = original[~self._mutable_mask]
        for index, feature in enumerate(self._feature_names):
            rule = self._feature_actionability[feature].lower()
            if rule == "same-or-increase":
                candidate[index] = torch.maximum(candidate[index], original[index])
            elif rule == "same-or-decrease":
                candidate[index] = torch.minimum(candidate[index], original[index])
            elif rule in {"none", "same"}:
                candidate[index] = original[index]
            if self._clamp:
                min_value, max_value = self._feature_ranges[feature]
                candidate[index] = candidate[index].clamp(min=min_value, max=max_value)

    def _is_successful_prediction(
        self, prediction: int, original_prediction: int
    ) -> bool:
        class_to_index = self._target_model.get_class_to_index()
        if prediction < 0 or prediction >= len(class_to_index):
            return False
        if self._desired_class is None:
            return prediction != original_prediction
        return prediction == class_to_index[self._desired_class]

    def _classification_loss(
        self, logits: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        if self._desired_class is None:
            return -F.cross_entropy(logits, target_tensor)
        return F.cross_entropy(logits, target_tensor)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        with seed_context(self._seed):
            counterfactual_rows: list[pd.Series] = []
            class_to_index = self._target_model.get_class_to_index()

            for _, row in factuals.iterrows():
                original = torch.tensor(
                    row.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                    device=self._device,
                )
                with torch.no_grad():
                    original_logits = self._target_model.forward(original.unsqueeze(0))
                    original_prediction = original_logits.argmax(dim=1).item()

                if (
                    self._desired_class is not None
                    and original_prediction == class_to_index[self._desired_class]
                ):
                    counterfactual_rows.append(
                        pd.Series(row.copy(deep=True), index=factuals.columns)
                    )
                    continue

                target_class = (
                    original_prediction
                    if self._desired_class is None
                    else class_to_index[self._desired_class]
                )
                target_tensor = torch.tensor(
                    [target_class], dtype=torch.long, device=self._device
                )

                candidate = original.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([candidate], lr=self._step_size)
                success = False

                for _ in range(self._max_iterations):
                    optimizer.zero_grad()
                    logits = self._target_model.forward(candidate.unsqueeze(0))
                    classification_loss = self._classification_loss(
                        logits, target_tensor
                    )
                    distance_loss = torch.mean(
                        torch.abs((candidate - original)[self._mutable_mask])
                    )
                    loss = classification_loss + self._lambda * distance_loss
                    loss.backward()

                    if candidate.grad is not None:
                        candidate.grad[~self._mutable_mask] = 0

                    optimizer.step()
                    with torch.no_grad():
                        self._apply_constraints(candidate, original)
                        prediction = (
                            self._target_model.forward(candidate.unsqueeze(0))
                            .argmax(dim=1)
                            .item()
                        )
                        if self._is_successful_prediction(
                            prediction, original_prediction
                        ):
                            success = True
                            break

                if success:
                    counterfactual_rows.append(
                        pd.Series(
                            candidate.detach().cpu().numpy(), index=factuals.columns
                        )
                    )
                else:
                    counterfactual_rows.append(
                        pd.Series(np.nan, index=factuals.columns, dtype="float64")
                    )

            return pd.DataFrame(
                counterfactual_rows,
                index=factuals.index,
                columns=factuals.columns,
            )
