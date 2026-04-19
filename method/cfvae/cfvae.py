from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.cfvae.support import (
    TorchModelTypes,
    ensure_supported_target_model,
    resolve_continuous_ranges,
    resolve_feature_groups,
    resolve_target_indices,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


class _CFVAE(nn.Module):
    def __init__(self, feature_num: int, encoded_size: int):
        super().__init__()
        self.feature_num = feature_num
        self.encoded_size = encoded_size

        self.encoder_mean = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.feature_num + 1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid(),
        )

        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size + 1, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.feature_num),
            nn.Sigmoid(),
        )

    def encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        mean = self.encoder_mean(x)
        var = 0.5 + self.encoder_var(x)
        return mean, var

    def sample_latent_code(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        mean, var = mean.float(), var.float()
        eps = torch.randn_like(var)
        return mean + torch.sqrt(var) * eps

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        return self.decoder_mean(z)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True):
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        z = self.sample_latent_code(mean, var) if sample else mean
        return self.decoder(torch.cat((z, conditions), dim=1))

    def forward_with_kl(
        self, x: torch.Tensor, conditions: torch.Tensor, sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, conditions = x.float(), conditions.view(-1, 1).float()
        mean, var = self.encoder(torch.cat((x, conditions), dim=1))
        kl_divergence = 0.5 * torch.mean(mean**2 + var - torch.log(var) - 1, dim=1)
        z = self.sample_latent_code(mean, var) if sample else mean
        x_pred = self.decoder(torch.cat((z, conditions), dim=1))
        return x_pred, kl_divergence


@register("cfvae")
class CfvaeMethod(MethodObject):
    _SAMPLING_METRIC_ORDER = (
        "target_class_validity",
        "constraint_feasibility_score",
        "cont_proximity",
        "cat_proximity",
    )

    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        encoded_size: int = 10,
        train: bool = True,
        pretrained_path: str | None = None,
        save_path: str | None = None,
        batch_size: int = 2048,
        epoch: int = 50,
        learning_rate: float = 1e-2,
        n_samples: int = 50,
        margin: float = 0.2,
        validity_reg: float = 20.0,
        constraint_loss_func: Any = None,
        preference_dataset: dict[str, Any] | None = None,
        constraint_reg: float = 1.0,
        preference_reg: float = 1.0,
        store_sampling_artifacts: bool = False,
        sampling_sample_sizes: list[int] | None = None,
        sampling_trials: int = 1,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, TorchModelTypes, "CfvaeMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        self._encoded_size = int(encoded_size)
        self._train_when_fit = bool(train)
        self._pretrained_path = pretrained_path
        self._save_path = save_path
        self._batch_size = int(batch_size)
        self._epoch = int(epoch)
        self._learning_rate = float(learning_rate)
        self._n_samples = int(n_samples)
        self._margin = float(margin)
        self._validity_reg = float(validity_reg)
        self._constraint_loss_func = constraint_loss_func
        self._preference_dataset = preference_dataset
        self._constraint_reg = float(constraint_reg)
        self._preference_reg = float(preference_reg)
        self._store_sampling_artifacts = bool(store_sampling_artifacts)
        self._sampling_sample_sizes = self._resolve_sampling_sample_sizes(
            sampling_sample_sizes
        )
        self._sampling_trials = int(sampling_trials)

        self._cf_model: _CFVAE | None = None
        self._logger = logging.getLogger(__name__)

        self._feature_names: list[str] = []
        self._continuous_features: list[str] = []
        self._continuous_indices: list[int] = []
        self._categorical_indices: list[int] = []
        self._binary_scalar_features: list[str] = []
        self._binary_scalar_indices: list[int] = []
        self._onehot_groups: list[list[str]] = []
        self._onehot_group_indices: list[list[int]] = []
        self._continuous_ranges: dict[int, float] = {}
        self._binary_value_ranges: dict[int, tuple[float, float]] = {}
        self._class_to_index: dict[int | str, int] | None = None
        self._pretrained_loaded = False
        self._metadata_ready = False
        self._loaded_feature_num: int | None = None

        if self._device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._epoch < 1:
            raise ValueError("epoch must be >= 1")
        if self._learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self._n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if self._margin < 0.0:
            raise ValueError("margin must be >= 0")
        if self._validity_reg < 0.0:
            raise ValueError("validity_reg must be >= 0")
        if self._constraint_reg < 0.0:
            raise ValueError("constraint_reg must be >= 0")
        if self._preference_reg < 0.0:
            raise ValueError("preference_reg must be >= 0")
        if self._sampling_trials < 1:
            raise ValueError("sampling_trials must be >= 1")
        if self._store_sampling_artifacts and not self._sampling_sample_sizes:
            raise ValueError(
                "sampling_sample_sizes must be provided when store_sampling_artifacts=True"
            )
        if not self._train_when_fit and self._pretrained_path is None:
            raise ValueError(
                "train=False requires pretrained_path because CfvaeMethod trains in fit()"
            )

        if self._pretrained_path is not None:
            checkpoint_path = Path(self._pretrained_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"pretrained_path does not exist: {checkpoint_path}"
                )
            state_dict = torch.load(checkpoint_path, map_location=self._device)
            inferred_feature_num = self._infer_feature_num_from_state_dict(state_dict)
            inferred_encoded_size = self._infer_encoded_size_from_state_dict(state_dict)
            if inferred_encoded_size != self._encoded_size:
                raise ValueError(
                    "encoded_size does not match pretrained checkpoint: "
                    f"{self._encoded_size} != {inferred_encoded_size}"
                )
            self._cf_model = _CFVAE(inferred_feature_num, inferred_encoded_size).to(
                self._device
            )
            self._cf_model.load_state_dict(state_dict)
            self._cf_model.eval()
            self._loaded_feature_num = inferred_feature_num
            self._pretrained_loaded = True
            self._is_trained = True
            self._logger.info(
                "Loaded CFVAE checkpoint from %s", checkpoint_path.as_posix()
            )

    @staticmethod
    def _resolve_sampling_sample_sizes(
        sample_sizes: list[int] | None,
    ) -> tuple[int, ...]:
        if sample_sizes is None:
            return tuple()

        resolved = []
        seen = set()
        for sample_size in sample_sizes:
            value = int(sample_size)
            if value < 1:
                raise ValueError("sampling_sample_sizes must contain values >= 1")
            if value in seen:
                continue
            seen.add(value)
            resolved.append(value)
        return tuple(resolved)

    @staticmethod
    def _infer_feature_num_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
        key = "encoder_mean.0.weight"
        if key not in state_dict:
            raise KeyError(f"Missing checkpoint key: {key}")
        input_dim = int(state_dict[key].shape[1])
        if input_dim < 2:
            raise ValueError("Invalid CFVAE checkpoint input dimension")
        return input_dim - 1

    @staticmethod
    def _infer_encoded_size_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
        key = "encoder_mean.16.weight"
        if key not in state_dict:
            raise KeyError(f"Missing checkpoint key: {key}")
        return int(state_dict[key].shape[0])

    def _seed_guard(self):
        if self._seed is None:
            return contextlib.nullcontext()
        return seed_context(self._seed)

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CfvaeMethod.fit()")
        if not getattr(self._target_model, "_is_trained", False):
            raise RuntimeError("Target model must be trained before CfvaeMethod.fit()")

        with self._seed_guard():
            feature_df = trainset.get(target=False).copy(deep=True)
            if feature_df.isna().any(axis=None):
                raise ValueError("CfvaeMethod.fit() does not support NaN values")

            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._continuous_features = list(feature_groups.continuous)
            self._binary_scalar_features = list(feature_groups.binary_scalar)
            self._onehot_groups = [
                list(group) for group in feature_groups.onehot_groups
            ]

            self._continuous_indices = [
                self._feature_names.index(feature_name)
                for feature_name in self._continuous_features
            ]
            self._binary_scalar_indices = [
                self._feature_names.index(feature_name)
                for feature_name in self._binary_scalar_features
            ]
            self._onehot_group_indices = [
                [self._feature_names.index(feature_name) for feature_name in group]
                for group in self._onehot_groups
            ]
            categorical_feature_set = {
                feature_name for group in self._onehot_groups for feature_name in group
            }
            categorical_feature_set.update(self._binary_scalar_features)
            self._categorical_indices = [
                self._feature_names.index(feature_name)
                for feature_name in self._feature_names
                if feature_name in categorical_feature_set
            ]
            self._continuous_ranges = {
                self._feature_names.index(feature_name): feature_range
                for feature_name, feature_range in resolve_continuous_ranges(
                    trainset, self._continuous_features
                ).items()
            }
            self._binary_value_ranges = {
                self._feature_names.index(feature_name): feature_range
                for feature_name, feature_range in feature_groups.binary_value_ranges.items()
            }

            class_to_index = self._target_model.get_class_to_index()
            if len(class_to_index) != 2:
                raise ValueError("CfvaeMethod supports binary classification only")
            if (
                self._desired_class is not None
                and self._desired_class not in class_to_index
            ):
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )
            self._class_to_index = class_to_index

            feature_count = len(self._feature_names)
            if self._pretrained_loaded:
                if self._cf_model is None or self._loaded_feature_num is None:
                    raise RuntimeError(
                        "Pretrained CFVAE checkpoint was marked as loaded but no network is available"
                    )
                if feature_count != self._loaded_feature_num:
                    raise ValueError(
                        "trainset feature count does not match pretrained CFVAE checkpoint: "
                        f"{feature_count} != {self._loaded_feature_num}"
                    )
                self._cf_model = self._cf_model.to(self._device)
                self._cf_model.eval()
                self._metadata_ready = True
                self._is_trained = True
                return

            self._cf_model = _CFVAE(feature_count, self._encoded_size).to(self._device)

            if not self._train_when_fit:
                raise ValueError(
                    "CfvaeMethod requires train=True when pretrained_path is not provided"
                )

            self._train_cfvae(feature_df)
            self._cf_model.eval()
            self._metadata_ready = True
            self._is_trained = True

    def _train_cfvae(self, train_features: pd.DataFrame) -> None:
        if self._cf_model is None:
            raise RuntimeError("CFVAE network has not been initialized")

        train_original_tensor = torch.tensor(
            train_features.loc[:, self._feature_names].to_numpy(dtype="float32"),
            dtype=torch.float32,
            device=self._device,
        )
        train_model_tensor = self._to_model_space_tensor(train_original_tensor)
        dataset_size = int(train_original_tensor.shape[0])

        if dataset_size < 2:
            raise ValueError(
                "CfvaeMethod requires at least 2 training rows because the "
                "underlying CFVAE network uses BatchNorm1d"
            )
        if self._batch_size < 2:
            raise ValueError(
                "CfvaeMethod requires batch_size >= 2 because the underlying "
                "CFVAE network uses BatchNorm1d"
            )

        drop_last = (
            dataset_size > self._batch_size and dataset_size % self._batch_size == 1
        )
        if drop_last:
            self._logger.info(
                "Dropping the last singleton CFVAE training batch to avoid BatchNorm1d failure"
            )

        train_loader = DataLoader(
            TensorDataset(train_original_tensor, train_model_tensor),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=self._device == "cuda",
        )

        optimizer = optim.SGD(
            self._cf_model.parameters(),
            lr=self._learning_rate,
            weight_decay=1e-2,
        )
        self._cf_model.train()
        self._logger.info("Starting CFVAE training on %d rows", dataset_size)

        epoch_iterator = tqdm(range(self._epoch), desc="cfvae-fit", leave=False)
        for _ in epoch_iterator:
            epoch_loss = 0.0
            for original_x, model_x in train_loader:
                original_x = original_x.to(self._device)
                model_x = model_x.to(self._device)
                optimizer.zero_grad()

                target_index = self._resolve_target_indices_tensor(original_x)
                reconstruction_loss = torch.zeros(model_x.size(0), device=self._device)
                kl_loss = torch.zeros(model_x.size(0), device=self._device)
                validity_loss = torch.zeros(1, device=self._device)
                constraint_loss = torch.zeros(1, device=self._device)
                preference_loss = torch.zeros(1, device=self._device)

                for _ in range(self._n_samples):
                    x_pred_model, kl_loss = self._cf_model.forward_with_kl(
                        model_x, target_index.to(dtype=torch.float32)
                    )
                    reconstruction_loss += self._compute_reconstruction_increment(
                        model_x, x_pred_model
                    )

                    x_pred_original = self._from_model_space_tensor(x_pred_model)
                    y_pred = self._target_model.forward(x_pred_original)
                    y_pred_pos = y_pred[target_index == 1, :]
                    y_pred_neg = y_pred[target_index == 0, :]

                    if y_pred_pos.shape[0] > 0:
                        validity_loss += F.hinge_embedding_loss(
                            y_pred_pos[:, 1] - y_pred_pos[:, 0],
                            torch.tensor(-1, device=self._device),
                            self._margin,
                            reduction="mean",
                        )
                    if y_pred_neg.shape[0] > 0:
                        validity_loss += F.hinge_embedding_loss(
                            y_pred_neg[:, 0] - y_pred_neg[:, 1],
                            torch.tensor(-1, device=self._device),
                            self._margin,
                            reduction="mean",
                        )

                    if self._constraint_loss_func is not None:
                        constraint_term = self._constraint_loss_func(
                            train_x=original_x,
                            x_pred=x_pred_original,
                        )
                        constraint_loss += self._reduce_optional_loss(constraint_term)

                    if self._preference_dataset is not None:
                        preference_loss += self._compute_preference_loss(
                            original_x, x_pred_original
                        )

                reconstruction_loss = reconstruction_loss / self._n_samples
                kl_loss = kl_loss / self._n_samples
                validity_loss = (
                    -1.0 * self._validity_reg * validity_loss / self._n_samples
                )
                constraint_loss = (
                    self._constraint_reg * constraint_loss / self._n_samples
                )
                preference_loss = (
                    self._preference_reg * preference_loss / self._n_samples
                )

                loss = (
                    -torch.mean(reconstruction_loss - kl_loss)
                    - validity_loss
                    + constraint_loss
                    + preference_loss
                )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item())

            epoch_iterator.set_postfix(loss=f"{epoch_loss:.4f}")

        if self._save_path is not None:
            save_dir = Path(self._save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = save_dir / "cfvae_model.pth"
            torch.save(self._cf_model.state_dict(), checkpoint_path)
            self._logger.info(
                "Saved CFVAE checkpoint to %s", checkpoint_path.as_posix()
            )

    def _resolve_target_indices_tensor(self, factuals: torch.Tensor) -> torch.Tensor:
        if self._class_to_index is None:
            raise RuntimeError("Target class mapping is unavailable")

        logits = self._target_model.forward(factuals)
        original_prediction = torch.argmax(logits, dim=1).detach().cpu().numpy()
        target_indices = resolve_target_indices(
            target_model=self._target_model,
            original_prediction=original_prediction,
            desired_class=self._desired_class,
        )
        return torch.tensor(target_indices, dtype=torch.long, device=self._device)

    def _compute_reconstruction_increment(
        self,
        factuals_model_space: torch.Tensor,
        candidates_model_space: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_increment = torch.zeros(
            factuals_model_space.size(0), device=self._device
        )

        if self._categorical_indices:
            reconstruction_increment += -torch.sum(
                torch.abs(
                    factuals_model_space[:, self._categorical_indices]
                    - candidates_model_space[:, self._categorical_indices]
                ),
                dim=1,
            )

        for index, feature_range in self._continuous_ranges.items():
            reconstruction_increment += -feature_range * torch.abs(
                factuals_model_space[:, index] - candidates_model_space[:, index]
            )

        for group in self._onehot_group_indices:
            reconstruction_increment += -torch.abs(
                1.0 - torch.sum(candidates_model_space[:, group], dim=1)
            )

        return reconstruction_increment

    def _to_model_space_tensor(self, values: torch.Tensor) -> torch.Tensor:
        output = values.clone().to(dtype=torch.float32)
        for index, (low_value, high_value) in self._binary_value_ranges.items():
            output[:, index] = (output[:, index] - low_value) / (high_value - low_value)
        return output

    def _from_model_space_tensor(self, values: torch.Tensor) -> torch.Tensor:
        output = values.clone().to(dtype=torch.float32)
        for index, (low_value, high_value) in self._binary_value_ranges.items():
            output[:, index] = output[:, index] * (high_value - low_value) + low_value
        return output

    def _project_model_output(self, values: torch.Tensor) -> torch.Tensor:
        output = values.clone().to(dtype=torch.float32)

        if self._continuous_indices:
            output[:, self._continuous_indices] = output[
                :, self._continuous_indices
            ].clamp(0.0, 1.0)

        for index in self._binary_scalar_indices:
            output[:, index] = torch.round(output[:, index]).clamp(0.0, 1.0)

        for group in self._onehot_group_indices:
            group_values = output[:, group]
            argmax_indices = torch.argmax(group_values, dim=1, keepdim=True)
            projected_group = torch.zeros_like(group_values)
            projected_group.scatter_(1, argmax_indices, 1.0)
            output[:, group] = projected_group

        return output

    def _reduce_optional_loss(self, loss_value: Any) -> torch.Tensor:
        if isinstance(loss_value, torch.Tensor):
            if loss_value.ndim == 0:
                return loss_value.to(self._device)
            return loss_value.to(self._device).mean()
        return torch.tensor(float(loss_value), dtype=torch.float32, device=self._device)

    def _compute_preference_loss(
        self,
        original_x: torch.Tensor,
        x_pred_original: torch.Tensor,
    ) -> torch.Tensor:
        if self._preference_dataset is None:
            return torch.zeros(1, device=self._device)

        x_prefer_map = self._preference_dataset.get("x_prefer", {})
        y_prefer_map = self._preference_dataset.get("y_prefer", {})

        positive_scores: list[torch.Tensor] = []
        negative_scores: list[torch.Tensor] = []

        for factual_row, candidate_row in zip(original_x, x_pred_original):
            key = tuple(float(value) for value in factual_row.detach().cpu().tolist())
            preferred_x = x_prefer_map.get(key, [])
            preferred_y = y_prefer_map.get(key, [])
            if len(preferred_x) != len(preferred_y):
                continue

            for preference_x, preference_y in zip(preferred_x, preferred_y):
                preference_tensor = torch.tensor(
                    preference_x,
                    dtype=torch.float32,
                    device=self._device,
                ).reshape(-1)
                score = torch.exp(
                    -0.5 * (preference_tensor - candidate_row).pow(2)
                ).mean()
                if int(preference_y) == 1:
                    positive_scores.append(score)
                else:
                    negative_scores.append(score)

        preference_loss = torch.zeros(1, device=self._device)
        if positive_scores:
            preference_loss += 1.0 - torch.stack(positive_scores).mean()
        if negative_scores:
            preference_loss += torch.stack(negative_scores).mean()
        return preference_loss

    def _generate_candidate_tensor(
        self,
        original_tensor: torch.Tensor,
        *,
        sample: bool,
        project: bool,
    ) -> torch.Tensor:
        if self._cf_model is None:
            raise RuntimeError("Method network is not initialized")
        model_input = self._to_model_space_tensor(original_tensor)
        target_index = self._resolve_target_indices_tensor(original_tensor)
        with torch.no_grad():
            candidate_model_space = self._cf_model(
                model_input,
                target_index.to(dtype=torch.float32),
                sample=sample,
            )
            if project:
                candidate_model_space = self._project_model_output(
                    candidate_model_space
                )
            return self._from_model_space_tensor(candidate_model_space)

    def _collect_sampling_artifacts(
        self,
        factuals: pd.DataFrame,
        batch_size: int,
    ) -> dict[str, dict[int, np.ndarray]]:
        num_rows = factuals.shape[0]
        num_features = len(self._feature_names)
        artifacts: dict[str, dict[int, np.ndarray]] = {
            metric_name: {
                sample_size: np.empty(
                    (self._sampling_trials, sample_size, num_rows, num_features),
                    dtype=np.float32,
                )
                for sample_size in self._sampling_sample_sizes
            }
            for metric_name in self._SAMPLING_METRIC_ORDER
        }

        with self._seed_guard():
            for trial_index in range(self._sampling_trials):
                for metric_name in self._SAMPLING_METRIC_ORDER:
                    for sample_size in self._sampling_sample_sizes:
                        for sample_index in range(sample_size):
                            sampled_batches = []
                            for start in range(0, num_rows, batch_size):
                                batch = factuals.iloc[start : start + batch_size]
                                original_tensor = torch.tensor(
                                    batch.to_numpy(dtype="float32"),
                                    dtype=torch.float32,
                                    device=self._device,
                                )
                                sampled_batches.append(
                                    self._generate_candidate_tensor(
                                        original_tensor,
                                        sample=True,
                                        project=False,
                                    )
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                            artifacts[metric_name][sample_size][
                                trial_index, sample_index
                            ] = np.concatenate(sampled_batches, axis=0)
        return artifacts

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained or self._cf_model is None:
            raise RuntimeError("Method is not trained")
        if not self._metadata_ready:
            raise RuntimeError(
                "Method metadata is not initialized; call fit() after loading pretrained weights"
            )
        if factuals.isna().any(axis=None):
            raise ValueError(
                "CfvaeMethod.get_counterfactuals() does not support NaN values"
            )
        if set(factuals.columns) != set(self._feature_names):
            raise ValueError(
                "factuals must contain the same feature columns used in fit()"
            )

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        if factuals.empty:
            return factuals.copy(deep=True)

        self._cf_model.eval()
        rows: list[list[float]] = []

        with self._seed_guard():
            for _, row in factuals.iterrows():
                original_row_tensor = torch.tensor(
                    row.to_numpy(dtype="float32").reshape(1, -1),
                    dtype=torch.float32,
                    device=self._device,
                )
                candidate_original = self._generate_candidate_tensor(
                    original_row_tensor,
                    sample=False,
                    project=True,
                )
                rows.append(candidate_original.view(-1).detach().cpu().tolist())

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> DatasetObject:
        output = super().predict(testset, batch_size=batch_size)
        if not self._store_sampling_artifacts:
            return output

        factuals = testset.get(target=False).loc[:, self._feature_names].copy(deep=True)
        sampling_artifacts = self._collect_sampling_artifacts(
            factuals=factuals,
            batch_size=batch_size,
        )

        enriched_output = output.clone()
        enriched_output.update("cfvae_sampling_artifacts", sampling_artifacts)
        enriched_output.update(
            "cfvae_sampling_sample_sizes",
            list(self._sampling_sample_sizes),
        )
        enriched_output.update("cfvae_sampling_trials", int(self._sampling_trials))
        enriched_output.freeze()
        return enriched_output

    @staticmethod
    def constraint_loss_func_example(train_x: torch.Tensor, x_pred: torch.Tensor):
        return F.hinge_embedding_loss(
            x_pred[:, 0] - train_x[:, 0],
            torch.tensor(-1, device=train_x.device),
            0.0,
        )
