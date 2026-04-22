from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.cfrl.cfrl_backend import set_seed
from method.cfrl.cfrl_support import (
    CfrlSchema,
    build_cfrl_schema,
    build_predictor,
    cfrl_to_benchmark_frame,
    frame_to_cfrl_array,
    resolve_immutable_features_and_ranges,
    resolve_target_indices,
    validate_counterfactuals,
)
from method.cfrl.cfrl_tabular import CounterfactualRLTabular as CFRLExplainer
from method.cfrl.cfrl_tabular import get_conditional_dim, get_he_preprocessor
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


class HeterogeneousEncoder(nn.Module):
    def __init__(
        self, hidden_dim: int, latent_dim: int, input_dim: int | None = None
    ) -> None:
        super().__init__()
        use_lazy = hasattr(nn, "LazyLinear") and input_dim is None
        if input_dim is None and not use_lazy:
            raise ValueError(
                "input_dim must be provided when torch.nn.LazyLinear is unavailable."
            )
        if use_lazy:
            self.fc1 = nn.LazyLinear(hidden_dim)  # type: ignore[attr-defined]
            self.fc2 = nn.LazyLinear(latent_dim)  # type: ignore[attr-defined]
        else:
            assert input_dim is not None
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class HeterogeneousDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dims: List[int],
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        use_lazy = hasattr(nn, "LazyLinear") and latent_dim is None
        if latent_dim is None and not use_lazy:
            raise ValueError(
                "latent_dim must be provided when torch.nn.LazyLinear is unavailable."
            )
        if use_lazy:
            self.fc1 = nn.LazyLinear(hidden_dim)  # type: ignore[attr-defined]
            self.heads = nn.ModuleList(
                [nn.LazyLinear(dim) for dim in output_dims]  # type: ignore[attr-defined]
            )
        else:
            assert latent_dim is not None
            self.fc1 = nn.Linear(latent_dim, hidden_dim)
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_dim, dim) for dim in output_dims]
            )

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        h = F.relu(self.fc1(z))
        return [head(h) for head in self.heads]


@register("cfrl")
class CfrlMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        autoencoder_batch_size: int = 128,
        autoencoder_target_steps: int = 100_000,
        autoencoder_lr: float = 1e-3,
        autoencoder_latent_dim: int = 15,
        autoencoder_hidden_dim: int = 128,
        coeff_sparsity: float = 0.5,
        coeff_consistency: float = 0.5,
        train_steps: int = 100_000,
        batch_size: int = 128,
        immutable_features: list[str] | None = None,
        constrained_ranges: dict[str, list[float]] | None = None,
        train: bool = True,
        save_path: str | None = None,
        **kwargs,
    ):
        if not isinstance(target_model, ModelObject):
            raise TypeError(
                f"CfrlMethod expects a ModelObject target model, received {type(target_model).__name__}"
            )

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._autoencoder_batch_size = int(autoencoder_batch_size)
        self._autoencoder_target_steps = int(autoencoder_target_steps)
        self._autoencoder_lr = float(autoencoder_lr)
        self._autoencoder_latent_dim = int(autoencoder_latent_dim)
        self._autoencoder_hidden_dim = int(autoencoder_hidden_dim)
        self._coeff_sparsity = float(coeff_sparsity)
        self._coeff_consistency = float(coeff_consistency)
        self._train_steps = int(train_steps)
        self._batch_size = int(batch_size)
        self._immutable_features_override = (
            None if immutable_features is None else list(immutable_features)
        )
        self._constrained_ranges_override = (
            None
            if constrained_ranges is None
            else {
                str(name): [float(bounds[0]), float(bounds[1])]
                for name, bounds in constrained_ranges.items()
            }
        )
        self._train_enabled = bool(train)
        self._save_path = save_path

        self._schema: CfrlSchema | None = None
        self._feature_names: list[str] = []
        self._feature_columns: list[str] = []
        self._encoder: HeterogeneousEncoder | None = None
        self._decoder: HeterogeneousDecoder | None = None
        self._encoder_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None
        self._decoder_inv_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None
        self._cf_model: CFRLExplainer | None = None

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._autoencoder_batch_size < 1:
            raise ValueError("autoencoder_batch_size must be >= 1")
        if self._autoencoder_target_steps < 1:
            raise ValueError("autoencoder_target_steps must be >= 1")
        if self._autoencoder_lr <= 0.0:
            raise ValueError("autoencoder_lr must be > 0")
        if self._autoencoder_latent_dim < 1:
            raise ValueError("autoencoder_latent_dim must be >= 1")
        if self._autoencoder_hidden_dim < 1:
            raise ValueError("autoencoder_hidden_dim must be >= 1")
        if self._coeff_sparsity < 0.0:
            raise ValueError("coeff_sparsity must be >= 0")
        if self._coeff_consistency < 0.0:
            raise ValueError("coeff_consistency must be >= 0")
        if self._train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def _train_autoencoder(self, X_pre: np.ndarray, X_zero: np.ndarray) -> None:
        if self._encoder is None or self._decoder is None or self._schema is None:
            raise RuntimeError("CFRL autoencoder components are not initialized")

        inputs = torch.tensor(X_pre, dtype=torch.float32, device=self._device)
        num_dim = len(self._schema.numerical_indices)
        num_targets = inputs[:, :num_dim] if num_dim > 0 else None
        cat_targets = [
            torch.tensor(
                X_zero[:, idx].astype(np.int64),
                dtype=torch.long,
                device=self._device,
            )
            for idx in self._schema.categorical_indices
        ]

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optimizer = optim.Adam(params, lr=self._autoencoder_lr)

        num_samples = inputs.size(0)
        steps_per_epoch = max(1, math.ceil(num_samples / self._autoencoder_batch_size))
        max_epochs = math.ceil(self._autoencoder_target_steps / steps_per_epoch)
        steps_run = 0

        with tqdm(
            total=self._autoencoder_target_steps, desc="cfrl-ae", leave=False
        ) as pbar:
            for _ in range(max_epochs):
                permutation = torch.randperm(num_samples, device=self._device)
                for start in range(0, num_samples, self._autoencoder_batch_size):
                    idx = permutation[start : start + self._autoencoder_batch_size]
                    batch_x = inputs[idx]
                    outputs = self._decoder(self._encoder(batch_x))

                    loss = torch.zeros((), device=self._device)
                    output_offset = 0
                    if num_dim > 0 and num_targets is not None:
                        recon_num = outputs[0]
                        target_num = num_targets[idx]
                        loss = loss + F.mse_loss(recon_num, target_num)
                        output_offset = 1

                    cat_outputs = outputs[output_offset:]
                    cat_batches = [target[idx] for target in cat_targets]
                    if cat_outputs and cat_batches:
                        weight = 1.0 / len(cat_outputs)
                        for logits, target in zip(cat_outputs, cat_batches):
                            loss = loss + weight * F.cross_entropy(logits, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    steps_run += 1
                    pbar.update(1)
                    if steps_run >= self._autoencoder_target_steps:
                        break

                if steps_run >= self._autoencoder_target_steps:
                    break

        self._encoder.eval()
        self._decoder.eval()

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CfrlMethod.fit()")
        if not getattr(self._target_model, "_is_trained", False):
            raise RuntimeError("Target model must be trained before CfrlMethod.fit()")
        if not self._train_enabled:
            raise ValueError(
                "CfrlMethod requires train=True because pretrained loading is not implemented yet"
            )

        with seed_context(self._seed):
            set_seed(self._seed if self._seed is not None else 13)

            feature_df = trainset.get(target=False).copy(deep=True)
            if feature_df.isna().any(axis=None):
                raise ValueError("CfrlMethod.fit() does not support NaN values")

            self._feature_columns = list(feature_df.columns)
            self._schema = build_cfrl_schema(trainset)
            self._feature_names = list(self._schema.feature_names)

            X_zero = frame_to_cfrl_array(feature_df, self._schema).astype(np.float32)
            (
                self._encoder_preprocessor,
                self._decoder_inv_preprocessor,
            ) = get_he_preprocessor(
                X=X_zero,
                feature_names=self._schema.feature_names,
                category_map=self._schema.category_map,
                feature_types=self._schema.feature_types,
            )

            X_pre = self._encoder_preprocessor(X_zero).astype(np.float32)
            input_dim = X_pre.shape[1]
            self._encoder = HeterogeneousEncoder(
                hidden_dim=self._autoencoder_hidden_dim,
                latent_dim=self._autoencoder_latent_dim,
                input_dim=input_dim,
            ).to(self._device)

            output_dims: list[int] = []
            num_dim = len(self._schema.numerical_indices)
            if num_dim > 0:
                output_dims.append(num_dim)
            output_dims.extend(
                len(self._schema.category_map[idx])
                for idx in self._schema.categorical_indices
            )

            self._decoder = HeterogeneousDecoder(
                hidden_dim=self._autoencoder_hidden_dim,
                output_dims=output_dims,
                latent_dim=self._autoencoder_latent_dim,
            ).to(self._device)

            self._train_autoencoder(X_pre, X_zero)
            predictor = build_predictor(self._target_model, self._schema)

            immutable_features, ranges = resolve_immutable_features_and_ranges(
                self._schema,
                immutable_features=self._immutable_features_override,
                constrained_ranges=self._constrained_ranges_override,
            )

            preds_shape = predictor(X_zero[:1]).shape
            num_classes = preds_shape[1] if len(preds_shape) == 2 else 1
            cond_dim = get_conditional_dim(
                self._schema.feature_names, self._schema.category_map
            )
            actor_input_dim = self._autoencoder_latent_dim + 2 * num_classes + cond_dim

            self._cf_model = CFRLExplainer(
                predictor=predictor,
                encoder=self._encoder,
                decoder=self._decoder,
                latent_dim=self._autoencoder_latent_dim,
                encoder_preprocessor=self._encoder_preprocessor,
                decoder_inv_preprocessor=self._decoder_inv_preprocessor,
                coeff_sparsity=self._coeff_sparsity,
                coeff_consistency=self._coeff_consistency,
                feature_names=self._schema.feature_names,
                category_map=self._schema.category_map,
                immutable_features=immutable_features,
                ranges=ranges,
                train_steps=self._train_steps,
                batch_size=self._batch_size,
                seed=self._seed if self._seed is not None else 0,
                actor_input_dim=actor_input_dim,
            )
            self._cf_model.fit(X_zero.astype(np.float32))
            self._is_trained = True

    def _generate_counterfactual_row(
        self, factual_row: pd.DataFrame, target_class: int
    ) -> pd.DataFrame:
        if self._schema is None or self._cf_model is None:
            raise RuntimeError("CFRL method has not been initialized")

        zero_input = frame_to_cfrl_array(factual_row, self._schema).astype(np.float32)
        explanation = self._cf_model.explain(
            X=zero_input,
            Y_t=np.array([target_class]),
            C=[],
        )
        cf_data = explanation.get("cf", {}).get("X")
        if cf_data is None:
            return factual_row.copy(deep=True)

        cf_array = np.asarray(cf_data, dtype=np.float32)
        if cf_array.ndim == 3:
            cf_array = cf_array[:, 0, :]
        cf_array = np.atleast_2d(cf_array)

        return cfrl_to_benchmark_frame(
            cf_array,
            self._schema,
            index=factual_row.index,
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if self._schema is None:
            raise RuntimeError("CFRL schema is unavailable")
        if factuals.isna().any(axis=None):
            raise ValueError(
                "CfrlMethod.get_counterfactuals() does not support NaN values"
            )
        if set(factuals.columns) != set(self._feature_columns):
            raise ValueError(
                "factuals must contain the same feature columns used in fit()"
            )

        factuals = factuals.loc[:, self._feature_columns].copy(deep=True)
        target_indices = resolve_target_indices(
            target_model=self._target_model,
            factuals=factuals,
            desired_class=self._desired_class,
        )

        results: list[pd.DataFrame] = []
        with seed_context(self._seed):
            set_seed(self._seed if self._seed is not None else 13)
            for row_position, (_, row) in enumerate(factuals.iterrows()):
                factual_row = pd.DataFrame([row], columns=self._feature_columns)
                cf_row = self._generate_counterfactual_row(
                    factual_row=factual_row,
                    target_class=int(target_indices[row_position]),
                )
                cf_row.index = factual_row.index
                results.append(cf_row)

        candidates = pd.concat(results, axis=0).reindex(
            index=factuals.index, columns=self._feature_columns
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
