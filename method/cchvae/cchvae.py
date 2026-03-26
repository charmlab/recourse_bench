from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.cchvae.architecture import (
    FeatureSpec,
    TypedCchvaeNet,
    normalize_feature_block,
)
from method.cchvae.support import (
    BlackBoxModelTypes,
    RecourseModelAdapter,
    ensure_supported_target_model,
    resolve_target_indices,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from preprocess.preprocess_utils import resolve_feature_metadata
from utils.registry import register
from utils.seed import seed_context


def _to_tensor(values: np.ndarray | pd.DataFrame, device: str) -> torch.Tensor:
    if isinstance(values, pd.DataFrame):
        array = values.to_numpy(dtype="float32")
    else:
        array = values.astype("float32", copy=False)
    return torch.tensor(array, dtype=torch.float32, device=device)


def reconstruct_binary_constraints(
    x: np.ndarray,
    binary_feature_indices: list[int],
) -> np.ndarray:
    if not binary_feature_indices:
        return x
    output = x.copy()
    output[:, binary_feature_indices] = np.clip(
        np.round(output[:, binary_feature_indices]), 0.0, 1.0
    )
    return output


@register("cchvae")
class CchvaeMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        dim_latent_s: int = 3,
        dim_latent_z: int = 2,
        dim_latent_y: int = 5,
        epochs: int = 80,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        search_samples: int = 1000,
        p_norm: int = 2,
        step_size: float = 0.5,
        max_step: int = 500,
        degree_active: float = 1.0,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "CchvaeMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._dim_latent_s = int(dim_latent_s)
        self._dim_latent_z = int(dim_latent_z)
        self._dim_latent_y = int(dim_latent_y)
        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._batch_size = int(batch_size)
        self._search_samples = int(search_samples)
        self._p_norm = int(p_norm)
        self._step_size = float(step_size)
        self._max_step = int(max_step)
        self._degree_active = float(degree_active)

        self._logger = logging.getLogger(__name__)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self._search_samples < 1:
            raise ValueError("search_samples must be >= 1")
        if self._p_norm not in {1, 2}:
            raise ValueError("p_norm must be either 1 or 2")
        if self._step_size <= 0:
            raise ValueError("step_size must be > 0")
        if self._max_step < 1:
            raise ValueError("max_step must be >= 1")

    def _resolve_cchvae_feature_type(
        self,
        trainset: DatasetObject,
    ) -> dict[str, str]:
        try:
            typed_feature_type = trainset.attr("cchvae_feature_type")
        except AttributeError:
            typed_feature_type = None

        raw_feature_type, _, _ = resolve_feature_metadata(trainset)
        resolved: dict[str, str] = {}
        for feature_name in self._feature_names:
            if typed_feature_type is not None and feature_name in typed_feature_type:
                likelihood = str(typed_feature_type[feature_name]).lower()
            else:
                raw_kind = str(raw_feature_type[feature_name]).lower()
                if raw_kind == "numerical":
                    likelihood = "real"
                elif raw_kind in {"categorical", "binary"}:
                    likelihood = "count"
                else:
                    raise ValueError(
                        f"Unsupported raw_feature_type for cchvae fallback: {raw_kind}"
                    )

            if likelihood not in {"real", "pos", "count"}:
                raise ValueError(
                    f"Unsupported cchvae_feature_type for {feature_name}: {likelihood}"
                )
            resolved[feature_name] = likelihood
        return resolved

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CchvaeMethod.fit()")

        with seed_context(self._seed):
            feature_df = trainset.get(target=False)
            self._feature_names = list(feature_df.columns)
            _, feature_mutability, _ = resolve_feature_metadata(trainset)
            self._conditionals = [
                feature_name
                for feature_name in self._feature_names
                if not bool(feature_mutability[feature_name])
            ]
            self._free_features = [
                feature_name
                for feature_name in self._feature_names
                if bool(feature_mutability[feature_name])
            ]
            if not self._free_features:
                raise ValueError("CchvaeMethod requires at least one mutable feature")
            self._typed_feature_type = self._resolve_cchvae_feature_type(trainset)

            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._free_specs = [
                FeatureSpec(name=name, likelihood=self._typed_feature_type[name])
                for name in self._free_features
            ]
            self._conditional_specs = [
                FeatureSpec(name=name, likelihood=self._typed_feature_type[name])
                for name in self._conditionals
            ]
            self._model = TypedCchvaeNet(
                free_specs=self._free_specs,
                conditional_specs=self._conditional_specs,
                latent_s_dim=self._dim_latent_s,
                latent_z_dim=self._dim_latent_z,
                latent_y_dim=self._dim_latent_y,
                device=self._device,
            )
            optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self._learning_rate
            )

            train_features = trainset.get(target=False).loc[:, self._feature_names]
            free_array = train_features.loc[:, self._free_features].to_numpy(
                dtype="float32"
            )
            conditional_array = train_features.loc[:, self._conditionals].to_numpy(
                dtype="float32"
            )
            usable_rows = (free_array.shape[0] // self._batch_size) * self._batch_size
            if usable_rows == 0:
                raise ValueError("Training data must contain at least one full batch")

            self._logger.info(
                "Starting author-style C-CHVAE training on %d usable rows",
                usable_rows,
            )
            epoch_iterator = tqdm(
                range(self._epochs),
                desc="cchvae-fit",
                leave=False,
            )
            for epoch in epoch_iterator:
                tau = max(1.0 - 0.001 * epoch, 1e-3)
                permutation = np.random.permutation(free_array.shape[0])[:usable_rows]
                epoch_loss = 0.0
                epoch_recon = 0.0
                epoch_kl_z = 0.0
                epoch_kl_s = 0.0
                num_batches = usable_rows // self._batch_size

                for batch_index in range(num_batches):
                    start = batch_index * self._batch_size
                    stop = start + self._batch_size
                    batch_indices = permutation[start:stop]

                    free_batch = _to_tensor(free_array[batch_indices], self._device)
                    conditional_batch = _to_tensor(
                        conditional_array[batch_indices], self._device
                    )

                    optimizer.zero_grad()
                    loss, recon, kl_z, kl_s = self._model.compute_loss(
                        free_values=free_batch,
                        conditional_values=conditional_batch,
                        tau=tau,
                    )
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.detach().cpu())
                    epoch_recon += float(recon.detach().cpu())
                    epoch_kl_z += float(kl_z.detach().cpu())
                    epoch_kl_s += float(kl_s.detach().cpu())

                epoch_iterator.set_postfix(
                    loss=f"{epoch_loss / num_batches:.4f}",
                    recon=f"{epoch_recon / num_batches:.4f}",
                    kl_z=f"{epoch_kl_z / num_batches:.4f}",
                    kl_s=f"{epoch_kl_s / num_batches:.4f}",
                )

            self._model.eval()
            self._is_trained = True
            self._logger.info("Finished C-CHVAE training")

    def _hyper_sphere_coordinates(
        self,
        instance: np.ndarray,
        high: float,
        low: float,
    ) -> np.ndarray:
        delta_instance = np.random.randn(self._search_samples, instance.shape[1])
        dist = np.random.rand(self._search_samples) * (high - low) + low
        norm_p = LA.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)
        return instance + np.multiply(delta_instance, d_norm)

    def _inference_stats(
        self,
        factuals: pd.DataFrame,
    ) -> tuple[
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        np.ndarray,
        StandardScaler,
    ]:
        free_tensor = _to_tensor(factuals.loc[:, self._free_features], self._device)
        conditional_tensor = _to_tensor(
            factuals.loc[:, self._conditionals], self._device
        )
        _, free_stats = normalize_feature_block(free_tensor, self._free_specs)
        _, conditional_stats = normalize_feature_block(
            conditional_tensor, self._conditional_specs
        )
        z_values = (
            self._model.encode_deterministic(
                free_values=free_tensor,
                conditional_values=conditional_tensor,
                free_stats=free_stats,
                conditional_stats=conditional_stats,
            )
            .detach()
            .cpu()
            .numpy()
        )
        scaler = StandardScaler().fit(
            factuals.loc[:, self._free_features].to_numpy(dtype="float32")
        )
        return free_stats, conditional_stats, z_values, scaler

    def _counterfactual_search(
        self,
        factual: pd.Series,
        latent_z: np.ndarray,
        free_stats: list[tuple[torch.Tensor, torch.Tensor]],
        scaler: StandardScaler,
    ) -> np.ndarray:
        factual_free = factual.loc[self._free_features].to_numpy(dtype="float32")
        factual_conditional = factual.loc[self._conditionals].to_numpy(dtype="float32")
        factual_full = factual.loc[self._feature_names].to_numpy(dtype="float32")
        factual_scaled = scaler.transform(factual_free.reshape(1, -1))[0]

        factual_label = self._adapter.predict_label_indices(
            factual_full.reshape(1, -1)
        ).reshape(-1)
        target_label = resolve_target_indices(
            target_model=self._target_model,
            original_prediction=factual_label,
            desired_class=self._desired_class,
        )[0]

        low = 0.0
        high = self._step_size
        count = 0

        while True:
            count += 1
            if count > self._max_step:
                return np.full(len(self._feature_names), np.nan, dtype=np.float32)

            latent_neighbourhood = self._hyper_sphere_coordinates(
                latent_z.reshape(1, -1),
                high=high,
                low=low,
            )
            decoded_free = (
                self._model.sample_from_latent(
                    _to_tensor(latent_neighbourhood, self._device),
                    free_stats=free_stats,
                )
                .detach()
                .cpu()
                .numpy()
            )
            scaled_candidates = scaler.transform(decoded_free)
            distance = np.abs(scaled_candidates - factual_scaled).sum(axis=1)
            candidates = np.concatenate(
                [
                    np.repeat(
                        factual_conditional.reshape(1, -1),
                        self._search_samples,
                        axis=0,
                    ),
                    decoded_free,
                ],
                axis=1,
            )
            candidate_prediction = self._adapter.predict_label_indices(candidates)
            success_indices = np.where(candidate_prediction == target_label)[0]

            if success_indices.size == 0:
                low = high
                high = low + self._step_size
                continue

            best_index = success_indices[np.argmin(distance[success_indices])]
            return candidates[best_index].astype(np.float32, copy=False)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("factuals must not contain NaN")
        if factuals.empty:
            return factuals.copy(deep=True)

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        with seed_context(self._seed):
            free_stats, _, latent_z, scaler = self._inference_stats(factuals)
            rows = []
            for index, (_, row) in enumerate(factuals.iterrows()):
                rows.append(
                    self._counterfactual_search(
                        factual=row,
                        latent_z=latent_z[index],
                        free_stats=free_stats,
                        scaler=scaler,
                    )
                )

        candidates = pd.DataFrame(
            rows,
            index=factuals.index,
            columns=self._feature_names,
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> DatasetObject:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if getattr(testset, "counterfactual", False):
            raise ValueError("testset must not already be marked as counterfactual")

        factuals = testset.get(target=False).loc[:, self._feature_names]
        full_counterfactuals = pd.DataFrame(
            np.nan,
            index=factuals.index,
            columns=factuals.columns,
        )

        evaluation_filter = None
        search_factuals = factuals
        if self._desired_class is not None:
            class_to_index = self._target_model.get_class_to_index()
            target_index = class_to_index[self._desired_class]
            predicted_label = (
                self._target_model.predict(testset, batch_size=batch_size)
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
            evaluation_filter = pd.DataFrame(
                predicted_label != target_index,
                index=factuals.index,
                columns=["evaluation_filter"],
                dtype=bool,
            )
            search_factuals = factuals.loc[evaluation_filter.iloc[:, 0]]

        if not search_factuals.empty:
            full_counterfactuals.loc[search_factuals.index, :] = (
                self.get_counterfactuals(search_factuals)
            )

        target_column = testset.target_column
        counterfactual_target = pd.DataFrame(
            -1.0,
            index=full_counterfactuals.index,
            columns=[target_column],
        )
        counterfactual_df = pd.concat(
            [full_counterfactuals, counterfactual_target], axis=1
        ).reindex(columns=testset.ordered_features())

        output = testset.clone()
        output.update("counterfactual", True, df=counterfactual_df)
        if evaluation_filter is not None:
            output.update("evaluation_filter", evaluation_filter)
        output.freeze()
        return output
