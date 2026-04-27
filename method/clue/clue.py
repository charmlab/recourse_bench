from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.clue.library import CLUE, VAE_gauss_cat_net, training
from method.clue.library.clue_ml.src.utils import (
    Ln_distance,
    register_checkpoint_aliases,
)
from method.clue.support import (
    ensure_supported_target_model,
    resolve_feature_names,
    resolve_input_dim_vec,
    resolve_vae_checkpoint_path,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.caching import get_cache_dir
from utils.registry import register
from utils.seed import seed_context


@register("clue")
class ClueMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        pretrained_path: str | None = None,
        save_name: str | None = None,
        uncertainty_weight: float = 1.0,
        aleatoric_weight: float = 0.0,
        epistemic_weight: float = 0.0,
        prior_weight: float = 0.0,
        distance_weight: float | None = None,
        latent_l2_weight: float = 0.0,
        prediction_similarity_weight: float = 0.0,
        lr: float = 1e-2,
        min_steps: int = 3,
        max_steps: int = 50,
        early_stop_patience: int = 3,
        sample_std: float = 0.15,
        num_explanations: int = 1,
        width: int = 300,
        depth: int = 3,
        latent_dim: int = 4,
        batch_size: int = 128,
        epochs: int = 2500,
        early_stop: int = 200,
        learning_rate: float = 1e-4,
        **kwargs,
    ):
        del kwargs
        ensure_supported_target_model(target_model, "ClueMethod")
        if desired_class is not None:
            raise ValueError("ClueMethod does not support desired_class")

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = None

        self._pretrained_path = pretrained_path
        self._save_name = save_name
        self._uncertainty_weight = float(uncertainty_weight)
        self._aleatoric_weight = float(aleatoric_weight)
        self._epistemic_weight = float(epistemic_weight)
        self._prior_weight = float(prior_weight)
        self._distance_weight = (
            None if distance_weight is None else float(distance_weight)
        )
        self._latent_l2_weight = float(latent_l2_weight)
        self._prediction_similarity_weight = float(prediction_similarity_weight)
        self._lr = float(lr)
        self._min_steps = int(min_steps)
        self._max_steps = int(max_steps)
        self._early_stop_patience = int(early_stop_patience)
        self._sample_std = float(sample_std)
        self._num_explanations = int(num_explanations)

        self._width = int(width)
        self._depth = int(depth)
        self._latent_dim = int(latent_dim)
        self._batch_size = int(batch_size)
        self._epochs = int(epochs)
        self._early_stop = int(early_stop)
        self._vae_learning_rate = float(learning_rate)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._lr <= 0:
            raise ValueError("lr must be > 0")
        if self._min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if self._max_steps < self._min_steps:
            raise ValueError("max_steps must be >= min_steps")
        if self._early_stop_patience < 0:
            raise ValueError("early_stop_patience must be >= 0")
        if self._num_explanations < 1:
            raise ValueError("num_explanations must be >= 1")

    def _resolve_vae_model_path(self, dataset_name: str) -> Path:
        checkpoint_path = resolve_vae_checkpoint_path(self._pretrained_path)
        if checkpoint_path is not None:
            return checkpoint_path

        save_stem = self._save_name or f"{dataset_name}_clue_vae"
        model_dir = Path(get_cache_dir("models")) / save_stem
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / "theta_best.dat"

    @staticmethod
    def _register_checkpoint_aliases() -> None:
        register_checkpoint_aliases()

    def _build_or_load_vae(self, train_features: np.ndarray, dataset_name: str) -> None:
        checkpoint_path = self._resolve_vae_model_path(dataset_name)
        model_dir = checkpoint_path.parent

        if self._pretrained_path is None:
            permutation = np.random.permutation(train_features.shape[0])
            split_index = max(1, int(np.ceil(0.1 * train_features.shape[0])))
            val_indices = permutation[:split_index]
            train_indices = permutation[split_index:]
            val_features = train_features[val_indices].copy()
            train_subset = train_features[train_indices].copy()
            if train_subset.shape[0] == 0:
                train_subset = train_features.copy()
            training(
                train_subset,
                val_features,
                self._input_dim_vec,
                model_dir.as_posix(),
                width=self._width,
                depth=self._depth,
                latent_dim=self._latent_dim,
                batch_size=self._batch_size,
                nb_epochs=self._epochs,
                lr=self._vae_learning_rate,
                early_stop=self._early_stop,
            )

        self._vae = VAE_gauss_cat_net(
            self._input_dim_vec,
            self._width,
            self._depth,
            self._latent_dim,
            pred_sig=False,
            lr=self._vae_learning_rate,
            cuda=self._device == "cuda",
            flatten=False,
        )
        self._register_checkpoint_aliases()
        self._vae.load(checkpoint_path.as_posix())

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ClueMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = resolve_feature_names(trainset)
            self._input_dim_vec = resolve_input_dim_vec(trainset)

            train_features_df = trainset.get(target=False).loc[:, self._feature_names]
            try:
                train_features = train_features_df.to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "ClueMethod requires numeric finalized features"
                ) from error

            dataset_name = getattr(trainset, "name", "dataset")
            self._build_or_load_vae(train_features, str(dataset_name))
            self._is_trained = True

    def _resolve_distance_weight(self, feature_dim: int) -> float:
        if self._distance_weight is not None:
            return self._distance_weight
        return 2.0 / float(feature_dim)

    def _generate_raw_counterfactual_array(
        self,
        factual_array: np.ndarray,
        *,
        uncertainty_weight: float,
        aleatoric_weight: float,
        epistemic_weight: float,
        prior_weight: float,
        distance_weight: float,
        latent_l2_weight: float,
        prediction_similarity_weight: float,
        lr: float,
        min_steps: int,
        max_steps: int,
        early_stop_patience: int,
        num_explanations: int,
        sample_std: float,
    ) -> np.ndarray:
        x_init = np.asarray(factual_array, dtype=np.float32).copy()
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_init_tensor = torch.tensor(
            x_init,
            dtype=torch.float32,
            device=self._device,
        )
        z_init = (
            self._vae.recongnition(x_init_tensor, flatten=False)
            .loc.detach()
            .cpu()
            .numpy()
        )
        clue_runner = CLUE(
            self._vae,
            self._target_model,
            x_init,
            uncertainty_weight=uncertainty_weight,
            aleatoric_weight=aleatoric_weight,
            epistemic_weight=epistemic_weight,
            prior_weight=prior_weight,
            distance_weight=distance_weight,
            latent_L2_weight=latent_l2_weight,
            prediction_similarity_weight=prediction_similarity_weight,
            lr=lr,
            desired_preds=None,
            cond_mask=None,
            distance_metric=Ln_distance(n=1, dim=(1)),
            z_init=z_init,
            norm_MNIST=False,
            flatten_BNN=False,
            regression=False,
            prob_BNN=True,
            cuda=self._device == "cuda",
        )
        if num_explanations == 1:
            _, x_vec, _, _, _, _, _ = clue_runner.optimise(
                min_steps=min_steps,
                max_steps=max_steps,
                n_early_stop=early_stop_patience,
            )
            return x_vec[-1].astype(np.float32, copy=False)

        full_x_vec, _, _, _, _, _, _ = clue_runner.sample_explanations(
            n_explanations=num_explanations,
            init_std=sample_std,
            min_steps=min_steps,
            max_steps=max_steps,
            n_early_stop=early_stop_patience,
        )
        return full_x_vec[0, -1].astype(np.float32, copy=False)

    def _select_best_counterfactuals(
        self,
        factual_array: np.ndarray,
        candidate_arrays: list[np.ndarray],
    ) -> np.ndarray:
        factual_frame = pd.DataFrame(factual_array, columns=self._feature_names)
        factual_probs = self._target_model.get_prediction(factual_frame, proba=True)
        factual_entropy = (
            -(factual_probs * torch.log(factual_probs + 1e-10)).sum(dim=1).cpu().numpy()
        )
        factual_prediction = factual_probs.argmax(dim=1).cpu().numpy()

        best_candidates = np.full_like(factual_array, np.nan, dtype=np.float32)
        best_entropy = np.full(factual_array.shape[0], np.inf, dtype=np.float32)

        for candidate_array in candidate_arrays:
            candidate_frame = pd.DataFrame(candidate_array, columns=self._feature_names)
            candidate_probs = self._target_model.get_prediction(
                candidate_frame, proba=True
            )
            candidate_entropy = (
                -(candidate_probs * torch.log(candidate_probs + 1e-10))
                .sum(dim=1)
                .cpu()
                .numpy()
            )
            candidate_prediction = candidate_probs.argmax(dim=1).cpu().numpy()
            success_mask = (candidate_entropy < factual_entropy) & (
                candidate_prediction == factual_prediction
            )
            replace_mask = success_mask & (candidate_entropy < best_entropy)
            best_candidates[replace_mask] = candidate_array[replace_mask]
            best_entropy[replace_mask] = candidate_entropy[replace_mask]

        return best_candidates

    def get_counterfactuals_clue(
        self,
        factuals: pd.DataFrame,
        *,
        uncertainty_weight: float | None = None,
        aleatoric_weight: float | None = None,
        epistemic_weight: float | None = None,
        prior_weight: float | None = None,
        distance_weight: float | None = None,
        latent_l2_weight: float | None = None,
        prediction_similarity_weight: float | None = None,
        lr: float | None = None,
        min_steps: int | None = None,
        max_steps: int | None = None,
        early_stop_patience: int | None = None,
        num_explanations: int | None = None,
        sample_std: float | None = None,
    ) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        factual_array = factuals.to_numpy(dtype="float32", copy=True)
        resolved_uncertainty_weight = (
            self._uncertainty_weight
            if uncertainty_weight is None
            else float(uncertainty_weight)
        )
        resolved_aleatoric_weight = (
            self._aleatoric_weight
            if aleatoric_weight is None
            else float(aleatoric_weight)
        )
        resolved_epistemic_weight = (
            self._epistemic_weight
            if epistemic_weight is None
            else float(epistemic_weight)
        )
        resolved_prior_weight = (
            self._prior_weight if prior_weight is None else float(prior_weight)
        )
        resolved_distance_weight = (
            self._resolve_distance_weight(factual_array.shape[1])
            if distance_weight is None
            else float(distance_weight)
        )
        resolved_latent_l2_weight = (
            self._latent_l2_weight
            if latent_l2_weight is None
            else float(latent_l2_weight)
        )
        resolved_prediction_similarity_weight = (
            self._prediction_similarity_weight
            if prediction_similarity_weight is None
            else float(prediction_similarity_weight)
        )
        resolved_lr = self._lr if lr is None else float(lr)
        resolved_min_steps = self._min_steps if min_steps is None else int(min_steps)
        resolved_max_steps = self._max_steps if max_steps is None else int(max_steps)
        resolved_early_stop_patience = (
            self._early_stop_patience
            if early_stop_patience is None
            else int(early_stop_patience)
        )
        resolved_num_explanations = (
            self._num_explanations
            if num_explanations is None
            else int(num_explanations)
        )
        resolved_sample_std = (
            self._sample_std if sample_std is None else float(sample_std)
        )

        if resolved_lr <= 0:
            raise ValueError("lr must be > 0")
        if resolved_min_steps < 1:
            raise ValueError("min_steps must be >= 1")
        if resolved_max_steps < resolved_min_steps:
            raise ValueError("max_steps must be >= min_steps")
        if resolved_early_stop_patience < 0:
            raise ValueError("early_stop_patience must be >= 0")
        if resolved_num_explanations < 1:
            raise ValueError("num_explanations must be >= 1")

        with seed_context(self._seed):
            raw_candidates = self._generate_raw_counterfactual_array(
                factual_array,
                uncertainty_weight=resolved_uncertainty_weight,
                aleatoric_weight=resolved_aleatoric_weight,
                epistemic_weight=resolved_epistemic_weight,
                prior_weight=resolved_prior_weight,
                distance_weight=resolved_distance_weight,
                latent_l2_weight=resolved_latent_l2_weight,
                prediction_similarity_weight=resolved_prediction_similarity_weight,
                lr=resolved_lr,
                min_steps=resolved_min_steps,
                max_steps=resolved_max_steps,
                early_stop_patience=resolved_early_stop_patience,
                num_explanations=resolved_num_explanations,
                sample_std=resolved_sample_std,
            )
        return pd.DataFrame(
            raw_candidates,
            index=factuals.index,
            columns=self._feature_names,
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        raw_candidates = self.get_counterfactuals_clue(factuals)
        selected_candidates = self._select_best_counterfactuals(
            factual_array=factuals.to_numpy(dtype="float32", copy=True),
            candidate_arrays=[raw_candidates.to_numpy(dtype="float32", copy=True)],
        )
        candidates = pd.DataFrame(
            selected_candidates,
            index=factuals.index,
            columns=self._feature_names,
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            feature_names=self._feature_names,
        )
