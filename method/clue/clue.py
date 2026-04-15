from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.clue.library import CLUE, VAE_gauss_cat_net, training
from method.clue.library.clue_ml.AE_models.AE import fc_gauss_cat as vendored_vae_fc
from method.clue.library.clue_ml.AE_models.AE import models as vendored_vae_models
from method.clue.library.clue_ml.src import gauss_cat as vendored_gauss_cat
from method.clue.library.clue_ml.src import layers as vendored_layers
from method.clue.library.clue_ml.src import probability as vendored_probability
from method.clue.library.clue_ml.src import radam as vendored_radam
from method.clue.library.clue_ml.src import utils as vendored_utils
from method.clue.library.clue_ml.src.utils import Ln_distance
from method.clue.support import (
    ClueModelAdapter,
    ensure_supported_target_model,
    resolve_feature_names,
    resolve_input_dim_vec,
    resolve_vae_checkpoint_path,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.mlp_bayesian.mlp_bayesian import MlpBayesianModel
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
        if "src" not in sys.modules:
            src_package = types.ModuleType("src")
            src_package.__path__ = []  # type: ignore[attr-defined]
            sys.modules["src"] = src_package
        if "VAE" not in sys.modules:
            vae_package = types.ModuleType("VAE")
            vae_package.__path__ = []  # type: ignore[attr-defined]
            sys.modules["VAE"] = vae_package
        sys.modules.setdefault("src.utils", vendored_utils)
        sys.modules.setdefault("src.radam", vendored_radam)
        sys.modules.setdefault("src.probability", vendored_probability)
        sys.modules.setdefault("src.gauss_cat", vendored_gauss_cat)
        sys.modules.setdefault("src.layers", vendored_layers)
        sys.modules.setdefault("VAE.fc_gauss_cat", vendored_vae_fc)
        sys.modules.setdefault("VAE.models", vendored_vae_models)

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
            self._adapter = ClueModelAdapter(self._target_model, self._feature_names)

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

    def _select_best_counterfactuals(
        self,
        factual_array: np.ndarray,
        candidate_arrays: list[np.ndarray],
    ) -> np.ndarray:
        factual_df = pd.DataFrame(factual_array, columns=self._feature_names)
        factual_entropy = self._adapter.entropy_numpy(factual_df)
        factual_prediction = self._adapter.predict_label_indices(factual_df)

        best_candidates = np.full_like(factual_array, np.nan, dtype=np.float32)
        best_entropy = np.full(factual_array.shape[0], np.inf, dtype=np.float32)

        for candidate_array in candidate_arrays:
            candidate_df = pd.DataFrame(candidate_array, columns=self._feature_names)
            candidate_entropy = self._adapter.entropy_numpy(candidate_df)
            candidate_prediction = self._adapter.predict_label_indices(candidate_df)
            success_mask = (candidate_entropy < factual_entropy) & (
                candidate_prediction == factual_prediction
            )
            replace_mask = success_mask & (candidate_entropy < best_entropy)
            best_candidates[replace_mask] = candidate_array[replace_mask]
            best_entropy[replace_mask] = candidate_entropy[replace_mask]

        return best_candidates

    def generate_raw_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        factual_array = factuals.to_numpy(dtype="float32", copy=True)

        with seed_context(self._seed):
            x_init = factual_array.copy()
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
                uncertainty_weight=self._uncertainty_weight,
                aleatoric_weight=self._aleatoric_weight,
                epistemic_weight=self._epistemic_weight,
                prior_weight=self._prior_weight,
                distance_weight=self._resolve_distance_weight(factual_array.shape[1]),
                latent_L2_weight=self._latent_l2_weight,
                prediction_similarity_weight=self._prediction_similarity_weight,
                lr=self._lr,
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

            if self._num_explanations == 1:
                _, x_vec, _, _, _, _, _ = clue_runner.optimise(
                    min_steps=self._min_steps,
                    max_steps=self._max_steps,
                    n_early_stop=self._early_stop_patience,
                )
                raw_candidates = x_vec[-1].astype(np.float32, copy=False)
            else:
                full_x_vec, _, _, _, _, _, _ = clue_runner.sample_explanations(
                    n_explanations=self._num_explanations,
                    init_std=self._sample_std,
                    min_steps=self._min_steps,
                    max_steps=self._max_steps,
                    n_early_stop=self._early_stop_patience,
                )
                raw_candidates = full_x_vec[0, -1].astype(np.float32, copy=False)

        return pd.DataFrame(
            raw_candidates,
            index=factuals.index,
            columns=self._feature_names,
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        raw_candidates = self.generate_raw_counterfactuals(factuals)

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
