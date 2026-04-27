from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from method.cchvae.autoencoder.models.csvae import CSVAE
from method.cruds.library.cruds import counterfactual_search
from method.cruds.support import CrudsTargetModelAdapter, check_counterfactuals
from utils.caching import get_cache_dir


class CRUDS:
    """
    Minimal native CRUDS integration around the imported CSVAE + path-search code.
    """

    def __init__(
        self,
        mlmodel: CrudsTargetModelAdapter,
        data_name: str,
        target_class: int = 1,
        lambda_param: float = 0.001,
        optimizer: str = "rmsprop",
        learning_rate: float = 0.008,
        max_iter: int = 2000,
        vae_hidden_dim: int = 64,
        vae_latent_dim: int = 6,
        vae_epochs: int = 5,
        vae_learning_rate: float = 1e-3,
        vae_batch_size: int = 32,
        device: str = "cpu",
    ):
        self._logger = logging.getLogger(__name__)
        self._mlmodel = mlmodel
        self.mlmodel = mlmodel
        self.device = device
        self._target_class = int(target_class)
        self._lambda_param = float(lambda_param)
        self._optimizer = str(optimizer).lower()
        self._learning_rate = float(learning_rate)
        self._max_iter = int(max_iter)
        self._vae_epochs = int(vae_epochs)
        self._vae_learning_rate = float(vae_learning_rate)
        self._vae_batch_size = int(vae_batch_size)

        mutable_dim = int(mlmodel.get_mutable_mask().sum())
        if mutable_dim < 1:
            raise ValueError("CRUDS requires at least one mutable feature")

        os.environ.setdefault("CF_MODELS", get_cache_dir("autoencoders"))
        self._csvae = CSVAE(
            data_name,
            [mutable_dim, int(vae_hidden_dim), int(vae_latent_dim)],
            mlmodel.get_mutable_mask(),
        )
        self._csvae.device = self.device
        self._csvae.to(self.device)

        train_df = self._mlmodel.data.df[
            self._mlmodel.feature_input_order + [self._mlmodel.data.target]
        ]
        self._logger.info(
            "Training CRUDS CSVAE on %d rows with %d mutable dimensions",
            train_df.shape[0],
            mutable_dim,
        )
        self._csvae.fit(
            data=train_df,
            epochs=self._vae_epochs,
            lr=self._vae_learning_rate,
            batch_size=self._vae_batch_size,
        )
        self._csvae.eval()
        for parameter in self._csvae.parameters():
            parameter.requires_grad_(False)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        ordered_factuals = self._mlmodel.get_ordered_features(factuals)
        if ordered_factuals.empty:
            return ordered_factuals.copy(deep=True)

        predicted_labels = np.asarray(self._mlmodel.predict(ordered_factuals)).reshape(
            -1
        )
        factuals_with_target = ordered_factuals.copy(deep=True)
        factuals_with_target[self._mlmodel.data.target] = predicted_labels

        columns = self._mlmodel.feature_input_order + [self._mlmodel.data.target]
        counterfactual_rows: list[np.ndarray] = []
        iterator = tqdm(
            factuals_with_target.iterrows(),
            total=factuals_with_target.shape[0],
            desc="cruds-search",
            leave=False,
        )
        for _, row in iterator:
            counterfactual_rows.append(
                counterfactual_search(
                    mlmodel=self._mlmodel,
                    csvae=self._csvae,
                    factual=row.to_numpy(dtype="float32", copy=True).reshape(1, -1),
                    categorical_groups=self._mlmodel.data.categorical_groups,
                    binary_feature_indices=self._mlmodel.data.binary_feature_indices,
                    target_class=self._target_class,
                    lambda_param=self._lambda_param,
                    optimizer=self._optimizer,
                    lr=self._learning_rate,
                    max_iter=self._max_iter,
                    device=self.device,
                )
            )

        counterfactual_frame = pd.DataFrame(
            np.asarray(counterfactual_rows),
            columns=columns,
            index=ordered_factuals.index.copy(),
        )
        checked = check_counterfactuals(
            self._mlmodel,
            counterfactual_frame.drop(columns=[self._mlmodel.data.target]),
            ordered_factuals.index,
            desired_class=self._target_class,
        )
        return self._mlmodel.get_ordered_features(checked)
