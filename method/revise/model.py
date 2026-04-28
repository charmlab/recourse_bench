from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from method.cchvae.autoencoder.models.vae import VariationalAutoencoder
from method.revise.support import ReviseTargetModelAdapter, check_counterfactuals
from utils.caching import get_cache_dir


def merge_default_parameters(
    hyperparams: dict[str, Any] | None,
    default: dict[str, Any],
) -> dict[str, Any]:
    if hyperparams is None:
        return deepcopy(default)

    output: dict[str, Any] = {}
    for key, default_value in default.items():
        if isinstance(default_value, dict):
            nested_params = hyperparams.get(key, {})
            if not isinstance(nested_params, dict):
                raise TypeError(f"hyperparams['{key}'] must be a dictionary")
            output[key] = merge_default_parameters(nested_params, default_value)
            continue

        if key not in hyperparams:
            output[key] = deepcopy(default_value)
            continue
        if hyperparams[key] is None:
            raise ValueError(f"For {key} in hyperparams is a value needed")
        output[key] = hyperparams[key]
    return output


def _project_onehot_group(group_tensor: torch.Tensor) -> torch.Tensor:
    winner = group_tensor.argmax(dim=1, keepdim=True)
    projected = torch.zeros_like(group_tensor)
    projected.scatter_(1, winner, 1.0)
    return projected


def _project_thermometer_group(group_tensor: torch.Tensor) -> torch.Tensor:
    size = int(group_tensor.shape[1])
    legal_patterns = torch.tril(
        torch.ones((size, size), dtype=group_tensor.dtype, device=group_tensor.device)
    )
    distances = torch.norm(
        legal_patterns.unsqueeze(0) - group_tensor.unsqueeze(1),
        dim=2,
    )
    best_index = distances.argmin(dim=1)
    return legal_patterns[best_index]


def reconstruct_encoding_constraints(
    x: torch.Tensor,
    categorical_groups: Sequence[Sequence[int]],
    binary_feature_indices: Sequence[int],
    thermometer_groups: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    x_enc = x.clone()

    for group in categorical_groups:
        if len(group) < 2:
            continue
        x_enc[:, list(group)] = _project_onehot_group(x_enc[:, list(group)])

    for group in thermometer_groups or []:
        if len(group) < 2:
            continue
        x_enc[:, list(group)] = _project_thermometer_group(x_enc[:, list(group)])

    if binary_feature_indices:
        x_enc[:, list(binary_feature_indices)] = torch.clamp(
            torch.round(x_enc[:, list(binary_feature_indices)]),
            0.0,
            1.0,
        )

    return x_enc


class Revise:
    _DEFAULT_HYPERPARAMS = {
        "data_name": None,
        "lambda": 0.5,
        "optimizer": "adam",
        "lr": 0.1,
        "max_iter": 1000,
        "target_class": [0, 1],
        "binary_cat_features": True,
        "vae_params": {
            "layers": None,
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

    def __init__(
        self,
        mlmodel: ReviseTargetModelAdapter,
        hyperparams: dict[str, Any] | None = None,
        vae: VariationalAutoencoder | None = None,
        device: str = "cpu",
    ) -> None:
        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        self._logger = logging.getLogger(__name__)
        self._mlmodel = mlmodel
        self.mlmodel = mlmodel
        self.device = str(device)
        self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        data = self._mlmodel.data

        self._target_column = data.target
        self._lambda = float(self._params["lambda"])
        self._optimizer = str(self._params["optimizer"]).lower()
        self._lr = float(self._params["lr"])
        self._max_iter = int(self._params["max_iter"])
        self._target_class = list(self._params["target_class"])
        self._desired_class = int(np.argmax(np.asarray(self._target_class)))
        self._binary_cat_features = bool(self._params["binary_cat_features"])

        if self._optimizer not in {"adam", "rmsprop"}:
            raise ValueError("REVISE optimizer must be 'adam' or 'rmsprop'")
        if self._max_iter < 1:
            raise ValueError("REVISE max_iter must be >= 1")

        vae_params = self._params["vae_params"]
        if self._params["data_name"] is None:
            raise ValueError("REVISE requires hyperparams['data_name']")
        if vae_params["layers"] is None:
            raise ValueError("REVISE requires hyperparams['vae_params']['layers']")

        os.environ["CF_MODELS"] = get_cache_dir("autoencoders")
        self.vae = (
            vae
            if vae is not None
            else VariationalAutoencoder(
                self._params["data_name"],
                list(vae_params["layers"]),
                mlmodel.get_mutable_mask(),
            )
        )
        self.vae.device = self.device
        self.vae.to(self.device)

        if vae_params["train"]:
            self._logger.info(
                "Training REVISE VAE on %d rows with %d mutable dimensions",
                data.df.shape[0],
                int(mlmodel.get_mutable_mask().sum()),
            )
            self.vae.fit(
                xtrain=data.df[mlmodel.feature_input_order],
                lambda_reg=float(vae_params["lambda_reg"]),
                epochs=int(vae_params["epochs"]),
                lr=float(vae_params["lr"]),
                batch_size=int(vae_params["batch_size"]),
            )
        else:
            try:
                self.vae.load(data.df.shape[1] - 1)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Loading of Autoencoder failed. {exc}"
                ) from exc
            self.vae.to(self.device)

        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        input_columns = list(factuals.columns)
        ordered_factuals = self._mlmodel.get_ordered_features(factuals).reindex(
            index=factuals.index,
            columns=self._mlmodel.feature_input_order,
        )
        if ordered_factuals.empty:
            return ordered_factuals.reindex(
                index=factuals.index,
                columns=input_columns,
            ).copy(deep=True)

        categorical_groups = getattr(self._mlmodel.data, "categorical_groups", [])
        thermometer_groups = getattr(self._mlmodel.data, "thermometer_groups", [])
        binary_feature_indices = (
            getattr(self._mlmodel.data, "binary_feature_indices", [])
            if self._binary_cat_features
            else []
        )
        list_cfs = self._counterfactual_optimization(
            categorical_groups=categorical_groups,
            thermometer_groups=thermometer_groups,
            binary_feature_indices=binary_feature_indices,
            df_fact=ordered_factuals,
        )

        cf_df = check_counterfactuals(
            self._mlmodel,
            list_cfs,
            factuals=ordered_factuals,
            desired_class=self._desired_class,
        )
        return cf_df.reindex(index=factuals.index, columns=input_columns).copy(
            deep=True
        )

    def _counterfactual_optimization(
        self,
        categorical_groups: Sequence[Sequence[int]],
        thermometer_groups: Sequence[Sequence[int]],
        binary_feature_indices: Sequence[int],
        df_fact: pd.DataFrame,
    ) -> list[np.ndarray]:
        test_loader = torch.utils.data.DataLoader(
            df_fact.to_numpy(dtype="float32", copy=False),
            batch_size=1,
            shuffle=False,
        )
        list_cfs: list[np.ndarray] = []
        target = torch.tensor(
            self._target_class,
            dtype=torch.float32,
            device=self.device,
        )
        target_prediction = int(np.argmax(np.asarray(self._target_class)))

        for query_instance in tqdm(
            test_loader,
            total=df_fact.shape[0],
            desc="revise-search",
            leave=False,
        ):
            query_instance = query_instance.float().to(self.device)

            z = self.vae.encode(query_instance[:, self.vae.mutable_mask])[0]
            z = torch.cat([z, query_instance[:, ~self.vae.mutable_mask]], dim=-1)
            z = z.clone().detach().requires_grad_(True)

            if self._optimizer == "adam":
                optim = torch.optim.Adam([z], lr=self._lr)
            else:
                optim = torch.optim.RMSprop([z], lr=self._lr)

            candidate_counterfactuals: list[np.ndarray] = []
            candidate_distances: list[float] = []

            for _ in range(self._max_iter):
                decoded_cf = self.vae.decode(z)
                cf_soft = query_instance.clone()
                cf_soft[:, self.vae.mutable_mask] = decoded_cf
                cf_hard = reconstruct_encoding_constraints(
                    cf_soft,
                    categorical_groups=categorical_groups,
                    thermometer_groups=thermometer_groups,
                    binary_feature_indices=binary_feature_indices,
                )

                output_soft = self._mlmodel.forward(cf_soft)[0]
                output_hard = self._mlmodel.predict_proba(cf_hard)[0]
                predicted = int(torch.argmax(output_hard).item())
                loss = self._compute_loss(output_soft, cf_soft, query_instance, target)

                if predicted == target_prediction:
                    candidate_counterfactuals.append(
                        cf_hard.detach().cpu().numpy().squeeze(axis=0)
                    )
                    candidate_distances.append(float(loss.detach().cpu().item()))

                loss.backward()
                optim.step()
                optim.zero_grad()

            if candidate_counterfactuals:
                best_index = int(np.argmin(np.asarray(candidate_distances)))
                cf_tensor = (
                    torch.tensor(
                        candidate_counterfactuals[best_index],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                cf_tensor = reconstruct_encoding_constraints(
                    cf_tensor,
                    categorical_groups=categorical_groups,
                    thermometer_groups=thermometer_groups,
                    binary_feature_indices=binary_feature_indices,
                )
                list_cfs.append(cf_tensor.detach().cpu().numpy().squeeze(axis=0))
            else:
                cf_tensor = reconstruct_encoding_constraints(
                    query_instance.detach().clone(),
                    categorical_groups=categorical_groups,
                    thermometer_groups=thermometer_groups,
                    binary_feature_indices=binary_feature_indices,
                )
                list_cfs.append(cf_tensor.detach().cpu().numpy().squeeze(axis=0))

        return list_cfs

    def _compute_loss(
        self,
        output: torch.Tensor,
        cf_initialize: torch.Tensor,
        query_instance: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss_function = nn.BCELoss()
        loss1 = loss_function(output, target)
        loss2 = torch.norm(cf_initialize - query_instance, 1)
        return loss1 + self._lambda * loss2
