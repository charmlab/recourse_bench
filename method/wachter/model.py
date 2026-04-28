from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from method.wachter.library.wachter import wachter_recourse
from method.wachter.support import WachterTargetModelAdapter, check_counterfactuals


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
            if default_value is None:
                raise ValueError(
                    f"For {key} is no default value defined, please pass this key "
                    "and its value in hyperparams"
                )
            output[key] = (
                None if default_value == "_optional_" else deepcopy(default_value)
            )
            continue
        if hyperparams[key] is None:
            if default_value == "_optional_":
                output[key] = None
                continue
            raise ValueError(f"For {key} in hyperparams is a value needed")
        output[key] = hyperparams[key]
    return output


class Wachter:
    """Gradient-based Wachter recourse on a differentiable target model adapter."""

    _DEFAULT_HYPERPARAMS = {
        "feature_cost": "_optional_",
        "lr": 0.01,
        "lambda_": 0.01,
        "n_iter": 1000,
        "t_max_min": 0.5,
        "norm": 1,
        "clamp": True,
        "loss_type": "BCE",
        "y_target": [0, 1],
    }

    def __init__(
        self,
        mlmodel: WachterTargetModelAdapter | None = None,
        hyperparams: dict[str, Any] | None = None,
    ):
        if mlmodel is None:
            raise ValueError("mlmodel is required")

        supported_backends = ["pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        self._mlmodel = mlmodel
        self.mlmodel = mlmodel
        self.device = str(getattr(mlmodel, "device", "cpu"))
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self._feature_cost = checked_hyperparams["feature_cost"]
        self._lr = checked_hyperparams["lr"]
        self._lambda_param = checked_hyperparams["lambda_"]
        self._n_iter = checked_hyperparams["n_iter"]
        self._t_max_min = checked_hyperparams["t_max_min"]
        self._norm = checked_hyperparams["norm"]
        self._clamp = checked_hyperparams["clamp"]
        self._loss_type = str(checked_hyperparams["loss_type"]).upper()
        self._y_target = checked_hyperparams["y_target"]

        if self._feature_cost is not None:
            feature_cost = np.asarray(self._feature_cost, dtype=np.float32).reshape(-1)
            if feature_cost.shape[0] != len(self._mlmodel.feature_input_order):
                raise ValueError(
                    "feature_cost length must match the finalized feature count"
                )
            self._feature_cost = feature_cost
        if self._lr <= 0:
            raise ValueError("lr must be > 0")
        if self._lambda_param < 0:
            raise ValueError("lambda_ must be >= 0")
        if self._n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if self._t_max_min <= 0:
            raise ValueError("t_max_min must be > 0")
        if int(self._norm) < 1:
            raise ValueError("norm must be >= 1")
        if self._loss_type not in {"BCE", "MSE"}:
            raise ValueError("loss_type must be 'BCE' or 'MSE'")

        if self._loss_type == "BCE":
            if len(self._y_target) != 2:
                raise ValueError("BCE y_target must have length 2")
            self._desired_class = int(np.argmax(np.asarray(self._y_target)))
        else:
            if len(self._y_target) != 1:
                raise ValueError("MSE y_target must have length 1")
            self._desired_class = int(float(self._y_target[0]) > 0.0)

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
        binary_feature_indices = getattr(
            self._mlmodel.data,
            "binary_feature_indices",
            [],
        )

        counterfactuals_list: list[np.ndarray] = []
        for _, factual in tqdm(
            ordered_factuals.iterrows(),
            total=ordered_factuals.shape[0],
            desc="wachter-search",
            leave=False,
        ):
            counterfactuals_list.append(
                wachter_recourse(
                    self._mlmodel,
                    factual.to_numpy(dtype="float32", copy=True).reshape(1, -1),
                    categorical_groups=categorical_groups,
                    thermometer_groups=thermometer_groups,
                    binary_feature_indices=binary_feature_indices,
                    feature_cost=self._feature_cost,
                    y_target=self._y_target,
                    lr=self._lr,
                    lambda_param=self._lambda_param,
                    n_iter=self._n_iter,
                    t_max_min=self._t_max_min,
                    norm=self._norm,
                    clamp=self._clamp,
                    loss_type=self._loss_type,
                )
            )

        df_cfs = check_counterfactuals(
            self._mlmodel,
            counterfactuals_list,
            factuals=ordered_factuals,
            desired_class=self._desired_class,
        )
        return df_cfs.reindex(
            index=factuals.index,
            columns=input_columns,
        ).copy(deep=True)
