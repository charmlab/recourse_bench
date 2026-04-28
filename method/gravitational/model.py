from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from method.gravitational.support import (
    GravitationalTargetModelAdapter,
    check_counterfactuals,
)


def merge_default_parameters(
    hyperparams: dict[str, Any] | None,
    default: dict[str, Any],
) -> dict[str, Any]:
    if hyperparams is None:
        return dict(default)

    output: dict[str, Any] = {}
    for key, default_value in default.items():
        if isinstance(default_value, dict):
            nested_params = hyperparams.get(key, {})
            if not isinstance(nested_params, dict):
                raise TypeError(f"hyperparams['{key}'] must be a dictionary")
            output[key] = merge_default_parameters(nested_params, default_value)
            continue

        if key not in hyperparams:
            output[key] = default_value
            continue
        if hyperparams[key] is None:
            raise ValueError(f"For {key} in hyperparams is a value needed")
        output[key] = hyperparams[key]
    return output


class Gravitational:
    """
    Implemention of Gravitational Recourse Algorithm

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.
    x_center : numpy.array
        A central or sensible point in the feature space of the target class.
        By default, the mean of the instances belonging to the target class will be assign to x_center.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    prediction_loss:
        Calculates the loss that measures how well the counterfactual is classified as the target class by the model.
    cost:
        Calculates the distance between the original input instance and the generated counterfactua.
    gravitational_penalty:
        Calculates the distance between the generated counterfactual and the central point.
    set_x_center:
        Sets x_center entry to x_center.
    reset_x_center:
        Sets the mean of the instances belonging to the target class to the x_center and return it.

    Notes
    -----
    - Restriction
        * Currently working only with PyTorch models

    -Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.
        * "prediction_loss_lambda": float, default: 1
            Controls the weight of the prediction loss in the total loss.
        * "original_dist_lambda": float, default: 0.5
            Controls the weight of the original distance in the total loss.
        * "grav_penalty_lambda": float, default: 1.5
            Controls the weight of the gravitational penalty in the total loss.
        * "learning_rate": float, default: 0.01
            Specifies the learning rate for the optimization algorithm.
        * "num_steps": int, default: 100
            Specifies the number of iterations for the optimization process.
        * "target_class": int (0 or 1), default: 1:
            Specifies the desired class for the counterfactual.
        * "scheduler_step_size": int, default: 100
            Step_size for "torch.optim.lr_scheduler.StepLR". Specifies the number of steps or epochs after
            which the learning rate should be decreased.
        * "scheduler_gamma": float, default: 0.5
            Gamma for "torch.optim.lr_scheduler.StepLR". Specifies the factor by which the learning rate is multiplied
            at each step when the scheduler is applied.

    Implemented from:
        "Endogenous Macrodynamics in Algorithmic Recourse"
        Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S. Liem
    """

    _DEFAULT_HYPERPARAMS = {
        "prediction_loss_lambda": 1,
        "original_dist_lambda": 0.5,
        "grav_penalty_lambda": 1.5,
        "learning_rate": 0.01,
        "num_steps": 100,
        "target_class": 1,
        "scheduler_step_size": 100,
        "scheduler_gamma": 0.5,
    }

    def __init__(
        self,
        mlmodel: GravitationalTargetModelAdapter | None = None,
        hyperparams: dict[str, Any] | None = None,
        x_center: np.ndarray | None = None,
    ):
        if mlmodel is None:
            raise ValueError("mlmodel is required")

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.device = torch.device(str(getattr(mlmodel, "device", "cpu")))
        self._mlmodel = mlmodel
        self.mlmodel = mlmodel
        self.prediction_loss_lambda = checked_hyperparams["prediction_loss_lambda"]
        self.original_dist_lambda = checked_hyperparams["original_dist_lambda"]
        self.grav_penalty_lambda = checked_hyperparams["grav_penalty_lambda"]
        self.learning_rate = checked_hyperparams["learning_rate"]
        self.num_steps = checked_hyperparams["num_steps"]
        self.target_class = checked_hyperparams["target_class"]
        self.scheduler_step_size = checked_hyperparams["scheduler_step_size"]
        self.scheduler_gamma = checked_hyperparams["scheduler_gamma"]

        self.x_center = x_center
        if self.x_center is None:
            self.x_center = self.reset_x_center()

        self.criterion = nn.CrossEntropyLoss()

    def _compute_default_x_center(self) -> np.ndarray:
        train_features = getattr(self.mlmodel, "train_features", None)
        train_labels = getattr(self.mlmodel, "train_labels", None)
        if train_features is None or train_labels is None:
            raise ValueError(
                "x_center is required when adapter training data is unavailable"
            )

        if isinstance(train_labels, pd.Series):
            label_array = train_labels.to_numpy(dtype="int64", copy=False)
        else:
            label_array = np.asarray(train_labels, dtype="int64").reshape(-1)

        mask = label_array == int(self.target_class)
        if mask.any():
            x_center = train_features.loc[mask].mean(axis=0).to_numpy(dtype=np.float32)
        else:
            y_pred = self.mlmodel.predict(train_features)
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.asarray(y_pred).reshape(-1) == int(self.target_class)
            if np.asarray(y_pred).sum() > 0:
                x_center = (
                    train_features.loc[y_pred].mean(axis=0).to_numpy(dtype=np.float32)
                )
            else:
                x_center = train_features.mean(axis=0).to_numpy(dtype=np.float32)

        return np.nan_to_num(x_center, nan=0.0, posinf=1e6, neginf=-1e6)

    def prediction_loss(
        self,
        model: GravitationalTargetModelAdapter,
        x_cf: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        x_cf = x_cf.to(self.device)
        output = model.predict_proba(x_cf)
        target_class_tensor = torch.tensor(
            [target_class] * output.size(0),
            dtype=torch.long,
            device=self.device,
        )
        return self.criterion(output, target_class_tensor)

    def cost(self, x_original: torch.Tensor, x_cf: torch.Tensor) -> torch.Tensor:
        return torch.norm(x_original - x_cf)

    def gravitational_penalty(
        self,
        x_cf: torch.Tensor,
        x_center: np.ndarray,
    ) -> torch.Tensor:
        x_center_tensor = torch.tensor(
            x_center,
            dtype=torch.float32,
            device=self.device,
        )
        return torch.norm(x_cf - x_center_tensor)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        factuals = self._mlmodel.get_ordered_features(factuals)
        factuals_index = factuals.index.copy()
        x_original = torch.tensor(
            factuals.to_numpy(dtype="float32"),
            dtype=torch.float32,
            device=self.device,
        )
        x_cf = x_original.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([x_cf], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )

        for _ in range(self.num_steps):
            optimizer.zero_grad()

            prediction_loss_value = self.prediction_loss(
                self.mlmodel,
                x_cf,
                self.target_class,
            )
            original_dist = self.cost(x_original, x_cf)
            grav_penalty = self.gravitational_penalty(x_cf, self.x_center)

            loss = (
                self.prediction_loss_lambda * prediction_loss_value
                + self.original_dist_lambda * original_dist
                + self.grav_penalty_lambda * grav_penalty
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

        x_cf_array = x_cf.detach().cpu().numpy()
        x_cf_df = pd.DataFrame(
            x_cf_array,
            columns=factuals.columns,
            index=factuals_index.copy(),
        )
        df_cfs = check_counterfactuals(
            self.mlmodel,
            x_cf_df,
            factuals=factuals,
            desired_class=int(self.target_class),
        )
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs.reindex(index=factuals_index, columns=factuals.columns)

    def set_x_center(self, x_center: np.ndarray) -> None:
        self.x_center = np.asarray(x_center, dtype=np.float32)

    def reset_x_center(self) -> np.ndarray:
        self.x_center = self._compute_default_x_center()
        return self.x_center
