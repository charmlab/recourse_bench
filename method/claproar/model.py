from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from method.claproar.support import ClaproarTargetModelAdapter, check_counterfactuals


class ClaPROAR:
    """
    Implemention of ClaPROAR Recourse Algorithm

    Parameters
    ----------
    mlmodel : model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    compute_costs:
        Compute sum of all costs
    compute_yloss:
        Compute the outcome loss between model's prediction for x_prime and the desired outcome y_star.
    compute_individual_cost:
        Compute the individual cost. (Euclidean distance between x and x_prime)
    compute_external_cost:
        Compute the external cost. (The change in model loss when the new point x_prime is added)

    Notes
    -----
    - Restriction
        * Currently working only with Pytorch models

    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.
        * "individual_cost_lambda": float, default: 0.1
            Controls the weight of the individual cost.
        * "external_cost_lambda": float, default: 0.1
            Controls the weight of the external cost.
        * "learning_rate": ifloat, default: 0.01
            Controls how large the steps taken by the optimizer.
        * "max_iter": int, default: 100
            Maximum number of iterations.
        * "tol": float, default: 1e-4
            This is the tolerance for convergence, which sets a threshold for the gradient norm. If the gradient norm falls below this value,
            the optimization process will stop
        * "target_class": int (0 or 1), default: 1
            Desired output class.

    Implemented from:
        "Endogenous Macrodynamics in Algorithmic Recourse"
        Patrick Altmeyer, Giovan Angela, Karol Dobiczek, Arie van Deursen, Cynthia C. S.
    """

    def __init__(
        self,
        mlmodel: ClaproarTargetModelAdapter,
        device: str = "cpu",
        individual_cost_lambda: float = 0.1,
        external_cost_lambda: float = 0.1,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-4,
        target_class: int = 1,
    ):
        self.device = torch.device(device)
        self._mlmodel = mlmodel
        self.mlmodel = mlmodel
        self.individual_cost_lambda = float(individual_cost_lambda)
        self.external_cost_lambda = float(external_cost_lambda)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.target_class = int(target_class)
        self.criterion = nn.CrossEntropyLoss()

    def compute_yloss(self, x_prime: torch.Tensor) -> torch.Tensor:
        x_prime = x_prime.to(self.device)
        output = self.mlmodel.predict_proba(x_prime)
        target_class = torch.tensor(
            [self.target_class] * output.size(0), dtype=torch.long
        ).to(self.device)
        yloss = self.criterion(output, target_class)
        return yloss

    def compute_individual_cost(
        self, x: torch.Tensor, x_prime: torch.Tensor
    ) -> torch.Tensor:
        return torch.norm(x - x_prime)

    def compute_external_cost(self, x_prime: torch.Tensor) -> torch.Tensor:
        x_prime = x_prime.to(self.device)
        output = self.mlmodel.predict_proba(x_prime)
        target_class = torch.tensor(
            [1 - self.target_class] * output.size(0), dtype=torch.long
        ).to(self.device)
        ext_cost = self.criterion(output, target_class)
        return ext_cost

    def compute_costs(self, x: torch.Tensor, x_prime: torch.Tensor) -> torch.Tensor:
        yloss = self.compute_yloss(x_prime)
        individual_cost = self.compute_individual_cost(x, x_prime)
        external_cost = self.compute_external_cost(x_prime)

        return (
            yloss
            + self.individual_cost_lambda * individual_cost
            + self.external_cost_lambda * external_cost
        )

    def get_counterfactuals(self, factuals: pd.DataFrame, raw_output: bool = False):
        factuals = self._mlmodel.get_ordered_features(factuals)

        x = torch.tensor(
            factuals.to_numpy(dtype="float32"),
            dtype=torch.float32,
            device=self.device,
        )

        x_prime = x.clone().detach().requires_grad_(True)
        optimizer_cf = optim.Adam([x_prime], lr=self.learning_rate)

        for _ in range(self.max_iter):
            optimizer_cf.zero_grad()

            objective = self.compute_costs(x, x_prime)

            objective.backward()

            optimizer_cf.step()

            if torch.norm(x_prime.grad) < self.tol:
                break

        cfs = x_prime.detach()
        df_cfs = pd.DataFrame(
            cfs.cpu().numpy(),
            columns=factuals.columns,
            index=factuals.index.copy(),
        )

        if not raw_output:
            df_cfs = check_counterfactuals(
                self._mlmodel,
                df_cfs,
                factuals.index,
                negative_label=1 - self.target_class,
            )
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
