from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject, process_nan
from model.model_utils import resolve_device
from utils.registry import register
from utils.seed import seed_context


@register("sklearn_logistic_regression")
class SklearnLogisticRegressionModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        cv: int = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        c_values: Iterable[float] | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = False
        self._is_trained = False
        self._solver = str(solver)
        self._max_iter = int(max_iter)
        self._cv = int(cv)
        self._scoring = str(scoring)
        self._n_jobs = int(n_jobs)
        self._c_values = (
            [float(value) for value in c_values]
            if c_values is not None
            else np.logspace(-4, 3).tolist()
        )
        self._model: LogisticRegression | None = None
        self._grid_search: GridSearchCV | None = None

        if self._device != "cpu":
            raise ValueError("SklearnLogisticRegressionModel only supports cpu")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self._cv < 2:
            raise ValueError("cv must be >= 2")
        if not self._c_values:
            raise ValueError("c_values must not be empty")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError(
                "trainset is required for SklearnLogisticRegressionModel.fit()"
            )

        with seed_context(self._seed):
            X, labels, _ = self.extract_training_data(trainset)
            estimator = LogisticRegression(
                max_iter=self._max_iter,
                solver=self._solver,
                random_state=self._seed,
            )
            grid = GridSearchCV(
                estimator=estimator,
                param_grid={"C": list(self._c_values)},
                cv=self._cv,
                scoring=self._scoring,
                return_train_score=True,
                n_jobs=self._n_jobs,
                refit=True,
            )
            grid.fit(X, labels.cpu().numpy())
            self._grid_search = grid
            self._model = grid.best_estimator_
            self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")

        with seed_context(self._seed):
            probabilities = torch.tensor(
                self._model.predict_proba(X), dtype=torch.float32
            )
            if proba:
                return probabilities
            indices = probabilities.argmax(dim=1)
            return torch.nn.functional.one_hot(
                indices, num_classes=probabilities.shape[1]
            ).to(dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise TypeError(
            "SklearnLogisticRegressionModel.forward() is unavailable because the underlying model is not torch-based"
        )
