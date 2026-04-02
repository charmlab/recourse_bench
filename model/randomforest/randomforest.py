from __future__ import annotations

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject, process_nan
from utils.registry import register
from utils.seed import seed_context


@register("randomforest")
class RandomForestModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        n_jobs: int | None = None,
        **kwargs,
    ):
        self._model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=seed,
            n_jobs=n_jobs,
        )
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for RandomForestModel.fit()")
        with seed_context(self._seed):
            X, labels, _ = self.extract_training_data(trainset)
            self._model.fit(X, labels.cpu().numpy())
            self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            if proba:
                probabilities = torch.tensor(
                    self._model.predict_proba(X), dtype=torch.float32
                )
                return probabilities

            predicted_labels = torch.tensor(
                self._model.predict(X),
                dtype=torch.long,
            )
            return torch.nn.functional.one_hot(
                predicted_labels,
                num_classes=len(self._model.classes_),
            ).to(dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        with seed_context(self._seed):
            raise TypeError(
                "RandomForestModel.forward() is unavailable because the underlying model is not torch-based"
            )
