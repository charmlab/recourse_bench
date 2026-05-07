from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject


class MethodObject(ABC):
    _target_model: ModelObject
    _seed: int | None = None
    _device: str
    _need_grad: bool
    _is_trained: bool = False
    _desired_class: int | str | None = None

    @abstractmethod
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def fit(self, trainset: DatasetObject | None):
        raise NotImplementedError

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> DatasetObject:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if getattr(testset, "counterfactual", False):
            raise ValueError("testset must not already be marked as counterfactual")

        factuals = testset.get(target=False)
        counterfactual_batches: list[pd.DataFrame] = []
        runtime_batches: list[pd.Series] = []

        for start in range(0, factuals.shape[0], batch_size):
            batch = factuals.iloc[start : start + batch_size]
            batch_start = perf_counter()
            counterfactual_batch = self.get_counterfactuals(batch)
            batch_elapsed = perf_counter() - batch_start

            if counterfactual_batch.shape[0] != batch.shape[0]:
                raise ValueError(
                    "get_counterfactuals() must preserve the input row count"
                )
            if set(counterfactual_batch.columns) != set(batch.columns):
                raise ValueError(
                    "get_counterfactuals() must preserve the input feature columns"
                )

            counterfactual_batch = counterfactual_batch.reindex(
                index=batch.index, columns=batch.columns
            )
            counterfactual_batches.append(counterfactual_batch)
            runtime_batches.append(
                pd.Series(
                    batch_elapsed / max(1, batch.shape[0]),
                    index=batch.index,
                    dtype="float64",
                )
            )

        if counterfactual_batches:
            counterfactual_features = pd.concat(counterfactual_batches, axis=0)
            counterfactual_features = counterfactual_features.reindex(
                index=factuals.index
            )
            runtime_seconds = pd.concat(runtime_batches, axis=0).reindex(
                index=factuals.index
            )
        else:
            counterfactual_features = factuals.iloc[0:0].copy(deep=True)
            runtime_seconds = pd.Series(index=factuals.index, dtype="float64")

        target_column = testset.target_column
        counterfactual_target = pd.DataFrame(
            -1.0,
            index=counterfactual_features.index,
            columns=[target_column],
        )
        counterfactual_df = pd.concat(
            [counterfactual_features, counterfactual_target], axis=1
        )
        counterfactual_df = counterfactual_df.reindex(
            columns=testset.ordered_features()
        )

        output = testset.clone()
        output.update("counterfactual", True, df=counterfactual_df)
        output.update(
            "runtime_seconds",
            pd.DataFrame(runtime_seconds, columns=["runtime_seconds"]),
        )
        output.update("runtime_total_seconds", float(runtime_seconds.sum()))

        prediction = self._target_model.predict(testset, batch_size=batch_size)
        predicted_label = prediction.argmax(dim=1).cpu().numpy().astype(np.int64)
        output.update(
            "factual_prediction_index",
            pd.DataFrame(
                predicted_label,
                index=counterfactual_df.index,
                columns=["factual_prediction_index"],
                dtype="int64",
            ),
        )

        class_to_index = self._target_model.get_class_to_index()
        if self._desired_class is None:
            if len(class_to_index) == 2:
                target_label = 1 - predicted_label
            else:
                target_label = predicted_label.copy()
        else:
            desired_index = int(class_to_index[self._desired_class])
            target_label = np.full(predicted_label.shape, desired_index, dtype=np.int64)
        output.update(
            "target_prediction_index",
            pd.DataFrame(
                target_label,
                index=counterfactual_df.index,
                columns=["target_prediction_index"],
                dtype="int64",
            ),
        )

        if self._desired_class is not None:
            desired_index = int(class_to_index[self._desired_class])
            evaluation_filter = pd.DataFrame(
                predicted_label != desired_index,
                index=counterfactual_df.index,
                columns=["evaluation_filter"],
                dtype=bool,
            )
            output.update("evaluation_filter", evaluation_filter)

        output.freeze()
        return output
