from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject


class MethodObject(ABC):
    """Base class for an algorithmic recourse method.

    A method wraps a trained target model and generates counterfactuals that
    flip (or move toward) a desired class. Subclasses implement :meth:`fit` and
    :meth:`get_counterfactuals`; the inherited :meth:`predict` wraps the raw
    counterfactual dataframe into a frozen counterfactual
    :class:`~dataset.dataset_object.DatasetObject` and attaches runtime,
    prediction, and target-label metadata.

    Attributes
    ----------
    _need_grad : bool
        Whether the method requires a differentiable target model.
    _desired_class : int or str or None
        Target class for recourse. ``None`` means "flip" for binary problems
        and "keep current label" otherwise.
    """

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
        """Configure the method against a target model.

        Parameters
        ----------
        target_model : ModelObject
            The (to-be-)trained classifier to generate recourse for. The method
            must check model-type and device compatibility.
        seed : int, optional
            Seed for any stochastic search.
        device : str, default "cpu"
            Device, which must match ``target_model``'s device.
        desired_class : int or str or None, optional
            Class label to steer counterfactuals toward.
        **kwargs
            Implementation-specific hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, trainset: DatasetObject | None):
        """Fit the method (train auxiliary models, build search structures).

        Parameters
        ----------
        trainset : DatasetObject or None
            Finalized training data. Methods that need no training should set
            ``self._is_trained = True`` in ``__init__``; otherwise set it here.
        """
        raise NotImplementedError

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """Generate counterfactuals for a batch of factual rows.

        Parameters
        ----------
        factuals : pandas.DataFrame
            Feature rows to explain (no target column).

        Returns
        -------
        pandas.DataFrame
            Same rows and feature columns as ``factuals``. Rows with no valid
            counterfactual must be filled with ``NaN``.
        """
        raise NotImplementedError

    def counterfactual_set_metadata(
        self,
        factual_index: pd.Index,
        feature_columns: pd.Index,
    ) -> dict[str, object] | None:
        return None

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> DatasetObject:
        """Generate counterfactuals over a dataset and package the result.

        Calls :meth:`get_counterfactuals` in batches, validates that row count
        and feature columns are preserved, and returns a frozen counterfactual
        dataset. Failed rows carry ``NaN`` features and target ``-1``. The
        returned object also stores ``runtime_seconds``, ``runtime_total_seconds``,
        ``factual_prediction_index``, ``target_prediction_index``, and (when
        ``desired_class`` is set) an ``evaluation_filter``.

        Parameters
        ----------
        testset : DatasetObject
            Frozen factual dataset. Must not already be a counterfactual dataset.
        batch_size : int, default 20
            Factual rows per call to :meth:`get_counterfactuals`.

        Returns
        -------
        DatasetObject
            A frozen counterfactual dataset aligned to ``testset``.

        Raises
        ------
        RuntimeError
            If the method is not trained.
        ValueError
            If ``batch_size < 1``, ``testset`` is already a counterfactual, or
            :meth:`get_counterfactuals` does not preserve rows/columns.
        """
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if getattr(testset, "counterfactual", False):
            raise ValueError("testset must not already be marked as counterfactual")

        factuals = testset.get(target=False)
        counterfactual_batches: list[pd.DataFrame] = []
        runtime_batches: list[pd.Series] = []
        metadata_batches: dict[str, list[object]] = {}

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
            counterfactual_set_metadata = self.counterfactual_set_metadata(
                factual_index=counterfactual_batch.index,
                feature_columns=counterfactual_batch.columns,
            )
            if counterfactual_set_metadata is not None:
                for flag, value in counterfactual_set_metadata.items():
                    metadata_batches.setdefault(flag, []).append(value)

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
        for flag, batches in metadata_batches.items():
            if all(isinstance(batch_value, list) for batch_value in batches):
                output.update(
                    flag,
                    [
                        item
                        for batch_value in batches
                        for item in batch_value
                    ],
                )
            else:
                if batches and all(batch_value == batches[0] for batch_value in batches):
                    output.update(flag, batches[0])
                else:
                    output.update(flag, batches)

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
