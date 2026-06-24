from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from dataset.dataset_object import DatasetObject


def process_nan():
    """Decorator factory for :meth:`ModelObject.get_prediction`.

    Wraps a prediction method so that any input row containing ``NaN`` (the
    convention for a failed counterfactual) is temporarily zero-filled before
    inference and its output is forced to ``-1`` afterwards, keeping invalid
    rows from contaminating predictions.

    Returns
    -------
    callable
        A decorator to apply to a ``get_prediction(self, X, ...)`` method.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, X: pd.DataFrame, *args, **kwargs):
            X_work = X.copy(deep=True)
            nan_rows = X_work.isna().any(axis=1)
            if nan_rows.any():
                X_work.loc[nan_rows, :] = 0.0
            y = func(self, X_work, *args, **kwargs)
            if nan_rows.any():
                y = y.clone()
                y[nan_rows.to_numpy()] = -1
            return y

        return wrapper

    return decorator


class ModelObject(ABC):
    """Base class for a target classifier that recourse methods explain.

    Subclasses implement :meth:`fit`, :meth:`get_prediction`, and
    :meth:`forward`. The :meth:`predict`/:meth:`predict_proba` methods are
    inherited batching wrappers over :meth:`get_prediction`. Differentiable
    (torch) models implement :meth:`forward` for gradient-based recourse;
    tree/sklearn models may raise from :meth:`forward`. ``device`` must match
    the recourse method's device.

    Attributes
    ----------
    _need_grad : bool
        Whether the model supports differentiable (gradient) access.
    _is_trained : bool
        Set to ``True`` by :meth:`fit`; guards inference.
    """

    _model: torch.nn.Module | RandomForestClassifier
    _seed: int | None = None
    _device: str
    _need_grad: bool
    _is_trained: bool = False
    _class_to_index: dict[int | str, int] | None = None

    @abstractmethod
    def __init__(self, seed: int | None = None, device: str = "cpu", **kwargs):
        """Configure the model.

        Parameters
        ----------
        seed : int, optional
            Seed for weight initialization and training.
        device : str, default "cpu"
            Device for torch models (``"cpu"`` or ``"cuda"``).
        **kwargs
            Implementation-specific hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, trainset: DatasetObject | None):
        """Train the model on a frozen training dataset.

        Parameters
        ----------
        trainset : DatasetObject or None
            Finalized training data. Implementations should set
            ``self._is_trained = True`` on success.
        """
        raise NotImplementedError

    @abstractmethod
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        """Predict on a feature dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature rows (no target column).
        proba : bool, default True
            If ``True`` return class probabilities; otherwise return logits.

        Returns
        -------
        torch.Tensor
            A ``(n_rows, n_classes)`` tensor. Typically decorated with
            :func:`process_nan`.
        """
        raise NotImplementedError

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> torch.Tensor:
        """Batched logits over a frozen dataset.

        Parameters
        ----------
        testset : DatasetObject
            Frozen dataset to predict on.
        batch_size : int, default 20
            Rows per inference batch.

        Returns
        -------
        torch.Tensor
            Concatenated ``(n_rows, n_classes)`` logits on CPU.

        Raises
        ------
        RuntimeError
            If the model is not trained.
        """
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        X = testset.get(target=False)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(X), batch_size):
            batch = X.iloc[start : start + batch_size]
            outputs.append(self.get_prediction(batch, proba=False).detach().cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0)

    def predict_proba(
        self, testset: DatasetObject, batch_size: int = 20
    ) -> torch.Tensor:
        """Batched class probabilities over a frozen dataset.

        Parameters
        ----------
        testset : DatasetObject
            Frozen dataset to predict on.
        batch_size : int, default 20
            Rows per inference batch.

        Returns
        -------
        torch.Tensor
            Concatenated ``(n_rows, n_classes)`` probabilities on CPU.
        """
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        X = testset.get(target=False)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(X), batch_size):
            batch = X.iloc[start : start + batch_size]
            outputs.append(self.get_prediction(batch, proba=True).detach().cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0)

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass on a feature tensor.

        Parameters
        ----------
        X : torch.Tensor
            Batch of feature rows.

        Returns
        -------
        torch.Tensor
            Logits. Non-differentiable models should raise ``RuntimeError``.
        """
        raise NotImplementedError

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(X)

    def extract_training_data(
        self,
        trainset: DatasetObject,
    ) -> tuple[pd.DataFrame, torch.Tensor, int]:
        """Split a trainset into features, integer labels, and output dimension.

        Builds and stores the class-to-index mapping used to translate model
        outputs back to dataset labels.

        Parameters
        ----------
        trainset : DatasetObject
            Frozen training dataset.

        Returns
        -------
        tuple[pandas.DataFrame, torch.Tensor, int]
            Features ``X``, integer ``labels``, and the number of output classes.
        """
        X = trainset.get(target=False)
        y = trainset.get(target=True)

        if y.shape[1] == 1:
            target = y.iloc[:, 0]
            if target.isna().any():
                raise ValueError("Target labels cannot contain NaN")

            unique_values = list(pd.Index(target.unique()))
            if len(unique_values) == 0:
                raise ValueError("Target labels cannot be empty")

            if all(isinstance(value, str) for value in unique_values):
                sorted_values = sorted(unique_values)
            elif all(
                isinstance(value, (int, np.integer))
                or (
                    isinstance(value, (float, np.floating))
                    and float(value).is_integer()
                )
                for value in unique_values
            ):
                target = target.map(int)
                sorted_values = sorted(pd.Index(target.unique()).tolist())
            else:
                raise TypeError(
                    "Single target_column must contain either all string or all integer labels"
                )

            self._class_to_index = {
                class_value: index for index, class_value in enumerate(sorted_values)
            }
            labels = torch.tensor(
                [self._class_to_index[value] for value in target.tolist()],
                dtype=torch.long,
            )
            output_dim = max(2, len(self._class_to_index))
            return X, labels, output_dim
        else:
            labels = torch.tensor(y.to_numpy().argmax(axis=1), dtype=torch.long)
            output_dim = y.shape[1]
            self._class_to_index = {index: index for index in range(output_dim)}
            return X, labels, output_dim

    def get_class_to_index(self) -> dict[int | str, int]:
        """Return the mapping from dataset label to model output index.

        Returns
        -------
        dict
            Copy of the label-to-index mapping established during :meth:`fit`.

        Raises
        ------
        RuntimeError
            If the mapping is unavailable (model not yet trained).
        """
        if self._class_to_index is None:
            raise RuntimeError("Target model class mapping is unavailable")
        return dict(self._class_to_index)
