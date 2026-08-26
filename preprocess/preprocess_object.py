from __future__ import annotations

from abc import ABC, abstractmethod

from dataset.dataset_object import DatasetObject


class PreProcessObject(ABC):
    """Base class for a preprocessing step that transforms a dataset.

    Steps are chained by :class:`~experiments.Experiment` (a typical pipeline is
    ``balance -> encode -> scale -> split -> finalize``). A step receives a
    mutable :class:`~dataset.dataset_object.DatasetObject`, edits a
    :meth:`~dataset.dataset_object.DatasetObject.snapshot` of its dataframe, and
    writes the result back with
    :meth:`~dataset.dataset_object.DatasetObject.update`. A step that splits the
    data may return a tuple of datasets (e.g. train and test).
    """

    _seed: int | None = None

    @abstractmethod
    def __init__(self, seed: int | None = None, **kwargs):
        """Configure the step.

        Parameters
        ----------
        seed : int, optional
            Seed for any stochastic behaviour (e.g. shuffling, sampling).
        **kwargs
            Implementation-specific options.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(
        self, input: DatasetObject
    ) -> DatasetObject | tuple[DatasetObject, ...]:
        """Apply the transformation to a dataset.

        Parameters
        ----------
        input : DatasetObject
            A mutable dataset to transform in place (or clone).

        Returns
        -------
        DatasetObject or tuple of DatasetObject
            The transformed dataset, or several datasets when the step splits
            the input (e.g. into trainset and testset).
        """
        raise NotImplementedError
