from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from dataset.dataset_object import DatasetObject


class EvaluationObject(ABC):
    """Base class for a metric comparing factual and counterfactual datasets.

    Each evaluation consumes the finalized factual dataset and the
    counterfactual dataset returned by
    :meth:`~method.method_object.MethodObject.predict`, and returns a one-row
    dataframe of named metrics. :class:`~experiments.Experiment` concatenates
    these one-row frames column-wise into the final metrics table.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Configure the metric.

        Parameters
        ----------
        **kwargs
            Implementation-specific options (e.g. a reference set or norm).
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        """Compute the metric.

        Parameters
        ----------
        factuals : DatasetObject
            The finalized factual dataset.
        counterfactuals : DatasetObject
            The counterfactual dataset. If it carries an ``evaluation_filter``,
            apply it before computing the metric.

        Returns
        -------
        pandas.DataFrame
            A single-row dataframe with stable, descriptive column names.
        """
        raise NotImplementedError
