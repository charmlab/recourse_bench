from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml


class DatasetObject(ABC):
    """Base class for a tabular dataset and its feature metadata.

    A dataset is created in a *mutable* state holding a raw, non-encoded,
    non-scaled dataframe (including the target column). Preprocessing steps
    mutate it through :meth:`snapshot`/:meth:`update`. Once :meth:`freeze` is
    called the object becomes *immutable* and the read interface
    (:meth:`get`, :meth:`ordered_features`, :meth:`__len__`,
    :meth:`__getitem__`) becomes available while the mutation interface is
    locked. This freeze/mutable state machine prevents recourse methods and
    evaluations from accidentally mutating finalized data.

    Attributes
    ----------
    target_column : str
        Name of the label column inside the dataframe.
    raw_feature_type : dict[str, str]
        Maps each feature name to its semantic type (e.g. ``"continuous"``,
        ``"categorical"``).
    raw_feature_mutability : dict[str, bool]
        Maps each feature name to whether it may be changed by a method.
    raw_feature_actionability : dict[str, str]
        Maps each feature name to its allowed direction of change.
    """

    _rawdf: pd.DataFrame
    _freeze: bool = False
    target_column: str
    raw_feature_type: dict[str, str]
    raw_feature_mutability: dict[str, bool]
    raw_feature_actionability: dict[str, str]

    @abstractmethod
    def __init__(self, path: str, **kwargs):
        """Load the raw dataframe and feature metadata from ``path``.

        Parameters
        ----------
        path : str
            Directory containing the offline data file and ``<name>.yaml``
            metadata for this dataset.
        **kwargs
            Implementation-specific options.
        """
        raise NotImplementedError

    @abstractmethod
    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError

    def _read_attrs(self, path: str) -> dict[str, object]:
        attrs_path = Path(path) / f"{Path(path).name}.yaml"
        with attrs_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _ensure_mutable(self) -> None:
        if self._freeze:
            raise RuntimeError("Dataset is frozen; snapshot()/update() are unavailable")

    def _ensure_frozen(self) -> None:
        if not self._freeze:
            raise RuntimeError(
                "Dataset is mutable; get()/ordered_features()/__len__()/__getitem__() are unavailable"
            )

    def snapshot(self) -> pd.DataFrame:
        """Return a deep copy of the current raw dataframe.

        Only available while the dataset is mutable. Preprocessing steps should
        edit this copy and write it back with :meth:`update`.

        Returns
        -------
        pandas.DataFrame
            A deep copy of the underlying dataframe.

        Raises
        ------
        RuntimeError
            If the dataset is frozen.
        """
        self._ensure_mutable()
        return self._rawdf.copy(deep=True)

    def update(self, flag: str, value: object, df: pd.DataFrame | None = None) -> bool:
        """Set an attribute (and optionally swap the dataframe) on a mutable dataset.

        Parameters
        ----------
        flag : str
            Name of the attribute to set. Commonly a boolean preprocessing flag
            (to guard against double application) or a metadata key.
        value : object
            Value to store; deep-copied before assignment.
        df : pandas.DataFrame, optional
            If provided, replaces the underlying dataframe (deep-copied).

        Returns
        -------
        bool
            ``True`` on success.

        Raises
        ------
        RuntimeError
            If the dataset is frozen.
        ValueError
            If ``flag`` is ``None``.
        """
        self._ensure_mutable()
        if flag is None:
            raise ValueError("flag must not be None")
        if df is not None:
            self._rawdf = df.copy(deep=True)
        setattr(self, flag, deepcopy(value))
        return True

    def attr(self, flag: str) -> object:
        """Return a deep copy of a public attribute (flag or metadata).

        Parameters
        ----------
        flag : str
            Public attribute name. Names starting with ``_`` are forbidden.

        Returns
        -------
        object
            Deep copy of the stored attribute value.

        Raises
        ------
        AttributeError
            If ``flag`` is protected or unknown.
        """
        if flag.startswith("_"):
            raise AttributeError(f"Access to protected member '{flag}' is forbidden")
        if not hasattr(self, flag):
            raise AttributeError(f"Unknown dataset attribute: {flag}")
        return deepcopy(getattr(self, flag))

    def freeze(self):
        """Lock the dataset, enabling the read interface and disabling mutation."""
        self._freeze = True

    def get(self, target: bool = False) -> pd.DataFrame:
        """Return the feature columns, or the target column.

        Only available once the dataset is frozen.

        Parameters
        ----------
        target : bool, default False
            If ``False`` return all feature columns (target excluded). If
            ``True`` return a single-column dataframe with the target.

        Returns
        -------
        pandas.DataFrame
            A deep copy of the requested columns.

        Raises
        ------
        RuntimeError
            If the dataset is not frozen.
        KeyError
            If the target column is missing.
        """
        self._ensure_frozen()
        target_column = self.target_column
        if target_column not in self._rawdf.columns:
            raise KeyError(f"Unknown target column: {target_column}")
        if target:
            return self._rawdf.loc[:, [target_column]].copy(deep=True)
        return self._rawdf.loc[:, self._rawdf.columns != target_column].copy(deep=True)

    def ordered_features(self) -> list[str]:
        """Return all column names (features and target) in dataframe order.

        Returns
        -------
        list[str]
            Column names. Only available once frozen.
        """
        self._ensure_frozen()
        return list(self._rawdf.columns)

    def __len__(self) -> int:
        """Return the number of rows (frozen datasets only)."""
        self._ensure_frozen()
        return int(self._rawdf.shape[0])

    def __getitem__(self, key) -> pd.DataFrame:
        """Index a frozen dataset by row (int/slice) or column name (str).

        Parameters
        ----------
        key : int or slice or str
            ``int``/``slice`` select rows; ``str`` selects a single column.

        Returns
        -------
        pandas.DataFrame
            A deep copy of the selected rows or column.
        """
        self._ensure_frozen()
        if isinstance(key, int):
            return self._rawdf.iloc[[key]].copy(deep=True)
        if isinstance(key, slice):
            return self._rawdf.iloc[key].copy(deep=True)
        if isinstance(key, str):
            if key not in self._rawdf.columns:
                raise KeyError(f"Unknown feature name: {key}")
            return self._rawdf.loc[:, [key]].copy(deep=True)
        raise TypeError(
            "DatasetObject only supports int/slice row indexing or str column indexing"
        )

    def clone(self) -> DatasetObject:
        """Return a deep, unfrozen copy of this dataset.

        The clone is always mutable regardless of this object's freeze state, so
        further preprocessing can be applied to it.

        Returns
        -------
        DatasetObject
            A deep copy with ``_freeze`` reset to ``False``.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__ = deepcopy(self.__dict__)
        clone._freeze = False
        return clone
