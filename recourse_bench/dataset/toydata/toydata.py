from __future__ import annotations

from pathlib import Path

import pandas as pd

from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.utils.registry import register


@register("toy_data")
class ToydataDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/toydata/", **kwargs):
        dataset_path = Path(path)
        if not dataset_path.exists():
            dataset_path = Path(__file__).resolve().parent

        self._rawdf = self._read_df(str(dataset_path))
        self._freeze = False

        for column in self._rawdf.columns:
            if not self._rawdf[column].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(self._rawdf[column]):
                self._rawdf[column] = self._rawdf[column].fillna(
                    self._rawdf[column].median()
                )
            else:
                self._rawdf[column] = self._rawdf[column].fillna(
                    self._rawdf[column].mode().iloc[0]
                )

        rawattrs = self._read_attrs(str(dataset_path))
        for flag, value in rawattrs.items():
            setattr(self, flag, value)

        feature_order = getattr(self, "feature_order", list(self._rawdf.columns))
        self._rawdf = self._rawdf.loc[:, feature_order].copy(deep=True)

    def _read_df(self, path: str) -> pd.DataFrame:
        df_path = Path(path) / "toydata.csv"
        return pd.read_csv(df_path)
