from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("news_popularity")
class NewsPopularityDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/news_popularity/", **kwargs):
        dataset_path = Path(path)
        if not dataset_path.exists():
            dataset_path = Path(__file__).resolve().parent

        self._rawdf = self._read_df(str(dataset_path))
        self._freeze = False

        rawattrs = self._read_attrs(str(dataset_path))
        for flag, value in rawattrs.items():
            setattr(self, flag, value)

        feature_order = getattr(self, "feature_order", list(self._rawdf.columns))
        self._rawdf = self._rawdf.loc[:, feature_order].copy(deep=True)

    def _read_df(self, path: str) -> pd.DataFrame:
        df_path = Path(path) / "news_popularity.csv"
        if not df_path.exists():
            raise FileNotFoundError(
                "Missing local news popularity dataset file: "
                f"{df_path}. Prepare the dataset locally before loading it."
            )
        return pd.read_csv(df_path)
