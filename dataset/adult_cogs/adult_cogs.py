from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("adult_cogs")
class AdultCoGSDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/adult_cogs/", **kwargs):
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
        df = pd.read_csv(
            Path(path) / "adult_cogs.csv",
            delimiter=",",
            skipinitialspace=True,
        )
        df = df.replace("?", pd.NA)
        df = df.dropna(axis=0).copy(deep=True)
        df = df.drop(columns=["fnlwgt", "education"]).copy(deep=True)
        df["income"] = df["income"].astype("int64")
        return df.reset_index(drop=True)
