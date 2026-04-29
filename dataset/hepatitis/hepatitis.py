from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("hepatitis")
class HepatitisDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/hepatitis/", **kwargs):
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
        data = pd.read_csv(Path(path) / "HepatitisCdata.csv")
        data = data.drop(["Unnamed: 0", "Sex", "Age"], axis=1).copy(deep=True)

        category = data["Category"].astype(str)
        positive = category.isin(["0=Blood Donor", "0s=suspect Blood Donor"])
        negative = category.isin(["1=Hepatitis", "2=Fibrosis", "3=Cirrhosis"])
        data.loc[negative, "Category"] = 0
        data.loc[positive, "Category"] = 1
        data["Category"] = data["Category"].astype("int64")

        for column in data.columns.tolist()[1:]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
            data[column] = data[column].fillna(data[column].mean())

        return data.reset_index(drop=True)
