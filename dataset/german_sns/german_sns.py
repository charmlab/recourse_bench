from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("german_sns")
class GermanSnsDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/german_sns/", **kwargs):
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
        dataset_path = Path(path)
        if (dataset_path / "german_sns.csv").exists():
            return pd.read_csv(dataset_path / "german_sns.csv")
        return pd.read_csv(dataset_path / "german.csv")

    def _read_attrs(self, path: str) -> dict[str, object]:
        attrs_path = Path(path) / "german_sns.yaml"
        with attrs_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
