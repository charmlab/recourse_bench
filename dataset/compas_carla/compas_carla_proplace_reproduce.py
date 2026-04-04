from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("compas_carla_proplace_reproduce")
class CompasCarlaProplaceReproduceDataset(DatasetObject):
    """Reference-faithful COMPAS loader for the ProPlace reproduction.

    This loader reads the same CSV and metadata as ``compas_carla`` but preserves the raw
    row order after dropping NaNs. The ProPlace Compas notebook relies on that order before
    its own seeded half-split, so the standard shuffled dataset is not faithful enough.
    """

    def __init__(self, path: str = "./dataset/compas_carla/", **kwargs):
        dataset_path = Path(path)
        if not dataset_path.exists():
            dataset_path = Path(__file__).resolve().parent

        self._rawdf = self._read_df(str(dataset_path))
        self._freeze = False

        rawattrs = self._read_attrs(str(dataset_path))
        for flag, value in rawattrs.items():
            setattr(self, flag, value)
        self.name = "compas_carla_proplace_reproduce"

        feature_order = getattr(self, "feature_order", list(self._rawdf.columns))
        self._rawdf = self._rawdf.loc[:, feature_order].copy(deep=True)

    def _read_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(Path(path) / "compas_carla.csv")
        return df.dropna().reset_index(drop=True)
