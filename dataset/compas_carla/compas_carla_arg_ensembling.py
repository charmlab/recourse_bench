from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("compas_carla_arg_ensembling")
class CompasCarlaArgEnsemblingDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/compas_carla/", **kwargs):
        dataset_path = Path(path)
        if not dataset_path.exists():
            dataset_path = Path(__file__).resolve().parent

        self._rawdf = self._read_df(str(dataset_path))
        self._freeze = False

        rawattrs = self._read_variant_attrs(dataset_path)
        for flag, value in rawattrs.items():
            setattr(self, flag, value)

        feature_order = getattr(self, "feature_order", list(self._rawdf.columns))
        self._rawdf = self._rawdf.loc[:, feature_order].copy(deep=True)

    def _read_df(self, path: str) -> pd.DataFrame:
        raw_df = pd.read_csv(Path(path) / "compas_carla.csv")
        transformed = pd.DataFrame(
            {
                "age": (
                    raw_df["age"] - raw_df["age"].min()
                ) / (raw_df["age"].max() - raw_df["age"].min()),
                "two_year_recid": raw_df["two_year_recid"].astype("float64"),
                "priors_count": (
                    raw_df["priors_count"] - raw_df["priors_count"].min()
                )
                / (raw_df["priors_count"].max() - raw_df["priors_count"].min()),
                "length_of_stay": (
                    raw_df["length_of_stay"] - raw_df["length_of_stay"].min()
                )
                / (raw_df["length_of_stay"].max() - raw_df["length_of_stay"].min()),
                "c_charge_degree_M": (
                    raw_df["c_charge_degree"] == "M"
                ).astype("float64"),
                "race_Other": (raw_df["race"] == "Other").astype("float64"),
                "sex_Male": (raw_df["sex"] == "Male").astype("float64"),
                "score": raw_df["score"].astype("float64"),
            }
        )
        return transformed

    def _read_variant_attrs(self, path: Path) -> dict[str, object]:
        attrs_path = path / "compas_carla_arg_ensembling.yaml"
        with attrs_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
