from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import dataset_has_attr, ensure_flag_absent
from utils.registry import register
from utils.seed import seed_context


@register("rbr_german_aggregate")
class RbrGermanAggregatePreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        feature_name: str = "personal_status_sex",
        **kwargs,
    ):
        self._seed = seed
        self._feature_name = str(feature_name)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "rbr_german_aggregated")
            if dataset_has_attr(input, "encoding"):
                raise ValueError(
                    "RbrGermanAggregatePreProcess must run before EncodePreProcess"
                )

            df = input.snapshot()
            if self._feature_name not in df.columns:
                raise ValueError(
                    f"RbrGermanAggregatePreProcess requires feature '{self._feature_name}'"
                )

            series = df[self._feature_name].astype("string").str.strip()
            allowed_values = {"1", "2", "3", "4", "5", "4_5"}
            observed_values = set(series.dropna().tolist())
            unexpected_values = sorted(observed_values - allowed_values)
            if unexpected_values:
                raise ValueError(
                    "RbrGermanAggregatePreProcess encountered unexpected "
                    f"{self._feature_name} values: {unexpected_values}"
                )

            df[self._feature_name] = series.replace({"4": "4_5", "5": "4_5"})
            input.update("rbr_german_aggregated", True, df=df)
            return input
