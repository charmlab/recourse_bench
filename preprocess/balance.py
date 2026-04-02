from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import dataset_has_attr, ensure_flag_absent
from utils.registry import register
from utils.seed import seed_context


@register("balance")
class BalancePreProcess(PreProcessObject):
    @staticmethod
    def _resolve_strategy(strategy: str) -> str:
        if not isinstance(strategy, str):
            raise TypeError("strategy must be str")
        strategy = strategy.lower()
        if strategy not in {"downsample", "upsample"}:
            raise ValueError(
                "BalancePreProcess supports strategy='downsample' or 'upsample' only"
            )
        return strategy

    @staticmethod
    def _resolve_round_to(round_to: int | None) -> int | None:
        if round_to is None:
            return None
        if not isinstance(round_to, int):
            raise TypeError("round_to must be int or None")
        if round_to < 1:
            raise ValueError("round_to must be >= 1 when provided")
        return round_to

    @staticmethod
    def _resolve_range(range: bool) -> bool:
        if not isinstance(range, bool):
            raise TypeError("range must be bool")
        return range

    @staticmethod
    def _compute_feature_ranges(
        df: pd.DataFrame, target_column: str
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        feature_min: dict[str, float] = {}
        feature_max: dict[str, float] = {}
        feature_range: dict[str, float] = {}
        for column in df.columns:
            if column == target_column:
                continue
            column_series = df[column]
            feature_min[column] = float(column_series.min())
            feature_max[column] = float(column_series.max())
            feature_range[column] = float(feature_max[column] - feature_min[column])
        return feature_min, feature_max, feature_range

    def __init__(
        self,
        seed: int | None = None,
        strategy: str = "downsample",
        round_to: int | None = None,
        shuffle: bool = True,
        range: bool = False,
        **kwargs,
    ):
        self._seed = seed
        self._strategy = self._resolve_strategy(strategy)
        self._round_to = self._resolve_round_to(round_to)
        self._shuffle = bool(shuffle)
        self._range = self._resolve_range(range)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            if getattr(input, "testset", False):
                return input

            ensure_flag_absent(input, "balanced")

            df = input.snapshot()
            target_column = input.target_column
            if target_column not in df.columns:
                raise KeyError(f"Unknown target column: {target_column}")

            target = df[target_column]
            class_counts = target.value_counts().sort_index()
            if class_counts.shape[0] < 2:
                raise ValueError(
                    "BalancePreProcess requires at least two target classes"
                )

            if self._strategy == "downsample":
                balanced_count_per_class = int(class_counts.min())
            else:
                balanced_count_per_class = int(class_counts.max())
            if self._round_to is not None:
                balanced_count_per_class = (
                    balanced_count_per_class // self._round_to * self._round_to
                )
            if balanced_count_per_class < 1:
                raise ValueError(
                    "BalancePreProcess requires balanced_count_per_class >= 1"
                )

            sampled_parts = []
            for class_value in class_counts.index.tolist():
                class_df = df.loc[target == class_value].copy(deep=True)
                sampled_parts.append(
                    class_df.sample(
                        n=balanced_count_per_class,
                        replace=self._strategy == "upsample",
                        random_state=self._seed,
                    )
                )

            balanced_df = pd.concat(sampled_parts, axis=0)
            if self._shuffle:
                balanced_df = balanced_df.sample(
                    frac=1.0, random_state=self._seed
                ).copy(deep=True)
            else:
                balanced_df = balanced_df.copy(deep=True)

            balanced_summary = {
                "strategy": self._strategy,
                "round_to": self._round_to,
                "shuffle": self._shuffle,
                "target_column": target_column,
                "applied": True,
                "original_counts": {
                    class_value: int(class_counts[class_value])
                    for class_value in class_counts.index.tolist()
                },
                "balanced_count_per_class": int(balanced_count_per_class),
                "balanced_counts": {
                    class_value: int(balanced_count_per_class)
                    for class_value in class_counts.index.tolist()
                },
            }

            if self._range:
                feature_min, feature_max, feature_range = self._compute_feature_ranges(
                    df=df,
                    target_column=target_column,
                )
                balanced_summary["feature_min"] = feature_min
                balanced_summary["feature_max"] = feature_max
                if not dataset_has_attr(input, "range"):
                    input.update("range", feature_range)

            input.update("balanced", balanced_summary, df=balanced_df)
            return input
