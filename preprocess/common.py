from __future__ import annotations

import math

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import (
    dataset_has_attr,
    ensure_flag_absent,
    resolve_feature_metadata,
)
from utils.registry import register
from utils.seed import seed_context


@register("encode")
class EncodePreProcess(PreProcessObject):
    @staticmethod
    def _format_category_suffix(category: object) -> str:
        if isinstance(category, (bool, np.bool_)):
            return str(int(category))
        if isinstance(category, (int, np.integer)):
            return str(int(category))
        if isinstance(category, (float, np.floating)) and float(category).is_integer():
            return str(int(category))
        return str(category)

    @staticmethod
    def _resolve_encoding(encoding: str | None) -> str | None:
        if encoding is None:
            return None
        if not isinstance(encoding, str):
            raise TypeError("encoding must be str or None")

        encoding = encoding.lower()
        if encoding not in {"none", "onehot", "thermometer", "mapping"}:
            raise ValueError(f"Unsupported encoding option: {encoding}")
        return encoding

    @staticmethod
    def _resolve_override(
        override: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if override is None:
            return None
        if not isinstance(override, dict):
            raise TypeError(
                "EncodePreProcess override must be a dict[str, str] or None"
            )

        resolved_override: dict[str, str] = {}
        for feature_name, mode in override.items():
            if not isinstance(feature_name, str):
                raise TypeError(
                    "EncodePreProcess override keys must be str feature names only"
                )
            if not isinstance(mode, str):
                raise TypeError(
                    "EncodePreProcess override values must be str encoding modes only"
                )
            resolved_mode = EncodePreProcess._resolve_encoding(mode)
            if resolved_mode is None:
                raise ValueError("EncodePreProcess override values must not be None")
            resolved_override[feature_name] = resolved_mode
        return resolved_override

    @staticmethod
    def _resolve_encode_modes(
        df: pd.DataFrame,
        target_column: str,
        raw_feature_type: dict[str, str],
        default_mode: str,
        override: dict[str, str] | None,
    ) -> dict[str, str]:
        categorical_columns = []
        for column in df.columns:
            if column == target_column:
                continue
            if raw_feature_type.get(column, "").lower() == "categorical":
                categorical_columns.append(column)

        resolved_modes = {column: default_mode for column in categorical_columns}
        if override is None:
            return resolved_modes

        for feature_name, mode in override.items():
            if feature_name == target_column:
                raise ValueError(
                    "EncodePreProcess override must not contain the target feature"
                )
            if feature_name not in df.columns:
                raise ValueError(
                    f"EncodePreProcess override contains unknown feature: {feature_name}"
                )
            if raw_feature_type.get(feature_name, "").lower() != "categorical":
                raise ValueError(
                    "EncodePreProcess override contains non-categorical feature: "
                    f"{feature_name}"
                )
            resolved_modes[feature_name] = mode
        return resolved_modes

    @staticmethod
    def _ensure_output_columns_available(
        new_columns: list[str],
        seen_columns: set[str],
        remaining_columns: set[str],
    ) -> None:
        for column in new_columns:
            if column in seen_columns:
                raise ValueError(
                    f"EncodePreProcess generated duplicated column: {column}"
                )
            if column in remaining_columns:
                raise ValueError(
                    "EncodePreProcess generated column collides with existing column: "
                    f"{column}"
                )

    @staticmethod
    def _build_onehot(series: pd.Series, feature_name: str) -> pd.DataFrame:
        categories = sorted(pd.Index(series.dropna().unique()).tolist())
        categorical = pd.Categorical(series, categories=categories, ordered=True)
        encoded = pd.get_dummies(categorical, dtype="float64")
        encoded.index = series.index
        encoded.columns = [
            f"{feature_name}_cat_{EncodePreProcess._format_category_suffix(category)}"
            for category in categories
        ]
        return encoded

    @staticmethod
    def _build_thermometer(series: pd.Series, feature_name: str) -> pd.DataFrame:
        categories = sorted(pd.Index(series.dropna().unique()).tolist())
        categorical = pd.Categorical(series, categories=categories, ordered=True)
        codes = pd.Series(categorical.codes, index=series.index)

        encoded_columns: dict[str, pd.Series] = {}
        for index, category in enumerate(categories):
            column_name = (
                f"{feature_name}_therm_"
                f"{EncodePreProcess._format_category_suffix(category)}"
            )
            encoded_columns[column_name] = (codes >= index).astype("float64")
        return pd.DataFrame(encoded_columns, index=series.index)

    @staticmethod
    def _build_mapping(
        series: pd.Series,
        feature_name: str,
    ) -> tuple[pd.DataFrame, dict[int, object]]:
        categories = sorted(pd.Index(series.dropna().unique()).tolist())
        categorical = pd.Categorical(series, categories=categories, ordered=True)
        codes = pd.Series(categorical.codes, index=series.index).astype("float64")
        value_mapping = {
            int(index): category for index, category in enumerate(categories)
        }
        return pd.DataFrame({feature_name: codes}, index=series.index), value_mapping

    @staticmethod
    def _build_encoded_metadata(
        dataset: DatasetObject,
        final_columns: list[str],
        encoded_sources: dict[str, str],
        encoded_feature_type_override: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
        raw_feature_type = dataset.attr("raw_feature_type")
        raw_feature_mutability = dataset.attr("raw_feature_mutability")
        raw_feature_actionability = dataset.attr("raw_feature_actionability")
        encoded_feature_type_override = encoded_feature_type_override or {}

        encoded_feature_type: dict[str, str] = {}
        encoded_feature_mutability: dict[str, bool] = {}
        encoded_feature_actionability: dict[str, str] = {}

        for column in final_columns:
            if column in encoded_feature_type_override:
                source_feature = encoded_sources.get(column, column)
                encoded_feature_type[column] = encoded_feature_type_override[column]
                encoded_feature_mutability[column] = bool(
                    raw_feature_mutability[source_feature]
                )
                encoded_feature_actionability[column] = str(
                    raw_feature_actionability[source_feature]
                )
                continue
            if column in encoded_sources:
                source_feature = encoded_sources[column]
                encoded_feature_type[column] = "binary"
                encoded_feature_mutability[column] = bool(
                    raw_feature_mutability[source_feature]
                )
                encoded_feature_actionability[column] = str(
                    raw_feature_actionability[source_feature]
                )
            else:
                encoded_feature_type[column] = str(raw_feature_type[column])
                encoded_feature_mutability[column] = bool(
                    raw_feature_mutability[column]
                )
                encoded_feature_actionability[column] = str(
                    raw_feature_actionability[column]
                )

        return (
            encoded_feature_type,
            encoded_feature_mutability,
            encoded_feature_actionability,
        )

    def __init__(
        self,
        seed: int | None = None,
        encoding: str | None = "onehot",
        override: dict[str, str] | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._encoding: str | None = self._resolve_encoding(encoding)
        self._override: dict[str, str] | None = self._resolve_override(override)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "encoding")

            if self._encoding is None:
                return input

            df = input.snapshot()
            target_column = input.target_column

            raw_feature_type = input.attr("raw_feature_type")
            selected_modes = self._resolve_encode_modes(
                df=df,
                target_column=target_column,
                raw_feature_type=raw_feature_type,
                default_mode=self._encoding,
                override=self._override,
            )

            final_parts: list[pd.DataFrame] = []
            encoded_sources: dict[str, str] = {}
            encoding_map: dict[str, list[str]] = {}
            encoded_value_mapping: dict[str, dict[int, object]] = {}
            encoded_feature_type_override: dict[str, str] = {}
            seen_columns: set[str] = set()
            remaining_columns = set(df.columns)
            for column in df.columns:
                remaining_columns.discard(column)

                if column == target_column:
                    passthrough = df.loc[:, [column]].copy(deep=True)
                    self._ensure_output_columns_available(
                        [column], seen_columns, remaining_columns
                    )
                    final_parts.append(passthrough)
                    seen_columns.add(column)
                elif column not in selected_modes:
                    passthrough = df.loc[:, [column]].copy(deep=True)
                    self._ensure_output_columns_available(
                        [column], seen_columns, remaining_columns
                    )
                    final_parts.append(passthrough)
                    seen_columns.add(column)
                else:
                    mode = selected_modes[column]
                    if mode == "onehot":
                        encoded = self._build_onehot(df[column], column)
                        new_columns = list(encoded.columns)
                        self._ensure_output_columns_available(
                            new_columns, seen_columns, remaining_columns
                        )
                        final_parts.append(encoded)
                        encoding_map[column] = new_columns
                        seen_columns.update(new_columns)
                        for encoded_column in new_columns:
                            encoded_sources[encoded_column] = column
                    elif mode == "thermometer":
                        encoded = self._build_thermometer(df[column], column)
                        new_columns = list(encoded.columns)
                        self._ensure_output_columns_available(
                            new_columns, seen_columns, remaining_columns
                        )
                        final_parts.append(encoded)
                        encoding_map[column] = new_columns
                        seen_columns.update(new_columns)
                        for encoded_column in new_columns:
                            encoded_sources[encoded_column] = column
                    elif mode == "mapping":
                        encoded, value_mapping = self._build_mapping(df[column], column)
                        new_columns = list(encoded.columns)
                        self._ensure_output_columns_available(
                            new_columns, seen_columns, remaining_columns
                        )
                        final_parts.append(encoded)
                        encoding_map[column] = new_columns
                        seen_columns.update(new_columns)
                        encoded_sources[column] = column
                        encoded_value_mapping[column] = value_mapping
                        encoded_feature_type_override[column] = "categorical"
                    elif mode == "none":
                        passthrough = df.loc[:, [column]].copy(deep=True)
                        self._ensure_output_columns_available(
                            [column], seen_columns, remaining_columns
                        )
                        final_parts.append(passthrough)
                        encoding_map[column] = [column]
                        seen_columns.add(column)
                    else:
                        raise ValueError(
                            f"Unsupported encoding option for feature {column}: {mode}"
                        )

            final_df = pd.concat(final_parts, axis=1)
            (
                encoded_feature_type,
                encoded_feature_mutability,
                encoded_feature_actionability,
            ) = self._build_encoded_metadata(
                dataset=input,
                final_columns=list(final_df.columns),
                encoded_sources=encoded_sources,
                encoded_feature_type_override=encoded_feature_type_override,
            )

            input.update("encoding", encoding_map, df=final_df)
            input.update("encoded_feature_type", encoded_feature_type)
            input.update("encoded_feature_mutability", encoded_feature_mutability)
            input.update("encoded_feature_actionability", encoded_feature_actionability)
            if encoded_value_mapping:
                input.update("encoded_value_mapping", encoded_value_mapping)
            return input


@register("scale")
class ScalePreProcess(PreProcessObject):
    @staticmethod
    def _resolve_scaling(scaling: str | None) -> str | None:
        if scaling is None:
            return None
        if not isinstance(scaling, str):
            raise TypeError("scaling must be str or None")

        scaling = scaling.lower()
        if scaling not in {"none", "standardize", "normalize"}:
            raise ValueError(f"Unsupported scaling option: {scaling}")
        return scaling

    @staticmethod
    def _resolve_override(
        override: dict[str, str] | None,
    ) -> dict[str, str] | None:
        if override is None:
            return None
        if not isinstance(override, dict):
            raise TypeError("ScalePreProcess override must be a dict[str, str] or None")

        resolved_override: dict[str, str] = {}
        for feature_name, mode in override.items():
            if not isinstance(feature_name, str):
                raise TypeError(
                    "ScalePreProcess override keys must be str feature names only"
                )
            if not isinstance(mode, str):
                raise TypeError(
                    "ScalePreProcess override values must be str scaling modes only"
                )
            resolved_mode = ScalePreProcess._resolve_scaling(mode)
            if resolved_mode is None:
                raise ValueError("ScalePreProcess override values must not be None")
            resolved_override[feature_name] = resolved_mode
        return resolved_override

    @staticmethod
    def _resolve_scale_modes(
        df: pd.DataFrame,
        target_column: str,
        feature_type: dict[str, str],
        default_mode: str,
        override: dict[str, str] | None,
    ) -> dict[str, str]:
        numerical_columns = []
        for column in df.columns:
            if column == target_column:
                continue
            if feature_type.get(column, "").lower() == "numerical":
                numerical_columns.append(column)

        resolved_modes = {column: default_mode for column in numerical_columns}
        if override is None:
            return resolved_modes

        for feature_name, mode in override.items():
            if feature_name == target_column:
                raise ValueError(
                    "ScalePreProcess override must not contain the target feature"
                )
            if feature_name not in df.columns:
                raise ValueError(
                    f"ScalePreProcess override contains unknown feature: {feature_name}"
                )
            if feature_type.get(feature_name, "").lower() != "numerical":
                raise ValueError(
                    "ScalePreProcess override contains non-numerical feature: "
                    f"{feature_name}"
                )
            resolved_modes[feature_name] = mode
        return resolved_modes

    @staticmethod
    def _compute_feature_ranges(
        df: pd.DataFrame, target_column: str, feature_type: dict[str, str]
    ) -> dict[str, float]:
        feature_ranges: dict[str, float] = {}
        for column in df.columns:
            if column == target_column:
                continue
            if feature_type.get(column, "").lower() == "numerical":
                feature_ranges[column] = float(df[column].max() - df[column].min())
        return feature_ranges

    @staticmethod
    def _compute_scaling_stats(
        df: pd.DataFrame,
        target_column: str,
        feature_type: dict[str, str],
        selected_modes: dict[str, str],
    ) -> dict[str, dict[str, float | str]]:
        scaling_stats: dict[str, dict[str, float | str]] = {}
        for column, mode in selected_modes.items():
            if column == target_column:
                continue
            if feature_type.get(column, "").lower() != "numerical":
                continue

            series = df[column].astype("float64")
            if mode == "standardize":
                scaling_stats[column] = {
                    "mode": mode,
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                }
            elif mode == "normalize":
                scaling_stats[column] = {
                    "mode": mode,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
            elif mode == "none":
                scaling_stats[column] = {"mode": mode}
            else:
                raise ValueError(
                    f"Unsupported scaling option for feature {column}: {mode}"
                )
        return scaling_stats

    def set_refset(self, refset: DatasetObject | None) -> None:
        self._refset = refset
        self._cached_scaling_stats = None

    @staticmethod
    def _resolve_ref_df(refset: DatasetObject) -> pd.DataFrame:
        if getattr(refset, "_freeze", False):
            return pd.concat(
                [refset.get(target=False), refset.get(target=True)], axis=1
            )
        return refset.snapshot()

    def __init__(
        self,
        seed: int | None = None,
        scaling: str | None = "standardize",
        override: dict[str, str] | None = None,
        range: bool = True,
        refset: DatasetObject | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._scaling: str | None = self._resolve_scaling(scaling)
        self._override: dict[str, str] | None = self._resolve_override(override)
        self._range: bool = range
        self._refset: DatasetObject | None = refset
        self._cached_scaling_stats: dict[str, dict[str, float | str]] | None = None

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "scaling")

            df = input.snapshot()
            target_column = input.target_column
            feature_type, _, _ = resolve_feature_metadata(input)
            fixed_scale_bounds = {}
            if dataset_has_attr(input, "scale_bounds"):
                candidate_bounds = input.attr("scale_bounds")
                if isinstance(candidate_bounds, dict):
                    fixed_scale_bounds = candidate_bounds

            if self._range:
                if dataset_has_attr(input, "range"):
                    pass
                else:
                    input.update(
                        "range",
                        self._compute_feature_ranges(df, target_column, feature_type),
                    )

            if self._scaling is None:
                return input

            selected_modes = self._resolve_scale_modes(
                df=df,
                target_column=target_column,
                feature_type=feature_type,
                default_mode=self._scaling,
                override=self._override,
            )

            scaling_stats: dict[str, dict[str, float | str]] | None = None
            if self._refset is not None:
                if self._cached_scaling_stats is None:
                    ref_df = self._resolve_ref_df(self._refset)
                    ref_target_column = self._refset.target_column
                    ref_feature_type, _, _ = resolve_feature_metadata(self._refset)
                    self._cached_scaling_stats = self._compute_scaling_stats(
                        df=ref_df,
                        target_column=ref_target_column,
                        feature_type=ref_feature_type,
                        selected_modes=selected_modes,
                    )
                scaling_stats = self._cached_scaling_stats

            scaled_df = df.copy(deep=True)
            for column, mode in selected_modes.items():
                series = scaled_df[column].astype("float64")

                if mode == "standardize":
                    if scaling_stats is None:
                        mean_value = float(series.mean())
                        std_value = float(series.std(ddof=0))
                    else:
                        mean_value = float(scaling_stats[column]["mean"])
                        std_value = float(scaling_stats[column]["std"])
                    if std_value == 0.0:
                        scaled_df[column] = 0.0
                    else:
                        scaled_df[column] = (series - mean_value) / std_value
                elif mode == "normalize":
                    if scaling_stats is None:
                        bounds = fixed_scale_bounds.get(column)
                        if bounds is not None:
                            if (
                                not isinstance(bounds, (list, tuple))
                                or len(bounds) != 2
                            ):
                                raise ValueError(
                                    "scale_bounds entries must be [min, max] pairs"
                                )
                            min_value = float(bounds[0])
                            max_value = float(bounds[1])
                        else:
                            min_value = float(series.min())
                            max_value = float(series.max())
                    else:
                        min_value = float(scaling_stats[column]["min"])
                        max_value = float(scaling_stats[column]["max"])
                    scale_value = max_value - min_value
                    if scale_value == 0.0:
                        scaled_df[column] = 0.0
                    else:
                        scaled_df[column] = (series - min_value) / scale_value
                elif mode == "none":
                    continue
                else:
                    raise ValueError(
                        f"Unsupported scaling option for feature {column}: {mode}"
                    )

            input.update("scaling", selected_modes, df=scaled_df)
            return input


@register("reorder")
class ReorderPreProcess(PreProcessObject):
    @staticmethod
    def _resolve_order(order: list[str] | None) -> list[str]:
        if order is None:
            raise ValueError("order must not be None")
        if not isinstance(order, list):
            raise TypeError("order must be a list[str]")
        if len(order) == 0:
            raise ValueError("order must not be empty")

        resolved_order: list[str] = []
        seen: set[str] = set()
        for feature_name in order:
            if not isinstance(feature_name, str):
                raise TypeError("order must contain str feature names only")
            if feature_name in seen:
                raise ValueError(f"order contains duplicated feature: {feature_name}")
            seen.add(feature_name)
            resolved_order.append(feature_name)
        return resolved_order

    def __init__(
        self,
        seed: int | None = None,
        order: list[str] | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._order: list[str] = self._resolve_order(order)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "reordered")

            df = input.snapshot()
            target_column = input.target_column

            feature_columns = [
                column for column in df.columns if column != target_column
            ]
            for feature_name in self._order:
                if feature_name == target_column:
                    raise ValueError(
                        "ReorderPreProcess order must not contain the target feature"
                    )
                if feature_name not in feature_columns:
                    raise ValueError(
                        f"ReorderPreProcess order contains unknown feature: {feature_name}"
                    )

            reordered_columns = list(self._order) + [target_column]
            reordered_df = df.loc[:, reordered_columns].copy(deep=True)

            input.update("reordered", list(self._order), df=reordered_df)
            return input


@register("split")
class SplitPreProcess(PreProcessObject):
    @staticmethod
    def _resolve_split(split: float | int) -> float | int:
        if isinstance(split, int):
            split = int(split)
            if split < 1:
                raise ValueError("integer split must be >= 1")
            return split
        if isinstance(split, float):
            split = float(split)
            if split <= 0 or split >= 1:
                raise ValueError("float split must satisfy 0 < split < 1")
            return split
        raise TypeError("split must be float or int")

    @staticmethod
    def _resolve_sample(sample: int | None) -> int | None:
        if sample is None:
            return None
        if isinstance(sample, int):
            sample = int(sample)
            if sample < 1:
                raise ValueError("sample must be >= 1 when provided")
            return sample
        raise TypeError("sample must be int or None")

    def __init__(
        self,
        seed: int | None = None,
        split: float | int = 0.2,
        sample: int | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._split: float | int = self._resolve_split(split)
        self._sample: int | None = self._resolve_sample(sample)

    def transform(self, input: DatasetObject) -> tuple[DatasetObject, DatasetObject]:
        with seed_context(self._seed):
            ensure_flag_absent(input, "trainset")
            ensure_flag_absent(input, "testset")

            df = input.snapshot()
            num_rows = int(df.shape[0])

            if isinstance(self._split, float):
                test_size = int(math.ceil(num_rows * self._split))
            elif isinstance(self._split, int):
                test_size = int(self._split)
            else:
                raise ValueError(f"Unsupported split option: {self._split}")

            if test_size < 1:
                raise ValueError("SplitPreProcess requires a test split size >= 1")
            if test_size >= num_rows:
                raise ValueError(
                    "SplitPreProcess requires at least one training sample"
                )

            shuffled_positions = np.random.permutation(num_rows)
            test_positions = shuffled_positions[:test_size]
            train_positions = shuffled_positions[test_size:]

            train_df = df.iloc[train_positions].copy(deep=True)
            test_df = df.iloc[test_positions].copy(deep=True)

            if self._sample is None:
                pass
            elif self._sample > test_df.shape[0]:
                raise ValueError("SplitPreProcess sample exceeds split testset size")
            else:
                sampled_positions = np.random.permutation(test_df.shape[0])[
                    : self._sample
                ]
                test_df = test_df.iloc[sampled_positions].copy(deep=True)

            testset = input.clone()
            trainset = input  # Reuse input

            trainset.update("trainset", True, df=train_df)
            testset.update("testset", True, df=test_df)

            return trainset, testset


@register("finalize")
class FinalizePreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, **kwargs):
        self._seed = seed

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            input.freeze()
            return input
