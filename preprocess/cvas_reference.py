from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from dataset.dataset_object import DatasetObject
from preprocess.common import EncodePreProcess, ScalePreProcess, SplitPreProcess
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import ensure_flag_absent
from utils.registry import get_registry
from utils.registry import register
from utils.seed import seed_context


@register("cvas_german_reference")
class CvasGermanReferencePreProcess(PreProcessObject):
    _PAIR_DATASET_NAME = {
        "german": "german_roar",
        "german_roar": "german",
    }

    _PERSONAL_STATUS_MAP = {
        "german": {
            1: "A91",
            2: "A92",
            3: "A94",
            5: "A93",
        },
        "german_roar": {
            1: "A91",
            2: "A92",
            3: "A93",
            4: "A94",
        },
    }

    def __init__(
        self,
        seed: int | None = None,
        paired_dataset_name: str | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._paired_dataset_name = paired_dataset_name

    def _resolve_dataset_name(self, input: DatasetObject) -> str:
        dataset_name = str(input.attr("name")).lower()
        if dataset_name not in self._PAIR_DATASET_NAME:
            raise ValueError(
                "CvasGermanReferencePreProcess supports ['german', 'german_roar'] only"
            )
        return dataset_name

    def _load_paired_snapshot(self, paired_dataset_name: str) -> pd.DataFrame:
        dataset_registry = get_registry("Dataset")
        if paired_dataset_name not in dataset_registry:
            raise ValueError(f"Unknown paired dataset: {paired_dataset_name}")
        paired_dataset = dataset_registry[paired_dataset_name]()
        return paired_dataset.snapshot()

    def _normalize_personal_status(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        target_column: str,
    ) -> pd.DataFrame:
        if "personal_status_sex" not in df.columns or target_column == "personal_status_sex":
            return df.copy(deep=True)

        mapping = self._PERSONAL_STATUS_MAP[dataset_name]
        series = df["personal_status_sex"]
        if series.dropna().map(lambda value: isinstance(value, str)).all():
            return df.copy(deep=True)

        unique_values = set(pd.Index(series.dropna().unique()).tolist())
        missing_values = sorted(unique_values.difference(mapping))
        if missing_values:
            raise ValueError(
                "CvasGermanReferencePreProcess found unmapped personal_status_sex "
                f"values for dataset '{dataset_name}': {missing_values}"
            )

        normalized = df.copy(deep=True)
        normalized["personal_status_sex"] = series.map(mapping)
        return normalized

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            for flag in (
                "range",
                "scaling",
                "encoding",
                "encoded_feature_type",
                "encoded_feature_mutability",
                "encoded_feature_actionability",
                "cvas_reference",
            ):
                ensure_flag_absent(input, flag)

            dataset_name = self._resolve_dataset_name(input)
            paired_dataset_name = (
                self._paired_dataset_name or self._PAIR_DATASET_NAME[dataset_name]
            ).lower()
            target_column = input.target_column

            raw_df = self._normalize_personal_status(
                input.snapshot(),
                dataset_name=dataset_name,
                target_column=target_column,
            )
            paired_df = self._normalize_personal_status(
                self._load_paired_snapshot(paired_dataset_name),
                dataset_name=paired_dataset_name,
                target_column=target_column,
            )
            joint_df = pd.concat([raw_df, paired_df], axis=0, ignore_index=True)

            raw_feature_type = input.attr("raw_feature_type")
            input.update(
                "range",
                ScalePreProcess._compute_feature_ranges(
                    raw_df,
                    target_column=target_column,
                    feature_type=raw_feature_type,
                ),
            )

            scaling_map: dict[str, str] = {}
            encoding_map: dict[str, list[str]] = {}
            encoded_sources: dict[str, str] = {}
            final_parts: list[pd.DataFrame] = []

            for column in raw_df.columns:
                if column == target_column:
                    continue

                feature_type = str(raw_feature_type.get(column, "")).lower()
                if feature_type == "numerical":
                    scaling_map[column] = "standardize"
                    series = raw_df[column].astype("float64")
                    joint_series = joint_df[column].astype("float64")
                    mean_value = float(joint_series.mean())
                    std_value = float(joint_series.std(ddof=0))
                    if std_value == 0.0:
                        scaled_series = pd.Series(0.0, index=series.index, name=column)
                    else:
                        scaled_series = (series - mean_value) / std_value
                    final_parts.append(scaled_series.to_frame(name=column))
                    encoding_map[column] = [column]
                    continue

                if feature_type == "categorical":
                    categories = sorted(pd.Index(joint_df[column].dropna().unique()).tolist())
                    categorical = pd.Categorical(
                        raw_df[column],
                        categories=categories,
                        ordered=True,
                    )
                    encoded = pd.get_dummies(categorical, dtype="float64")
                    encoded.index = raw_df.index
                    encoded.columns = [
                        (
                            f"{column}_cat_"
                            f"{EncodePreProcess._format_category_suffix(category)}"
                        )
                        for category in categories
                    ]
                    final_parts.append(encoded)
                    encoding_map[column] = list(encoded.columns)
                    for encoded_column in encoded.columns:
                        encoded_sources[encoded_column] = column
                    continue

                passthrough = raw_df.loc[:, [column]].copy(deep=True)
                final_parts.append(passthrough)
                encoding_map[column] = [column]

            final_parts.append(raw_df.loc[:, [target_column]].copy(deep=True))
            final_df = pd.concat(final_parts, axis=1)

            (
                encoded_feature_type,
                encoded_feature_mutability,
                encoded_feature_actionability,
            ) = EncodePreProcess._build_encoded_metadata(
                dataset=input,
                final_columns=list(final_df.columns),
                encoded_sources=encoded_sources,
            )

            input.update("scaling", scaling_map)
            input.update("encoding", encoding_map, df=final_df)
            input.update("encoded_feature_type", encoded_feature_type)
            input.update("encoded_feature_mutability", encoded_feature_mutability)
            input.update("encoded_feature_actionability", encoded_feature_actionability)
            input.update("cvas_reference", True)
            return input


@register("stratified_split")
class StratifiedSplitPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        split: float | int = 0.2,
        sample: int | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._split: float | int = SplitPreProcess._resolve_split(split)
        self._sample: int | None = SplitPreProcess._resolve_sample(sample)

    def transform(self, input: DatasetObject) -> tuple[DatasetObject, DatasetObject]:
        with seed_context(self._seed):
            ensure_flag_absent(input, "trainset")
            ensure_flag_absent(input, "testset")

            df = input.snapshot()
            target = df.loc[:, input.target_column]
            train_df, test_df = train_test_split(
                df,
                test_size=self._split,
                random_state=self._seed,
                stratify=target,
                shuffle=True,
            )
            train_df = train_df.copy(deep=True)
            test_df = test_df.copy(deep=True)

            if self._sample is None:
                pass
            elif self._sample > test_df.shape[0]:
                raise ValueError(
                    "StratifiedSplitPreProcess sample exceeds split testset size"
                )
            else:
                test_df = test_df.sample(
                    n=self._sample,
                    random_state=self._seed,
                ).copy(deep=True)

            testset = input.clone()
            trainset = input
            trainset.update("trainset", True, df=train_df)
            testset.update("testset", True, df=test_df)
            return trainset, testset
