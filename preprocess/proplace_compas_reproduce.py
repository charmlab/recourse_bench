from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import ensure_flag_absent
from utils.registry import register
from utils.seed import seed_context


@register("proplace_compas_reproduce")
class ProplaceCompasReproducePreProcess(PreProcessObject):
    """Reference-faithful COMPAS preprocessing for the ProPlace reproduction only.

    The reference notebook uses CARLA's ``OneHot_drop_binary`` COMPAS view, which yields
    seven numeric non-target features. This preprocess rebuilds that representation from
    the raw CSV directly so the experiment is not coupled to the shared dataset class's
    load-time shuffle or the benchmark's default one-hot encoding.
    """

    def __init__(
        self,
        seed: int | None = None,
        source_path: str = "./dataset/compas_carla/compas_carla.csv",
        target_column: str = "score",
        numerical_features: list[str] | None = None,
        binary_positive_category: dict[str, str] | None = None,
        dropna: bool = True,
        **kwargs,
    ):
        self._seed = seed
        self._source_path = Path(source_path)
        self._target_column = str(target_column)
        self._numerical_features = list(
            numerical_features
            or ["age", "two_year_recid", "priors_count", "length_of_stay"]
        )
        self._binary_positive_category = dict(
            binary_positive_category
            or {
                # CARLA OnlineCatalog defaults to OneHot_drop_binary.
                "c_charge_degree": "M",
                "race": "Other",
                "sex": "Male",
            }
        )
        self._dropna = bool(dropna)

    def _load_source_frame(self) -> pd.DataFrame:
        source_path = self._source_path
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[1] / source_path
        df = pd.read_csv(source_path)
        if self._dropna:
            df = df.dropna().reset_index(drop=True)
        return df

    def _normalize_feature(self, series: pd.Series) -> tuple[pd.Series, tuple[float, float]]:
        minimum = float(series.min())
        maximum = float(series.max())
        if maximum == minimum:
            normalized = pd.Series(0.0, index=series.index, dtype="float64")
        else:
            normalized = (series.astype("float64") - minimum) / (maximum - minimum)
        return normalized.astype("float64"), (minimum, maximum)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "proplace_compas_reproduce")

            raw_df = self._load_source_frame()
            target_column = self._target_column
            if target_column not in raw_df.columns:
                raise KeyError(f"Missing target column in raw COMPAS CSV: {target_column}")

            feature_columns = [
                *self._numerical_features,
                *self._binary_positive_category.keys(),
            ]
            missing_columns = [
                column
                for column in [*feature_columns, target_column]
                if column not in raw_df.columns
            ]
            if missing_columns:
                raise KeyError(
                    f"COMPAS CSV is missing required reproduction columns: {missing_columns}"
                )

            processed = pd.DataFrame(index=raw_df.index)
            range_map: dict[str, tuple[float, float]] = {}
            scaling_map: dict[str, str] = {}
            for column in self._numerical_features:
                processed[column], range_map[column] = self._normalize_feature(
                    raw_df[column]
                )
                scaling_map[column] = "normalize"

            for column, positive_category in self._binary_positive_category.items():
                categories = sorted(pd.Index(raw_df[column].dropna().unique()).tolist())
                if positive_category not in categories:
                    raise ValueError(
                        f"Configured positive category '{positive_category}' is invalid for "
                        f"{column}; available categories: {categories}"
                    )
                processed[column] = (
                    raw_df[column].astype(str) == str(positive_category)
                ).astype("float64")

            processed[target_column] = (
                pd.Series(raw_df[target_column], index=raw_df.index)
                .astype("float64")
                .round()
                .astype("int64")
            )

            raw_feature_mutability = input.attr("raw_feature_mutability")
            raw_feature_actionability = input.attr("raw_feature_actionability")
            raw_feature_type = input.attr("raw_feature_type")

            encoded_feature_type: dict[str, str] = {}
            encoded_feature_mutability: dict[str, bool] = {}
            encoded_feature_actionability: dict[str, str] = {}

            for column in processed.columns:
                source_column = column
                if column == target_column:
                    encoded_feature_type[column] = str(raw_feature_type[target_column])
                    encoded_feature_mutability[column] = bool(
                        raw_feature_mutability[target_column]
                    )
                    encoded_feature_actionability[column] = str(
                        raw_feature_actionability[target_column]
                    )
                    continue

                if column in self._numerical_features:
                    encoded_feature_type[column] = "numerical"
                else:
                    encoded_feature_type[column] = "binary"
                encoded_feature_mutability[column] = bool(
                    raw_feature_mutability[source_column]
                )
                encoded_feature_actionability[column] = str(
                    raw_feature_actionability[source_column]
                )

            final_columns = [*feature_columns, target_column]
            processed = processed.loc[:, final_columns].copy(deep=True)

            input.update(
                "proplace_compas_reproduce",
                {
                    "source_path": str(self._source_path),
                    "numerical_features": list(self._numerical_features),
                    "binary_positive_category": dict(self._binary_positive_category),
                },
                df=processed,
            )
            input.update(
                "range",
                {name: list(bounds) for name, bounds in range_map.items()},
            )
            input.update("scaling", scaling_map)
            input.update(
                "encoding",
                {name: [name] for name in self._binary_positive_category},
            )
            input.update("encoded_feature_type", encoded_feature_type)
            input.update("encoded_feature_mutability", encoded_feature_mutability)
            input.update("encoded_feature_actionability", encoded_feature_actionability)
            input.update("feature_order", final_columns)
            return input
