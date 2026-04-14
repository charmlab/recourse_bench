from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.mace.library.mace import generateExplanations
from method.mace.support import ensure_supported_target_model, validate_counterfactuals
from method.method_object import MethodObject
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from utils.caching import get_cache_dir
from utils.registry import register
from utils.seed import seed_context


def _dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    else:
        return True


def _is_integer_valued(series: pd.Series) -> bool:
    values = series.dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return False
    return bool(np.allclose(values, np.round(values)))


def _normalize_actionability(value: object) -> str:
    normalized = str(value).lower()
    if normalized == "same":
        return "none"
    if normalized not in {"none", "any", "same-or-increase", "same-or-decrease"}:
        raise ValueError(f"Unsupported MACE actionability: {value}")
    return normalized


def _format_epsilon_string(epsilon: float) -> str:
    mantissa, exponent = f"{epsilon:.0e}".split("e")
    formatted = f"{mantissa}e{int(exponent)}"
    if len(formatted) != 4:
        raise ValueError("MaceMethod epsilon must format like 1e-1, 1e-3, or 1e-5 only")
    return formatted


@dataclass
class _MaceAttribute:
    attr_name_kurz: str
    attr_type: str
    lower_bound: float
    upper_bound: float
    mutability: bool
    actionability: str


class _MaceDatasetWrapper:
    def __init__(
        self,
        dataset_name: str,
        feature_names: list[str],
        feature_types: dict[str, str],
        bounds: dict[str, tuple[float, float]],
        mutability: dict[str, bool],
        actionability: dict[str, str],
    ):
        self.dataset_name = str(dataset_name)
        self.is_one_hot = False
        self._feature_names = list(feature_names)
        self._feature_types = dict(feature_types)
        self._short_names = {
            feature: f"x{i}" for i, feature in enumerate(feature_names)
        }
        self._inverse_short_names = {
            short_name: feature_name
            for feature_name, short_name in self._short_names.items()
        }
        self.attributes_kurz: dict[str, _MaceAttribute] = {}

        for feature_name in feature_names:
            short_name = self._short_names[feature_name]
            lower_bound, upper_bound = bounds[feature_name]
            self.attributes_kurz[short_name] = _MaceAttribute(
                attr_name_kurz=short_name,
                attr_type=str(feature_types[feature_name]),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                mutability=bool(mutability[feature_name]),
                actionability=str(actionability[feature_name]).lower(),
            )
        self.attributes_kurz["y"] = _MaceAttribute(
            attr_name_kurz="y",
            attr_type="binary",
            lower_bound=0.0,
            upper_bound=1.0,
            mutability=False,
            actionability="none",
        )

    def getInputAttributeNames(self, kind: str = "kurz"):
        return [self._short_names[feature_name] for feature_name in self._feature_names]

    def getOutputAttributeNames(self, kind: str = "kurz"):
        return ["y"]

    def getInputOutputAttributeNames(self, kind: str = "kurz"):
        return self.getInputAttributeNames(kind) + self.getOutputAttributeNames(kind)

    def getMutableAttributeNames(self, kind: str = "kurz"):
        return [
            self._short_names[feature_name]
            for feature_name in self._feature_names
            if self.attributes_kurz[self._short_names[feature_name]].mutability
        ]

    def getOneHotAttributesNames(self, kind: str = "kurz"):
        return []

    def getNonHotAttributesNames(self, kind: str = "kurz"):
        return self.getInputAttributeNames(kind)

    def getSiblingsFor(self, attr_name_kurz: str):
        return [attr_name_kurz]

    def getDictOfSiblings(self, kind: str = "kurz"):
        return {"cat": {}, "ord": {}}

    def factual_to_short_dict(
        self,
        factual: pd.Series,
        predicted_label: int,
    ) -> dict[str, int | float | bool]:
        output: dict[str, int | float | bool] = {}
        for feature_name in self._feature_names:
            value = factual[feature_name]
            feature_type = self._feature_types[feature_name]
            if feature_type == "numeric-real":
                output[self._short_names[feature_name]] = float(value)
            else:
                output[self._short_names[feature_name]] = int(round(float(value)))
        output["y"] = bool(predicted_label)
        return output

    def short_dict_to_feature_row(self, sample: dict[str, object]) -> pd.Series:
        row = {}
        for short_name, feature_name in self._inverse_short_names.items():
            value = sample.get(short_name, np.nan)
            if value is None:
                row[feature_name] = np.nan
            else:
                feature_type = self._feature_types[feature_name]
                if feature_type == "numeric-real":
                    row[feature_name] = float(value)
                else:
                    row[feature_name] = int(round(float(value)))
        return pd.Series(row, index=self._feature_names)


@register("mace")
class MaceMethod(MethodObject):
    @staticmethod
    def _resolve_norm_type(norm_type: str | list[str] | None) -> list[str]:
        if norm_type is None:
            return ["zero_norm"]
        if isinstance(norm_type, str):
            resolved = [norm_type.lower()]
        elif isinstance(norm_type, list) and norm_type:
            resolved = [str(value).lower() for value in norm_type]
        else:
            raise TypeError("norm_type must be str, non-empty list[str], or None")

        valid = {"zero_norm", "one_norm", "infty_norm"}
        invalid = [value for value in resolved if value not in valid]
        if invalid:
            raise ValueError(f"Unsupported MACE norm_type values: {invalid}")
        return resolved

    @staticmethod
    def _resolve_epsilon(epsilon: float) -> float:
        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        _format_epsilon_string(epsilon)
        return epsilon

    @staticmethod
    def _resolve_feature_metadata(
        dataset: DatasetObject,
        feature_df: pd.DataFrame,
    ) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
        has_mace_type = _dataset_has_attr(dataset, "mace_feature_type")
        has_mace_mutability = _dataset_has_attr(dataset, "mace_feature_mutability")
        has_mace_actionability = _dataset_has_attr(
            dataset, "mace_feature_actionability"
        )
        has_any_mace_metadata = (
            has_mace_type or has_mace_mutability or has_mace_actionability
        )
        has_all_mace_metadata = (
            has_mace_type and has_mace_mutability and has_mace_actionability
        )
        if has_any_mace_metadata and not has_all_mace_metadata:
            raise ValueError(
                "MACE metadata must define mace_feature_type, "
                "mace_feature_mutability, and mace_feature_actionability together"
            )

        if has_all_mace_metadata:
            feature_type_raw = dataset.attr("mace_feature_type")
            feature_mutability_raw = dataset.attr("mace_feature_mutability")
            feature_actionability_raw = dataset.attr("mace_feature_actionability")
            feature_type = {
                feature_name: str(feature_type_raw[feature_name]).lower()
                for feature_name in feature_df.columns
            }
        else:
            raw_feature_type = dataset.attr("raw_feature_type")
            feature_mutability_raw = dataset.attr("raw_feature_mutability")
            feature_actionability_raw = dataset.attr("raw_feature_actionability")
            feature_type = {}
            for feature_name in feature_df.columns:
                raw_type = str(raw_feature_type[feature_name]).lower()
                if raw_type == "numerical":
                    if _is_integer_valued(feature_df[feature_name]):
                        feature_type[feature_name] = "numeric-int"
                    else:
                        feature_type[feature_name] = "numeric-real"
                elif raw_type in {"binary", "categorical"}:
                    feature_type[feature_name] = raw_type
                else:
                    raise ValueError(
                        f"Unsupported raw_feature_type for MACE fallback: {raw_type}"
                    )

        valid_feature_types = {
            "binary",
            "categorical",
            "ordinal",
            "numeric-int",
            "numeric-real",
        }
        invalid_feature_types = [
            feature_name
            for feature_name, feature_type_value in feature_type.items()
            if feature_type_value not in valid_feature_types
        ]
        if invalid_feature_types:
            raise ValueError(
                "Unsupported MACE feature types for features: "
                f"{invalid_feature_types}"
            )

        feature_mutability = {
            feature_name: bool(feature_mutability_raw[feature_name])
            for feature_name in feature_df.columns
        }
        feature_actionability = {
            feature_name: _normalize_actionability(
                feature_actionability_raw[feature_name]
            )
            for feature_name in feature_df.columns
        }
        return feature_type, feature_mutability, feature_actionability

    @staticmethod
    def _resolve_bounds(
        dataset: DatasetObject,
        feature_df: pd.DataFrame,
    ) -> dict[str, tuple[float, float]]:
        feature_min: dict[str, float] | None = None
        feature_max: dict[str, float] | None = None
        if _dataset_has_attr(dataset, "balanced"):
            balanced = dataset.attr("balanced")
            if isinstance(balanced, dict):
                raw_feature_min = balanced.get("feature_min")
                raw_feature_max = balanced.get("feature_max")
                if isinstance(raw_feature_min, dict) and isinstance(
                    raw_feature_max, dict
                ):
                    feature_min = {
                        feature_name: float(raw_feature_min[feature_name])
                        for feature_name in feature_df.columns
                    }
                    feature_max = {
                        feature_name: float(raw_feature_max[feature_name])
                        for feature_name in feature_df.columns
                    }

        bounds: dict[str, tuple[float, float]] = {}
        for feature_name in feature_df.columns:
            if feature_min is not None and feature_max is not None:
                bounds[feature_name] = (
                    float(feature_min[feature_name]),
                    float(feature_max[feature_name]),
                )
            else:
                bounds[feature_name] = (
                    float(feature_df[feature_name].min()),
                    float(feature_df[feature_name].max()),
                )
        return bounds

    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        norm_type: str | list[str] | None = None,
        epsilon: float = 1e-5,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, (RandomForestModel,), "MaceMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._norm_type = self._resolve_norm_type(norm_type)
        self._epsilon = self._resolve_epsilon(epsilon)
        self._approach_string = f"MACE_eps_{_format_epsilon_string(self._epsilon)}"

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for MaceMethod.fit()")
        if _dataset_has_attr(trainset, "encoding"):
            raise ValueError("MaceMethod does not support encoded datasets")
        if _dataset_has_attr(trainset, "scaling"):
            raise ValueError("MaceMethod does not support scaled datasets")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            feature_df = trainset.get(target=False)
            feature_type, mutability, actionability = self._resolve_feature_metadata(
                trainset,
                feature_df,
            )
            bounds = self._resolve_bounds(trainset, feature_df)
            dataset_name = (
                str(trainset.attr("name"))
                if _dataset_has_attr(trainset, "name")
                else ""
            )

            if self._desired_class is not None:
                class_to_index = self._target_model.get_class_to_index()
                if len(class_to_index) != 2:
                    raise ValueError(
                        "MaceMethod desired_class is supported for binary classification only"
                    )
                if self._desired_class not in class_to_index:
                    raise ValueError(
                        "desired_class is invalid for the trained target model"
                    )
                self._desired_index = int(class_to_index[self._desired_class])
            else:
                self._desired_index = None

            self._dataset_wrapper = _MaceDatasetWrapper(
                dataset_name=dataset_name,
                feature_names=self._feature_names,
                feature_types=feature_type,
                bounds=bounds,
                mutability=mutability,
                actionability=actionability,
            )
            self._explanation_dir = Path(get_cache_dir("mace")) / "__explanation_log"
            self._explanation_dir.mkdir(parents=True, exist_ok=True)
            self._is_trained = True

    def _predict_label(self, factuals: pd.DataFrame) -> np.ndarray:
        prediction = self._target_model.get_prediction(factuals, proba=False)
        return prediction.detach().cpu().numpy().argmax(axis=1)

    def get_counterfactuals(self, factuals: pd.DataFrame):
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("MaceMethod factuals must not contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        predicted_labels = self._predict_label(factuals)
        candidates = pd.DataFrame(
            np.nan,
            index=factuals.index,
            columns=self._feature_names,
            dtype="float64",
        )

        already_desired_mask = np.zeros(factuals.shape[0], dtype=bool)
        if self._desired_index is not None:
            already_desired_mask = predicted_labels == self._desired_index
            if already_desired_mask.any():
                candidates.loc[
                    factuals.index[already_desired_mask], self._feature_names
                ] = factuals.loc[
                    factuals.index[already_desired_mask], self._feature_names
                ].to_numpy(
                    dtype="float64"
                )

        solve_indices = [
            row_index
            for row_index in range(factuals.shape[0])
            if not already_desired_mask[row_index]
        ]

        with seed_context(self._seed):
            for row_index in tqdm(
                solve_indices,
                desc="mace-generate",
                leave=False,
            ):
                row = factuals.iloc[row_index]
                factual_sample = self._dataset_wrapper.factual_to_short_dict(
                    row, int(predicted_labels[row_index])
                )
                explanation_file_name = str(
                    self._explanation_dir / f"sample_{factuals.index[row_index]}.txt"
                )
                found_row = None
                for norm_type in self._norm_type:
                    result = generateExplanations(
                        self._approach_string,
                        explanation_file_name,
                        self._target_model._model,
                        self._dataset_wrapper,
                        factual_sample,
                        norm_type,
                    )
                    cfe_sample = (
                        result.get("cfe_sample") if isinstance(result, dict) else None
                    )
                    if cfe_sample:
                        found_row = self._dataset_wrapper.short_dict_to_feature_row(
                            cfe_sample
                        )
                        break
                if found_row is None:
                    found_row = pd.Series(
                        np.nan, index=self._feature_names, dtype="float64"
                    )
                candidates.loc[factuals.index[row_index], self._feature_names] = (
                    found_row.reindex(self._feature_names).to_numpy(dtype="float64")
                )

        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
