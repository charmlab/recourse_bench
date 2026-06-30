from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.method.mace.library.mace import generateExplanations
from recourse_bench.method.mace.support import (
    BlackBoxModelTypes,
    ensure_supported_target_model,
    validate_counterfactuals,
)
from recourse_bench.method.method_object import MethodObject
from recourse_bench.model.model_object import ModelObject
from recourse_bench.utils.caching import get_cache_dir
from recourse_bench.utils.registry import register
from recourse_bench.utils.seed import seed_context


def _dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    return True


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
        raise ValueError("MaceMethod epsilon must format like 1e-1, 1e-3, or 1e-5")
    return formatted


@dataclass
class _MaceAttribute:
    attr_name_kurz: str
    attr_type: str
    lower_bound: float
    upper_bound: float
    mutability: bool
    actionability: str
    parent_name_kurz: str | int = -1


class MaceDatasetWrapper:
    def __init__(
        self,
        dataset_name: str,
        feature_names: list[str],
        feature_types: dict[str, str],
        bounds: dict[str, tuple[float, float]],
        mutability: dict[str, bool],
        actionability: dict[str, str],
        encoded_parent: dict[str, str],
    ):
        self.dataset_name = str(dataset_name)
        self.is_one_hot = True
        self._feature_names = list(feature_names)
        self._short_names = {
            feature: f"x{i}" for i, feature in enumerate(self._feature_names)
        }
        self._inverse_short_names = {
            short_name: feature_name
            for feature_name, short_name in self._short_names.items()
        }
        self._parent_short_names: dict[str, str | int] = {}
        self.attributes_kurz: dict[str, _MaceAttribute] = {}

        for feature_name in self._feature_names:
            short_name = self._short_names[feature_name]
            parent_name = encoded_parent.get(feature_name, feature_name)
            parent_short_name: str | int
            if parent_name == feature_name:
                parent_short_name = -1
            elif parent_name in self._short_names:
                parent_short_name = self._short_names[parent_name]
            else:
                same_parent_columns = [
                    column
                    for column, parent in encoded_parent.items()
                    if parent == parent_name and column in self._short_names
                ]
                parent_short_name = (
                    self._short_names[same_parent_columns[0]]
                    if same_parent_columns
                    else -1
                )
            self._parent_short_names[short_name] = parent_short_name
            lower_bound, upper_bound = bounds[feature_name]
            self.attributes_kurz[short_name] = _MaceAttribute(
                attr_name_kurz=short_name,
                attr_type=str(feature_types[feature_name]),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                mutability=bool(mutability[feature_name]),
                actionability=_normalize_actionability(actionability[feature_name]),
                parent_name_kurz=parent_short_name,
            )
        self.attributes_kurz["y"] = _MaceAttribute(
            attr_name_kurz="y",
            attr_type="binary",
            lower_bound=0.0,
            upper_bound=1.0,
            mutability=False,
            actionability="none",
            parent_name_kurz=-1,
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

    def getDictOfSiblings(self, kind: str = "kurz"):
        siblings: dict[str, dict[str, list[str]]] = {"cat": {}, "ord": {}}
        for feature_name in self._feature_names:
            short_name = self._short_names[feature_name]
            attr = self.attributes_kurz[short_name]
            if attr.attr_type == "sub-categorical":
                parent = str(attr.parent_name_kurz)
                siblings["cat"].setdefault(parent, []).append(short_name)
            elif attr.attr_type == "sub-ordinal":
                parent = str(attr.parent_name_kurz)
                siblings["ord"].setdefault(parent, []).append(short_name)

        for group in siblings.values():
            for parent, values in group.items():
                group[parent] = sorted(
                    values, key=lambda value: int(value[1:].split("_")[0])
                )
        return siblings

    def getOneHotAttributesNames(self, kind: str = "kurz"):
        siblings = self.getDictOfSiblings(kind)
        names: list[str] = []
        for group in siblings.values():
            for sibling_names in group.values():
                names.extend(sibling_names)
        return names

    def getNonHotAttributesNames(self, kind: str = "kurz"):
        one_hot = set(self.getOneHotAttributesNames(kind))
        return [
            attr_name
            for attr_name in self.getInputAttributeNames(kind)
            if attr_name not in one_hot
        ]

    def getSiblingsFor(self, attr_name_kurz: str):
        for group in self.getDictOfSiblings("kurz").values():
            for siblings in group.values():
                if attr_name_kurz in siblings:
                    return siblings
        return [attr_name_kurz]

    def factual_to_short_dict(
        self,
        factual: pd.Series,
        predicted_label: int,
    ) -> dict[str, int | float | bool]:
        output: dict[str, int | float | bool] = {}
        for feature_name in self._feature_names:
            value = factual[feature_name]
            attr_type = self.attributes_kurz[self._short_names[feature_name]].attr_type
            if attr_type == "numeric-real":
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
                attr_type = self.attributes_kurz[short_name].attr_type
                if attr_type == "numeric-real":
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

        valid = {"zero_norm", "one_norm", "two_norm", "infty_norm"}
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
    def _unwrap_sklearn_model(target_model: ModelObject):
        model = getattr(target_model, "_model", None)
        if model is None:
            raise RuntimeError("Target model has not been initialized")
        return model

    @staticmethod
    def _resolve_feature_metadata(
        dataset: DatasetObject,
    ) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
        if _dataset_has_attr(dataset, "mace_encoded_attr_type"):
            feature_type = dataset.attr("mace_encoded_attr_type")
            feature_mutability = dataset.attr("encoded_feature_mutability")
            feature_actionability = dataset.attr("encoded_feature_actionability")
        elif _dataset_has_attr(dataset, "mace_feature_type"):
            feature_type = dataset.attr("mace_feature_type")
            feature_mutability = dataset.attr("mace_feature_mutability")
            feature_actionability = dataset.attr("mace_feature_actionability")
        else:
            feature_type = dataset.attr("raw_feature_type")
            feature_mutability = dataset.attr("raw_feature_mutability")
            feature_actionability = dataset.attr("raw_feature_actionability")
        return (
            {str(key): str(value).lower() for key, value in feature_type.items()},
            {str(key): bool(value) for key, value in feature_mutability.items()},
            {
                str(key): _normalize_actionability(value)
                for key, value in feature_actionability.items()
            },
        )

    @staticmethod
    def _resolve_bounds(
        dataset: DatasetObject,
        feature_df: pd.DataFrame,
    ) -> dict[str, tuple[float, float]]:
        if _dataset_has_attr(dataset, "mace_encoded_bounds"):
            encoded_bounds = dataset.attr("mace_encoded_bounds")
            return {
                feature_name: tuple(map(float, encoded_bounds[feature_name]))
                for feature_name in feature_df.columns
            }
        if _dataset_has_attr(dataset, "balanced"):
            balanced = dataset.attr("balanced")
            if isinstance(balanced, dict):
                raw_feature_min = balanced.get("feature_min")
                raw_feature_max = balanced.get("feature_max")
                if isinstance(raw_feature_min, dict) and isinstance(
                    raw_feature_max, dict
                ):
                    return {
                        feature_name: (
                            float(raw_feature_min[feature_name]),
                            float(raw_feature_max[feature_name]),
                        )
                        for feature_name in feature_df.columns
                    }
        return {
            feature_name: (
                float(feature_df[feature_name].min()),
                float(feature_df[feature_name].max()),
            )
            for feature_name in feature_df.columns
        }

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
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "MaceMethod")
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
        if not _dataset_has_attr(trainset, "mace_encoding"):
            raise ValueError("MaceMethod expects MaceEncodePreProcess before fit()")
        if _dataset_has_attr(trainset, "scaling"):
            raise ValueError("MaceMethod does not support scaled datasets")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            feature_df = trainset.get(target=False)
            feature_type, mutability, actionability = self._resolve_feature_metadata(
                trainset
            )
            bounds = self._resolve_bounds(trainset, feature_df)
            encoded_parent = trainset.attr("mace_encoded_parent")
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

            self._dataset_wrapper = MaceDatasetWrapper(
                dataset_name=dataset_name,
                feature_names=self._feature_names,
                feature_types=feature_type,
                bounds=bounds,
                mutability=mutability,
                actionability=actionability,
                encoded_parent=encoded_parent,
            )
            self._sklearn_model = self._unwrap_sklearn_model(self._target_model)
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
            for row_index in tqdm(solve_indices, desc="mace-generate", leave=False):
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
                        self._sklearn_model,
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
