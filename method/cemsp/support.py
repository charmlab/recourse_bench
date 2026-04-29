from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from z3 import Bool, Not, Or, Solver, is_false, unsat

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from preprocess.preprocess_utils import resolve_feature_metadata

TorchModelTypes = (LinearModel, MlpModel)
BlackBoxModelTypes = (LinearModel, MlpModel, RandomForestModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    supported_types: Sequence[type[ModelObject]],
    method_name: str,
) -> None:
    if isinstance(target_model, tuple(supported_types)):
        return

    if isinstance(target_model, ModelObject):
        required_attributes = [
            "_device",
            "get_prediction",
            "get_class_to_index",
            "predict",
            "predict_proba",
        ]
        missing_attributes = [
            attribute
            for attribute in required_attributes
            if not hasattr(target_model, attribute)
        ]
        if not missing_attributes:
            return

    supported_names = ", ".join(cls.__name__ for cls in supported_types)
    raise TypeError(
        f"{method_name} supports target models [{supported_names}] or a compatible "
        f"ModelObject subclass only, "
        f"received {target_model.__class__.__name__}"
    )


def to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


@dataclass
class FeatureGroups:
    feature_names: list[str]
    feature_type: dict[str, str]
    continuous: list[str]
    categorical: list[str]
    binary: list[str]
    immutable: list[str]
    mutable: list[str]
    plausibility_constraints: list[str | None]


def resolve_feature_groups(dataset: DatasetObject) -> FeatureGroups:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, feature_mutability, feature_actionability = resolve_feature_metadata(
        dataset
    )

    continuous: list[str] = []
    categorical: list[str] = []
    binary: list[str] = []
    immutable: list[str] = []
    mutable: list[str] = []
    plausibility_constraints: list[str | None] = []

    for feature_name in feature_names:
        feature_kind = str(feature_type[feature_name]).lower()
        is_mutable = bool(feature_mutability[feature_name])
        actionability = str(feature_actionability[feature_name]).lower()

        if feature_kind == "numerical":
            continuous.append(feature_name)
        else:
            categorical.append(feature_name)
            if feature_kind == "binary":
                binary.append(feature_name)

        constraint: str | None = None
        if (not is_mutable) or actionability in {"none", "same"}:
            immutable.append(feature_name)
            constraint = "="
        else:
            mutable.append(feature_name)
            if actionability in {"same-or-increase", "increase"}:
                constraint = ">="
            elif actionability in {"same-or-decrease", "decrease"}:
                constraint = "<="
        plausibility_constraints.append(constraint)

    return FeatureGroups(
        feature_names=feature_names,
        feature_type={name: str(feature_type[name]) for name in feature_names},
        continuous=continuous,
        categorical=categorical,
        binary=binary,
        immutable=immutable,
        mutable=mutable,
        plausibility_constraints=plausibility_constraints,
    )


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)
        class_to_index = target_model.get_class_to_index()
        self._index_to_class = {
            index: class_value for class_value, index in class_to_index.items()
        }
        self.classes_ = np.array(
            [self._index_to_class[index] for index in sorted(self._index_to_class)]
        )

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict(self, X: pd.DataFrame | np.ndarray | torch.Tensor) -> np.ndarray:
        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=False)
        if isinstance(prediction, torch.Tensor):
            label_indices = prediction.detach().cpu().numpy().argmax(axis=1)
        else:
            label_indices = np.asarray(prediction).argmax(axis=1)
        return np.asarray(
            [self._index_to_class[int(index)] for index in label_indices],
            dtype=object,
        )

    def predict_label_indices(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        features = self.get_ordered_features(X)
        probabilities = self._target_model.get_prediction(features, proba=True)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.detach().cpu().numpy().argmax(axis=1)
        return np.asarray(probabilities).argmax(axis=1)


def resolve_target_classes(
    target_model: ModelObject,
    original_predictions: np.ndarray,
    desired_class: int | str | None,
) -> np.ndarray:
    class_to_index = target_model.get_class_to_index()
    index_to_class = {index: label for label, index in class_to_index.items()}

    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return np.full(
            shape=original_predictions.shape,
            fill_value=desired_class,
            dtype=object,
        )

    if len(class_to_index) != 2:
        raise ValueError(
            "desired_class=None is supported for binary classification only"
        )
    opposite_indices = 1 - original_predictions.astype(np.int64, copy=False)
    return np.asarray(
        [index_to_class[int(index)] for index in opposite_indices],
        dtype=object,
    )


def validate_counterfactuals(
    target_model: ModelObject,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    desired_class: int | str | None = None,
) -> pd.DataFrame:
    if list(candidates.columns) != list(factuals.columns):
        candidates = candidates.reindex(columns=factuals.columns)
    candidates = candidates.copy(deep=True)

    if candidates.shape[0] != factuals.shape[0]:
        raise ValueError("Candidates must preserve the number of factual rows")

    valid_rows = ~candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return candidates

    adapter = RecourseModelAdapter(target_model, factuals.columns)
    original_predictions = adapter.predict_label_indices(factuals)
    target_classes = resolve_target_classes(
        target_model=target_model,
        original_predictions=original_predictions,
        desired_class=desired_class,
    )

    candidate_predictions = adapter.predict(candidates.loc[valid_rows])
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = candidate_predictions == target_classes[
        valid_rows.to_numpy()
    ]
    candidates.loc[~success_mask, :] = np.nan
    return candidates


def parse_feature_vector(
    values: dict[str, object] | Sequence[object] | np.ndarray | pd.Series | pd.DataFrame,
    feature_names: Sequence[str],
    argument_name: str,
) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        if values.shape[0] != 1:
            raise ValueError(f"{argument_name} DataFrame must contain exactly one row")
        series = values.iloc[0]
    elif isinstance(values, pd.Series):
        series = values
    elif isinstance(values, dict):
        series = pd.Series(values)
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"{argument_name} must be a 1D vector")
        if array.shape[0] != len(feature_names):
            raise ValueError(
                f"{argument_name} length must match the number of feature columns"
            )
        return array.astype(np.float64, copy=False)

    series = series.reindex(feature_names)
    if series.isna().any():
        missing = series.index[series.isna()].tolist()
        raise ValueError(f"{argument_name} is missing feature values for: {missing}")
    return series.to_numpy(dtype=np.float64, copy=True)


def build_change_mask(
    factual: np.ndarray,
    replacement: np.ndarray,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> np.ndarray:
    factual = np.asarray(factual, dtype=np.float64).reshape(-1)
    replacement = np.asarray(replacement, dtype=np.float64).reshape(-1)
    return ~np.isclose(factual, replacement, atol=atol, rtol=rtol)


class MapSolver:
    def __init__(self, n_constraints: int):
        self.solver = Solver()
        self.n = int(n_constraints)
        self.all_n = set(range(self.n))

    def next_seed(self) -> list[int] | None:
        if self.solver.check() == unsat:
            return None
        seed = self.all_n.copy()
        model = self.solver.model()
        for variable in model:
            if is_false(model[variable]):
                seed.remove(int(variable.name()))
        return list(seed)

    def complement(self, subset: Sequence[int]) -> set[int]:
        return self.all_n.difference(subset)

    def prune_superset(self, from_point: Sequence[int]) -> None:
        self.solver.add(Or([Not(Bool(str(index))) for index in from_point]))

    def prune_subset(self, from_point: Sequence[int]) -> None:
        complement = self.complement(from_point)
        self.solver.add(Or([Bool(str(index)) for index in complement]))


class CFSolver:
    def __init__(
        self,
        candidate_indices: Sequence[int],
        model: RecourseModelAdapter,
        input_x: np.ndarray,
        replacement_x: np.ndarray,
        desired_pred: int | str,
    ):
        self.candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        self.n = int(self.candidate_indices.shape[0])
        self.model = model
        self.input_x = np.asarray(input_x, dtype=np.float64).reshape(1, -1)
        self.replacement_x = np.asarray(replacement_x, dtype=np.float64).reshape(1, -1)
        self.desired_pred = desired_pred

    def check_cf(self, mask: np.ndarray) -> bool:
        cf = self.mask_to_cf(mask)
        output = self.model.predict(cf)
        return bool(output[0] == self.desired_pred)

    def set_to_array(self, seed: Sequence[int]) -> np.ndarray:
        mask = np.zeros(self.n, dtype=np.float32)
        mask[list(seed)] = 1.0
        return mask

    def mask_to_cf(self, mask: np.ndarray) -> np.ndarray:
        cf = self.input_x.copy()
        if self.n == 0:
            return cf

        mask = np.asarray(mask, dtype=bool).reshape(-1)
        active_indices = self.candidate_indices[mask]
        cf[:, active_indices] = self.replacement_x[:, active_indices]
        return cf

    def complement(self, subset: Sequence[int]) -> set[int]:
        return set(range(self.n)).difference(subset)

    def shrink(self, seed: Sequence[int]) -> set[int]:
        current = set(seed)
        for feature_index in list(seed):
            if feature_index not in current:
                continue
            current.remove(feature_index)
            if self.check_cf(self.set_to_array(current)):
                continue
            current.add(feature_index)
        return current

    def grow(self, seed: Sequence[int]) -> list[int]:
        current = list(seed)
        for feature_index in self.complement(current):
            current.append(feature_index)
            if self.check_cf(self.set_to_array(current)):
                current.pop()
        return current


def find_cf(cfsolver: CFSolver, mapsolver: MapSolver):
    while True:
        seed = mapsolver.next_seed()
        if seed is None:
            return

        mask = cfsolver.set_to_array(seed)
        if not cfsolver.check_cf(mask):
            a_hat = cfsolver.grow(seed)
            mapsolver.prune_subset(a_hat)
        else:
            a_star = cfsolver.shrink(seed)
            mask = cfsolver.set_to_array(a_star)
            cf = cfsolver.mask_to_cf(mask)
            yield ("Counterfactual Explanation", cf, mask)
            mapsolver.prune_superset(a_star)
