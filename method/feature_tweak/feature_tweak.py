from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.feature_tweak.support import (
    RecourseModelAdapter,
    ensure_supported_target_model,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from preprocess.preprocess_utils import resolve_feature_metadata
from utils.registry import register
from utils.seed import seed_context


def _l2_cost(a, b):
    return np.linalg.norm(a - b, ord=2)


def _tweaked_feature_rate_cost(a, b):
    return float(np.mean(~np.isclose(a, b)))


def _cosine_distance_cost(a, b):
    a_norm = float(np.linalg.norm(a, ord=2))
    b_norm = float(np.linalg.norm(b, ord=2))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0 if np.allclose(a, b) else 1.0
    cosine_similarity = float(np.dot(a, b) / (a_norm * b_norm))
    return 1.0 - cosine_similarity


def _jaccard_distance_cost(a, b):
    a_mask = ~np.isclose(a, 0.0)
    b_mask = ~np.isclose(b, 0.0)
    union = int(np.logical_or(a_mask, b_mask).sum())
    if union == 0:
        return 0.0
    intersection = int(np.logical_and(a_mask, b_mask).sum())
    return 1.0 - (intersection / union)


def _pearson_distance_cost(a, b):
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        return 0.0 if np.allclose(a, b) else 1.0
    correlation = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(correlation):
        return 1.0
    return 1.0 - correlation


def _resolve_cost_func(cost: str) -> Callable[[np.ndarray, np.ndarray], float]:
    normalized_cost = str(cost).lower()
    cost_map: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "euclidean": _l2_cost,
        "l2": _l2_cost,
        "tweaked_feature_rate": _tweaked_feature_rate_cost,
        "cosine": _cosine_distance_cost,
        "jaccard": _jaccard_distance_cost,
        "pearson": _pearson_distance_cost,
    }
    if normalized_cost not in cost_map:
        raise ValueError(f"Unsupported FeatureTweak cost: {cost}")
    return cost_map[normalized_cost]


def search_path(tree, desired_class: int):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    path_info = {}

    def _dfs(node_id: int, path_conditions: list[tuple[int, int, int, float]]) -> None:
        if children_left[node_id] == -1:
            node_values = values[node_id]
            if node_values.ndim > 1:
                node_values = node_values[0]
            if int(np.argmax(node_values)) != desired_class:
                return
            path_info[node_id] = {
                "node_id": [condition[0] for condition in path_conditions],
                "inequality_symbol": [condition[1] for condition in path_conditions],
                "feature": [condition[2] for condition in path_conditions],
                "threshold": [condition[3] for condition in path_conditions],
            }
            return

        left_node = int(children_left[node_id])
        right_node = int(children_right[node_id])
        feature_idx = int(feature[node_id])
        threshold_value = float(threshold[node_id])

        _dfs(
            left_node,
            path_conditions
            + [(node_id, 0, feature_idx, threshold_value)],
        )
        _dfs(
            right_node,
            path_conditions
            + [(node_id, 1, feature_idx, threshold_value)],
        )

    _dfs(0, [])
    return path_info


@register("feature_tweak")
class FeatureTweakMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        eps: float = 0.1,
        cost: str = "euclidean",
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (RandomForestModel,),
            "FeatureTweakMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._eps = float(eps)
        self._cost_name = str(cost).lower()
        self._cost_func = _resolve_cost_func(self._cost_name)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._eps <= 0:
            raise ValueError("eps must be > 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for FeatureTweakMethod.fit()")

        with seed_context(self._seed):
            self._feature_names = list(trainset.get(target=False).columns)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            (
                _feature_type,
                feature_mutability,
                feature_actionability,
            ) = resolve_feature_metadata(trainset)
            self._feature_mutability = {
                feature_name: bool(feature_mutability[feature_name])
                for feature_name in self._feature_names
            }
            self._feature_actionability = {
                feature_name: str(feature_actionability[feature_name]).lower()
                for feature_name in self._feature_names
            }
            self._forest = self._target_model._model
            self._trees = list(self._forest.estimators_)
            self._is_trained = True

    @staticmethod
    def _satisfies_condition(
        value: float, threshold_value: float, inequality_symbol: int
    ) -> bool:
        if inequality_symbol == 0:
            return bool(value <= threshold_value)
        if inequality_symbol == 1:
            return bool(value > threshold_value)
        raise ValueError(f"Unsupported inequality symbol: {inequality_symbol}")

    def _resolve_actionable_value(
        self,
        original_value: float,
        current_value: float,
        required_value: float,
        threshold_value: float,
        inequality_symbol: int,
        feature_name: str,
    ) -> float | None:
        actionability = self._feature_actionability[feature_name]
        is_mutable = self._feature_mutability[feature_name]

        if (not is_mutable) or actionability in {"none", "same"}:
            if self._satisfies_condition(
                current_value,
                threshold_value,
                inequality_symbol,
            ):
                return current_value
            return None

        if actionability == "same-or-increase" and required_value < original_value:
            if self._satisfies_condition(
                current_value,
                threshold_value,
                inequality_symbol,
            ) and current_value >= original_value:
                return current_value
            return None

        if actionability == "same-or-decrease" and required_value > original_value:
            if self._satisfies_condition(
                current_value,
                threshold_value,
                inequality_symbol,
            ) and current_value <= original_value:
                return current_value
            return None

        return required_value

    def _satisfies_path(self, x: np.ndarray, path_info) -> bool:
        for feature_idx, threshold_value, inequality_symbol in zip(
            path_info["feature"],
            path_info["threshold"],
            path_info["inequality_symbol"],
        ):
            if not self._satisfies_condition(
                float(x[int(feature_idx)]),
                float(threshold_value),
                int(inequality_symbol),
            ):
                return False
        return True

    def _esatisfactory_instance(self, x: np.ndarray, path_info):
        esatisfactory = copy.deepcopy(x)
        for index in range(len(path_info["feature"])):
            feature_idx = int(path_info["feature"][index])
            feature_name = self._feature_names[feature_idx]
            threshold_value = float(path_info["threshold"][index])
            inequality_symbol = int(path_info["inequality_symbol"][index])
            if inequality_symbol == 0:
                required_value = threshold_value - self._eps
            elif inequality_symbol == 1:
                required_value = threshold_value + self._eps
            else:
                raise ValueError(f"Unsupported inequality symbol: {inequality_symbol}")

            next_value = self._resolve_actionable_value(
                original_value=float(x[feature_idx]),
                current_value=float(esatisfactory[feature_idx]),
                required_value=float(required_value),
                threshold_value=threshold_value,
                inequality_symbol=inequality_symbol,
                feature_name=feature_name,
            )
            if next_value is None:
                return None
            esatisfactory[feature_idx] = next_value

        if not self._satisfies_path(esatisfactory, path_info):
            return None
        return esatisfactory

    @staticmethod
    def _as_array(x: np.ndarray) -> np.ndarray:
        array = np.asarray(x, dtype="float32")
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array

    def _as_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(self._as_array(x), columns=self._feature_names)

    def _forest_predict_label(self, x: np.ndarray) -> int:
        return int(self._forest.predict(self._as_dataframe(x))[0])

    def _tree_predict_label(self, tree, x: np.ndarray) -> int:
        return int(tree.predict(self._as_array(x))[0])

    def _feature_tweaking(self, x: np.ndarray, cf_label: int) -> np.ndarray:
        x_out = copy.deepcopy(x)
        delta_min = float("inf")
        forest_prediction = self._forest_predict_label(x)

        for tree in self._trees:
            estimator_prediction = self._tree_predict_label(tree, x)
            if forest_prediction != estimator_prediction or estimator_prediction == cf_label:
                continue

            paths_info = search_path(tree, desired_class=cf_label)
            for path_info in paths_info.values():
                es_instance = self._esatisfactory_instance(x, path_info)
                if es_instance is None:
                    continue
                if self._tree_predict_label(tree, es_instance) != cf_label:
                    continue
                if self._forest_predict_label(es_instance) != cf_label:
                    continue

                candidate_cost = float(self._cost_func(x, es_instance))
                if candidate_cost < delta_min:
                    x_out = es_instance
                    delta_min = candidate_cost
        return x_out

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        class_to_index = self._target_model.get_class_to_index()
        if self._desired_class is None:
            if len(class_to_index) != 2:
                raise ValueError(
                    "FeatureTweakMethod requires desired_class for non-binary targets"
                )
            cf_label = 1
        else:
            cf_label = int(class_to_index[self._desired_class])

        rows = []
        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                rows.append(
                    self._feature_tweaking(
                        row.to_numpy(dtype="float32"),
                        cf_label,
                    )
                )

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
