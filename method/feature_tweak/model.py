from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from method.feature_tweak.support import (
    FeatureTweakContext,
    FeatureTweakTargetModelAdapter,
    check_counterfactuals,
    project_candidate_features,
)


def _L1_cost_func(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=1))


def _L2_cost_func(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord=2))


def search_path(
    tree: DecisionTreeClassifier,
    class_labels: list[int] | np.ndarray,
    target_class: int,
) -> dict[int, dict[str, list[int] | list[float]]]:
    if not isinstance(tree, DecisionTreeClassifier):
        raise TypeError(
            "FeatureTweak supports sklearn.tree.DecisionTreeClassifier only"
        )

    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    leaf_nodes = np.where(children_left == -1)[0]
    leaf_values = values[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
    leaf_classes = np.argmax(leaf_values, axis=-1)
    leaf_nodes = leaf_nodes[np.where(leaf_classes == int(target_class))[0]]

    paths: dict[int, tuple[list[int], list[int]]] = {}
    for leaf_node in leaf_nodes:
        if int(leaf_node) == 0:
            paths[int(leaf_node)] = ([], [])
            continue

        child_node = int(leaf_node)
        parent_node = -100
        parents_left: list[int] = []
        parents_right: list[int] = []
        while parent_node != 0:
            left_parents = np.where(children_left == child_node)[0]
            if left_parents.shape == (0,):
                parent_left = -1
                parent_right = int(np.where(children_right == child_node)[0][0])
                parent_node = parent_right
            else:
                parent_right = -1
                parent_left = int(left_parents[0])
                parent_node = parent_left

            parents_left.append(parent_left)
            parents_right.append(parent_right)
            child_node = parent_node

        paths[int(leaf_node)] = (parents_left, parents_right)

    return get_path_info(paths, threshold, feature)


def get_path_info(
    paths: dict[int, tuple[list[int], list[int]]],
    threshold: np.ndarray,
    feature: np.ndarray,
) -> dict[int, dict[str, list[int] | list[float]]]:
    path_info: dict[int, dict[str, list[int] | list[float]]] = {}
    for leaf_node, (parents_left, parents_right) in paths.items():
        node_ids: list[int] = []
        inequality_symbols: list[int] = []
        thresholds: list[float] = []
        features: list[int] = []

        for idx in range(len(parents_left)):

            def do_appends(node_id: int) -> None:
                node_ids.append(node_id)
                thresholds.append(float(threshold[node_id]))
                features.append(int(feature[node_id]))

            if parents_left[idx] != -1:
                node_id = int(parents_left[idx])
                inequality_symbols.append(0)
                do_appends(node_id)
            elif parents_right[idx] != -1:
                node_id = int(parents_right[idx])
                inequality_symbols.append(1)
                do_appends(node_id)

        path_info[int(leaf_node)] = {
            "node_id": node_ids,
            "inequality_symbol": inequality_symbols,
            "threshold": thresholds,
            "feature": features,
        }
    return path_info


class FeatureTweak:
    def __init__(
        self,
        mlmodel: FeatureTweakTargetModelAdapter,
        context: FeatureTweakContext,
        desired_class: int,
        eps: float = 0.1,
        cost_func: Callable[[np.ndarray, np.ndarray], float] = _L2_cost_func,
    ):
        self.model = mlmodel
        self.context = context
        self.desired_class = int(desired_class)
        self.eps = float(eps)
        self.cost_func = cost_func
        self.class_labels = np.asarray(self.model.classes_, dtype=np.int64)
        self._tree_paths = tuple(
            search_path(
                tree=tree,
                class_labels=self.class_labels,
                target_class=self.desired_class,
            )
            for tree in self.model.tree_iterator
        )

    def esatisfactory_instance(
        self,
        x: np.ndarray,
        path_info: dict[str, list[int] | list[float]],
    ) -> np.ndarray:
        esatisfactory = np.asarray(x, dtype="float64").copy()
        features = path_info["feature"]
        thresholds = path_info["threshold"]
        inequality_symbols = path_info["inequality_symbol"]
        for idx in range(len(features)):
            feature_idx = int(features[idx])
            threshold_value = float(thresholds[idx])
            inequality_symbol = int(inequality_symbols[idx])
            if inequality_symbol == 0:
                esatisfactory[feature_idx] = threshold_value - self.eps
            elif inequality_symbol == 1:
                esatisfactory[feature_idx] = threshold_value + self.eps
            else:
                raise ValueError(
                    f"Unsupported inequality symbol in path: {inequality_symbol}"
                )
        return esatisfactory

    @staticmethod
    def _predict_tree(tree: DecisionTreeClassifier, x: np.ndarray) -> int:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names",
                category=UserWarning,
            )
            prediction = tree.predict(np.asarray(x, dtype="float64").reshape(1, -1))
        return int(prediction[0])

    def _predict_forest(self, x: np.ndarray) -> int:
        prediction = self.model.predict(np.asarray(x, dtype="float64").reshape(1, -1))
        return int(np.asarray(prediction, dtype=np.int64)[0])

    def feature_tweaking(
        self,
        x: np.ndarray,
        factual_prediction: int | None = None,
    ) -> np.ndarray:
        x = np.asarray(x, dtype="float64").copy()
        if factual_prediction is None:
            factual_prediction = self._predict_forest(x)
        if int(factual_prediction) == self.desired_class:
            return x

        best_forest_candidate: np.ndarray | None = None
        best_forest_cost = float("inf")
        best_tree_candidate: np.ndarray | None = None
        best_tree_cost = float("inf")

        for tree, paths_info in zip(self.model.tree_iterator, self._tree_paths):
            estimator_prediction = self._predict_tree(tree, x)
            if estimator_prediction != int(factual_prediction):
                continue
            if estimator_prediction == self.desired_class:
                continue

            for path_info in paths_info.values():
                es_instance = self.esatisfactory_instance(x, path_info)
                projected_candidate = project_candidate_features(
                    candidate=es_instance,
                    factual=x,
                    context=self.context,
                )

                if self._predict_tree(tree, projected_candidate) != self.desired_class:
                    continue

                candidate_cost = self.cost_func(x, projected_candidate)
                if candidate_cost < best_tree_cost:
                    best_tree_candidate = projected_candidate.copy()
                    best_tree_cost = candidate_cost

                if self._predict_forest(projected_candidate) == self.desired_class:
                    if candidate_cost < best_forest_cost:
                        best_forest_candidate = projected_candidate.copy()
                        best_forest_cost = candidate_cost

        if best_forest_candidate is not None:
            return best_forest_candidate
        if best_tree_candidate is not None:
            return best_tree_candidate
        return x

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        instances = self.model.get_ordered_features(factuals)
        if instances.shape[0] == 0:
            return instances.copy(deep=True)

        counterfactuals: list[np.ndarray] = []
        for _, row in instances.iterrows():
            factual = row.to_numpy(dtype="float64", copy=True)
            factual_prediction = self._predict_forest(factual)
            counterfactual = self.feature_tweaking(
                factual,
                factual_prediction=factual_prediction,
            )
            counterfactuals.append(counterfactual)

        return check_counterfactuals(
            mlmodel=self.model,
            counterfactuals=counterfactuals,
            factuals=instances,
            desired_class=self.desired_class,
        )
