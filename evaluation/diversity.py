from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register


def _nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if np.isnan(array).all():
        return float("nan")
    return float(np.nanmean(array))


def _pairwise_distances(values: np.ndarray, ord_value: int | float) -> np.ndarray:
    if values.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    distances = []
    for left_index in range(values.shape[0]):
        for right_index in range(left_index + 1, values.shape[0]):
            distances.append(
                float(
                    np.linalg.norm(
                        values[left_index] - values[right_index],
                        ord=ord_value,
                    )
                )
            )
    return np.asarray(distances, dtype=np.float64)


def _unique_row_count(values: np.ndarray, decimals: int) -> int:
    if values.shape[0] == 0:
        return 0
    rounded = np.round(values.astype(np.float64, copy=False), decimals=decimals)
    return int(np.unique(rounded, axis=0).shape[0])


def _resolve_counterfactual_sets(counterfactuals: DatasetObject) -> list[pd.DataFrame]:
    try:
        counterfactual_sets = counterfactuals.attr("counterfactual_sets")
    except AttributeError:
        return []
    if not isinstance(counterfactual_sets, list):
        raise TypeError("counterfactual_sets must be a list of pandas DataFrames")
    for counterfactual_set in counterfactual_sets:
        if not isinstance(counterfactual_set, pd.DataFrame):
            raise TypeError("counterfactual_sets must contain only pandas DataFrames")
    return counterfactual_sets


def _resolve_validity_masks(
    counterfactuals: DatasetObject,
    counterfactual_sets: list[pd.DataFrame],
) -> list[pd.Series]:
    try:
        raw_validity = counterfactuals.attr("counterfactual_set_validity")
    except AttributeError:
        return [
            ~counterfactual_set.isna().any(axis=1)
            for counterfactual_set in counterfactual_sets
        ]

    if not isinstance(raw_validity, list):
        raise TypeError("counterfactual_set_validity must be a list of pandas Series")
    if len(raw_validity) != len(counterfactual_sets):
        raise ValueError("counterfactual_set_validity must align with counterfactual_sets")

    validity_masks = []
    for validity_mask, counterfactual_set in zip(
        raw_validity,
        counterfactual_sets,
        strict=True,
    ):
        if not isinstance(validity_mask, pd.Series):
            raise TypeError("counterfactual_set_validity must contain pandas Series")
        validity_masks.append(validity_mask.astype(bool).reindex(counterfactual_set.index))
    return validity_masks


@register("diversity")
class DiversityEvaluation(EvaluationObject):
    def __init__(self, unique_decimals: int = 8, **kwargs):
        del kwargs
        self._unique_decimals = int(unique_decimals)
        if self._unique_decimals < 0:
            raise ValueError("unique_decimals must be >= 0")

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            _,
            evaluation_mask,
            _success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        counterfactual_sets = _resolve_counterfactual_sets(counterfactuals)
        if not counterfactual_sets:
            return pd.DataFrame(
                [
                    {
                        "cf_set_size": float("nan"),
                        "cf_set_validity": float("nan"),
                        "cf_set_unique_size": float("nan"),
                        "cf_set_unique_fraction": float("nan"),
                        "cf_pairwise_l1_mean": float("nan"),
                        "cf_pairwise_l1_min": float("nan"),
                        "cf_pairwise_l2_mean": float("nan"),
                        "cf_pairwise_l2_min": float("nan"),
                        "cf_set_l1_to_factual_mean": float("nan"),
                        "cf_set_l1_to_factual_min": float("nan"),
                        "cf_set_l2_to_factual_mean": float("nan"),
                        "cf_set_l2_to_factual_min": float("nan"),
                    }
                ]
            )
        if len(counterfactual_sets) != factual_features.shape[0]:
            raise ValueError("counterfactual_sets must align with factual rows")

        validity_masks = _resolve_validity_masks(counterfactuals, counterfactual_sets)

        metrics: dict[str, list[float]] = {
            "cf_set_size": [],
            "cf_set_validity": [],
            "cf_set_unique_size": [],
            "cf_set_unique_fraction": [],
            "cf_pairwise_l1_mean": [],
            "cf_pairwise_l1_min": [],
            "cf_pairwise_l2_mean": [],
            "cf_pairwise_l2_min": [],
            "cf_set_l1_to_factual_mean": [],
            "cf_set_l1_to_factual_min": [],
            "cf_set_l2_to_factual_mean": [],
            "cf_set_l2_to_factual_min": [],
        }

        selected_positions = np.flatnonzero(evaluation_mask.to_numpy())
        for position in selected_positions:
            factual_row = factual_features.iloc[position].to_numpy(dtype=np.float64)
            counterfactual_set = counterfactual_sets[position].reindex(
                columns=factual_features.columns
            )
            validity_mask = validity_masks[position].reindex(
                counterfactual_set.index
            ).fillna(False)

            set_size = int(counterfactual_set.shape[0])
            valid_set = counterfactual_set.loc[validity_mask.to_numpy()]
            valid_set = valid_set.loc[~valid_set.isna().any(axis=1)]
            valid_size = int(valid_set.shape[0])

            metrics["cf_set_size"].append(float(set_size))
            metrics["cf_set_validity"].append(
                float(valid_size / set_size) if set_size else 0.0
            )

            if valid_size == 0:
                for key in metrics:
                    if key not in {"cf_set_size", "cf_set_validity"}:
                        metrics[key].append(float("nan"))
                continue

            valid_values = valid_set.to_numpy(dtype=np.float64)
            unique_count = _unique_row_count(valid_values, self._unique_decimals)
            metrics["cf_set_unique_size"].append(float(unique_count))
            metrics["cf_set_unique_fraction"].append(float(unique_count / valid_size))

            for norm_name, ord_value in (("l1", 1), ("l2", 2)):
                pairwise = _pairwise_distances(valid_values, ord_value)
                metrics[f"cf_pairwise_{norm_name}_mean"].append(
                    float(pairwise.mean()) if pairwise.size else float("nan")
                )
                metrics[f"cf_pairwise_{norm_name}_min"].append(
                    float(pairwise.min()) if pairwise.size else float("nan")
                )

                factual_distances = np.linalg.norm(
                    valid_values - factual_row.reshape(1, -1),
                    ord=ord_value,
                    axis=1,
                )
                metrics[f"cf_set_{norm_name}_to_factual_mean"].append(
                    float(factual_distances.mean())
                )
                metrics[f"cf_set_{norm_name}_to_factual_min"].append(
                    float(factual_distances.min())
                )

        return pd.DataFrame([{key: _nanmean(values) for key, values in metrics.items()}])
