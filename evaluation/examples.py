from __future__ import annotations

import json
import random
from numbers import Integral

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register
from utils.seed import seed_context

_EMPTY_EXAMPLE = "{}"


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _to_python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


@register("examples")
class ExamplesEvaluation(EvaluationObject):
    def __init__(
        self,
        seed: int | None = None,
        num_examples: int = 1,
        iloc: list[int] | None = None,
        **kwargs,
    ):
        if not _is_integer(num_examples):
            raise TypeError("num_examples must be an integer")
        if int(num_examples) < 1:
            raise ValueError("num_examples must be >= 1")

        if iloc is not None:
            if not isinstance(iloc, list):
                raise TypeError("iloc must be a list of integers")
            if len(iloc) != int(num_examples):
                raise ValueError("iloc length must match num_examples")
            for row_position in iloc:
                if not _is_integer(row_position):
                    raise TypeError("iloc must contain only integers")
                if int(row_position) < 0:
                    raise ValueError("iloc must contain only non-negative integers")

        self._num_examples = int(num_examples)
        self._iloc = (
            None if iloc is None else [int(row_position) for row_position in iloc]
        )
        self._seed = seed

    @staticmethod
    def _serialize_counterfactual(row: pd.Series, feature_names: list[str]) -> str:
        payload = {
            feature_name: _to_python_scalar(row.loc[feature_name])
            for feature_name in feature_names
        }
        return json.dumps(payload)

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            _,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        feature_names = list(counterfactual_features.columns)
        selected_mask = evaluation_mask & success_mask
        selected_positions = list(np.flatnonzero(selected_mask.to_numpy()))
        selected_position_set = set(selected_positions)

        examples = [_EMPTY_EXAMPLE] * self._num_examples
        if self._iloc is None:
            sample_size = min(self._num_examples, len(selected_positions))
            with seed_context(self._seed):
                sampled_positions = random.sample(selected_positions, k=sample_size)

            for output_position, row_position in enumerate(sampled_positions):
                examples[output_position] = self._serialize_counterfactual(
                    counterfactual_features.iloc[row_position],
                    feature_names,
                )
        else:
            num_rows = counterfactual_features.shape[0]
            for output_position, row_position in enumerate(self._iloc):
                if (
                    row_position >= num_rows
                    or row_position not in selected_position_set
                ):
                    continue
                examples[output_position] = self._serialize_counterfactual(
                    counterfactual_features.iloc[row_position],
                    feature_names,
                )

        return pd.DataFrame(
            [
                {
                    f"counterfactual_example_{index + 1}": example
                    for index, example in enumerate(examples)
                }
            ]
        )
