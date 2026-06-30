from __future__ import annotations

import pandas as pd

from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.evaluation.evaluation_object import EvaluationObject
from recourse_bench.evaluation.evaluation_utils import resolve_evaluation_inputs
from recourse_bench.utils.registry import register


@register("validity")
class ValidityEvaluation(EvaluationObject):
    def __init__(self, **kwargs):
        pass

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        evaluated_success_mask = success_mask.loc[evaluation_mask.to_numpy()]
        if evaluated_success_mask.shape[0] == 0:
            validity = float("nan")
        else:
            validity = float(evaluated_success_mask.astype(float).mean())

        return pd.DataFrame([{"validity": validity}])
