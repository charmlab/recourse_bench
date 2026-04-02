from __future__ import annotations

import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs, to_float_tensor
from utils.registry import register


@register("distance")
class DistanceEvaluation(EvaluationObject):
    @staticmethod
    def _resolve_metrics(metrics: list[str] | None) -> list[str]:
        resolved_metrics = [
            metric.lower() for metric in (metrics or ["l0", "l1", "l2", "linf"])
        ]
        invalid = [
            metric
            for metric in resolved_metrics
            if metric not in {"l0", "l1", "l2", "linf"}
        ]
        if invalid:
            raise ValueError(f"Unsupported distance metrics: {invalid}")
        return resolved_metrics

    def __init__(self, metrics: list[str] | None = None, **kwargs):
        self._metrics = self._resolve_metrics(metrics)

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        selected_mask = evaluation_mask & success_mask
        results: dict[str, float] = {}
        if selected_mask.sum() == 0:
            for metric in self._metrics:
                results[f"distance_{metric}"] = float("nan")
            return pd.DataFrame([results])

        factual_success = factual_features.loc[selected_mask.to_numpy()]
        counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]
        diff = torch.abs(
            to_float_tensor(counterfactual_success) - to_float_tensor(factual_success)
        )

        for metric in self._metrics:
            if metric == "l0":
                zero = torch.zeros(1, dtype=diff.dtype, device=diff.device)
                changed = ~torch.isclose(diff, zero)
                value = changed.sum(dim=1).to(dtype=torch.float32).mean().item()
            elif metric == "l1":
                value = diff.sum(dim=1).mean().item()
            elif metric == "l2":
                value = torch.linalg.vector_norm(diff, ord=2, dim=1).mean().item()
            else:
                value = diff.max(dim=1).values.mean().item()
            results[f"distance_{metric}"] = float(value)

        return pd.DataFrame([results])
