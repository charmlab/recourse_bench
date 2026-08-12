from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.utils import write_reproduction_report  # noqa: E402
from method.claproar.model import ClaPROAR  # noqa: E402


REPORT_PATH = Path(__file__).with_name("reproduction_report.json")
FEATURE_NAMES = ["x0", "x1"]


class LinearLogitAdapter:
    def __init__(self) -> None:
        self.feature_input_order = list(FEATURE_NAMES)
        self.data = type("Data", (), {"target": "target"})()
        self.weight = torch.tensor([2.5, 0.8], dtype=torch.float32)
        self.bias = torch.tensor(-0.2, dtype=torch.float32)

    def get_ordered_features(
        self,
        values: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            return values.loc[:, self.feature_input_order].copy(deep=True)
        if isinstance(values, torch.Tensor):
            array = values.detach().cpu().numpy()
        else:
            array = np.asarray(values)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return pd.DataFrame(array, columns=self.feature_input_order)

    def predict_proba(
        self,
        values: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if isinstance(values, torch.Tensor):
            logits = values.to(dtype=torch.float32) @ self.weight + self.bias
            return torch.stack([-logits, logits], dim=1)

        array = self.get_ordered_features(values).to_numpy(dtype="float32")
        logits_np = array @ self.weight.numpy() + float(self.bias.item())
        return np.stack([-logits_np, logits_np], axis=1)

    def predict(self, values: pd.DataFrame | np.ndarray | torch.Tensor) -> np.ndarray:
        probabilities = self.predict_proba(values)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.detach().cpu().numpy().argmax(axis=1)
        return probabilities.argmax(axis=1)


def _factuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x0": [-0.8, -0.7, -0.6, -0.5],
            "x1": [0.0, 0.1, -0.1, 0.05],
        }
    )


def _run_variant(external_cost_lambda: float) -> dict[str, float]:
    adapter = LinearLogitAdapter()
    factuals = _factuals()
    method = ClaPROAR(
        mlmodel=adapter,
        device="cpu",
        individual_cost_lambda=0.05,
        external_cost_lambda=external_cost_lambda,
        learning_rate=0.03,
        max_iter=300,
        target_class=1,
    )
    counterfactuals = method.get_counterfactuals(factuals, raw_output=True)
    predicted = adapter.predict(counterfactuals)
    target_center = np.asarray([1.0, 0.0], dtype="float64")
    cf_array = counterfactuals.to_numpy(dtype="float64")
    factual_array = factuals.to_numpy(dtype="float64")
    return {
        "validity": float(np.mean(predicted == 1)),
        "mean_recourse_distance": float(
            np.linalg.norm(cf_array - factual_array, axis=1).mean()
        ),
        "mean_target_center_distance": float(
            np.linalg.norm(cf_array - target_center.reshape(1, -1), axis=1).mean()
        ),
        "mean_positive_logit": float(adapter.predict_proba(counterfactuals)[:, 1].mean()),
    }


@pytest.mark.fast
def test_reproduce() -> None:
    """
    Scoped reproduction of the ClaPROAR mitigation claim.

    The original paper evaluates repeated retraining at population scale. This
    deterministic probe isolates the paper's classifier-preserving penalty:
    adding the external-cost term should keep counterfactuals valid while
    avoiding larger-than-needed movement into regions that would perturb the
    classifier more strongly when added back as target-class data.
    """
    baseline = _run_variant(external_cost_lambda=0.0)
    claproar = _run_variant(external_cost_lambda=0.5)

    assert baseline["validity"] == pytest.approx(1.0)
    assert claproar["validity"] == pytest.approx(1.0)
    assert claproar["mean_recourse_distance"] < baseline["mean_recourse_distance"]
    assert claproar["mean_positive_logit"] < baseline["mean_positive_logit"]

    write_reproduction_report(
        REPORT_PATH,
        paper_id="Endogenous Macrodynamics in Algorithmic Recourse",
        reproduction_metadata={
            "source_script": Path(__file__).name,
            "paper_reference": "reference/gravitaional-claproar.pdf",
            "paper_anchor": "ClaPROAR external classifier-preserving penalty, Eq. 6 and Section VII mitigation results",
            "scope": "Scoped deterministic toy check; not the full repeated-retraining simulation.",
            "timestamp_utc": datetime.now(timezone.utc),
        },
        experiments_data={
            "claproar_classifier_preserving_penalty": {
                "configuration": {
                    "baseline_external_cost_lambda": 0.0,
                    "claproar_external_cost_lambda": 0.5,
                    "individual_cost_lambda": 0.05,
                    "learning_rate": 0.03,
                    "max_iter": 300,
                    "num_factuals": len(_factuals()),
                },
                "metrics": {
                    "baseline_validity": {
                        "original": None,
                        "reproduced": baseline["validity"],
                    },
                    "claproar_validity": {
                        "original": None,
                        "reproduced": claproar["validity"],
                    },
                    "baseline_mean_recourse_distance": {
                        "original": None,
                        "reproduced": baseline["mean_recourse_distance"],
                    },
                    "claproar_mean_recourse_distance": {
                        "original": None,
                        "reproduced": claproar["mean_recourse_distance"],
                    },
                    "baseline_mean_positive_logit": {
                        "original": None,
                        "reproduced": baseline["mean_positive_logit"],
                    },
                    "claproar_mean_positive_logit": {
                        "original": None,
                        "reproduced": claproar["mean_positive_logit"],
                    },
                },
            }
        },
    )


if __name__ == "__main__":
    test_reproduce()
