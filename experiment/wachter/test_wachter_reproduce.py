from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.utils import write_reproduction_report  # noqa: E402
from method.wachter.library.wachter import wachter_recourse  # noqa: E402


REPORT_PATH = Path(__file__).with_name("reproduction_report.json")


class TinyRiskModel(torch.nn.Module):
    device = "cpu"

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        logits = 4.0 * values[:, 0] + values[:, 1] - 2.0
        positive_probability = torch.sigmoid(logits)
        return torch.stack([1.0 - positive_probability, positive_probability], dim=1)


def _positive_probability(model: TinyRiskModel, values: np.ndarray) -> float:
    tensor = torch.tensor(values.reshape(1, -1), dtype=torch.float32)
    return float(model(tensor).detach().cpu().numpy()[0, 1])


@pytest.mark.fast
def test_reproduce() -> None:
    """
    Scoped reproduction of Wachter-style closest counterfactual search.

    The paper's LSAT/Pima examples are not bundled with executable artifacts.
    This probe checks the optimizer property behind Equation 2 and the
    MAD-weighted L1 discussion: find a close target-class point while respecting
    feature-specific costs.
    """
    model = TinyRiskModel()
    factual = np.asarray([0.2, 0.2], dtype=np.float32)
    unweighted = wachter_recourse(
        torch_model=model,
        x=factual,
        categorical_groups=[],
        thermometer_groups=[],
        binary_feature_indices=[],
        feature_cost=None,
        lr=0.05,
        lambda_param=0.2,
        y_target=[0.0, 1.0],
        n_iter=400,
        t_max_min=0.05,
        norm=1,
        clamp=True,
        loss_type="BCE",
    )
    mad_weighted = wachter_recourse(
        torch_model=model,
        x=factual,
        categorical_groups=[],
        thermometer_groups=[],
        binary_feature_indices=[],
        feature_cost=[1.0, 8.0],
        lr=0.05,
        lambda_param=0.2,
        y_target=[0.0, 1.0],
        n_iter=400,
        t_max_min=0.05,
        norm=1,
        clamp=True,
        loss_type="BCE",
    )

    unweighted_delta = np.abs(unweighted - factual)
    mad_weighted_delta = np.abs(mad_weighted - factual)
    unweighted_probability = _positive_probability(model, unweighted)
    mad_weighted_probability = _positive_probability(model, mad_weighted)

    assert unweighted_probability > 0.5
    assert mad_weighted_probability > 0.5
    assert mad_weighted_delta[1] < unweighted_delta[1]
    assert mad_weighted_delta[0] > mad_weighted_delta[1]

    write_reproduction_report(
        REPORT_PATH,
        paper_id="Counterfactual Explanations Without Opening the Black Box",
        reproduction_metadata={
            "source_script": Path(__file__).name,
            "paper_reference": "reference/wachter.pdf",
            "paper_anchor": "Section III optimization objective and MAD-weighted L1 distance",
            "scope": "Scoped deterministic optimizer check; original LSAT/Pima experiment data are unavailable.",
            "timestamp_utc": datetime.now(timezone.utc),
        },
        experiments_data={
            "mad_weighted_l1_counterfactual": {
                "configuration": {
                    "factual": factual.tolist(),
                    "feature_cost": [1.0, 8.0],
                    "decision_threshold": 0.5,
                    "norm": 1,
                },
                "metrics": {
                    "unweighted_target_probability": {
                        "original": None,
                        "reproduced": unweighted_probability,
                    },
                    "mad_weighted_target_probability": {
                        "original": None,
                        "reproduced": mad_weighted_probability,
                    },
                    "unweighted_changed_feature_count": {
                        "original": None,
                        "reproduced": int(np.sum(unweighted_delta > 1e-6)),
                    },
                    "mad_weighted_changed_feature_count": {
                        "original": None,
                        "reproduced": int(np.sum(mad_weighted_delta > 1e-6)),
                    },
                    "unweighted_second_feature_change": {
                        "original": None,
                        "reproduced": float(unweighted_delta[1]),
                    },
                    "mad_weighted_second_feature_change": {
                        "original": None,
                        "reproduced": float(mad_weighted_delta[1]),
                    },
                },
            }
        },
    )


if __name__ == "__main__":
    test_reproduce()
