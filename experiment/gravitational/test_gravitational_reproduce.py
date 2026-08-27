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
from method.gravitational.model import Gravitational  # noqa: E402


REPORT_PATH = Path(__file__).with_name("reproduction_report.json")
FEATURE_NAMES = ["x0", "x1"]
TARGET_CENTER = np.asarray([1.0, 0.0], dtype=np.float32)


class _FeatureGroup:
    def __init__(self, index: int) -> None:
        self.indices = (index,)
        self.mutable = True
        self.actionability = "any"
        self.encoding_kind = "scalar"
        self.valid_codes = None
        self.feature_kind = "numerical"
        self.lower_bound = -4.0
        self.upper_bound = 4.0
        self.integer_valued = False


class _FeatureContext:
    def __init__(self) -> None:
        self.feature_names = tuple(FEATURE_NAMES)
        self.target_column = "target"
        self.groups = tuple(_FeatureGroup(index) for index in range(len(FEATURE_NAMES)))


class LinearLogitAdapter:
    def __init__(self) -> None:
        self.feature_input_order = list(FEATURE_NAMES)
        self.feature_context = _FeatureContext()
        self.data = type("Data", (), {"target": "target"})()
        self.device = "cpu"
        self.weight = torch.tensor([2.5, 0.8], dtype=torch.float32)
        self.bias = torch.tensor(-0.2, dtype=torch.float32)
        self.train_features = pd.DataFrame(
            {"x0": [0.9, 1.0, 1.1, 1.0], "x1": [-0.1, 0.0, 0.1, 0.05]}
        )
        self.train_labels = np.ones(self.train_features.shape[0], dtype=np.int64)

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

    def predict(self, values: pd.DataFrame | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        probabilities = self.predict_proba(values)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.argmax(dim=1)
        return probabilities.argmax(axis=1)


def _factuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x0": [-0.8, -0.7, -0.6, -0.5],
            "x1": [1.6, 1.4, 1.8, 1.5],
        }
    )


def _run_variant(grav_penalty_lambda: float) -> dict[str, float]:
    adapter = LinearLogitAdapter()
    factuals = _factuals()
    method = Gravitational(
        mlmodel=adapter,
        hyperparams={
            "prediction_loss_lambda": 1.0,
            "original_dist_lambda": 0.05,
            "grav_penalty_lambda": grav_penalty_lambda,
            "learning_rate": 0.03,
            "num_steps": 300,
            "target_class": 1,
            "scheduler_step_size": 100,
            "scheduler_gamma": 0.8,
        },
        x_center=TARGET_CENTER,
    )
    counterfactuals = method.get_counterfactuals(factuals)
    valid_mask = ~counterfactuals.isna().any(axis=1)
    cf_array = counterfactuals.loc[valid_mask].to_numpy(dtype="float64")
    factual_array = factuals.loc[valid_mask].to_numpy(dtype="float64")
    return {
        "validity": float(valid_mask.mean()),
        "mean_recourse_distance": float(
            np.linalg.norm(cf_array - factual_array, axis=1).mean()
        ),
        "mean_target_center_distance": float(
            np.linalg.norm(cf_array - TARGET_CENTER.reshape(1, -1), axis=1).mean()
        ),
    }


@pytest.mark.fast
def test_reproduce() -> None:
    """
    Scoped reproduction of the Gravitational recourse claim.

    The paper proposes penalizing distance to a representative target-domain
    point. This small check verifies the expected local trend: the
    gravitational penalty moves valid counterfactuals much closer to the target
    class center than a boundary-only variant.
    """
    boundary_only = _run_variant(grav_penalty_lambda=0.0)
    gravitational = _run_variant(grav_penalty_lambda=0.5)

    assert boundary_only["validity"] == pytest.approx(1.0)
    assert gravitational["validity"] == pytest.approx(1.0)
    assert (
        gravitational["mean_target_center_distance"]
        < boundary_only["mean_target_center_distance"]
    )

    write_reproduction_report(
        REPORT_PATH,
        paper_id="Endogenous Macrodynamics in Algorithmic Recourse",
        reproduction_metadata={
            "source_script": Path(__file__).name,
            "paper_reference": "reference/gravitaional-claproar.pdf",
            "paper_anchor": "Gravitational counterfactual penalty, Eq. 7 and Section VII mitigation results",
            "scope": "Scoped deterministic toy check; not the full repeated-retraining simulation.",
            "timestamp_utc": datetime.now(timezone.utc),
        },
        experiments_data={
            "gravitational_target_domain_penalty": {
                "configuration": {
                    "boundary_only_grav_penalty_lambda": 0.0,
                    "gravitational_grav_penalty_lambda": 0.5,
                    "target_center": TARGET_CENTER.tolist(),
                    "num_factuals": len(_factuals()),
                },
                "metrics": {
                    "boundary_only_validity": {
                        "original": None,
                        "reproduced": boundary_only["validity"],
                    },
                    "gravitational_validity": {
                        "original": None,
                        "reproduced": gravitational["validity"],
                    },
                    "boundary_only_mean_target_center_distance": {
                        "original": None,
                        "reproduced": boundary_only["mean_target_center_distance"],
                    },
                    "gravitational_mean_target_center_distance": {
                        "original": None,
                        "reproduced": gravitational["mean_target_center_distance"],
                    },
                    "boundary_only_mean_recourse_distance": {
                        "original": None,
                        "reproduced": boundary_only["mean_recourse_distance"],
                    },
                    "gravitational_mean_recourse_distance": {
                        "original": None,
                        "reproduced": gravitational["mean_recourse_distance"],
                    },
                },
            }
        },
    )


if __name__ == "__main__":
    test_reproduce()
