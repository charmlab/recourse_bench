from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.utils import write_reproduction_report  # noqa: E402


REPORT_PATH = Path(__file__).with_name("reproduction_report.json")
TARGET_MODES = np.asarray([[1.0, -1.0], [1.0, 1.0]], dtype="float64")
SIGMA = 0.18


def _target_probability(point: np.ndarray) -> float:
    return float(1.0 / (1.0 + np.exp(-12.0 * (point[0] - 0.0))))


def _target_density(point: np.ndarray) -> float:
    point = np.asarray(point, dtype="float64")
    squared_distances = np.sum((TARGET_MODES - point.reshape(1, -1)) ** 2, axis=1)
    component_density = np.exp(-0.5 * squared_distances / (SIGMA**2))
    return float(component_density.mean())


def _nearest_boundary_recourse(factual: np.ndarray) -> np.ndarray:
    counterfactual = factual.copy()
    counterfactual[0] = 0.05
    return counterfactual


def _cruds_style_sampled_recourse(factual: np.ndarray) -> np.ndarray:
    candidates = np.asarray(
        [
            [1.0, -1.0],
            [1.0, 1.0],
            [0.95, -0.9],
            [0.95, 0.9],
        ],
        dtype="float64",
    )
    valid_candidates = candidates[
        np.asarray([_target_probability(candidate) > 0.5 for candidate in candidates])
    ]
    densities = np.asarray([_target_density(candidate) for candidate in valid_candidates])
    distances = np.linalg.norm(valid_candidates - factual.reshape(1, -1), axis=1)
    density_rank = densities / densities.max()
    score = density_rank - 0.05 * distances
    return valid_candidates[int(np.argmax(score))]


@pytest.mark.fast
def test_reproduce() -> None:
    """
    Scoped reproduction of CRUDS' low-density failure-mode claim.

    The original CRUDS code/data are unavailable. This deterministic mixture
    experiment mirrors the paper's Figure 1/Section 5 setup: a nearest-boundary
    counterfactual can flip the classifier while landing between target-class
    modes, whereas target-class sampling prefers high-density support.
    """
    factual = np.asarray([-1.0, 0.0], dtype="float64")
    boundary_cf = _nearest_boundary_recourse(factual)
    cruds_cf = _cruds_style_sampled_recourse(factual)

    boundary_validity = float(_target_probability(boundary_cf) > 0.5)
    cruds_validity = float(_target_probability(cruds_cf) > 0.5)
    boundary_density = _target_density(boundary_cf)
    cruds_density = _target_density(cruds_cf)
    boundary_distance = float(np.linalg.norm(boundary_cf - factual))
    cruds_distance = float(np.linalg.norm(cruds_cf - factual))

    assert boundary_validity == pytest.approx(1.0)
    assert cruds_validity == pytest.approx(1.0)
    assert cruds_density > boundary_density
    assert cruds_distance > boundary_distance

    write_reproduction_report(
        REPORT_PATH,
        paper_id="CRUDS: Counterfactual Recourse Using Disentangled Subspaces",
        reproduction_metadata={
            "source_script": Path(__file__).name,
            "paper_reference": "reference/cruds.pdf",
            "paper_anchor": "Section 3/Figure 1 low-density failure modes and Section 5 target-class sampling experiments",
            "scope": "Scoped synthetic mixture check; original CRUDS code/data are unavailable.",
            "timestamp_utc": datetime.now(timezone.utc),
        },
        experiments_data={
            "mixture_low_density_failure_mode": {
                "configuration": {
                    "factual": factual.tolist(),
                    "target_modes": TARGET_MODES.tolist(),
                    "target_density_sigma": SIGMA,
                    "nearest_boundary_counterfactual": boundary_cf.tolist(),
                    "cruds_style_counterfactual": cruds_cf.tolist(),
                },
                "metrics": {
                    "nearest_boundary_validity": {
                        "original": None,
                        "reproduced": boundary_validity,
                    },
                    "cruds_style_validity": {
                        "original": None,
                        "reproduced": cruds_validity,
                    },
                    "nearest_boundary_target_density": {
                        "original": None,
                        "reproduced": boundary_density,
                    },
                    "cruds_style_target_density": {
                        "original": None,
                        "reproduced": cruds_density,
                    },
                    "nearest_boundary_distance": {
                        "original": None,
                        "reproduced": boundary_distance,
                    },
                    "cruds_style_distance": {
                        "original": None,
                        "reproduced": cruds_distance,
                    },
                },
            }
        },
    )


if __name__ == "__main__":
    test_reproduce()
