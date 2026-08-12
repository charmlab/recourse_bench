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


REPORT_PATH = Path(__file__).with_name("reproduction_report.json")


def _decode(latent: torch.Tensor) -> torch.Tensor:
    return torch.stack([latent, latent * latent], dim=1)


def _target_probability(features: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(8.0 * (features[:, 0] - 0.2))


def _manifold_residual(features: torch.Tensor) -> float:
    values = features.detach().cpu().numpy()[0]
    return float(abs(values[1] - values[0] ** 2))


def _optimize_in_input_space(factual: torch.Tensor) -> torch.Tensor:
    candidate = factual.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        probability = _target_probability(candidate)
        loss = torch.nn.functional.binary_cross_entropy(
            probability,
            torch.ones_like(probability),
        ) + 0.05 * torch.norm(candidate - factual)
        loss.backward()
        optimizer.step()
    return candidate.detach()


def _optimize_in_latent_space(factual_z: torch.Tensor, factual: torch.Tensor) -> torch.Tensor:
    latent = factual_z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        candidate = _decode(latent)
        probability = _target_probability(candidate)
        loss = torch.nn.functional.binary_cross_entropy(
            probability,
            torch.ones_like(probability),
        ) + 0.05 * torch.norm(candidate - factual)
        loss.backward()
        optimizer.step()
    return _decode(latent).detach()


@pytest.mark.fast
def test_reproduce() -> None:
    """
    Scoped reproduction of the REVISE latent-manifold recourse claim.

    The original paper's credit and causal-system artifacts are unavailable.
    This deterministic parabola fixture checks the central mechanism: optimize
    through a latent decoder so counterfactuals remain on the learned manifold.
    """
    factual_z = torch.tensor([-0.6], dtype=torch.float32)
    factual = _decode(factual_z).detach()

    input_space_cf = _optimize_in_input_space(factual)
    latent_space_cf = _optimize_in_latent_space(factual_z, factual)

    input_probability = float(_target_probability(input_space_cf).item())
    latent_probability = float(_target_probability(latent_space_cf).item())
    input_residual = _manifold_residual(input_space_cf)
    latent_residual = _manifold_residual(latent_space_cf)
    input_distance = float(torch.norm(input_space_cf - factual).item())
    latent_distance = float(torch.norm(latent_space_cf - factual).item())

    assert input_probability > 0.5
    assert latent_probability > 0.5
    assert latent_residual == pytest.approx(0.0)
    assert latent_residual < input_residual

    write_reproduction_report(
        REPORT_PATH,
        paper_id="Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems",
        reproduction_metadata={
            "source_script": Path(__file__).name,
            "paper_reference": "reference/revise.pdf",
            "paper_anchor": "Algorithm 1 and Section 3 latent-space optimization over a generative model",
            "scope": "Scoped synthetic manifold check; original credit and causal experiments are unavailable.",
            "timestamp_utc": datetime.now(timezone.utc),
        },
        experiments_data={
            "latent_manifold_counterfactual": {
                "configuration": {
                    "factual_latent": factual_z.tolist(),
                    "decoder": "x = [z, z^2]",
                    "decision_rule": "sigmoid(8 * (x0 - 0.2))",
                    "optimizer_steps": 200,
                },
                "metrics": {
                    "input_space_target_probability": {
                        "original": None,
                        "reproduced": input_probability,
                    },
                    "latent_space_target_probability": {
                        "original": None,
                        "reproduced": latent_probability,
                    },
                    "input_space_manifold_residual": {
                        "original": None,
                        "reproduced": input_residual,
                    },
                    "latent_space_manifold_residual": {
                        "original": 0,
                        "reproduced": latent_residual,
                    },
                    "input_space_distance": {
                        "original": None,
                        "reproduced": input_distance,
                    },
                    "latent_space_distance": {
                        "original": None,
                        "reproduced": latent_distance,
                    },
                },
            }
        },
    )


if __name__ == "__main__":
    test_reproduce()
