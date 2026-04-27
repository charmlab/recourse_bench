from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Trigger local registrations.
import dataset  # noqa: E402,F401
import method  # noqa: E402,F401
import model  # noqa: E402,F401
import preprocess  # noqa: E402,F401
from dataset.compas_clue.compas_clue import CompasClueDataset  # noqa: E402
from method.clue.clue import ClueMethod  # noqa: E402
from method.clue.library.clue_ml.interpret.FIDO import mask_explainer  # noqa: E402
from method.clue.library.clue_ml.interpret.functionally_grounded import (  # noqa: E402
    evaluate_aleatoric_explanation_cat,
    evaluate_epistemic_explanation_cat,
    get_BNN_uncertainties,
    get_VAEAC_px_gauss_cat,
)
from method.clue.library.clue_ml.interpret.generate_data import (
    sample_artificial_dataset,
)  # noqa: E402
from method.clue.library.clue_ml.src.gauss_cat import selective_softmax  # noqa: E402
from method.clue.library.clue_ml.src.utils import (  # noqa: E402
    generate_ind_batch,
    register_checkpoint_aliases,
)
from method.clue.library.clue_ml.VAEAC.fc_gauss_cat import (
    VAEAC_gauss_cat_net,
)  # noqa: E402
from method.clue.library.clue_ml.VAEAC.under_net import under_VAEAC  # noqa: E402
from method.clue.support import resolve_vae_checkpoint_path  # noqa: E402
from model.mlp_bayesian.mlp_bayesian import MlpBayesianModel  # noqa: E402
from preprocess.common import (
    EncodePreProcess,  # noqa: E402
    FinalizePreProcess,
    ReorderPreProcess,
    ScalePreProcess,
    SplitPreProcess,
)

LOGGER = logging.getLogger("clue-reproduce")
SEED = 42
EXPECTED_TRAIN = 5554
EXPECTED_TEST = 618
COMPAS_OFFICIAL_ORDER = [
    "age_cat_cat_25 - 45",
    "age_cat_cat_Greater than 45",
    "age_cat_cat_Less than 25",
    "race_cat_African-American",
    "race_cat_Asian",
    "race_cat_Caucasian",
    "race_cat_Hispanic",
    "race_cat_Native American",
    "race_cat_Other",
    "sex_cat_Female",
    "sex_cat_Male",
    "c_charge_degree_cat_F",
    "c_charge_degree_cat_M",
    "is_recid_cat_0",
    "is_recid_cat_1",
    "priors_count",
    "time_served",
]
COMPAS_INPUT_DIM_VEC = [3, 6, 2, 2, 2, 1, 1]
COMPAS_INPUT_DIM_VEC_XY = [3, 6, 2, 2, 2, 1, 1, 2]
COMPAS_TEST_DIMS = [17, 18]
COMPAS_ARTIFICIAL_TRAIN = 5554
COMPAS_ARTIFICIAL_TEST = 618
COMPAS_ARTIFICIAL_POINTS = COMPAS_ARTIFICIAL_TRAIN + COMPAS_ARTIFICIAL_TEST
COMPAS_ART_ALEATORIC_THRESHOLD = 0.15
COMPAS_ART_EPISTEMIC_THRESHOLD = 0.02
COMPAS_CLUE_LR = 0.1
COMPAS_CLUE_LAMBDAS = np.logspace(np.log(0.0001), np.log(30), 30, base=np.e)
COMPAS_FIDO_LAMBDAS = np.logspace(np.log(0.00001), np.log(100), 30, base=np.e)
COMPAS_FIDO_BATCH_SIZE = 3000
COMPAS_FIDO_EPOCHS = 30
COMPAS_FIDO_MASK_SAMPLES = 20
COMPAS_FIDO_MASK_SAMPLES2 = 10
COMPAS_PAPER_RESULTS = {
    "table1_epistemic": 0.71,
    "table1_aleatoric": 0.18,
    "table2_epistemic": 0.707,
    "table2_aleatoric": 0.044,
}
COMPAS_PAPER_TOLERANCES = {
    "table1_epistemic": 0.01,
    "table1_aleatoric": 0.05,
    "table2_epistemic": 0.01,
    "table2_aleatoric": 0.05,
}


class _OfficialStyleBNNAdapter:
    def __init__(self, model: MlpBayesianModel):
        self.model = model

    def sample_predict(self, x, Nsamples=0, grad=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(
                x,
                dtype=torch.float32,
                device=self.model._device,
            )
        return self.model.sample_predict(x, Nsamples=Nsamples, grad=grad)

    def set_mode_train(self, train=False):
        del train
        return None


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return requested


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_compas_train_schema(seed: int) -> dict[str, Any]:
    dataset = CompasClueDataset()
    trainset, testset = SplitPreProcess(seed=seed, split=0.1).transform(dataset)

    scale = ScalePreProcess(
        seed=seed,
        scaling="standardize",
        range=True,
        refset=trainset,
    )
    trainset = scale.transform(trainset)
    testset = scale.transform(testset)

    encode = EncodePreProcess(seed=seed, encoding="onehot")
    trainset = encode.transform(trainset)
    testset = encode.transform(testset)

    reorder = ReorderPreProcess(seed=seed, order=COMPAS_OFFICIAL_ORDER)
    trainset = reorder.transform(trainset)
    testset = reorder.transform(testset)

    finalize = FinalizePreProcess(seed=seed)
    trainset = finalize.transform(trainset)
    testset = finalize.transform(testset)

    train_features = trainset.get(target=False)
    test_features = testset.get(target=False)
    if list(train_features.columns) != COMPAS_OFFICIAL_ORDER:
        raise AssertionError(
            "Finalized train feature order does not match OFFICIAL_ORDER"
        )
    if train_features.shape[1] != 17 or test_features.shape[1] != 17:
        raise AssertionError("Finalized COMPAS feature dimension must be 17")
    if len(trainset) != EXPECTED_TRAIN:
        raise AssertionError(
            f"Expected COMPAS train size {EXPECTED_TRAIN}, observed {len(trainset)}"
        )
    if len(testset) != EXPECTED_TEST:
        raise AssertionError(
            f"Expected COMPAS test size {EXPECTED_TEST}, observed {len(testset)}"
        )

    return {
        "trainset": trainset,
        "testset": testset,
    }


def _load_compas_reproduction_models(
    target_model: MlpBayesianModel,
    device: str,
    *,
    vaeac_art_path: str | Path,
    vaeac_gt_path: str | Path,
    under_vaeac_gt_path: str | Path,
) -> dict[str, Any]:
    cuda = device == "cuda"
    register_checkpoint_aliases()

    resolved_vaeac_art = resolve_vae_checkpoint_path(str(vaeac_art_path))
    resolved_vaeac_gt = resolve_vae_checkpoint_path(str(vaeac_gt_path))
    resolved_under = resolve_vae_checkpoint_path(str(under_vaeac_gt_path))
    for label, path in {
        "vaeac_art_path": resolved_vaeac_art,
        "vaeac_gt_path": resolved_vaeac_gt,
        "under_vaeac_gt_path": resolved_under,
    }.items():
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing required checkpoint for {label}: {path}")

    vaeac_gt = VAEAC_gauss_cat_net(
        COMPAS_INPUT_DIM_VEC_XY,
        350,
        3,
        4,
        pred_sig=False,
        lr=3e-4,
        cuda=cuda,
        flatten=False,
    )
    vaeac_gt.load(resolved_vaeac_gt.as_posix())

    under_vaeac_gt = under_VAEAC(
        vaeac_gt.model,
        150,
        2,
        4,
        lr=3e-4,
        cuda=cuda,
    )
    under_vaeac_gt.load(resolved_under.as_posix())

    vaeac_art = VAEAC_gauss_cat_net(
        COMPAS_INPUT_DIM_VEC,
        350,
        3,
        4,
        pred_sig=False,
        lr=1e-4,
        cuda=cuda,
        flatten=False,
    )
    vaeac_art.load(resolved_vaeac_art.as_posix())

    return {
        "art_bnn": _OfficialStyleBNNAdapter(target_model),
        "vaeac_art": vaeac_art,
        "vaeac_gt": vaeac_gt,
        "under_vaeac_gt": under_vaeac_gt,
    }


def _generate_artificial_compas(
    under_vaeac_gt: under_VAEAC,
) -> dict[str, np.ndarray]:
    x_art, y_art, xy_art = sample_artificial_dataset(
        under_vaeac_gt,
        COMPAS_TEST_DIMS,
        COMPAS_ARTIFICIAL_POINTS,
        u_dims=4,
        sig=False,
        softmax=False,
    )
    x_art = selective_softmax(
        x_art,
        COMPAS_INPUT_DIM_VEC,
        grad=False,
        cat_probs=True,
        prob_sample=True,
    )
    xy_art = selective_softmax(
        xy_art,
        COMPAS_INPUT_DIM_VEC_XY,
        grad=False,
        cat_probs=True,
        prob_sample=True,
    )
    y_art = selective_softmax(
        y_art,
        [2],
        grad=False,
        cat_probs=True,
        prob_sample=True,
    )
    _, y_art_bnn = y_art.max(dim=1)

    x_art_test = x_art[COMPAS_ARTIFICIAL_TRAIN:, :].cpu().numpy()
    y_art_test = y_art[COMPAS_ARTIFICIAL_TRAIN:, :].cpu().numpy()
    y_art_bnn_test = y_art_bnn[COMPAS_ARTIFICIAL_TRAIN:].cpu().numpy()

    if x_art_test.shape != (COMPAS_ARTIFICIAL_TEST, 17):
        raise AssertionError(
            "Expected artificial COMPAS test shape "
            f"({COMPAS_ARTIFICIAL_TEST}, 17), observed {x_art_test.shape}"
        )
    if y_art_test.shape != (COMPAS_ARTIFICIAL_TEST, 2):
        raise AssertionError(
            "Expected artificial COMPAS target shape "
            f"({COMPAS_ARTIFICIAL_TEST}, 2), observed {y_art_test.shape}"
        )

    return {
        "x_art_test": x_art_test,
        "y_art_test": y_art_test,
        "y_art_bnn_test": y_art_bnn_test,
    }


def _compute_artificial_baselines(
    art_data: dict[str, np.ndarray],
    models: dict[str, Any],
    device: str,
    *,
    vaeac_samples: int,
    likelihood_samples: int,
) -> dict[str, Any]:
    x_art_test = art_data["x_art_test"]

    _, aleatoric_stack, epistemic_stack = get_BNN_uncertainties(
        models["art_bnn"],
        x_art_test,
        regression=False,
        batch_size=1024,
        norm_MNIST=False,
        flatten=False,
        return_probs=False,
        prob_BNN=True,
    )
    aleatoric_mask = (
        aleatoric_stack.detach().cpu().numpy() >= COMPAS_ART_ALEATORIC_THRESHOLD
    )
    epistemic_mask = (
        epistemic_stack.detach().cpu().numpy() >= COMPAS_ART_EPISTEMIC_THRESHOLD
    )
    if not bool(aleatoric_mask.any()):
        raise AssertionError(
            "Aleatoric threshold selected zero artificial COMPAS samples"
        )
    if not bool(epistemic_mask.any()):
        raise AssertionError(
            "Epistemic threshold selected zero artificial COMPAS samples"
        )

    og_aleatoric_uncert = evaluate_aleatoric_explanation_cat(
        models["vaeac_gt"],
        torch.tensor(x_art_test[aleatoric_mask], dtype=torch.float32),
        COMPAS_TEST_DIMS,
        N_target_samples=vaeac_samples,
        batch_size=1024,
    )
    gt_test_err, _ = evaluate_epistemic_explanation_cat(
        models["art_bnn"],
        models["vaeac_gt"],
        torch.tensor(
            x_art_test[epistemic_mask],
            dtype=torch.float32,
            device=device,
        ),
        COMPAS_TEST_DIMS,
        outer_batch_size=2000,
        inner_batch_size=1024,
        VAEAC_samples=vaeac_samples,
    )
    log_px_aleatoric = get_VAEAC_px_gauss_cat(
        models["under_vaeac_gt"],
        x_art_test[aleatoric_mask],
        COMPAS_INPUT_DIM_VEC_XY,
        COMPAS_TEST_DIMS,
        override_y_dims=1,
        Nsamples=likelihood_samples,
    )
    log_px_epistemic = get_VAEAC_px_gauss_cat(
        models["under_vaeac_gt"],
        x_art_test[epistemic_mask],
        COMPAS_INPUT_DIM_VEC_XY,
        COMPAS_TEST_DIMS,
        override_y_dims=1,
        Nsamples=likelihood_samples,
    )

    return {
        "aleatoric_mask": aleatoric_mask,
        "epistemic_mask": epistemic_mask,
        "og_aleatoric_uncert": og_aleatoric_uncert,
        "gt_test_err": float(gt_test_err),
        "baseline_logpx_aleatoric": float(log_px_aleatoric.mean().cpu().item()),
        "baseline_logpx_epistemic": float(log_px_epistemic.mean().cpu().item()),
    }


def _compute_clue_curve(
    clue_method: ClueMethod,
    x_init_batch: np.ndarray,
    *,
    mode: str,
    models: dict[str, Any],
    baselines: dict[str, Any],
    clue_lambdas: np.ndarray,
    device: str,
    vaeac_samples: int,
    likelihood_samples: int,
) -> dict[str, np.ndarray]:
    x_dim = x_init_batch.shape[1]
    factual_frame = pd.DataFrame(x_init_batch, columns=COMPAS_OFFICIAL_ORDER)
    delta_x_values: list[float] = []
    benefit_values: list[float] = []
    logpx_values: list[float] = []

    if mode == "aleatoric":
        max_steps = 55
        aleatoric_weight = 1.0
        epistemic_weight = 0.0
    elif mode == "epistemic":
        max_steps = 65
        aleatoric_weight = 0.0
        epistemic_weight = 1.0
    else:
        raise ValueError(f"Unknown CLUE mode: {mode}")

    for distance_weight in tqdm(
        clue_lambdas / x_dim,
        desc=f"clue-{mode}",
        leave=False,
    ):
        x_clue_df = clue_method.get_counterfactuals_clue(
            factual_frame,
            uncertainty_weight=0.0,
            aleatoric_weight=aleatoric_weight,
            epistemic_weight=epistemic_weight,
            prior_weight=0.0,
            distance_weight=float(distance_weight),
            latent_l2_weight=0.0,
            prediction_similarity_weight=0.0,
            lr=COMPAS_CLUE_LR,
            min_steps=3,
            max_steps=max_steps,
            early_stop_patience=3,
            num_explanations=1,
        )
        if list(x_clue_df.columns) != COMPAS_OFFICIAL_ORDER:
            raise AssertionError("CLUE raw output columns deviated from OFFICIAL_ORDER")
        if x_clue_df.shape != factual_frame.shape:
            raise AssertionError(
                f"CLUE raw output shape mismatch: expected {factual_frame.shape}, "
                f"observed {x_clue_df.shape}"
            )
        x_clue = x_clue_df.to_numpy(dtype="float32", copy=True)
        if np.isnan(x_clue).any():
            raise AssertionError(f"Raw CLUE output contains NaN values for mode={mode}")

        delta_x_values.append(float(np.abs(x_init_batch - x_clue).sum(axis=1).mean()))
        if mode == "aleatoric":
            clue_uncert = evaluate_aleatoric_explanation_cat(
                models["vaeac_gt"],
                torch.tensor(x_clue, dtype=torch.float32),
                COMPAS_TEST_DIMS,
                N_target_samples=vaeac_samples,
                batch_size=1024,
            )
            benefit_values.append(
                float(
                    (baselines["og_aleatoric_uncert"] - clue_uncert)
                    .detach()
                    .cpu()
                    .numpy()
                    .mean()
                )
            )
            log_px = get_VAEAC_px_gauss_cat(
                models["under_vaeac_gt"],
                x_clue,
                COMPAS_INPUT_DIM_VEC_XY,
                COMPAS_TEST_DIMS,
                override_y_dims=1,
                Nsamples=likelihood_samples,
            )
        else:
            clue_test_err, _ = evaluate_epistemic_explanation_cat(
                models["art_bnn"],
                models["vaeac_gt"],
                torch.tensor(x_clue, dtype=torch.float32, device=device),
                COMPAS_TEST_DIMS,
                outer_batch_size=2000,
                inner_batch_size=1024,
                VAEAC_samples=vaeac_samples,
            )
            benefit_values.append(float(baselines["gt_test_err"] - clue_test_err))
            log_px = get_VAEAC_px_gauss_cat(
                models["under_vaeac_gt"],
                x_clue,
                COMPAS_INPUT_DIM_VEC_XY,
                COMPAS_TEST_DIMS,
                override_y_dims=1,
                Nsamples=likelihood_samples,
            )
        logpx_values.append(float(log_px.mean().cpu().item()))

    return {
        "delta_x": np.asarray(delta_x_values, dtype=np.float64),
        "benefit": np.asarray(benefit_values, dtype=np.float64),
        "logpx": np.asarray(logpx_values, dtype=np.float64),
    }


def _compute_fido_curve(
    x_init_batch: np.ndarray,
    *,
    mode: str,
    models: dict[str, Any],
    baselines: dict[str, Any],
    device: str,
    fido_lambdas: np.ndarray,
    fido_batch_size: int,
    fido_epochs: int,
    fido_mask_samples: int,
    fido_mask_samples2: int,
    vaeac_samples: int,
    likelihood_samples: int,
) -> dict[str, np.ndarray]:
    delta_x_values: list[float] = []
    benefit_values: list[float] = []
    logpx_values: list[float] = []

    for l1_weight in tqdm(fido_lambdas, desc=f"fido-{mode}", leave=False):
        explanations = []
        aleatoric_coeff = 1 if mode == "aleatoric" else 0
        epistemic_coeff = 1 if mode == "epistemic" else 0
        aux_loader = generate_ind_batch(
            x_init_batch.shape[0],
            fido_batch_size,
            random=False,
            roundup=True,
        )
        for indices in aux_loader:
            explainer, _, _, _ = mask_explainer.train_mask(
                torch.tensor(x_init_batch[indices], dtype=torch.float32),
                models["art_bnn"],
                models["vaeac_art"],
                aleatoric_coeff=aleatoric_coeff,
                epistemic_coeff=epistemic_coeff,
                L1w=float(l1_weight),
                N_epochs=fido_epochs,
                mask_samples=fido_mask_samples,
                mask_samples2=fido_mask_samples2,
                flatten_ims=False,
                test_dims=None,
                cat=True,
                cuda=device == "cuda",
            )
            explanation, _ = explainer.mask_inpaint(
                torch.tensor(x_init_batch[indices], dtype=torch.float32),
                models["vaeac_art"],
                flatten_ims=False,
                test_dims=None,
                cat=True,
            )
            explanations.append(explanation.cpu())

        x_fido = torch.cat(explanations, dim=0).numpy()
        delta_x_values.append(float(np.abs(x_init_batch - x_fido).sum(axis=1).mean()))
        if mode == "aleatoric":
            fido_uncert = evaluate_aleatoric_explanation_cat(
                models["vaeac_gt"],
                torch.tensor(x_fido, dtype=torch.float32),
                COMPAS_TEST_DIMS,
                N_target_samples=vaeac_samples,
                batch_size=1024,
            )
            benefit_values.append(
                float(
                    (baselines["og_aleatoric_uncert"] - fido_uncert)
                    .detach()
                    .cpu()
                    .numpy()
                    .mean()
                )
            )
            log_px = get_VAEAC_px_gauss_cat(
                models["under_vaeac_gt"],
                x_fido,
                COMPAS_INPUT_DIM_VEC_XY,
                COMPAS_TEST_DIMS,
                override_y_dims=1,
                Nsamples=likelihood_samples,
            )
        else:
            fido_test_err, _ = evaluate_epistemic_explanation_cat(
                models["art_bnn"],
                models["vaeac_gt"],
                torch.tensor(x_fido, dtype=torch.float32, device=device),
                COMPAS_TEST_DIMS,
                outer_batch_size=2000,
                inner_batch_size=1024,
                VAEAC_samples=vaeac_samples,
            )
            benefit_values.append(float(baselines["gt_test_err"] - fido_test_err))
            log_px = get_VAEAC_px_gauss_cat(
                models["under_vaeac_gt"],
                x_fido,
                COMPAS_INPUT_DIM_VEC_XY,
                COMPAS_TEST_DIMS,
                override_y_dims=1,
                Nsamples=likelihood_samples,
            )
        logpx_values.append(float(log_px.mean().cpu().item()))

    return {
        "delta_x": np.asarray(delta_x_values, dtype=np.float64),
        "benefit": np.asarray(benefit_values, dtype=np.float64),
        "logpx": np.asarray(logpx_values, dtype=np.float64),
    }


def _save_curve(
    output_path: Path,
    clue_x: np.ndarray,
    clue_y: np.ndarray,
    fido_x: np.ndarray,
    fido_y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    baseline_x: float | None = None,
    invert_x: bool = False,
) -> None:
    plt.figure(dpi=100)
    plt.plot(clue_x, clue_y, "--.", c="r")
    plt.plot(fido_x, fido_y, "--.", c="g")
    if baseline_x is not None:
        ax = plt.gca()
        ax.axvline(x=baseline_x, c="k", linewidth=1.0)
        if invert_x:
            ax.invert_xaxis()
    plt.title(title)
    plt.legend(["CLUE", "U-FIDO"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def _table1_scalar(
    clue_x: np.ndarray,
    clue_benefit: np.ndarray,
    fido_x: np.ndarray,
    fido_benefit: np.ndarray,
) -> float:
    max_x = float(max(np.max(clue_x), np.max(fido_x)))
    max_benefit = float(max(np.max(clue_benefit), np.max(fido_benefit)))
    if max_benefit <= 0 or not np.isfinite(max_benefit):
        return float(1 / np.sqrt(2.0))
    if max_x <= 0 or not np.isfinite(max_x):
        return float(1 / np.sqrt(2.0))
    x_scaled = clue_x / (np.sqrt(2.0) * max_x)
    y_scaled = (max_benefit - clue_benefit) / (np.sqrt(2.0) * max_benefit)
    return float(np.sqrt(x_scaled**2 + y_scaled**2).min())


def _table2_scalar(
    clue_logpx: np.ndarray,
    clue_benefit: np.ndarray,
    fido_logpx: np.ndarray,
    fido_benefit: np.ndarray,
    baseline_logpx: float,
) -> float:
    clue_cost = np.maximum(0.0, baseline_logpx - clue_logpx)
    fido_cost = np.maximum(0.0, baseline_logpx - fido_logpx)
    max_cost = float(max(np.max(clue_cost), np.max(fido_cost)))
    max_benefit = float(max(np.max(clue_benefit), np.max(fido_benefit)))
    if max_benefit <= 0 or not np.isfinite(max_benefit):
        return float(1 / np.sqrt(2.0))
    if max_cost <= 0 or not np.isfinite(max_cost):
        return float(1 / np.sqrt(2.0))
    x_scaled = clue_cost / (np.sqrt(2.0) * max_cost)
    y_scaled = (max_benefit - clue_benefit) / (np.sqrt(2.0) * max_benefit)
    return float(np.sqrt(x_scaled**2 + y_scaled**2).min())


def _run_compas_reproduction(
    clue_method: ClueMethod,
    target_model: MlpBayesianModel,
    device: str,
    output_dir: str | Path,
    *,
    vaeac_art_path: str | Path,
    vaeac_gt_path: str | Path,
    under_vaeac_gt_path: str | Path,
    assert_paper: bool = True,
    clue_lambdas: np.ndarray | None = None,
    fido_lambdas: np.ndarray | None = None,
    vaeac_samples: int = 500,
    likelihood_samples: int = 1000,
    fido_batch_size: int = COMPAS_FIDO_BATCH_SIZE,
    fido_epochs: int = COMPAS_FIDO_EPOCHS,
    fido_mask_samples: int = COMPAS_FIDO_MASK_SAMPLES,
    fido_mask_samples2: int = COMPAS_FIDO_MASK_SAMPLES2,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(SEED)
    clue_lambdas = (
        COMPAS_CLUE_LAMBDAS if clue_lambdas is None else np.asarray(clue_lambdas)
    )
    fido_lambdas = (
        COMPAS_FIDO_LAMBDAS if fido_lambdas is None else np.asarray(fido_lambdas)
    )

    LOGGER.info("Loading COMPAS reproduction helpers on device=%s", device)
    models = _load_compas_reproduction_models(
        target_model,
        device,
        vaeac_art_path=vaeac_art_path,
        vaeac_gt_path=vaeac_gt_path,
        under_vaeac_gt_path=under_vaeac_gt_path,
    )
    art_data = _generate_artificial_compas(models["under_vaeac_gt"])
    baselines = _compute_artificial_baselines(
        art_data,
        models,
        device,
        vaeac_samples=vaeac_samples,
        likelihood_samples=likelihood_samples,
    )
    LOGGER.info(
        "Selected artificial subsets: aleatoric=%d epistemic=%d",
        int(baselines["aleatoric_mask"].sum()),
        int(baselines["epistemic_mask"].sum()),
    )

    clue_aleatoric = _compute_clue_curve(
        clue_method,
        art_data["x_art_test"][baselines["aleatoric_mask"]],
        mode="aleatoric",
        models=models,
        baselines=baselines,
        clue_lambdas=clue_lambdas,
        device=device,
        vaeac_samples=vaeac_samples,
        likelihood_samples=likelihood_samples,
    )
    clue_epistemic = _compute_clue_curve(
        clue_method,
        art_data["x_art_test"][baselines["epistemic_mask"]],
        mode="epistemic",
        models=models,
        baselines=baselines,
        clue_lambdas=clue_lambdas,
        device=device,
        vaeac_samples=vaeac_samples,
        likelihood_samples=likelihood_samples,
    )
    fido_aleatoric = _compute_fido_curve(
        art_data["x_art_test"][baselines["aleatoric_mask"]],
        mode="aleatoric",
        models=models,
        baselines=baselines,
        device=device,
        fido_lambdas=fido_lambdas,
        fido_batch_size=fido_batch_size,
        fido_epochs=fido_epochs,
        fido_mask_samples=fido_mask_samples,
        fido_mask_samples2=fido_mask_samples2,
        vaeac_samples=vaeac_samples,
        likelihood_samples=likelihood_samples,
    )
    fido_epistemic = _compute_fido_curve(
        art_data["x_art_test"][baselines["epistemic_mask"]],
        mode="epistemic",
        models=models,
        baselines=baselines,
        device=device,
        fido_lambdas=fido_lambdas,
        fido_batch_size=fido_batch_size,
        fido_epochs=fido_epochs,
        fido_mask_samples=fido_mask_samples,
        fido_mask_samples2=fido_mask_samples2,
        vaeac_samples=vaeac_samples,
        likelihood_samples=likelihood_samples,
    )

    _save_curve(
        output_dir / "compas_epistemic_delta_x.png",
        clue_epistemic["delta_x"],
        -clue_epistemic["benefit"],
        fido_epistemic["delta_x"],
        -fido_epistemic["benefit"],
        "compas epistemic",
        "L1 change in X",
        "change in err",
    )
    _save_curve(
        output_dir / "compas_epistemic_logpx.png",
        clue_epistemic["logpx"],
        -clue_epistemic["benefit"],
        fido_epistemic["logpx"],
        -fido_epistemic["benefit"],
        "compas epistemic",
        "log p(x)",
        "change in err",
        baseline_x=baselines["baseline_logpx_epistemic"],
        invert_x=True,
    )
    _save_curve(
        output_dir / "compas_aleatoric_delta_x.png",
        clue_aleatoric["delta_x"],
        -clue_aleatoric["benefit"],
        fido_aleatoric["delta_x"],
        -fido_aleatoric["benefit"],
        "compas aleatoric",
        "L1 change in X",
        "change in H",
    )
    _save_curve(
        output_dir / "compas_aleatoric_logpx.png",
        clue_aleatoric["logpx"],
        -clue_aleatoric["benefit"],
        fido_aleatoric["logpx"],
        -fido_aleatoric["benefit"],
        "compas aleatoric",
        "log p(x)",
        "change in H",
        baseline_x=baselines["baseline_logpx_aleatoric"],
        invert_x=True,
    )
    np.savez_compressed(
        output_dir / "compas_curves.npz",
        clue_epistemic_delta_x=clue_epistemic["delta_x"],
        clue_epistemic_logpx=clue_epistemic["logpx"],
        clue_epistemic_benefit=clue_epistemic["benefit"],
        fido_epistemic_delta_x=fido_epistemic["delta_x"],
        fido_epistemic_logpx=fido_epistemic["logpx"],
        fido_epistemic_benefit=fido_epistemic["benefit"],
        clue_aleatoric_delta_x=clue_aleatoric["delta_x"],
        clue_aleatoric_logpx=clue_aleatoric["logpx"],
        clue_aleatoric_benefit=clue_aleatoric["benefit"],
        fido_aleatoric_delta_x=fido_aleatoric["delta_x"],
        fido_aleatoric_logpx=fido_aleatoric["logpx"],
        fido_aleatoric_benefit=fido_aleatoric["benefit"],
    )

    comparison = {
        "table1_epistemic": _table1_scalar(
            clue_epistemic["delta_x"],
            clue_epistemic["benefit"],
            fido_epistemic["delta_x"],
            fido_epistemic["benefit"],
        ),
        "table1_aleatoric": _table1_scalar(
            clue_aleatoric["delta_x"],
            clue_aleatoric["benefit"],
            fido_aleatoric["delta_x"],
            fido_aleatoric["benefit"],
        ),
        "table2_epistemic": _table2_scalar(
            clue_epistemic["logpx"],
            clue_epistemic["benefit"],
            fido_epistemic["logpx"],
            fido_epistemic["benefit"],
            baselines["baseline_logpx_epistemic"],
        ),
        "table2_aleatoric": _table2_scalar(
            clue_aleatoric["logpx"],
            clue_aleatoric["benefit"],
            fido_aleatoric["logpx"],
            fido_aleatoric["benefit"],
            baselines["baseline_logpx_aleatoric"],
        ),
    }
    abs_errors = {
        key: abs(value - COMPAS_PAPER_RESULTS[key]) for key, value in comparison.items()
    }
    summary = {
        "device": device,
        "output_dir": output_dir.as_posix(),
        "paper_style_targets": COMPAS_PAPER_RESULTS,
        "paper_style_comparison": comparison,
        "paper_style_abs_errors": abs_errors,
        "selected_counts": {
            "aleatoric": int(baselines["aleatoric_mask"].sum()),
            "epistemic": int(baselines["epistemic_mask"].sum()),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    expected_outputs = [
        output_dir / "compas_epistemic_delta_x.png",
        output_dir / "compas_epistemic_logpx.png",
        output_dir / "compas_aleatoric_delta_x.png",
        output_dir / "compas_aleatoric_logpx.png",
        output_dir / "compas_curves.npz",
        output_dir / "summary.json",
    ]
    missing_outputs = [
        path.as_posix() for path in expected_outputs if not path.exists()
    ]
    if missing_outputs:
        raise AssertionError(
            "Missing expected reproduction artifacts:\n" + "\n".join(missing_outputs)
        )

    if assert_paper:
        failing_metrics = [
            (
                metric,
                comparison[metric],
                COMPAS_PAPER_RESULTS[metric],
                COMPAS_PAPER_TOLERANCES[metric],
            )
            for metric in COMPAS_PAPER_RESULTS
            if abs_errors[metric] > COMPAS_PAPER_TOLERANCES[metric]
        ]
        if failing_metrics:
            formatted = "\n".join(
                f"{metric}: observed={observed:.6f}, target={target:.6f}, tol={tol:.6f}"
                for metric, observed, target, tol in failing_metrics
            )
            raise AssertionError(
                "COMPAS CLUE reproduction deviated from expected paper tolerances:\n"
                + formatted
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bnn-art-path", type=Path, required=True)
    parser.add_argument("--vae-art-path", type=Path, required=True)
    parser.add_argument("--vaeac-art-path", type=Path, required=True)
    parser.add_argument("--vaeac-gt-path", type=Path, required=True)
    parser.add_argument("--under-vaeac-gt-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "clue" / "compas_reproduce",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--no-assert-paper",
        action="store_true",
        help="Skip paper-value tolerance assertions.",
    )
    args = parser.parse_args()

    _setup_logging()
    device = _resolve_device(args.device)
    LOGGER.info("Using device=%s", device)

    schema = _build_compas_train_schema(seed=SEED)
    trainset = schema["trainset"]
    LOGGER.info(
        "Prepared finalized COMPAS schema: train=%d test=%d",
        len(trainset),
        len(schema["testset"]),
    )

    target_model = MlpBayesianModel(
        seed=SEED,
        device=device,
        layers=[200, 200],
        pretrained_path=args.bnn_art_path.as_posix(),
    )
    target_model.fit(trainset)
    LOGGER.info("Loaded ART Bayesian MLP checkpoint")

    clue_method = ClueMethod(
        target_model=target_model,
        seed=SEED,
        device=device,
        pretrained_path=args.vae_art_path.as_posix(),
    )
    clue_method.fit(trainset)
    LOGGER.info("Loaded ART CLUE VAE checkpoint")

    summary = _run_compas_reproduction(
        clue_method,
        target_model,
        device,
        args.output_dir,
        vaeac_art_path=args.vaeac_art_path,
        vaeac_gt_path=args.vaeac_gt_path,
        under_vaeac_gt_path=args.under_vaeac_gt_path,
        assert_paper=not args.no_assert_paper,
    )
    LOGGER.info(
        "Reproduction summary:\n%s", json.dumps(summary, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
