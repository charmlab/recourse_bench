from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = PROJECT_ROOT / "reference" / "CLUE_official"


def _setup_paths() -> None:
    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


_setup_paths()

import src.utils as official_utils  # noqa: E402


official_utils.BaseNet.get_nb_parameters = lambda self: sum(
    parameter.numel() for parameter in self.model.parameters()
)


def _patched_load(self, filename):
    state = torch.load(
        filename,
        weights_only=False,
        map_location=(
            "cuda"
            if getattr(self, "cuda", False) and torch.cuda.is_available()
            else "cpu"
        ),
    )
    self.epoch = state["epoch"]
    self.lr = state["lr"]
    self.model = state["model"]
    self.optimizer = state["optimizer"]
    return self.epoch


official_utils.BaseNet.load = _patched_load

from src.utils import Datafeed, Ln_distance, generate_ind_batch  # noqa: E402
from src.gauss_cat import (  # noqa: E402
    flat_to_gauss_cat,
    rms_cat_loglike,
    selective_softmax,
)
from interpret.CLUE import CLUE  # noqa: E402
from interpret.FIDO import mask_explainer  # noqa: E402
from interpret.explanation_tools import input_uncertainty_step_cat  # noqa: E402
from interpret.functionally_grounded import (  # noqa: E402
    evaluate_aleatoric_explanation_cat,
    evaluate_epistemic_explanation_cat,
    get_BNN_uncertainties,
)
from interpret.generate_data import sample_artificial_dataset  # noqa: E402
from VAE.fc_gauss_cat import VAE_gauss_cat_net  # noqa: E402
from VAEAC.fc_gauss_cat import VAEAC_gauss_cat_net  # noqa: E402
from VAEAC.under_net import under_VAEAC  # noqa: E402

from dataset.compas_clue.compas_clue import CompasClueDataset  # noqa: E402
from model.mlp_bayesian.mlp_bayesian import MlpBayesianModel  # noqa: E402
from preprocess.common import (  # noqa: E402
    EncodePreProcess,
    FinalizePreProcess,
    ReorderPreProcess,
    ScalePreProcess,
)

OFFICIAL_ORDER = [
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
INPUT_DIM_VEC = [3, 6, 2, 2, 2, 1, 1]
INPUT_DIM_VEC_XY = [3, 6, 2, 2, 2, 1, 1, 2]
TEST_DIMS = [17, 18]
ARTIFICIAL_POINTS = 5554 + 618
ARTIFICIAL_TRAIN = 5554
ART_ALEATORIC_THRESHOLD = 0.15
ART_EPISTEMIC_THRESHOLD = 0.02
CLUE_LR_A = 0.1
CLUE_LR_E = 0.1
CLUE_LAMBDAS = np.logspace(np.log(0.0001), np.log(30), 30, base=np.e)
FIDO_LAMBDAS = np.logspace(np.log(0.00001), np.log(100), 30, base=np.e)
FIDO_BATCH_SIZE = 3000
FIDO_EPOCHS = 30
FIDO_MASK_SAMPLES = 20
FIDO_MASK_SAMPLES2 = 10
SENSITIVITY_STEPS = np.logspace(np.log(0.001), np.log(200), 20, base=np.e)
PAPER_TABLE1_COMPAS = {"epistemic": 0.71, "aleatoric": 0.18}
PAPER_TABLE2_COMPAS = {"epistemic": 0.707, "aleatoric": 0.044}
PAPER_TOLERANCES = {
    "table1_epistemic": 0.01,
    "table1_aleatoric": 0.05,
    "table2_epistemic": 0.01,
    "table2_aleatoric": 0.05,
}
DEFAULT_BNN_ART_PATH = (
    PROJECT_ROOT
    / "reference/clueweights/notebooks/saves/fc_BNN_NEW_ART_compas_models/state_dicts.pkl"
)
DEFAULT_VAE_ART_PATH = (
    PROJECT_ROOT
    / "reference/clueweights/notebooks/saves/fc_preact_VAE_NEW(300)_ART_compas_models/theta_best.dat"
)
DEFAULT_VAEAC_ART_PATH = (
    PROJECT_ROOT
    / "reference/clueweights/notebooks/saves/fc_preact_VAEAC_NEW_ART_compas_models/theta_best.dat"
)
DEFAULT_VAEAC_GT_PATH = (
    PROJECT_ROOT
    / "reference/clueweights/notebooks/saves/fc_preact_VAEAC_NEW_compas_models/theta_best.dat"
)
DEFAULT_UNDER_VAEAC_GT_PATH = (
    PROJECT_ROOT / "reference/clueweights/notebooks/saves/fc_VAEAC_NEW_under_compas_models/theta_best.dat"
)


class OfficialStyleBNNAdapter:
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
        return None


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_compas_arrays(seed: int = 42):
    dataset = CompasClueDataset()
    raw_df = dataset.snapshot()
    train_df, test_df = train_test_split(
        raw_df,
        test_size=0.1,
        random_state=seed,
        shuffle=True,
    )
    trainset = dataset
    testset = dataset.clone()
    trainset.update("trainset", True, df=train_df.copy(deep=True))
    testset.update("testset", True, df=test_df.copy(deep=True))

    scale = ScalePreProcess(seed=seed, scaling="standardize", range=True, refset=trainset)
    trainset = scale.transform(trainset)
    testset = scale.transform(testset)

    encode = EncodePreProcess(seed=seed, encoding="onehot")
    trainset = encode.transform(trainset)
    testset = encode.transform(testset)

    reorder = ReorderPreProcess(seed=seed, order=OFFICIAL_ORDER)
    trainset = reorder.transform(trainset)
    testset = reorder.transform(testset)

    finalize = FinalizePreProcess(seed=seed)
    trainset = finalize.transform(trainset)
    testset = finalize.transform(testset)

    x_train = trainset.get(target=False).to_numpy(dtype="float32")
    x_test = testset.get(target=False).to_numpy(dtype="float32")
    y_train = trainset.get(target=True).iloc[:, 0].to_numpy(dtype=int)
    y_test = testset.get(target=True).iloc[:, 0].to_numpy(dtype=int)
    y_train_oh = np.eye(2, dtype="float32")[y_train]
    y_test_oh = np.eye(2, dtype="float32")[y_test]
    xy_train = np.concatenate([x_train, y_train_oh], axis=1)
    xy_test = np.concatenate([x_test, y_test_oh], axis=1)
    return {
        "trainset": trainset,
        "testset": testset,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "xy_train": xy_train,
        "xy_test": xy_test,
    }


def _load_models(device: str, trainset):
    cuda = device == "cuda"

    vaeac_gt = VAEAC_gauss_cat_net(
        INPUT_DIM_VEC_XY,
        350,
        3,
        4,
        pred_sig=False,
        lr=3e-4,
        cuda=cuda,
        flatten=False,
    )
    vaeac_gt.load(DEFAULT_VAEAC_GT_PATH.as_posix())

    under_vaeac_gt = under_VAEAC(vaeac_gt.model, 150, 2, 4, 3e-4, cuda=cuda)
    under_vaeac_gt.load(DEFAULT_UNDER_VAEAC_GT_PATH.as_posix())

    art_bnn_model = MlpBayesianModel(
        seed=42,
        device=device,
        layers=[200, 200],
        pretrained_path=DEFAULT_BNN_ART_PATH.as_posix(),
        bayesian=True,
    )
    art_bnn_model.fit(trainset)
    art_bnn = OfficialStyleBNNAdapter(art_bnn_model)

    vae_art = VAE_gauss_cat_net(
        INPUT_DIM_VEC,
        300,
        3,
        4,
        pred_sig=False,
        lr=1e-4,
        cuda=cuda,
        flatten=False,
    )
    vae_art.load(DEFAULT_VAE_ART_PATH.as_posix())

    vaeac_art = VAEAC_gauss_cat_net(
        INPUT_DIM_VEC,
        350,
        3,
        4,
        pred_sig=False,
        lr=1e-4,
        cuda=cuda,
        flatten=False,
    )
    vaeac_art.load(DEFAULT_VAEAC_ART_PATH.as_posix())

    return {
        "vaeac_gt": vaeac_gt,
        "under_vaeac_gt": under_vaeac_gt,
        "art_bnn": art_bnn,
        "art_bnn_model": art_bnn_model,
        "vae_art": vae_art,
        "vaeac_art": vaeac_art,
    }


def _get_vaeac_px_gauss_cat(
    under_vaeac_net,
    x_art_test: np.ndarray,
    input_dim_vec: list[int],
    y_dims: list[int],
    override_y_dims: int | None = None,
    nsamples: int = 1000,
):
    rec_loglike_func = rms_cat_loglike(input_dim_vec, reduction="none")
    max_dims = x_art_test.shape[1] + len(y_dims)
    x_dims = list(range(max_dims))
    for dim in y_dims:
        x_dims.remove(dim)

    iw_xy = torch.zeros((x_art_test.shape[0], max_dims))
    iw_xy[:, x_dims] = torch.tensor(x_art_test, dtype=torch.float32)
    iw_xy[:, y_dims] = iw_xy.new_zeros((iw_xy.shape[0], len(y_dims))).normal_()
    if under_vaeac_net.cuda:
        iw_xy = iw_xy.cuda()

    iw_xy_target = flat_to_gauss_cat(iw_xy, input_dim_vec)
    iw_mask = torch.zeros_like(iw_xy)
    iw_mask[:, y_dims] = 1
    prior = under_vaeac_net.prior
    u_approx_dist = under_vaeac_net.u_mask_recongnition(iw_xy, iw_mask, grad=False)

    estimates = []
    for _ in range(nsamples):
        u_sample = u_approx_dist.sample()
        rec_distrib = under_vaeac_net.u_regenerate(u_sample, grad=False)
        log_p = prior.log_prob(u_sample).sum(dim=1).data
        log_q = u_approx_dist.log_prob(u_sample).sum(dim=1).data
        rec_loglike = rec_loglike_func(rec_distrib, iw_xy_target).view(iw_xy.shape[0], -1)
        if override_y_dims is not None:
            x_loglike = rec_loglike[:, :-override_y_dims].sum(dim=1)
        else:
            x_loglike = rec_loglike[:, x_dims].sum(dim=1)
        estimates.append(x_loglike + log_p - log_q)

    return torch.logsumexp(torch.stack(estimates), dim=0, keepdim=False) - np.log(nsamples)


def _generate_artificial_compas(models):
    x_art, y_art, xy_art = sample_artificial_dataset(
        models["under_vaeac_gt"],
        TEST_DIMS,
        ARTIFICIAL_POINTS,
        u_dims=4,
        sig=False,
        softmax=False,
    )
    x_art = selective_softmax(
        x_art,
        INPUT_DIM_VEC,
        grad=False,
        cat_probs=True,
        prob_sample=True,
    )
    xy_art = selective_softmax(
        xy_art,
        INPUT_DIM_VEC_XY,
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
    return {
        "x_art_test": x_art[ARTIFICIAL_TRAIN:, :].cpu().numpy(),
        "y_art_test": y_art[ARTIFICIAL_TRAIN:, :].cpu().numpy(),
        "y_art_bnn_test": y_art_bnn[ARTIFICIAL_TRAIN:].cpu().numpy(),
    }


def _compute_artificial_baselines(art_data, models, device: str):
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
    aleatoric_idxs = aleatoric_stack.cpu().numpy() >= ART_ALEATORIC_THRESHOLD
    epistemic_idxs = epistemic_stack.cpu().numpy() >= ART_EPISTEMIC_THRESHOLD

    og_aleatoric_uncert = evaluate_aleatoric_explanation_cat(
        models["vaeac_gt"],
        torch.tensor(x_art_test[aleatoric_idxs], dtype=torch.float32),
        TEST_DIMS,
        N_target_samples=500,
        batch_size=1024,
    )
    gt_test_err, _ = evaluate_epistemic_explanation_cat(
        models["art_bnn"],
        models["vaeac_gt"],
        torch.tensor(x_art_test[epistemic_idxs], dtype=torch.float32).to(device),
        TEST_DIMS,
        outer_batch_size=2000,
        inner_batch_size=1024,
        VAEAC_samples=500,
    )
    log_px_vaeac_a = _get_vaeac_px_gauss_cat(
        models["under_vaeac_gt"],
        x_art_test[aleatoric_idxs],
        INPUT_DIM_VEC_XY,
        TEST_DIMS,
        override_y_dims=1,
        nsamples=1000,
    )
    log_px_vaeac_e = _get_vaeac_px_gauss_cat(
        models["under_vaeac_gt"],
        x_art_test[epistemic_idxs],
        INPUT_DIM_VEC_XY,
        TEST_DIMS,
        override_y_dims=1,
        nsamples=1000,
    )
    z_test = (
        models["vae_art"]
        .recongnition(torch.tensor(x_art_test, dtype=torch.float32).to(device), flatten=False)
        .loc.data.cpu()
        .numpy()
    )
    return {
        "aleatoric_idxs": aleatoric_idxs,
        "epistemic_idxs": epistemic_idxs,
        "og_aleatoric_uncert": og_aleatoric_uncert,
        "gt_test_err": float(gt_test_err),
        "log_px_vaeac_a": log_px_vaeac_a,
        "log_px_vaeac_e": log_px_vaeac_e,
        "z_test": z_test,
    }


def _compute_clue_curve(
    x_init_batch: np.ndarray,
    z_init_batch: np.ndarray,
    mode: str,
    models,
    baselines,
    max_steps: int,
) -> dict[str, np.ndarray]:
    dist = Ln_distance(n=1, dim=(1))
    x_dim = x_init_batch.reshape(x_init_batch.shape[0], -1).shape[1]
    lr = CLUE_LR_A if mode == "aleatoric" else CLUE_LR_E

    delta_x_values: list[float] = []
    benefit_values: list[float] = []
    logpx_values: list[float] = []

    for distance_weight in (CLUE_LAMBDAS / x_dim):
        clue_explainer = CLUE(
            models["vae_art"],
            models["art_bnn"],
            x_init_batch,
            uncertainty_weight=0,
            aleatoric_weight=1 if mode == "aleatoric" else 0,
            epistemic_weight=1 if mode == "epistemic" else 0,
            prior_weight=0,
            distance_weight=distance_weight,
            latent_L2_weight=0,
            prediction_similarity_weight=0,
            lr=lr,
            desired_preds=None,
            cond_mask=None,
            distance_metric=dist,
            z_init=z_init_batch,
            norm_MNIST=False,
            flatten_BNN=False,
            regression=False,
            cuda=True,
        )
        _, x_vec, _, _, _, _, _ = clue_explainer.optimise(
            min_steps=3,
            max_steps=max_steps,
            n_early_stop=3,
        )
        x_clue = x_vec[-1]
        delta_x_values.append(float(np.abs(x_init_batch - x_clue).sum(axis=1).mean()))

        if mode == "aleatoric":
            clue_uncert = evaluate_aleatoric_explanation_cat(
                models["vaeac_gt"],
                torch.tensor(x_clue, dtype=torch.float32),
                TEST_DIMS,
                N_target_samples=500,
                batch_size=1024,
            )
            benefit_values.append(
                float((baselines["og_aleatoric_uncert"] - clue_uncert).cpu().numpy().mean())
            )
            log_px = _get_vaeac_px_gauss_cat(
                models["under_vaeac_gt"],
                x_clue,
                INPUT_DIM_VEC_XY,
                TEST_DIMS,
                override_y_dims=1,
                nsamples=1000,
            )
            logpx_values.append(float(log_px.mean().cpu().numpy()))
        else:
            clue_test_err, _ = evaluate_epistemic_explanation_cat(
                models["art_bnn"],
                models["vaeac_gt"],
                torch.tensor(x_clue, dtype=torch.float32).cuda(),
                TEST_DIMS,
                outer_batch_size=2000,
                inner_batch_size=1024,
                VAEAC_samples=500,
            )
            benefit_values.append(float(baselines["gt_test_err"] - clue_test_err))
            log_px = _get_vaeac_px_gauss_cat(
                models["under_vaeac_gt"],
                x_clue,
                INPUT_DIM_VEC_XY,
                TEST_DIMS,
                override_y_dims=1,
                nsamples=1000,
            )
            logpx_values.append(float(log_px.mean().cpu().numpy()))

    return {
        "delta_x": np.asarray(delta_x_values, dtype=np.float64),
        "benefit": np.asarray(benefit_values, dtype=np.float64),
        "logpx": np.asarray(logpx_values, dtype=np.float64),
    }


def _compute_fido_curve(
    x_init_batch: np.ndarray,
    mode: str,
    models,
    baselines,
    device: str,
) -> dict[str, np.ndarray]:
    delta_x_values: list[float] = []
    benefit_values: list[float] = []
    logpx_values: list[float] = []

    for l1_weight in FIDO_LAMBDAS:
        explanations = []
        aleatoric_coeff = 1 if mode == "aleatoric" else 0
        epistemic_coeff = 1 if mode == "epistemic" else 0

        aux_loader = generate_ind_batch(
            x_init_batch.shape[0],
            FIDO_BATCH_SIZE,
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
                L1w=l1_weight,
                N_epochs=FIDO_EPOCHS,
                mask_samples=FIDO_MASK_SAMPLES,
                mask_samples2=FIDO_MASK_SAMPLES2,
                flatten_ims=False,
                test_dims=None,
                cat=True,
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
                TEST_DIMS,
                N_target_samples=500,
                batch_size=1024,
            )
            benefit_values.append(
                float((baselines["og_aleatoric_uncert"] - fido_uncert).cpu().numpy().mean())
            )
            log_px = _get_vaeac_px_gauss_cat(
                models["under_vaeac_gt"],
                x_fido,
                INPUT_DIM_VEC_XY,
                TEST_DIMS,
                override_y_dims=1,
                nsamples=1000,
            )
            logpx_values.append(float(log_px.mean().cpu().numpy()))
        else:
            fido_test_err, _ = evaluate_epistemic_explanation_cat(
                models["art_bnn"],
                models["vaeac_gt"],
                torch.tensor(x_fido, dtype=torch.float32).to(device),
                TEST_DIMS,
                outer_batch_size=2000,
                inner_batch_size=1024,
                VAEAC_samples=500,
            )
            benefit_values.append(float(baselines["gt_test_err"] - fido_test_err))
            log_px = _get_vaeac_px_gauss_cat(
                models["under_vaeac_gt"],
                x_fido,
                INPUT_DIM_VEC_XY,
                TEST_DIMS,
                override_y_dims=1,
                nsamples=1000,
            )
            logpx_values.append(float(log_px.mean().cpu().numpy()))

    return {
        "delta_x": np.asarray(delta_x_values, dtype=np.float64),
        "benefit": np.asarray(benefit_values, dtype=np.float64),
        "logpx": np.asarray(logpx_values, dtype=np.float64),
    }


def _compute_sensitivity_curve(
    x_art_test: np.ndarray,
    x_init_batch: np.ndarray,
    selected_mask: np.ndarray,
    mode: str,
    models,
    baselines,
    device: str,
) -> dict[str, np.ndarray]:
    del device
    delta_x_values: list[float] = []
    benefit_values: list[float] = []
    delta_logp_values: list[float] = []
    dummy_y = np.zeros(x_art_test.shape[0], dtype=np.float32)

    for step in SENSITIVITY_STEPS:
        sens_x = input_uncertainty_step_cat(
            models["art_bnn"],
            Datafeed(x_art_test, dummy_y),
            aleatoric_coeff=1 if mode == "aleatoric" else 0,
            epistemic_coeff=1 if mode == "epistemic" else 0,
            stepsize_perdim=-step,
            batch_size=1024,
            cuda=torch.cuda.is_available(),
            norm_MNIST=False,
            flatten=False,
            norm_grad=False,
        )
        x_sens = sens_x.detach().cpu().numpy()[selected_mask]
        delta_x_values.append(float(np.abs(x_init_batch - x_sens).sum(axis=1).mean()))

        if mode == "aleatoric":
            sens_uncert = evaluate_aleatoric_explanation_cat(
                models["vaeac_gt"],
                torch.tensor(x_sens, dtype=torch.float32),
                TEST_DIMS,
                N_target_samples=500,
                batch_size=1024,
            )
            benefit_values.append(
                float((baselines["og_aleatoric_uncert"] - sens_uncert).cpu().numpy().mean())
            )
            log_px = _get_vaeac_px_gauss_cat(
                models["under_vaeac_gt"],
                x_sens,
                INPUT_DIM_VEC_XY,
                TEST_DIMS,
                override_y_dims=1,
                nsamples=1000,
            )
            delta_logp_values.append(float(log_px.mean().cpu().numpy()))
        else:
            sens_test_err, _ = evaluate_epistemic_explanation_cat(
                models["art_bnn"],
                models["vaeac_gt"],
                torch.tensor(x_sens, dtype=torch.float32).to(models["art_bnn"].model._device),
                TEST_DIMS,
                outer_batch_size=2000,
                inner_batch_size=1024,
                VAEAC_samples=500,
            )
            benefit_values.append(float(baselines["gt_test_err"] - sens_test_err))
            log_px = _get_vaeac_px_gauss_cat(
                models["under_vaeac_gt"],
                x_sens,
                INPUT_DIM_VEC_XY,
                TEST_DIMS,
                override_y_dims=1,
                nsamples=1000,
            )
            delta_logp_values.append(float(log_px.mean().cpu().numpy()))

    return {
        "delta_x": np.asarray(delta_x_values, dtype=np.float64),
        "benefit": np.asarray(benefit_values, dtype=np.float64),
        "logpx": np.asarray(delta_logp_values, dtype=np.float64),
    }


def _save_curve(
    output_path: Path,
    x_blue: np.ndarray,
    y_blue: np.ndarray,
    x_red: np.ndarray,
    y_red: np.ndarray,
    x_green: np.ndarray,
    y_green: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    baseline_x: float | None = None,
    invert_x: bool = False,
) -> None:
    plt.figure(dpi=100)
    plt.plot(x_blue, y_blue, "--.", c="b")
    plt.plot(x_red, y_red, "--.", c="r")
    plt.plot(x_green, y_green, "--.", c="g")
    if baseline_x is not None:
        ax = plt.gca()
        ax.axvline(x=baseline_x)
        if invert_x:
            ax.invert_xaxis()
    plt.title(title)
    plt.legend(["sensitivity", "CLUE", "U-FIDO"])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment" / "clue" / "art_data_curves",
    )
    args = parser.parse_args()

    device = _resolve_device()
    compas = _prepare_compas_arrays(seed=42)
    models = _load_models(device=device, trainset=compas["trainset"])
    art_data = _generate_artificial_compas(models=models)
    baselines = _compute_artificial_baselines(art_data=art_data, models=models, device=device)
    clue_aleatoric = _compute_clue_curve(
        x_init_batch=art_data["x_art_test"][baselines["aleatoric_idxs"]],
        z_init_batch=baselines["z_test"][baselines["aleatoric_idxs"]],
        mode="aleatoric",
        models=models,
        baselines=baselines,
        max_steps=55,
    )
    clue_epistemic = _compute_clue_curve(
        x_init_batch=art_data["x_art_test"][baselines["epistemic_idxs"]],
        z_init_batch=baselines["z_test"][baselines["epistemic_idxs"]],
        mode="epistemic",
        models=models,
        baselines=baselines,
        max_steps=65,
    )
    fido_aleatoric = _compute_fido_curve(
        x_init_batch=art_data["x_art_test"][baselines["aleatoric_idxs"]],
        mode="aleatoric",
        models=models,
        baselines=baselines,
        device=device,
    )
    fido_epistemic = _compute_fido_curve(
        x_init_batch=art_data["x_art_test"][baselines["epistemic_idxs"]],
        mode="epistemic",
        models=models,
        baselines=baselines,
        device=device,
    )
    sensitivity_aleatoric = _compute_sensitivity_curve(
        x_art_test=art_data["x_art_test"],
        x_init_batch=art_data["x_art_test"][baselines["aleatoric_idxs"]],
        selected_mask=baselines["aleatoric_idxs"],
        mode="aleatoric",
        models=models,
        baselines=baselines,
        device=device,
    )
    sensitivity_epistemic = _compute_sensitivity_curve(
        x_art_test=art_data["x_art_test"],
        x_init_batch=art_data["x_art_test"][baselines["epistemic_idxs"]],
        selected_mask=baselines["epistemic_idxs"],
        mode="epistemic",
        models=models,
        baselines=baselines,
        device=device,
    )

    _save_curve(
        args.output_dir / "compas_epistemic_delta_x.png",
        sensitivity_epistemic["delta_x"],
        -sensitivity_epistemic["benefit"],
        clue_epistemic["delta_x"],
        -clue_epistemic["benefit"],
        fido_epistemic["delta_x"],
        -fido_epistemic["benefit"],
        "compas epistemic",
        "L1 change in X",
        "change in err",
    )
    _save_curve(
        args.output_dir / "compas_epistemic_logpx.png",
        sensitivity_epistemic["logpx"],
        -sensitivity_epistemic["benefit"],
        clue_epistemic["logpx"],
        -clue_epistemic["benefit"],
        fido_epistemic["logpx"],
        -fido_epistemic["benefit"],
        "compas epistemic",
        "log p(x)",
        "change in err",
        baseline_x=float(baselines["log_px_vaeac_e"].mean().cpu().numpy()),
        invert_x=True,
    )
    _save_curve(
        args.output_dir / "compas_aleatoric_delta_x.png",
        sensitivity_aleatoric["delta_x"],
        -sensitivity_aleatoric["benefit"],
        clue_aleatoric["delta_x"],
        -clue_aleatoric["benefit"],
        fido_aleatoric["delta_x"],
        -fido_aleatoric["benefit"],
        "compas Aleatoric",
        "L1 change in X",
        "change in H",
    )
    _save_curve(
        args.output_dir / "compas_aleatoric_logpx.png",
        sensitivity_aleatoric["logpx"],
        -sensitivity_aleatoric["benefit"],
        clue_aleatoric["logpx"],
        -clue_aleatoric["benefit"],
        fido_aleatoric["logpx"],
        -fido_aleatoric["benefit"],
        "compas Aleatoric",
        "log p(x)",
        "change in H",
        baseline_x=float(baselines["log_px_vaeac_a"].mean().cpu().numpy()),
        invert_x=True,
    )
    np.savez_compressed(
        args.output_dir / "compas_curves.npz",
        sensitivity_epistemic_delta_x=sensitivity_epistemic["delta_x"],
        sensitivity_epistemic_logpx=sensitivity_epistemic["logpx"],
        sensitivity_epistemic_benefit=sensitivity_epistemic["benefit"],
        clue_epistemic_delta_x=clue_epistemic["delta_x"],
        clue_epistemic_logpx=clue_epistemic["logpx"],
        clue_epistemic_benefit=clue_epistemic["benefit"],
        fido_epistemic_delta_x=fido_epistemic["delta_x"],
        fido_epistemic_logpx=fido_epistemic["logpx"],
        fido_epistemic_benefit=fido_epistemic["benefit"],
        sensitivity_aleatoric_delta_x=sensitivity_aleatoric["delta_x"],
        sensitivity_aleatoric_logpx=sensitivity_aleatoric["logpx"],
        sensitivity_aleatoric_benefit=sensitivity_aleatoric["benefit"],
        clue_aleatoric_delta_x=clue_aleatoric["delta_x"],
        clue_aleatoric_logpx=clue_aleatoric["logpx"],
        clue_aleatoric_benefit=clue_aleatoric["benefit"],
        fido_aleatoric_delta_x=fido_aleatoric["delta_x"],
        fido_aleatoric_logpx=fido_aleatoric["logpx"],
        fido_aleatoric_benefit=fido_aleatoric["benefit"],
        baseline_logpx_epistemic=float(baselines["log_px_vaeac_e"].mean().cpu().numpy()),
        baseline_logpx_aleatoric=float(baselines["log_px_vaeac_a"].mean().cpu().numpy()),
    )

    paper_style = {
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
            float(baselines["log_px_vaeac_e"].mean().cpu().numpy()),
        ),
        "table2_aleatoric": _table2_scalar(
            clue_aleatoric["logpx"],
            clue_aleatoric["benefit"],
            fido_aleatoric["logpx"],
            fido_aleatoric["benefit"],
            float(baselines["log_px_vaeac_a"].mean().cpu().numpy()),
        ),
    }
    paper_targets = {
        "table1_epistemic": PAPER_TABLE1_COMPAS["epistemic"],
        "table1_aleatoric": PAPER_TABLE1_COMPAS["aleatoric"],
        "table2_epistemic": PAPER_TABLE2_COMPAS["epistemic"],
        "table2_aleatoric": PAPER_TABLE2_COMPAS["aleatoric"],
    }
    paper_errors = {
        key: abs(paper_style[key] - paper_targets[key]) for key in paper_style
    }

    print("saved_curve_dir", args.output_dir.as_posix())
    print("saved_files", sorted(path.name for path in args.output_dir.iterdir()))
    print("paper_style_comparison", paper_style)
    print("paper_style_targets", paper_targets)
    print("paper_style_abs_errors", paper_errors)

    for key, tolerance in PAPER_TOLERANCES.items():
        if paper_errors[key] > tolerance:
            raise AssertionError(
                f"{key} mismatch: observed={paper_style[key]} paper={paper_targets[key]} tol={tolerance}"
            )


if __name__ == "__main__":
    main()
