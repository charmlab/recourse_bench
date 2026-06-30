import numpy as np
import torch
import torch.distributions as dists
from torch import nn


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=-1)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean((y_true - y_pred) ** 2, axis=-1)


def csvae_loss(csvae, x_train, y_train):
    x = x_train.clone().float()
    y = y_train.clone().float()

    (
        x_mu,
        x_logvar,
        zw,
        y_pred,
        w_mu_encoder,
        w_logvar_encoder,
        w_mu_prior,
        w_logvar_prior,
        z_mu,
        z_logvar,
    ) = csvae.forward(x, y)

    x_recon = nn.MSELoss()(x_mu, x)

    w_dist = dists.MultivariateNormal(
        w_mu_encoder.flatten(), torch.diag(w_logvar_encoder.flatten().exp())
    )
    w_prior = dists.MultivariateNormal(
        w_mu_prior.flatten(), torch.diag(w_logvar_prior.flatten().exp())
    )
    w_kl = dists.kl.kl_divergence(w_dist, w_prior)

    z_dist = dists.MultivariateNormal(
        z_mu.flatten(), torch.diag(z_logvar.flatten().exp())
    )
    z_prior = dists.MultivariateNormal(
        torch.zeros(csvae.z_dim * z_mu.size()[0], device=z_mu.device),
        torch.eye(csvae.z_dim * z_mu.size()[0], device=z_mu.device),
    )
    z_kl = dists.kl.kl_divergence(z_dist, z_prior)

    y_pred_negentropy = (
        y_pred.log() * y_pred + (1 - y_pred).log() * (1 - y_pred)
    ).mean()

    class_label = torch.argmax(y, dim=1)
    y_recon = (
        100.0
        * torch.where(
            class_label == 1, -torch.log(y_pred[:, 1]), -torch.log(y_pred[:, 0])
        )
    ).mean()

    ELBO = 40 * x_recon + 0.2 * z_kl + 1 * w_kl + 110 * y_pred_negentropy

    return ELBO, x_recon, w_kl, z_kl, y_pred_negentropy, y_recon
