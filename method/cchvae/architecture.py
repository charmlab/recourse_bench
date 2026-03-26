from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    likelihood: str


def transform_by_likelihood(values: torch.Tensor, likelihood: str) -> torch.Tensor:
    if likelihood in {"pos", "count"}:
        return torch.log1p(values)
    if likelihood == "real":
        return values
    raise ValueError(f"Unsupported likelihood: {likelihood}")


def normalize_feature_block(
    values: torch.Tensor,
    specs: list[FeatureSpec],
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    if not specs:
        empty = values.new_zeros((values.shape[0], 0))
        return empty, []

    normalized_columns = []
    stats: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, spec in enumerate(specs):
        column = values[:, index : index + 1]
        transformed = transform_by_likelihood(column, spec.likelihood)
        mean = transformed.mean(dim=0, keepdim=True)
        var = transformed.var(dim=0, unbiased=False, keepdim=True).clamp_min(EPSILON)
        normalized_columns.append((transformed - mean) / torch.sqrt(var))
        stats.append((mean, var))
    return torch.cat(normalized_columns, dim=1), stats


def normalize_with_stats(
    values: torch.Tensor,
    specs: list[FeatureSpec],
    stats: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if not specs:
        return values.new_zeros((values.shape[0], 0))

    normalized_columns = []
    for index, spec in enumerate(specs):
        column = values[:, index : index + 1]
        transformed = transform_by_likelihood(column, spec.likelihood)
        mean, var = stats[index]
        normalized_columns.append((transformed - mean) / torch.sqrt(var))
    return torch.cat(normalized_columns, dim=1)


class TypedCchvaeNet(nn.Module):
    def __init__(
        self,
        free_specs: list[FeatureSpec],
        conditional_specs: list[FeatureSpec],
        latent_s_dim: int,
        latent_z_dim: int,
        latent_y_dim: int,
        device: str,
    ):
        super().__init__()
        self.free_specs = free_specs
        self.conditional_specs = conditional_specs
        self.s_dim = int(latent_s_dim)
        self.z_dim = int(latent_z_dim)
        self.y_dim = int(latent_y_dim)
        self.device = torch.device(device)

        if not self.free_specs:
            raise ValueError(
                "TypedCchvaeNet requires at least one mutable/free feature"
            )

        self.s_encoder = nn.Linear(
            len(self.free_specs) + len(self.conditional_specs),
            self.s_dim,
        )
        self.qz_mean_heads = nn.ModuleList(
            [
                nn.Linear(1 + self.s_dim + len(self.conditional_specs), self.z_dim)
                for _ in self.free_specs
            ]
        )
        self.qz_logvar_heads = nn.ModuleList(
            [
                nn.Linear(1 + self.s_dim + len(self.conditional_specs), self.z_dim)
                for _ in self.free_specs
            ]
        )
        self.pz_mean = nn.Linear(self.s_dim, self.z_dim)
        self.y_projection = nn.Linear(self.z_dim, self.y_dim * len(self.free_specs))

        self.real_mean_heads = nn.ModuleDict()
        self.real_logvar_heads = nn.ModuleDict()
        self.pos_mean_heads = nn.ModuleDict()
        self.pos_logvar_heads = nn.ModuleDict()
        self.count_lambda_heads = nn.ModuleDict()
        for spec in self.free_specs:
            if spec.likelihood == "real":
                self.real_mean_heads[spec.name] = nn.Linear(self.y_dim, 1)
                self.real_logvar_heads[spec.name] = nn.Linear(self.y_dim, 1)
            elif spec.likelihood == "pos":
                self.pos_mean_heads[spec.name] = nn.Linear(self.y_dim, 1)
                self.pos_logvar_heads[spec.name] = nn.Linear(self.y_dim, 1)
            elif spec.likelihood == "count":
                self.count_lambda_heads[spec.name] = nn.Linear(self.y_dim, 1)
            else:
                raise ValueError(f"Unsupported free likelihood: {spec.likelihood}")

        self._reset_parameters()
        self.to(self.device)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                nn.init.zeros_(module.bias)

    def _poe_qz(
        self,
        normalized_free: torch.Tensor,
        normalized_conditional: torch.Tensor,
        samples_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_qz = []
        logvar_qz = []
        for index in range(normalized_free.shape[1]):
            inputs = torch.cat(
                [
                    normalized_free[:, index : index + 1],
                    samples_s,
                    normalized_conditional,
                ],
                dim=1,
            )
            mean_qz.append(self.qz_mean_heads[index](inputs))
            logvar_qz.append(self.qz_logvar_heads[index](inputs))

        mean_qz.append(torch.zeros_like(mean_qz[0]))
        logvar_qz.append(torch.zeros_like(logvar_qz[0]))

        stacked_mean = torch.stack(mean_qz, dim=0)
        stacked_logvar = torch.stack(logvar_qz, dim=0)
        joint_logvar = -torch.logsumexp(-stacked_logvar, dim=0)
        joint_mean = torch.exp(joint_logvar) * torch.sum(
            stacked_mean * torch.exp(-stacked_logvar), dim=0
        )
        return joint_mean, joint_logvar

    def _theta_from_z(
        self, z: torch.Tensor
    ) -> list[tuple[str, tuple[torch.Tensor, ...]]]:
        y = self.y_projection(z)
        grouped_y = torch.split(y, self.y_dim, dim=1)
        theta: list[tuple[str, tuple[torch.Tensor, ...]]] = []
        for chunk, spec in zip(grouped_y, self.free_specs):
            if spec.likelihood == "real":
                theta.append(
                    (
                        spec.name,
                        (
                            self.real_mean_heads[spec.name](chunk),
                            self.real_logvar_heads[spec.name](chunk),
                        ),
                    )
                )
            elif spec.likelihood == "pos":
                theta.append(
                    (
                        spec.name,
                        (
                            self.pos_mean_heads[spec.name](chunk),
                            self.pos_logvar_heads[spec.name](chunk),
                        ),
                    )
                )
            else:
                theta.append((spec.name, (self.count_lambda_heads[spec.name](chunk),)))
        return theta

    def _log_prob_feature(
        self,
        feature_values: torch.Tensor,
        spec: FeatureSpec,
        theta: tuple[torch.Tensor, ...],
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mean, var = stats
        if spec.likelihood == "real":
            est_mean, est_logvar = theta
            est_var = var * F.softplus(est_logvar).clamp(min=EPSILON, max=1.0)
            est_mean = torch.sqrt(var) * est_mean + mean
            return -0.5 * (
                ((feature_values - est_mean) ** 2) / est_var
                + torch.log(2 * math.pi * est_var)
            ).sum(dim=1)

        if spec.likelihood == "pos":
            transformed = torch.log1p(feature_values)
            est_mean, est_logvar = theta
            est_var = var * F.softplus(est_logvar).clamp(min=EPSILON, max=1.0)
            est_mean = torch.sqrt(var) * est_mean + mean
            return -0.5 * (
                ((transformed - est_mean) ** 2) / est_var
                + torch.log(2 * math.pi * est_var)
            ).sum(dim=1) - transformed.sum(dim=1)

        if spec.likelihood == "count":
            (raw_lambda,) = theta
            est_lambda = F.softplus(raw_lambda).clamp_min(EPSILON)
            return (
                torch.distributions.Poisson(est_lambda)
                .log_prob(feature_values)
                .sum(dim=1)
            )

        raise ValueError(f"Unsupported free likelihood: {spec.likelihood}")

    def _sample_feature(
        self,
        spec: FeatureSpec,
        theta: tuple[torch.Tensor, ...],
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mean, var = stats
        if spec.likelihood == "real":
            est_mean, est_logvar = theta
            est_var = var * F.softplus(est_logvar).clamp(min=EPSILON, max=1.0)
            est_mean = torch.sqrt(var) * est_mean + mean
            return torch.normal(est_mean, torch.sqrt(est_var))

        if spec.likelihood == "pos":
            est_mean, est_logvar = theta
            est_var = var * F.softplus(est_logvar).clamp(min=EPSILON, max=1.0)
            est_mean = torch.sqrt(var) * est_mean + mean
            sample = torch.normal(est_mean, torch.sqrt(est_var))
            return torch.exp(sample) - 1.0

        if spec.likelihood == "count":
            (raw_lambda,) = theta
            est_lambda = F.softplus(raw_lambda).clamp_min(EPSILON)
            return torch.poisson(est_lambda)

        raise ValueError(f"Unsupported free likelihood: {spec.likelihood}")

    def compute_loss(
        self,
        free_values: torch.Tensor,
        conditional_values: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_free, free_stats = normalize_feature_block(
            free_values, self.free_specs
        )
        normalized_conditional, _ = normalize_feature_block(
            conditional_values, self.conditional_specs
        )

        log_pi = self.s_encoder(
            torch.cat([normalized_free, normalized_conditional], dim=1)
        )
        samples_s = F.gumbel_softmax(log_pi, tau=tau, hard=False)
        mean_qz, logvar_qz = self._poe_qz(
            normalized_free=normalized_free,
            normalized_conditional=normalized_conditional,
            samples_s=samples_s,
        )
        eps = torch.randn_like(mean_qz)
        samples_z = mean_qz + torch.exp(0.5 * logvar_qz) * eps

        mean_pz = self.pz_mean(samples_s)
        logvar_pz = torch.zeros_like(mean_pz)

        theta = self._theta_from_z(samples_z)
        log_p_x = []
        for index, (feature_name, feature_theta) in enumerate(theta):
            spec = self.free_specs[index]
            if spec.name != feature_name:
                raise RuntimeError("Decoder feature order mismatch")
            log_p_x.append(
                self._log_prob_feature(
                    feature_values=free_values[:, index : index + 1],
                    spec=spec,
                    theta=feature_theta,
                    stats=free_stats[index],
                )
            )

        reconstruction = torch.stack(log_p_x, dim=0).sum(dim=0)
        pi = torch.softmax(log_pi, dim=1)
        kl_s = torch.sum(pi * torch.log(pi.clamp_min(EPSILON)), dim=1) + math.log(
            float(self.s_dim)
        )
        kl_z = -0.5 * self.z_dim + 0.5 * torch.sum(
            torch.exp(logvar_qz - logvar_pz)
            + ((mean_pz - mean_qz) ** 2) / torch.exp(logvar_pz)
            - logvar_qz
            + logvar_pz,
            dim=1,
        )
        elbo = torch.mean(1.20 * reconstruction - (kl_z + kl_s))
        loss = -elbo
        return loss, reconstruction.mean(), kl_z.mean(), kl_s.mean()

    @torch.no_grad()
    def encode_deterministic(
        self,
        free_values: torch.Tensor,
        conditional_values: torch.Tensor,
        free_stats: list[tuple[torch.Tensor, torch.Tensor]],
        conditional_stats: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        normalized_free = normalize_with_stats(free_values, self.free_specs, free_stats)
        normalized_conditional = normalize_with_stats(
            conditional_values, self.conditional_specs, conditional_stats
        )
        log_pi = self.s_encoder(
            torch.cat([normalized_free, normalized_conditional], dim=1)
        )
        samples_s = F.one_hot(log_pi.argmax(dim=1), num_classes=self.s_dim).to(
            dtype=free_values.dtype
        )
        mean_qz, _ = self._poe_qz(
            normalized_free=normalized_free,
            normalized_conditional=normalized_conditional,
            samples_s=samples_s,
        )
        return mean_qz

    @torch.no_grad()
    def sample_from_latent(
        self,
        z_values: torch.Tensor,
        free_stats: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        theta = self._theta_from_z(z_values)
        outputs = []
        for index, (_, feature_theta) in enumerate(theta):
            outputs.append(
                self._sample_feature(
                    spec=self.free_specs[index],
                    theta=feature_theta,
                    stats=free_stats[index],
                )
            )
        return torch.cat(outputs, dim=1).clamp_min(0.0)
