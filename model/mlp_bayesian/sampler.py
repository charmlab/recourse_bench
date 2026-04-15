from __future__ import annotations

import torch
from numpy.random import gamma
from torch.optim import Optimizer


class H_SA_SGHMC(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        base_C: float = 0.05,
        gauss_sig: float = 0.1,
        alpha0: float = 10,
        beta0: float = 10,
    ):
        self.eps = 1e-6
        self.alpha0 = alpha0
        self.beta0 = beta0

        if gauss_sig == 0:
            self.weight_decay = 0
        else:
            self.weight_decay = 1 / (gauss_sig**2)

        if self.weight_decay <= 0.0:
            raise ValueError(f"Invalid weight_decay value: {self.weight_decay}")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if base_C < 0:
            raise ValueError(f"Invalid friction term: {base_C}")

        defaults = dict(lr=lr, base_C=base_C)
        super().__init__(params, defaults)

    def step(
        self,
        burn_in: bool = False,
        resample_momentum: bool = False,
        resample_prior: bool = False,
    ):
        loss = None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["V_hat"] = torch.ones_like(parameter)
                    state["v_momentum"] = torch.zeros_like(parameter)
                    state["weight_decay"] = self.weight_decay

                state["iteration"] += 1

                if resample_prior:
                    alpha = self.alpha0 + parameter.data.nelement() / 2
                    beta = self.beta0 + (parameter.data**2).sum().item() / 2
                    gamma_sample = gamma(shape=alpha, scale=1 / beta, size=None)
                    state["weight_decay"] = gamma_sample

                base_C = group["base_C"]
                lr = group["lr"]
                weight_decay = state["weight_decay"]
                tau, g, V_hat = state["tau"], state["g"], state["V_hat"]

                grad = parameter.grad.data
                if weight_decay != 0:
                    grad = grad.add(parameter.data, alpha=weight_decay)

                if burn_in:
                    tau.add_(-tau * (g**2) / (V_hat + self.eps) + 1)
                    tau_inv = 1.0 / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * grad)
                    V_hat.add_(-tau_inv * V_hat + tau_inv * (grad**2))

                V_inv_sqrt = 1.0 / (torch.sqrt(V_hat) + self.eps)

                if resample_momentum:
                    state["v_momentum"] = torch.normal(
                        mean=torch.zeros_like(grad),
                        std=torch.sqrt((lr**2) * V_inv_sqrt),
                    )
                v_momentum = state["v_momentum"]

                noise_var = 2.0 * (lr**2) * V_inv_sqrt * base_C - (lr**4)
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                noise_sample = torch.normal(
                    mean=torch.zeros_like(grad),
                    std=torch.ones_like(grad) * noise_std,
                )

                v_momentum.add_(
                    -(lr**2) * V_inv_sqrt * grad - base_C * v_momentum + noise_sample
                )
                parameter.data.add_(v_momentum)

        return loss
