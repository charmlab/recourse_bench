# methods/catalog/rbr/library.py
import math
from typing import Any, Optional, Sequence

import numpy as np
import torch
from sklearn.utils import check_random_state

"""
This code is largely ported over from the original authors codebase.
Light restructuring and modifications have been made in order to make it compatible with CARLAs structure.

Original code can be found at: https://github.com/VinAIResearch/robust-bayesian-recourse
"""

# ---------- low-level helpers & projections ----------


@torch.no_grad()
def l2_projection(x: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Euclidean projection onto an L2-ball for last axis.
    x: shape (..., d)
    radius: scalar
    """
    norm = torch.linalg.norm(x, ord=2, axis=-1)
    # avoid divide by zero
    denom = torch.max(norm, torch.tensor(radius, device=x.device))
    scale = (radius / denom).unsqueeze(1)
    return scale * x


def reconstruct_encoding_constraints(
    x: torch.Tensor,
    cat_pos: Optional[Sequence[Sequence[int]]],
) -> torch.Tensor:
    if not cat_pos:
        return x

    x_enc = x.clone()
    squeeze_output = False
    if x_enc.ndim == 1:
        x_enc = x_enc.unsqueeze(0)
        squeeze_output = True

    for positions in cat_pos:
        position_list = list(positions)
        if len(position_list) == 0:
            continue
        if len(position_list) == 1:
            x_enc[:, position_list[0]] = torch.round(x_enc[:, position_list[0]])
            continue

        group_tensor = x_enc[:, position_list]
        max_indices = torch.argmax(group_tensor, dim=1)
        x_enc[:, position_list] = 0.0
        row_indices = torch.arange(x_enc.shape[0], device=x_enc.device)
        absolute_feature_indices = torch.tensor(
            position_list,
            device=x_enc.device,
        )[max_indices]
        x_enc[row_indices, absolute_feature_indices] = 1.0

    if squeeze_output:
        return x_enc.squeeze(0)
    return x_enc


# ---------- likelihood modules ----------


class OptimisticLikelihood(torch.nn.Module):
    def __init__(
        self,
        x_dim: torch.Tensor,
        epsilon_op: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.x_dim = x_dim.to(self.device)
        self.epsilon_op = epsilon_op.to(self.device)
        self.sigma = sigma.to(self.device)

    @torch.no_grad()
    def projection(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clone()
        v = torch.max(v, torch.tensor(0, device=self.device))
        result = l2_projection(v, float(self.epsilon_op))
        return result.to(self.device)

    def _forward(self, v: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim
        L = (
            torch.log(d)
            + (c - v[..., 0]) ** 2 / (2 * d**2)
            + (p - 1) * torch.log(self.sigma)
        )
        return L

    def forward(self, v: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim

        L = (
            torch.log(d)
            + (c - v[..., 0]) ** 2 / (2 * d**2)
            + (p - 1) * torch.log(self.sigma)
        )

        v_grad = torch.zeros_like(v, device=self.device)
        v_grad[..., 0] = -(c - v[..., 0]) / d**2
        v_grad[..., 1] = 1 / d - (c - v[..., 0]) ** 2 / d**3

        return L, v_grad

    def optimize(
        self,
        x: torch.Tensor,
        x_feas: torch.Tensor,
        max_iter: int = int(1e3),
        verbose: bool = False,
    ):
        v = torch.zeros([x.shape[0], 2], device=self.device)
        lr = 1 / torch.sqrt(torch.tensor(max_iter, device=self.device).float())

        loss_diff = 1.0
        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(v, x, x_feas)
            v = self.projection(v - lr * grad)

            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum
            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)
            if verbose and (t % 200 == 0):
                print(f"[Optimistic] iter {t} loss {loss_sum:.6f}")
        return v


class PessimisticLikelihood(torch.nn.Module):
    def __init__(
        self,
        x_dim: torch.Tensor,
        epsilon_pe: torch.Tensor,
        sigma: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.epsilon_pe = epsilon_pe.to(self.device)
        self.sigma = sigma.to(self.device)
        self.x_dim = x_dim.to(self.device)

    @torch.no_grad()
    def projection(self, u: torch.Tensor) -> torch.Tensor:
        u = u.clone()
        u = torch.max(u, torch.tensor(0, device=self.device))
        result = l2_projection(u, float(self.epsilon_pe) / math.sqrt(float(self.x_dim)))
        return result.to(self.device)

    def _forward(
        self, u: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor, zeta: float = 1e-6
    ):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        # p = p.float()
        sqrt_p = torch.sqrt(p.float())

        inside = (zeta + self.epsilon_pe**2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (
            p - 1
        )
        # f = torch.sqrt(torch.maximum(inside, torch.tensor(1e-12, device=self.device)))
        f = torch.sqrt(inside)

        L = (
            -torch.log(d)
            - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d**2)
            - (p - 1) * torch.log(f + self.sigma)
        )
        return L

    def forward(
        self, u: torch.Tensor, x: torch.Tensor, x_feas: torch.Tensor, zeta: float = 1e-6
    ):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim

        # p = p.float() # issue with support with int tensors when taking sqrt?

        sqrt_p = torch.sqrt(p.float())
        inside = (zeta + self.epsilon_pe**2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (
            p - 1
        )
        # f = torch.sqrt(torch.maximum(inside, torch.tensor(1e-12, device=self.device)))
        f = torch.sqrt(inside)

        L = (
            -torch.log(d)
            - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d**2)
            - (p - 1) * torch.log(f + self.sigma)
        )

        u_grad = torch.zeros_like(u, device=self.device)
        u_grad[..., 0] = -sqrt_p * (c + sqrt_p * u[..., 0]) / d**2 - (
            p * u[..., 0]
        ) / (f * (f + self.sigma))
        u_grad[..., 1] = (
            -1 / d
            + (c + sqrt_p * u[..., 0]) ** 2 / d**3
            + u[..., 1] / (f * (f + self.sigma))
        )

        return L, u_grad

    def optimize(
        self,
        x: torch.Tensor,
        x_feas: torch.Tensor,
        max_iter: int = int(1e3),
        verbose: bool = False,
    ):
        u = torch.zeros([x.shape[0], 2], device=self.device)
        lr = 1.0 / torch.sqrt(torch.tensor(max_iter, device=self.device).float())

        loss_diff = 1.0
        min_loss = float("inf")
        num_stable_iter = 0
        max_stable_iter = 10

        for t in range(max_iter):
            F, grad = self.forward(u, x, x_feas)
            u = self.projection(u - lr * grad)

            loss_sum = F.sum().data.item()
            loss_diff = min_loss - loss_sum

            if loss_diff <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= max_stable_iter:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)
            if verbose and (t % 200 == 0):
                print(f"[Pessimistic] iter {t} loss {loss_sum:.6f}")
        return u


# ---------- RBRLoss wrapper ----------


class RBRLoss(torch.nn.Module):
    def __init__(
        self,
        X_feas: torch.Tensor,
        X_feas_pos: torch.Tensor,
        X_feas_neg: torch.Tensor,
        epsilon_op: float,
        epsilon_pe: float,
        sigma: float,
        device: torch.device,
        verbose: bool = False,
    ):
        super(RBRLoss, self).__init__()
        self.device = device
        self.verbose = verbose

        self.X_feas = X_feas.to(self.device)
        self.X_feas_pos = X_feas_pos.to(self.device)
        self.X_feas_neg = X_feas_neg.to(self.device)

        self.epsilon_op = torch.tensor(epsilon_op, device=self.device)
        self.epsilon_pe = torch.tensor(epsilon_pe, device=self.device)
        self.sigma = torch.tensor(sigma, device=self.device)
        self.x_dim = torch.tensor(X_feas.shape[-1], device=self.device)

        # print("This is epsilon op: ", self.epsilon_op)
        # print("This is epsilon pe: ", self.epsilon_pe)

        self.op_likelihood = OptimisticLikelihood(
            self.x_dim, self.epsilon_op, self.sigma, self.device
        )
        self.pe_likelihood = PessimisticLikelihood(
            self.x_dim, self.epsilon_pe, self.sigma, self.device
        )

    def forward(self, x: torch.Tensor, verbose: bool = False):
        if verbose or self.verbose:
            print(f"N_neg: {self.X_feas_neg.shape}, N_pos: {self.X_feas_pos.shape}")

        # pessimistic part
        if self.X_feas_pos.shape[0] > 0:
            u = self.pe_likelihood.optimize(
                x.detach().clone().expand([self.X_feas_pos.shape[0], -1]),
                self.X_feas_pos,
                verbose=self.verbose,
            )
            F_pe = self.pe_likelihood._forward(
                u, x.expand([self.X_feas_pos.shape[0], -1]), self.X_feas_pos
            )
            denom = torch.logsumexp(F_pe, -1)
        else:
            denom = torch.tensor(0.0, device=self.device)

        # optimistic part
        if self.X_feas_neg.shape[0] > 0:
            v = self.op_likelihood.optimize(
                x.detach().clone().expand([self.X_feas_neg.shape[0], -1]),
                self.X_feas_neg,
                verbose=self.verbose,
            )
            F_op = self.op_likelihood._forward(
                v, x.expand([self.X_feas_neg.shape[0], -1]), self.X_feas_neg
            )
            numer = torch.logsumexp(-F_op, -1)
        else:
            numer = torch.tensor(0.0, device=self.device)

        result = numer - denom
        return result, denom, numer


# ---------- high-level RBR generator (callable used by CARLA wrapper) ----------


def robust_bayesian_recourse(
    raw_model: Any,
    x0: np.ndarray,
    y_target: int = 1,
    cat_features_indices: Optional[Sequence[Sequence[int]]] = None,
    train_data: Optional[np.ndarray] = None,
    train_t: Optional[torch.Tensor] = None,
    train_label: Optional[torch.Tensor] = None,
    num_samples: int = 200,
    perturb_radius: float = 0.2,
    delta_plus: float = 1.0,
    sigma: float = 1.0,
    epsilon_op: float = 0.5,
    epsilon_pe: float = 1.0,
    max_iter: int = 1000,
    dev: str = "cpu",
    random_state: Optional[int] = 42,
    verbose: bool = False,
) -> np.ndarray:
    device = torch.device(dev)

    def predict_label_indices(
        x: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        if hasattr(raw_model, "predict_label_indices"):
            labels = raw_model.predict_label_indices(x)
        else:
            predictions = raw_model.predict(x)
            if isinstance(predictions, torch.Tensor):
                labels = predictions.detach().cpu().numpy()
            else:
                labels = np.asarray(predictions)

            if labels.ndim == 2 and labels.shape[1] > 1:
                labels = labels.argmax(axis=1)
            else:
                labels = np.asarray(labels).reshape(-1)
                if labels.dtype.kind == "f":
                    labels = (labels >= 0.5).astype(np.int64)
                else:
                    labels = labels.astype(np.int64, copy=False)

        labels = np.asarray(labels)
        if labels.ndim == 0:
            labels = labels.reshape(1)
        return labels.astype(np.int64, copy=False)

    def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(a - b, ord=1, axis=-1)

    def uniform_ball(
        x: torch.Tensor,
        radius: float,
        n: int,
        rng_state,
    ) -> torch.Tensor:
        rng_local = check_random_state(rng_state)
        dimension = x.shape[0]
        samples = rng_local.randn(n, dimension)
        samples = samples / np.linalg.norm(samples, axis=1).reshape(-1, 1)
        samples = samples * (rng_local.random(n) ** (1.0 / dimension)).reshape(-1, 1)
        samples = samples * radius + x.detach().cpu().numpy()
        return torch.from_numpy(samples).float().to(device)

    def simplex_projection(x: torch.Tensor, delta: float) -> torch.Tensor:
        (dimension,) = x.shape
        if torch.linalg.norm(x, ord=1) == delta and torch.all(x >= 0):
            return x
        sorted_x, _ = torch.sort(x, descending=True)
        cumulative_sum = torch.cumsum(sorted_x, 0)
        rho = torch.nonzero(
            sorted_x
            * torch.arange(1, dimension + 1, device=device)
            > (cumulative_sum - delta)
        )[-1, 0]
        theta = (cumulative_sum[rho] - delta) / (rho + 1.0)
        return torch.clip(x - theta, min=0)

    def projection(x: torch.Tensor, delta: float) -> torch.Tensor:
        x_abs = torch.abs(x)
        if x_abs.sum() <= delta:
            return x

        projected = simplex_projection(x_abs, delta=delta)
        projected *= torch.sign(x)
        return projected

    rng = check_random_state(random_state)
    if train_t is None and train_data is None:
        raise ValueError(
            "train_data or train_t must be provided to robust_bayesian_recourse"
        )

    x0_t = torch.from_numpy(np.asarray(x0, dtype=np.float32).copy()).float().to(device)
    if train_t is None:
        train_t = torch.tensor(train_data, dtype=torch.float32, device=device)
    else:
        train_t = train_t.detach().clone().to(device)

    if train_label is None:
        train_label = torch.tensor(
            predict_label_indices(train_t),
            dtype=torch.long,
            device=device,
        )
    else:
        train_label = train_label.detach().clone().to(device)

    x_label = int(predict_label_indices(x0_t)[0])
    target_label = int(y_target)
    opposite_label = 1 - x_label

    dists = dist(train_t, x0_t)
    order = torch.argsort(dists)
    candidates = train_t[order[train_label[order] == opposite_label]][:1000]

    best_x_boundary: Optional[torch.Tensor] = None
    best_distance = torch.tensor(float("inf"), device=device)
    for candidate in candidates:
        lambdas = torch.linspace(0, 1, 100, device=device)
        for lam in lambdas:
            x_boundary = (1 - lam) * x0_t + lam * candidate
            label = int(predict_label_indices(x_boundary)[0])
            if label == opposite_label:
                current_distance = dist(x0_t, x_boundary)
                if current_distance < best_distance:
                    best_x_boundary = x_boundary.detach().clone()
                    best_distance = current_distance.detach().clone()
                break

    if best_x_boundary is None:
        if candidates.shape[0] == 0:
            return x0.copy()
        best_x_boundary = candidates[0].detach().clone()
        best_distance = dist(x0_t, best_x_boundary)

    delta = best_distance + delta_plus
    X_feas = uniform_ball(best_x_boundary, perturb_radius, num_samples, rng).float()
    if cat_features_indices:
        X_feas = reconstruct_encoding_constraints(X_feas, cat_features_indices)

    y_feas = predict_label_indices(X_feas)
    X_feas_pos = X_feas[y_feas == target_label].reshape(
        [int(np.sum(y_feas == target_label)), -1]
    )
    X_feas_neg = X_feas[y_feas == (1 - target_label)].reshape(
        [int(np.sum(y_feas == (1 - target_label))), -1]
    )

    loss_fn = RBRLoss(
        X_feas,
        X_feas_pos,
        X_feas_neg,
        epsilon_op,
        epsilon_pe,
        sigma,
        device=device,
        verbose=verbose,
    )

    x_t = best_x_boundary.detach().clone()
    x_t.requires_grad_(True)
    min_loss = float("inf")
    num_stable_iter = 0
    max_stable_iter = 10
    step = 1.0 / math.sqrt(1e3)

    for _ in range(max_iter):
        if x_t.grad is not None:
            x_t.grad.data.zero_()

        objective, _, _ = loss_fn(x_t)
        objective.backward()

        if torch.ge(dist(x_t.detach(), x0_t), delta):
            break

        with torch.no_grad():
            x_new = x_t - step * x_t.grad
            x_new = projection(x_new - x0_t, float(delta)) + x0_t

        if cat_features_indices:
            x_new = reconstruct_encoding_constraints(x_new, cat_features_indices)

        for index, element in enumerate(x_new.data):
            x_t.data[index] = element

        loss_sum = objective.sum().data.item()
        loss_diff = min_loss - loss_sum
        if loss_diff <= 1e-10:
            num_stable_iter += 1
            if num_stable_iter >= max_stable_iter:
                break
        else:
            num_stable_iter = 0

        min_loss = min(min_loss, loss_sum)

    return x_t.detach().cpu().numpy().squeeze()
