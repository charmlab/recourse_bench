from __future__ import annotations

import contextlib
import copy
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from model.mlp_bayesian.sampler import H_SA_SGHMC
from model.model_object import ModelObject, process_nan
from model.model_utils import resolve_device
from utils.caching import get_cache_dir
from utils.registry import register
from utils.seed import seed_context

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore


class BayesianMlpNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, layers: list[int], output_dim: int):
        super().__init__()
        blocks: list[torch.nn.Module] = []
        current_dim = input_dim
        for hidden_dim in layers:
            blocks.append(torch.nn.Linear(current_dim, hidden_dim))
            blocks.append(torch.nn.ReLU())
            current_dim = hidden_dim
        blocks.append(torch.nn.Linear(current_dim, output_dim))
        self.block = torch.nn.Sequential(*blocks)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.block(X)


def _load_pickle_object(path: str):
    with Path(path).open("rb") as file:
        try:
            return pickle.load(file)
        except Exception:
            file.seek(0)
            try:
                return pickle.load(file, encoding="latin1")
            except Exception:
                file.seek(0)
                return pickle.load(file, encoding="bytes")


def _decode_state_dict_keys(state_dict: dict) -> dict[str, torch.Tensor]:
    decoded: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(key, bytes):
            decoded_key = key.decode("utf-8")
        else:
            decoded_key = str(key)
        decoded[decoded_key] = value
    return decoded


@register("mlp_bayesian")
class MlpBayesianModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        epochs: int = 2400,
        learning_rate: float = 0.01,
        batch_size: int = 512,
        layers: list[int] | None = None,
        optimizer: str = "sghmc",
        criterion: str = "cross_entropy",
        output_activation: str = "softmax",
        pretrained_path: str | None = None,
        save_name: str | None = None,
        burn_in: int = 120,
        sim_steps: int = 20,
        n_saves: int = 100,
        resample_its: int = 10,
        resample_prior_its: int = 50,
        re_burn: int = 10**7,
        grad_std_mul: float = 30.0,
        base_C: float = 0.05,
        gauss_sig: float = 0.1,
        subsample: int = 1,
        **kwargs,
    ):
        self._model: BayesianMlpNetwork
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._bayesian = True

        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._batch_size = int(batch_size)
        self._layers = list(layers or [200, 200])
        self._optimizer_name = str(optimizer).lower()
        self._criterion_name = str(criterion).lower()
        self._output_activation_name = str(output_activation).lower()
        self._pretrained_path = pretrained_path
        self._save_name = save_name
        self._burn_in = int(burn_in)
        self._sim_steps = int(sim_steps)
        self._n_saves = int(n_saves)
        self._resample_its = int(resample_its)
        self._resample_prior_its = int(resample_prior_its)
        self._re_burn = int(re_burn)
        self._grad_std_mul = float(grad_std_mul)
        self._base_C = float(base_C)
        self._gauss_sig = float(gauss_sig)
        self._subsample = int(subsample)

        self._posterior_samples: list[dict[str, torch.Tensor]] = []
        self._posterior_samples_device: list[dict[str, torch.Tensor]] | None = None
        self._output_dim: int | None = None

        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._criterion_name != "cross_entropy":
            raise ValueError(
                "MlpBayesianModel currently supports criterion='cross_entropy' only"
            )
        if self._output_activation_name != "softmax":
            raise ValueError(
                "MlpBayesianModel currently supports output_activation='softmax' only"
            )
        if self._optimizer_name not in {"sghmc", "h_sa_sghmc"}:
            raise ValueError(
                "MlpBayesianModel currently supports optimizer='sghmc' only"
            )
        if self._subsample < 1:
            raise ValueError("subsample must be >= 1")

    def _build_model(self, input_dim: int, output_dim: int) -> BayesianMlpNetwork:
        return BayesianMlpNetwork(
            input_dim=input_dim, layers=self._layers, output_dim=output_dim
        )

    def _resolve_pretrained_path(self) -> Path | None:
        if self._pretrained_path is None:
            return None
        path = Path(self._pretrained_path)
        if path.is_dir():
            path = path / "state_dicts.pkl"
        return path

    def _save_posterior_samples(self) -> None:
        if self._save_name is None:
            return
        save_dir = Path(get_cache_dir("models"))
        save_path = save_dir / f"{self._save_name}.pkl"
        with save_path.open("wb") as file:
            pickle.dump(self._posterior_samples, file, pickle.HIGHEST_PROTOCOL)

    def _load_posterior_samples(self, path: Path) -> None:
        raw_samples = _load_pickle_object(path.as_posix())
        if not isinstance(raw_samples, list):
            raise TypeError("Bayesian posterior samples must load as a list")
        decoded = [_decode_state_dict_keys(state) for state in raw_samples]
        self._posterior_samples = decoded[:: self._subsample]
        self._posterior_samples_device = None
        if len(self._posterior_samples) == 0:
            raise ValueError("No posterior samples available after subsampling")

    def _sample_state_dicts_for_device(self) -> list[dict[str, torch.Tensor]]:
        if self._posterior_samples_device is None:
            samples: list[dict[str, torch.Tensor]] = []
            for state in self._posterior_samples:
                samples.append(
                    {
                        key: value.detach().to(self._device)
                        for key, value in state.items()
                    }
                )
            self._posterior_samples_device = samples
        return self._posterior_samples_device

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for MlpBayesianModel.fit()")

        with seed_context(self._seed):
            X, labels, output_dim = self.extract_training_data(trainset)
            input_dim = X.shape[1]
            self._output_dim = output_dim
            self._model = self._build_model(
                input_dim=input_dim, output_dim=output_dim
            ).to(self._device)

            pretrained_path = self._resolve_pretrained_path()
            if pretrained_path is not None and pretrained_path.exists():
                self._load_posterior_samples(pretrained_path)
                self._is_trained = True
                self._model.eval()
                return

            X_tensor = torch.tensor(X.to_numpy(dtype="float32"), dtype=torch.float32)
            y_tensor = labels.detach().cpu()
            train_loader = DataLoader(
                TensorDataset(X_tensor, y_tensor),
                batch_size=self._batch_size,
                shuffle=True,
            )

            optimizer = H_SA_SGHMC(
                params=self._model.parameters(),
                lr=self._learning_rate,
                base_C=self._base_C,
                gauss_sig=self._gauss_sig,
            )
            grad_buffer: list[float] = []
            max_grad = 1e20
            iteration_count = 0

            self._model.train()
            for epoch_index in tqdm(
                range(self._epochs),
                desc="mlp-bayesian-fit",
                leave=False,
            ):
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self._device)
                    batch_y = batch_y.to(self._device)
                    optimizer.zero_grad()
                    logits = self._model(batch_X)
                    loss = F.cross_entropy(logits, batch_y, reduction="mean")
                    loss = loss * X_tensor.shape[0]
                    loss.backward()

                    if len(grad_buffer) > 1000:
                        grad_array = torch.tensor(grad_buffer, dtype=torch.float32)
                        max_grad = float(
                            grad_array.mean().item()
                            + self._grad_std_mul * grad_array.std(unbiased=False).item()
                        )
                        grad_buffer.pop(0)

                    clipped_grad = torch.nn.utils.clip_grad_norm_(
                        parameters=self._model.parameters(),
                        max_norm=max_grad,
                        norm_type=2,
                    )
                    grad_value = float(clipped_grad.detach().cpu().item())
                    if grad_value < max_grad:
                        grad_buffer.append(grad_value)

                    optimizer.step(
                        burn_in=(epoch_index % self._re_burn) < self._burn_in,
                        resample_momentum=(iteration_count % self._resample_its == 0),
                        resample_prior=(
                            iteration_count % self._resample_prior_its == 0
                        ),
                    )
                    iteration_count += 1

                if (
                    epoch_index % self._re_burn >= self._burn_in
                    and epoch_index % self._sim_steps == 0
                ):
                    if len(self._posterior_samples) >= self._n_saves:
                        self._posterior_samples.pop(0)
                    self._posterior_samples.append(
                        {
                            key: value.detach().cpu().clone()
                            for key, value in self._model.state_dict().items()
                        }
                    )

            if len(self._posterior_samples) == 0:
                self._posterior_samples.append(
                    {
                        key: value.detach().cpu().clone()
                        for key, value in self._model.state_dict().items()
                    }
                )

            self._posterior_samples_device = None
            self._save_posterior_samples()
            self._is_trained = True
            self._model.eval()

    def posterior_sample_count(self) -> int:
        return len(self._posterior_samples)

    def sample_predict(
        self,
        X: torch.Tensor,
        Nsamples: int = 0,
        grad: bool = False,
    ) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        if X.ndim == 1:
            X = X.unsqueeze(0)

        X = X.to(self._device)
        sample_count = Nsamples if Nsamples > 0 else len(self._posterior_samples)
        effective_count = sample_count if sample_count <= 1 else sample_count - 1
        state_dicts = self._sample_state_dicts_for_device()[:effective_count]
        if len(state_dicts) == 0:
            raise RuntimeError("No posterior samples available")

        probs: list[torch.Tensor] = []
        context = contextlib.nullcontext() if grad else torch.no_grad()
        with context:
            for state_dict in state_dicts:
                logits = functional_call(self._model, state_dict, (X,))
                probs.append(torch.softmax(logits, dim=1))
        return torch.stack(probs, dim=0)

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")

        with seed_context(self._seed):
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"), dtype=torch.float32, device=self._device
            )
            probabilities = self.sample_predict(X_tensor, Nsamples=0, grad=False).mean(
                dim=0
            )
            probabilities = probabilities.detach().cpu()
            if proba:
                return probabilities
            indices = probabilities.argmax(dim=1)
            return torch.nn.functional.one_hot(
                indices, num_classes=probabilities.shape[1]
            ).to(dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        probabilities = self.sample_predict(X, Nsamples=0, grad=X.requires_grad).mean(
            dim=0
        )
        return torch.log(probabilities.clamp_min(1e-12))
