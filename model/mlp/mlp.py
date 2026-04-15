from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject, process_nan
from model.model_utils import (
    build_optimizer,
    logits_to_prediction,
    resolve_device,
    save_torch_model,
)
from utils.registry import register
from utils.seed import seed_context


@register("mlp")
class MlpModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        epochs: int = 160,
        learning_rate: float = 0.01,
        batch_size: int | None = 16,
        layers: list[int] | None = None,
        l2_lambda: float = 0.0,
        optimizer: str = "adam",
        criterion: str = "cross_entropy",
        output_activation: str = "softmax",
        pretrained_path: str | None = None,
        save_name: str | None = None,
        weight_decay: float = 0.0,
        loss_reduction: str = "mean",
        xavier_uniform_init: bool = False,
        early_stop_tol: float | None = None,
        early_stop_patience: int | None = None,
        **kwargs,
    ):
        self._model: torch.nn.Module
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._epochs: int = int(epochs)
        self._learning_rate: float = float(learning_rate)
        self._batch_size: int | None = (
            None if batch_size is None else int(batch_size)
        )
        self._layers: list[int] = list(layers or [32, 16])
        self._l2_lambda: float = float(l2_lambda)
        self._optimizer_name: str = optimizer
        self._criterion_name: str = criterion.lower()
        self._output_activation_name: str = output_activation.lower()
        self._pretrained_path: str | None = pretrained_path
        self._save_name: str | None = save_name
        self._weight_decay: float = float(weight_decay)
        self._loss_reduction: str = str(loss_reduction).lower()
        self._xavier_uniform_init: bool = bool(xavier_uniform_init)
        self._early_stop_tol: float | None = (
            None if early_stop_tol is None else float(early_stop_tol)
        )
        self._early_stop_patience: int | None = (
            None if early_stop_patience is None else int(early_stop_patience)
        )
        self._output_dim: int | None = None

        if self._batch_size is not None and self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._l2_lambda < 0.0:
            raise ValueError("l2_lambda must be >= 0")
        if self._weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self._loss_reduction not in {"mean", "sum"}:
            raise ValueError("loss_reduction must be 'mean' or 'sum'")
        if self._early_stop_patience is not None and self._early_stop_patience < 1:
            raise ValueError("early_stop_patience must be >= 1")
        if self._early_stop_tol is not None and self._early_stop_tol < 0:
            raise ValueError("early_stop_tol must be >= 0")
        if self._criterion_name not in {"cross_entropy", "bce"}:
            raise ValueError(
                "MlpModel supports criterion='cross_entropy' or criterion='bce' only"
            )
        if self._output_activation_name not in {"softmax", "sigmoid"}:
            raise ValueError(
                "MlpModel output_activation must be 'softmax' or 'sigmoid'"
            )
        if self._output_activation_name == "sigmoid" and self._criterion_name != "bce":
            raise ValueError(
                "MlpModel output_activation='sigmoid' requires criterion='bce'"
            )
        if self._output_activation_name == "softmax" and self._criterion_name == "bce":
            raise ValueError(
                "MlpModel output_activation='softmax' is incompatible with criterion='bce'"
            )

    def _build_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        blocks: list[torch.nn.Module] = []
        current_dim = input_dim
        for hidden_dim in self._layers:
            blocks.append(torch.nn.Linear(current_dim, hidden_dim))
            blocks.append(torch.nn.ReLU())
            current_dim = hidden_dim
        blocks.append(torch.nn.Linear(current_dim, output_dim))
        model = torch.nn.Sequential(*blocks)
        if self._xavier_uniform_init:
            for parameter in model.parameters():
                if parameter.ndim > 1:
                    torch.nn.init.xavier_uniform_(parameter)
        return model

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for MlpModel.fit()")

        with seed_context(self._seed):
            X, labels, output_dim = self.extract_training_data(trainset)
            if self._output_activation_name == "sigmoid":
                if len(self.get_class_to_index()) != 2:
                    raise ValueError(
                        "MlpModel output_activation='sigmoid' supports binary classification only"
                    )
                network_output_dim = 1
            else:
                network_output_dim = output_dim
            input_dim = X.shape[1]
            self._output_dim = network_output_dim
            self._model = self._build_model(input_dim, network_output_dim).to(
                self._device
            )

            if (
                self._pretrained_path is not None
                and Path(self._pretrained_path).exists()
            ):
                state_dict = torch.load(
                    self._pretrained_path, map_location=self._device
                )
                self._model.load_state_dict(state_dict)
                self._model.eval()
                self._is_trained = True
                return

            optimizer = build_optimizer(
                self._optimizer_name,
                self._model.parameters(),
                self._learning_rate,
                weight_decay=self._weight_decay,
            )
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"), dtype=torch.float32, device=self._device
            )
            if self._criterion_name == "cross_entropy":
                criterion = torch.nn.CrossEntropyLoss(
                    reduction=self._loss_reduction
                )
                y_tensor = labels.to(self._device)
            else:
                criterion = torch.nn.BCELoss(reduction=self._loss_reduction)
                y_tensor = labels.to(self._device).to(dtype=torch.float32).unsqueeze(1)

            effective_batch_size = (
                X_tensor.shape[0]
                if self._batch_size is None
                else self._batch_size
            )
            previous_loss = float("inf")
            stable_iterations = 0

            self._model.train()
            for _ in tqdm(range(self._epochs), desc="mlp-fit", leave=False):
                permutation = torch.randperm(X_tensor.shape[0], device=self._device)
                for start in range(0, X_tensor.shape[0], effective_batch_size):
                    batch_indices = permutation[start : start + effective_batch_size]
                    batch_X = X_tensor[batch_indices]
                    batch_y = y_tensor[batch_indices]
                    optimizer.zero_grad()
                    logits = self._model(batch_X)
                    loss_input = (
                        torch.sigmoid(logits)
                        if self._criterion_name == "bce"
                        else logits
                    )
                    loss = criterion(loss_input, batch_y)
                    if self._l2_lambda > 0.0:
                        l2_norm = sum(
                            parameter.pow(2.0).sum()
                            for parameter in self._model.parameters()
                        )
                        loss = loss + self._l2_lambda * l2_norm
                    loss.backward()
                    optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                if self._early_stop_tol is None or self._early_stop_patience is None:
                    continue
                if previous_loss - loss_value <= self._early_stop_tol:
                    stable_iterations += 1
                    if stable_iterations >= self._early_stop_patience:
                        break
                else:
                    stable_iterations = 0
                previous_loss = loss_value

            self._model.eval()
            self._is_trained = True
            save_torch_model(self._model, self._save_name)

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"), dtype=torch.float32, device=self._device
            )
            with torch.no_grad():
                logits = self._model(X_tensor)
                prediction = logits_to_prediction(
                    logits,
                    proba=proba,
                    output_activation=self._output_activation_name,
                )
            return prediction.detach().cpu()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            logits = self._model(X.to(self._device))
            if self._output_activation_name == "sigmoid":
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)
                return torch.cat([-logits, logits], dim=1)
            return logits
