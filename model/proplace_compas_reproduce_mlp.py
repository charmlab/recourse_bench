from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from model.mlp.mlp import MlpModel
from model.model_utils import resolve_device, save_torch_model
from utils.registry import register
from utils.seed import seed_context


@register("proplace_compas_reproduce_mlp")
class ProplaceCompasReproduceMlpModel(MlpModel):
    """PyTorch MLP matching the reference ProPlace notebook training loop."""

    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        hidden_size: int = 20,
        epochs: int = 50,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        l2_lambda: float = 0.0001,
        pretrained_path: str | None = None,
        save_name: str | None = None,
        **kwargs,
    ):
        self._model: torch.nn.Module
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._batch_size = int(batch_size)
        self._hidden_size = int(hidden_size)
        self._layers = [self._hidden_size, self._hidden_size]
        self._optimizer_name = "adam"
        self._criterion_name = "bce"
        self._output_activation_name = "sigmoid"
        self._pretrained_path = pretrained_path
        self._save_name = save_name
        self._output_dim: int | None = None
        self._l2_lambda = float(l2_lambda)

        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self._hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if self._epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self._learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self._l2_lambda < 0.0:
            raise ValueError("l2_lambda must be >= 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError(
                "trainset is required for ProplaceCompasReproduceMlpModel.fit()"
            )

        with seed_context(self._seed):
            X, labels, _ = self.extract_training_data(trainset)
            if len(self.get_class_to_index()) != 2:
                raise ValueError(
                    "ProplaceCompasReproduceMlpModel supports binary classification only"
                )

            input_dim = X.shape[1]
            self._output_dim = 1
            self._model = self._build_model(input_dim, 1).to(self._device)

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

            features = torch.tensor(
                X.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            targets = labels.to(dtype=torch.float32).unsqueeze(1)
            train_loader = DataLoader(
                TensorDataset(features, targets),
                batch_size=self._batch_size,
                shuffle=True,
            )

            optimizer = torch.optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate,
            )
            criterion = torch.nn.BCELoss()

            self._model.train()
            epoch_iterator = tqdm(
                range(self._epochs),
                desc="proplace-compas-mlp-fit",
                leave=False,
            )
            for _ in epoch_iterator:
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self._device)
                    batch_y = batch_y.to(self._device)

                    optimizer.zero_grad()
                    logits = self._model(batch_x)
                    positive_probability = torch.sigmoid(logits)
                    loss_pred = criterion(positive_probability, batch_y)
                    l2_norm = sum(
                        parameter.pow(2.0).sum()
                        for parameter in self._model.parameters()
                    )
                    loss = loss_pred + self._l2_lambda * l2_norm
                    loss.backward()
                    optimizer.step()

            self._model.eval()
            self._is_trained = True
            save_torch_model(self._model, self._save_name)
