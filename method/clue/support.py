from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.clue.library.clue_ml.src.probability import decompose_entropy_cat
from model.mlp_bayesian.mlp_bayesian import MlpBayesianModel
from model.model_object import ModelObject
from preprocess.preprocess_utils import dataset_has_attr


def ensure_supported_target_model(
    target_model: ModelObject,
    method_name: str,
) -> None:
    if not isinstance(target_model, MlpBayesianModel):
        raise TypeError(
            f"{method_name} supports MlpBayesianModel only, "
            f"received {target_model.__class__.__name__}"
        )
    if not bool(getattr(target_model, "_bayesian", False)):
        raise ValueError(f"{method_name} requires target_model._bayesian=True")


def to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


def resolve_feature_names(dataset: DatasetObject) -> list[str]:
    return list(dataset.get(target=False).columns)


def resolve_input_dim_vec(dataset: DatasetObject) -> list[int]:
    feature_names = list(dataset.get(target=False).columns)
    if not dataset_has_attr(dataset, "encoding"):
        return [1] * len(feature_names)

    encoding_map = dataset.attr("encoding")
    source_by_column: dict[str, str] = {}
    for source_feature, encoded_columns in encoding_map.items():
        for encoded_column in encoded_columns:
            source_by_column[encoded_column] = source_feature

    input_dim_vec: list[int] = []
    current_source: str | None = None
    current_width = 0
    for feature_name in feature_names:
        source_feature = source_by_column.get(feature_name, feature_name)
        if current_source is None:
            current_source = source_feature
            current_width = 1
            continue
        if source_feature == current_source:
            current_width += 1
            continue
        input_dim_vec.append(current_width)
        current_source = source_feature
        current_width = 1
    if current_source is not None:
        input_dim_vec.append(current_width)
    return input_dim_vec


def resolve_vae_checkpoint_path(path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "theta_best.dat"
    return resolved


class ClueModelAdapter:
    def __init__(self, target_model: MlpBayesianModel, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)

    def sample_predict(
        self,
        X: torch.Tensor | np.ndarray | pd.DataFrame,
        grad: bool = False,
    ) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            tensor = X
        else:
            features = to_feature_dataframe(X, self._feature_names)
            tensor = torch.tensor(
                features.to_numpy(dtype="float32"),
                dtype=torch.float32,
                device=self._target_model._device,
            )
        return self._target_model.sample_predict(tensor, Nsamples=0, grad=grad)

    def predict_proba(
        self, X: torch.Tensor | np.ndarray | pd.DataFrame
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X, torch.Tensor):
            return self.sample_predict(X, grad=X.requires_grad).mean(dim=0)

        features = to_feature_dataframe(X, self._feature_names)
        prediction = self._target_model.get_prediction(features, proba=True)
        return prediction.detach().cpu().numpy()

    def predict_label_indices(
        self, X: torch.Tensor | np.ndarray | pd.DataFrame
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.argmax(dim=1).detach().cpu().numpy()
        return np.asarray(probabilities).argmax(axis=1)

    def entropy_tensor(self, X: torch.Tensor) -> torch.Tensor:
        prob_samples = self.sample_predict(X, grad=X.requires_grad)
        total_entropy, _, _ = decompose_entropy_cat(prob_samples)
        return total_entropy

    def entropy_numpy(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if isinstance(probabilities, torch.Tensor):
            probs = probabilities.detach().cpu().numpy()
        else:
            probs = probabilities
        return -(probs * np.log(probs + 1e-10)).sum(axis=1)


def validate_counterfactuals(
    target_model: MlpBayesianModel,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_names: Sequence[str],
    entropy_margin: float = 1e-8,
) -> pd.DataFrame:
    aligned_candidates = candidates.reindex(
        index=factuals.index, columns=list(feature_names)
    )
    aligned_candidates = aligned_candidates.copy(deep=True)

    valid_rows = ~aligned_candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return aligned_candidates

    adapter = ClueModelAdapter(target_model, feature_names)
    factual_subset = factuals.loc[valid_rows]
    candidate_subset = aligned_candidates.loc[valid_rows]

    factual_entropy = adapter.entropy_numpy(factual_subset)
    candidate_entropy = adapter.entropy_numpy(candidate_subset)
    factual_prediction = adapter.predict_label_indices(factual_subset)
    candidate_prediction = adapter.predict_label_indices(candidate_subset)

    success_mask = (candidate_entropy + entropy_margin < factual_entropy) & (
        candidate_prediction == factual_prediction
    )
    invalid_index = candidate_subset.index[~success_mask]
    aligned_candidates.loc[invalid_index, :] = np.nan
    return aligned_candidates
