from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import clingo
import numpy as np
import pandas as pd
import torch

from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from utils.registry import get_registry

BlackBoxModelTypes = (LinearModel, MlpModel, RandomForestModel)

BAF_ENCODING = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Encodings for BAFs
% to compute: d-admissible,
%             c-admissible,
%             s-admissible,
%             d-preferred,
%             c-preferred and
%             s-preferred extensions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

adm :- d_adm, baf, not input_error.
adm :- prefex, not baf, not input_error.
comp :- ground, not input_error.
prefex :- d_prefex, baf, not input_error.
d_adm :- d_prefex, baf, not input_error.
closed :- c_adm, baf, not input_error.
safe :- s_adm, not input_error.
s_adm :- s_prefex, baf, not input_error.
c_adm :- c_prefex, baf, not input_error.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   support and defeat for BAF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

support(X,Z) :- support(X,Y), support(Y,Z).

supported(X) :- support(Y,X), in(Y).

defeat(X,Y) :- att(Z,Y), support(X,Z), baf.
defeat(X,Y) :- att(X,Y), baf.
defeat(X,Y) :- att(X,Z), support(Z,Y), baf.

defeat(X,Y) :- att(X,Y).

in(X) :- not out(X), arg(X).
out(X) :- not in(X), arg(X).

:- in(X), in(Y), defeat(X,Y).

defeated(X) :- in(Y), defeat(Y,X).

not_defended(X) :- defeat(Y,X), not defeated(Y).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% special semantics for BAF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:- supported(B), defeated(B), safe.
:- defeated(B), in(B), safe.

:- in(X), not_defended(X), s_adm.

:- support(X,Y), out(Y),in(X), closed.
:- support(X,Y), in(Y), out(X), closed.

:- in(X), not_defended(X), c_adm.

:- in(X), not_defended(X), d_adm.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% successor relation with infinum and supremum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lt(X,Y) :- arg(X),arg(Y), X<Y, not input_error.
nsucc(X,Z) :- lt(X,Y), lt(Y,Z).
succ(X,Y) :- lt(X,Y), not nsucc(X,Y).
ninf(X) :- lt(Y,X).
nsup(X) :- lt(X,Y).
inf(X) :- not ninf(X), arg(X).
sup(X) :- not nsup(X), arg(X).

inN(X) | outN(X) :- out(X), prefex, not input_error.
inN(X) :- in(X), prefex, not input_error.

inN(X) | outN(X) :- out(X), s_prefex, not input_error.
inN(X) :- in(X), s_prefex.

inN(X) | outN(X) :- out(X), c_prefex, not input_error.
inN(X) :- in(X), c_prefex.

inN(X) | outN(X) :- out(X), d_prefex, not input_error.
inN(X) :- in(X), d_prefex.

eq_upto(Y) :- inf(Y), in(Y), inN(Y), not semis.
eq_upto(Y) :- inf(Y), out(Y), outN(Y), not semis.

eq_upto(Y) :- succ(Z,Y), in(Y), inN(Y), eq_upto(Z).
eq_upto(Y) :- succ(Z,Y), out(Y), outN(Y), eq_upto(Z).

eq :- sup(Y), eq_upto(Y).

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X), not defeat(Y,X), prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), s_prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X), not defeat(Y,X), s_prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), c_prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X), not defeat(Y,X), c_prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), d_prefex.
undefeated_upto(X,Y) :- inf(Y), outN(X), not defeat(Y,X), d_prefex.

undefeated_upto(X,Y) :- inf(Y), outN(X), outN(Y), semis.
undefeated_upto(X,Y) :- inf(Y), outN(X), not defeat(Y,X), semis.

undefeated_upto(X,Y) :- succ(Z,Y), undefeated_upto(X,Z), outN(Y).
undefeated_upto(X,Y) :- succ(Z,Y), undefeated_upto(X,Z), not defeat(Y,X).

undefeated(X) :- sup(Y), undefeated_upto(X,Y).

spoil :- eq.

spoil :- inN(X), inN(Y), defeat(X,Y), c_prefex.
spoil :- inN(X), inN(Y), defeat(X,Y), d_prefex.
spoil :- inN(X), inN(Y), defeat(X,Y), prefex.

supportedN(X) :- support(Y,X), inN(Y).

spoil :- supportedN(B), defeat(X,B), inN(X), s_prefex.
spoil :- defeat(X,B), inN(X), inN(B), s_prefex.

spoil :- support(X,Y), outN(Y), inN(X), c_prefex.
spoil :- support(X,Y), inN(Y), outN(X), c_prefex.

spoil :- inN(X), outN(Y), defeat(Y,X), undefeated(Y).

inN(X) :- spoil, arg(X), not input_error.
outN(X) :- spoil, arg(X), not input_error.

:- not spoil, prefex.
:- not spoil, s_prefex.
:- not spoil, c_prefex.
:- not spoil, d_prefex.

#show in/1.
"""


@dataclass(frozen=True)
class ExtensionResult:
    model_indices: list[int]
    ce_indices: list[int]
    all_indices: list[int]
    answer_status: int
    same_as_majority: bool


def ensure_supported_target_model(
    target_model: ModelObject,
    supported_types: Sequence[type[ModelObject]],
    method_name: str,
) -> None:
    if isinstance(target_model, tuple(supported_types)):
        return

    supported_names = ", ".join(cls.__name__ for cls in supported_types)
    raise TypeError(
        f"{method_name} supports target models [{supported_names}] only, "
        f"received {target_model.__class__.__name__}"
    )


def instantiate_models_from_configs(
    model_configs: Sequence[dict],
    default_device: str,
) -> list[ModelObject]:
    if not model_configs:
        return []

    registry = get_registry("model")
    models: list[ModelObject] = []
    for model_config in model_configs:
        if not isinstance(model_config, dict):
            raise TypeError("ensemble_model_configs entries must be dictionaries")
        item_cfg = dict(model_config)
        model_name = item_cfg.pop("name", None)
        if model_name is None:
            raise ValueError("Each ensemble model config must include a model name")
        if model_name not in registry:
            raise KeyError(f"Unknown ensemble model name: {model_name}")
        item_cfg.setdefault("device", default_device)
        model_class = registry[model_name]
        models.append(model_class(**item_cfg))
    return models


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


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)

    def get_ordered_features(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=True)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction)

    def predict_label_indices(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.asarray(probabilities).argmax(axis=1).astype(np.int64, copy=False)


def validate_ensemble_models(
    models: Sequence[ModelObject],
    device: str,
    method_name: str,
) -> None:
    if not models:
        raise ValueError("At least one ensemble model is required")

    for model in models:
        ensure_supported_target_model(model, BlackBoxModelTypes, method_name)
        if model._device != device:
            raise ValueError("All ensemble models must share the method device")


def ensure_class_mapping_alignment(models: Sequence[ModelObject]) -> None:
    if not models:
        raise ValueError("At least one ensemble model is required")

    reference_mapping = models[0].get_class_to_index()
    for model in models[1:]:
        if model.get_class_to_index() != reference_mapping:
            raise ValueError("All ensemble models must share the same class mapping")


def predict_label_indices(
    adapter: RecourseModelAdapter,
    X: pd.DataFrame | np.ndarray | torch.Tensor,
) -> np.ndarray:
    return adapter.predict_label_indices(X)


def build_model_adapters(
    models: Sequence[ModelObject],
    feature_names: Sequence[str],
) -> list[RecourseModelAdapter]:
    return [RecourseModelAdapter(model, feature_names) for model in models]


def compute_model_accuracy_scores(
    adapters: Sequence[RecourseModelAdapter],
    features: pd.DataFrame,
    encoded_target: np.ndarray,
) -> np.ndarray:
    scores = []
    for adapter in adapters:
        prediction = predict_label_indices(adapter, features)
        scores.append(float(np.mean(prediction == encoded_target)))
    return np.asarray(scores, dtype=np.float64)


def _resolve_model_complexity(model: ModelObject) -> float:
    if isinstance(model, (LinearModel, MlpModel)):
        return float(
            sum(parameter.numel() for parameter in model._model.parameters())
        )
    if isinstance(model, RandomForestModel):
        forest = model._model
        return float(sum(tree.tree_.node_count for tree in forest.estimators_))
    return float("inf")


def compute_model_simplicity_scores(models: Sequence[ModelObject]) -> np.ndarray:
    complexities = np.asarray(
        [_resolve_model_complexity(model) for model in models],
        dtype=np.float64,
    )
    if complexities.shape[0] == 1:
        return np.ones(1, dtype=np.float64)
    min_complexity = float(complexities.min())
    max_complexity = float(complexities.max())
    if np.isclose(min_complexity, max_complexity):
        return np.ones_like(complexities, dtype=np.float64)
    return 1.0 - (complexities - min_complexity) / (max_complexity - min_complexity)


def validate_score_vector(
    scores: Sequence[float] | None,
    expected_size: int,
    name: str,
) -> np.ndarray | None:
    if scores is None:
        return None
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != expected_size:
        raise ValueError(f"{name} must have length {expected_size}")
    return values


def nearest_neighbor_counterfactual(
    factual: pd.Series,
    train_features: pd.DataFrame,
    train_predictions: np.ndarray,
    original_prediction: int,
    desired_prediction: int | None = None,
) -> pd.Series | None:
    if desired_prediction is None:
        candidate_mask = train_predictions != int(original_prediction)
    else:
        candidate_mask = train_predictions == int(desired_prediction)

    if not bool(np.any(candidate_mask)):
        return None

    candidate_frame = train_features.loc[candidate_mask]
    factual_array = factual.to_numpy(dtype=np.float64, copy=False)
    candidate_array = candidate_frame.to_numpy(dtype=np.float64, copy=False)
    distances = np.linalg.norm(candidate_array - factual_array, axis=1)
    best_index = int(np.argmin(distances))
    return candidate_frame.iloc[best_index].copy(deep=True)


def preferred_over(
    accuracy_scores: np.ndarray,
    simplicity_scores: np.ndarray,
    source_index: int,
    target_index: int,
    preference_mode: str,
) -> bool:
    if preference_mode == "none":
        return True
    if preference_mode == "accuracy":
        return bool(accuracy_scores[source_index] >= accuracy_scores[target_index])
    if preference_mode == "simplicity":
        return bool(simplicity_scores[source_index] >= simplicity_scores[target_index])
    if preference_mode == "accuracy_simplicity":
        return not (
            accuracy_scores[source_index] < accuracy_scores[target_index]
            and simplicity_scores[source_index] < simplicity_scores[target_index]
        )
    raise ValueError(f"Unsupported preference_mode: {preference_mode}")


def build_baf_program(
    factual_predictions: np.ndarray,
    counterfactual_predictions: np.ndarray,
    accuracy_scores: np.ndarray,
    simplicity_scores: np.ndarray,
    semantics: str = "s",
    preference_mode: str = "none",
) -> str:
    semantics = semantics.lower()
    if semantics not in {"s", "d"}:
        raise ValueError("semantics must be 's' or 'd'")

    lines = [BAF_ENCODING, "baf."]
    lines.append("s_prefex." if semantics == "s" else "d_prefex.")

    num_models = int(factual_predictions.shape[0])
    for model_index in range(num_models):
        lines.append(f"arg(m{model_index}).")
        lines.append(f"arg(c{model_index}).")

    for candidate_index in range(num_models):
        lines.append(f"support(m{candidate_index},c{candidate_index}).")
        lines.append(f"support(c{candidate_index},m{candidate_index}).")

        for model_index in range(num_models):
            if (
                int(factual_predictions[model_index])
                == int(counterfactual_predictions[model_index, candidate_index])
            ):
                if preferred_over(
                    accuracy_scores,
                    simplicity_scores,
                    model_index,
                    candidate_index,
                    preference_mode,
                ):
                    lines.append(f"att(m{model_index},c{candidate_index}).")
                if preferred_over(
                    accuracy_scores,
                    simplicity_scores,
                    candidate_index,
                    model_index,
                    preference_mode,
                ):
                    lines.append(f"att(c{candidate_index},m{model_index}).")

            if model_index < candidate_index:
                continue
            if int(factual_predictions[candidate_index]) != int(
                factual_predictions[model_index]
            ):
                if preferred_over(
                    accuracy_scores,
                    simplicity_scores,
                    model_index,
                    candidate_index,
                    preference_mode,
                ):
                    lines.append(f"att(m{model_index},m{candidate_index}).")
                if preferred_over(
                    accuracy_scores,
                    simplicity_scores,
                    candidate_index,
                    model_index,
                    preference_mode,
                ):
                    lines.append(f"att(m{candidate_index},m{model_index}).")

    return "\n".join(lines) + "\n"


def _extract_extension_partitioned(
    model: clingo.Model,
) -> tuple[list[int], list[int], list[str]]:
    model_indices: list[int] = []
    ce_indices: list[int] = []
    all_arguments: list[str] = []
    for atom in model.symbols(atoms=True):
        if atom.name != "in" or not atom.arguments:
            continue
        argument_name = str(atom.arguments[0])
        if not argument_name:
            continue
        all_arguments.append(argument_name)
        if argument_name.startswith("m"):
            model_indices.append(int(argument_name[1:]))
        elif argument_name.startswith("c"):
            ce_indices.append(int(argument_name[1:]))
    return sorted(set(model_indices)), sorted(set(ce_indices)), sorted(set(all_arguments))


def _extension_prediction(
    factual_predictions: np.ndarray,
    model_indices: Sequence[int],
) -> int | None:
    if not model_indices:
        return None
    values = factual_predictions[list(model_indices)]
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def solve_argumentative_extension(
    program: str,
    factual_predictions: np.ndarray,
) -> ExtensionResult | None:
    control = clingo.Control(["--warn=none"])
    control.add("base", [], program)
    control.ground([("base", [])])
    control.configuration.solve.models = "0"

    candidate_extensions: list[tuple[list[int], list[int], list[str]]] = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            candidate_extensions.append(_extract_extension_partitioned(model))

    if not candidate_extensions:
        return None

    max_size = max(len(models) + len(ces) for models, ces, _ in candidate_extensions)
    max_extensions = [
        extension
        for extension in candidate_extensions
        if len(extension[0]) + len(extension[1]) == max_size
    ]
    max_extensions.sort(key=lambda item: (tuple(item[0]), tuple(item[1]), tuple(item[2])))

    majority_prediction = _extension_prediction(
        factual_predictions=factual_predictions,
        model_indices=list(range(len(factual_predictions))),
    )
    non_empty_extensions = [
        extension for extension in max_extensions if len(extension[0]) > 0 and len(extension[1]) > 0
    ]
    if not non_empty_extensions:
        return ExtensionResult([], [], [], 3, False)

    for model_indices, ce_indices, all_arguments in non_empty_extensions:
        if _extension_prediction(factual_predictions, model_indices) == majority_prediction:
            return ExtensionResult(
                model_indices=model_indices,
                ce_indices=ce_indices,
                all_indices=[
                    int(argument[1:])
                    for argument in all_arguments
                    if argument.startswith(("m", "c"))
                ],
                answer_status=0,
                same_as_majority=True,
            )

    model_indices, ce_indices, all_arguments = non_empty_extensions[0]
    return ExtensionResult(
        model_indices=model_indices,
        ce_indices=ce_indices,
        all_indices=[
            int(argument[1:])
            for argument in all_arguments
            if argument.startswith(("m", "c"))
        ],
        answer_status=2,
        same_as_majority=False,
    )


def select_best_accepted_counterfactual(
    factual: pd.Series,
    counterfactuals: pd.DataFrame,
    accepted_ce_indices: Sequence[int],
) -> pd.Series | None:
    if counterfactuals.empty:
        return None

    candidate_indices = [
        int(index)
        for index in accepted_ce_indices
        if 0 <= int(index) < counterfactuals.shape[0]
        and not counterfactuals.iloc[int(index)].isna().any()
    ]
    if not candidate_indices:
        return None

    factual_array = factual.to_numpy(dtype=np.float64, copy=False)
    best_index = min(
        candidate_indices,
        key=lambda index: (
            float(
                np.linalg.norm(
                    counterfactuals.iloc[index].to_numpy(dtype=np.float64, copy=False)
                    - factual_array
                )
            ),
            int(index),
        ),
    )
    return counterfactuals.iloc[best_index].copy(deep=True)
