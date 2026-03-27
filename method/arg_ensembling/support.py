from __future__ import annotations

from typing import Sequence

import clingo
import numpy as np
import pandas as pd

from method.face.support import (
    BlackBoxModelTypes,
    RecourseModelAdapter,
    ensure_supported_target_model,
)
from model.model_object import ModelObject

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


def validate_ensemble_models(
    models: Sequence[ModelObject],
    device: str,
    method_name: str,
) -> None:
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


def build_baf_program(
    factual_predictions: np.ndarray,
    counterfactual_predictions: np.ndarray,
) -> str:
    lines = ["baf.", "s_prefex."]
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
                lines.append(f"att(m{model_index},c{candidate_index}).")
                lines.append(f"att(c{candidate_index},m{model_index}).")

            if model_index < candidate_index:
                continue
            if int(factual_predictions[candidate_index]) != int(
                factual_predictions[model_index]
            ):
                lines.append(f"att(m{model_index},m{candidate_index}).")
                lines.append(f"att(m{candidate_index},m{model_index}).")

    return BAF_ENCODING + "\n" + "\n".join(lines) + "\n"


def solve_largest_extension(program: str) -> list[int]:
    model_indices, ce_indices = solve_largest_extension_partitioned(program)
    ordered = list(model_indices)
    for index in ce_indices:
        if index not in ordered:
            ordered.append(index)
    return ordered


def solve_largest_extension_partitioned(program: str) -> tuple[list[int], list[int]]:
    control = clingo.Control(["--warn=none"])
    control.add("base", [], program)
    control.ground([("base", [])])
    control.configuration.solve.models = "0"

    best_models: list[int] = []
    best_ces: list[int] = []
    best_size = -1
    with control.solve(yield_=True) as handle:
        for model in handle:
            model_indices, ce_indices = _extract_extension_partitioned(model)
            extension_size = len(model_indices) + len(ce_indices)
            if extension_size >= best_size:
                best_models = model_indices
                best_ces = ce_indices
                best_size = extension_size
    return best_models, best_ces


def select_best_accepted_counterfactual(
    factual: pd.Series,
    counterfactuals: pd.DataFrame,
    accepted_indices: Sequence[int],
) -> pd.Series | None:
    candidate_indices = [
        int(index)
        for index in accepted_indices
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
            index,
        ),
    )
    return counterfactuals.iloc[best_index].copy(deep=True)


def build_model_adapters(
    models: Sequence[ModelObject],
    feature_names: Sequence[str],
) -> list[RecourseModelAdapter]:
    return [RecourseModelAdapter(model, feature_names) for model in models]


def predict_label_indices(
    adapter: RecourseModelAdapter,
    X: pd.DataFrame,
) -> np.ndarray:
    return np.asarray(adapter.predict_label_indices(X), dtype=np.int64)


def _extract_extension_indices(model: clingo.Model) -> list[int]:
    model_indices, ce_indices = _extract_extension_partitioned(model)
    ordered = list(model_indices)
    for index in ce_indices:
        if index not in ordered:
            ordered.append(index)
    return ordered


def _extract_extension_partitioned(model: clingo.Model) -> tuple[list[int], list[int]]:
    model_indices: list[int] = []
    ce_indices: list[int] = []
    accepted_indices: list[int] = []
    for atom in model.symbols(atoms=True):
        if atom.name != "in" or not atom.arguments:
            continue
        argument_name = str(atom.arguments[0])
        if not argument_name or len(argument_name) < 2:
            continue
        argument_index = int(argument_name[1:])
        if argument_name.startswith("m"):
            if argument_index not in model_indices:
                model_indices.append(argument_index)
        elif argument_name.startswith("c"):
            if argument_index not in ce_indices:
                ce_indices.append(argument_index)
        elif argument_index not in accepted_indices:
            accepted_indices.append(argument_index)

    if accepted_indices:
        for argument_index in accepted_indices:
            if argument_index not in model_indices:
                model_indices.append(argument_index)
            if argument_index not in ce_indices:
                ce_indices.append(argument_index)
    return model_indices, ce_indices
