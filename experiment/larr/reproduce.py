from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from dataset.german.german import GermanDataset
from method.larr.library.larr import LARRecourse, RecourseCost
from model.mlp.mlp import MlpModel
from preprocess.common import EncodePreProcess, FinalizePreProcess, ScalePreProcess

SEED = 0
N_FOLDS = 5
NUM_FACTUALS = 5
ALPHA = 0.5
# Keep this value fixed so the migrated script matches the current asserted targets.
BETA = 0.518
ROBUSTNESS_BOUNDS = (0.27, 0.29)
CONSISTENCY_BOUNDS = (0.40, 0.41)


class ReferencePredictAdapter:
    def __init__(self, model: MlpModel, feature_names: list[str]):
        self._model = model
        self._feature_names = list(feature_names)

    def _to_feature_df(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self._feature_names].copy(deep=True)

        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return pd.DataFrame(array, columns=self._feature_names)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        features = self._to_feature_df(X)
        probabilities = self._model.get_prediction(features, proba=True)
        return probabilities.detach().cpu().numpy().argmax(axis=1)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        features = self._to_feature_df(X)
        probabilities = self._model.get_prediction(features, proba=True)
        return probabilities.detach().cpu().numpy()


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_preprocessed_german_dataset() -> GermanDataset:
    dataset = GermanDataset()
    shuffled_df = dataset.snapshot().sample(frac=1, random_state=SEED)
    dataset.update("shuffled", True, df=shuffled_df)
    dataset = ScalePreProcess(
        seed=SEED,
        scaling="standardize",
        range=True,
    ).transform(dataset)
    dataset = EncodePreProcess(seed=SEED, encoding="onehot").transform(dataset)
    dataset = FinalizePreProcess(seed=SEED).transform(dataset)
    return dataset


def _build_frozen_dataset(template, df: pd.DataFrame, marker: str):
    dataset = template.clone()
    dataset.update(marker, True, df=df.copy(deep=True))
    dataset.freeze()
    return dataset


def _split_fold(full_df: pd.DataFrame, fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = int(fold_index / N_FOLDS * len(full_df))
    end = int((fold_index + 1) / N_FOLDS * len(full_df))
    test_df = full_df.iloc[start:end].copy(deep=True)
    train_df = pd.concat([full_df.iloc[:start], full_df.iloc[end:]], axis=0)
    return train_df.copy(deep=True), test_df


def _select_recourse_needed(
    predict_fn,
    X: np.ndarray,
    y_target: float = 1,
) -> np.ndarray:
    indices = np.where(predict_fn(X) == 1 - y_target)
    return X[indices]


def _build_model(trainset, batch_size: int) -> MlpModel:
    model = MlpModel(
        seed=SEED,
        device=_resolve_device(),
        epochs=100,
        learning_rate=0.001,
        batch_size=batch_size,
        layers=[50, 100, 200],
        optimizer="adam",
        criterion="bce",
        output_activation="sigmoid",
        save_name=None,
    )
    model.fit(trainset)
    return model


def _evaluate_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    predict_adapter: ReferencePredictAdapter,
) -> tuple[float, float, float, int]:
    recourse_needed_X_train = _select_recourse_needed(
        predict_adapter.predict,
        X_train.values,
    )
    recourse_needed_X_test = _select_recourse_needed(
        predict_adapter.predict,
        X_test.values,
    )

    if recourse_needed_X_train.shape[0] == 0:
        raise ValueError("No training instances require recourse in this fold")
    if recourse_needed_X_test.shape[0] < NUM_FACTUALS:
        raise ValueError("Not enough test instances require recourse in this fold")

    larr_recourse = LARRecourse(weights=None, bias=None, alpha=ALPHA)
    larr_recourse.choose_lambda(
        recourse_needed_X_train,
        predict_adapter.predict,
        X_train.values,
    )

    fold_robustness = 0.0
    fold_consistency = 0.0
    for factual_index in range(NUM_FACTUALS):
        x_0 = recourse_needed_X_test[factual_index]
        objective = RecourseCost(x_0, larr_recourse.lamb)

        np.random.seed(factual_index)
        weights_0, bias_0 = larr_recourse.lime_explanation(
            predict_adapter.predict,
            X_train.values,
            x_0,
        )
        weights_0 = np.round(weights_0, 4)
        bias_0 = np.round(bias_0, 4)

        larr_recourse.weights = weights_0
        larr_recourse.bias = bias_0

        x_r = larr_recourse.get_recourse(x_0, beta=1.0)
        weights_r, bias_r = larr_recourse.calc_theta_adv(x_r)
        J_r_opt = objective.eval(x_r, weights_r, bias_r)

        theta_p = (deepcopy(weights_r), deepcopy(bias_r))

        x_c = larr_recourse.get_recourse(x_0, beta=0.0, theta_p=theta_p)
        J_c_opt = objective.eval(x_c, *theta_p)

        x = larr_recourse.get_recourse(x_0, beta=BETA, theta_p=theta_p)
        weights_r, bias_r = larr_recourse.calc_theta_adv(x)

        J_r = objective.eval(x, weights_r, bias_r)
        J_c = objective.eval(x, *theta_p)

        fold_robustness += float(np.asarray(J_r - J_r_opt).reshape(-1)[0])
        fold_consistency += float(np.asarray(J_c - J_c_opt).reshape(-1)[0])

    return (
        fold_robustness,
        fold_consistency,
        float(larr_recourse.lamb),
        int(recourse_needed_X_test.shape[0]),
    )


def run_reproduction() -> tuple[float, float]:
    dataset = _build_preprocessed_german_dataset()
    full_df = pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    feature_names = list(dataset.get(target=False).columns)

    running_robustness = 0.0
    running_consistency = 0.0
    counter = 0

    for fold_index in range(N_FOLDS):
        train_df, test_df = _split_fold(full_df, fold_index)
        trainset = _build_frozen_dataset(dataset, train_df, "trainset")
        model = _build_model(trainset, batch_size=len(train_df))
        adapter = ReferencePredictAdapter(model, feature_names)

        X_train = train_df.loc[:, feature_names]
        X_test = test_df.loc[:, feature_names]
        fold_robustness, fold_consistency, lambda_value, negative_test = _evaluate_fold(
            X_train,
            X_test,
            adapter,
        )
        print(f"fold {fold_index} lambda {lambda_value} neg_test {negative_test}")

        running_robustness += fold_robustness
        running_consistency += fold_consistency
        counter += NUM_FACTUALS

    avge_robustness = running_robustness / counter
    avge_consistency = running_consistency / counter
    print(
        f"Avge Robustness: {avge_robustness} and Avge Consistency: {avge_consistency}"
    )

    assert ROBUSTNESS_BOUNDS[0] < avge_robustness < ROBUSTNESS_BOUNDS[1]
    assert CONSISTENCY_BOUNDS[0] < avge_consistency < CONSISTENCY_BOUNDS[1]
    return avge_robustness, avge_consistency


def test_run_experiment():
    return run_reproduction()


def main() -> None:
    run_reproduction()


if __name__ == "__main__":
    main()
