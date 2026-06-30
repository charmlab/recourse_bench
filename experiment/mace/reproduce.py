from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import recourse_bench.dataset  # noqa: F401
import recourse_bench.evaluation  # noqa: F401
import recourse_bench.method  # noqa: F401
import recourse_bench.model  # noqa: F401
import recourse_bench.preprocess  # noqa: F401
from recourse_bench.dataset.adult.adult import AdultDataset
from recourse_bench.dataset.credit.credit import CreditDataset
from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.evaluation.evaluation_object import EvaluationObject
from recourse_bench.evaluation.evaluation_utils import resolve_evaluation_inputs
from recourse_bench.evaluation.validity import ValidityEvaluation
from recourse_bench.method.mace.library import normalizedDistance
from recourse_bench.method.mace.mace import MaceDatasetWrapper, MaceMethod
from recourse_bench.model.sklearn_logistic_regression.sklearn_logistic_regression import (
    SklearnLogisticRegressionModel,
)
from recourse_bench.preprocess.balance import BalancePreProcess
from recourse_bench.preprocess.common import FinalizePreProcess, SplitPreProcess
from recourse_bench.preprocess.preprocess_object import PreProcessObject
from recourse_bench.preprocess.preprocess_utils import ensure_flag_absent
from recourse_bench.utils.caching import set_cache_dir
from recourse_bench.utils.registry import register
from recourse_bench.utils.seed import seed_context

SEED = 54321
DEFAULTS = {
    "adult": {
        "dataset_cls": AdultDataset,
        "split": 6601,
        "targets": {"zero_norm": 62.0, "one_norm": 92.0, "infty_norm": 86.0},
    },
    "credit": {
        "dataset_cls": CreditDataset,
        "split": 3900,
        "targets": {"zero_norm": 80.0, "one_norm": 82.0, "infty_norm": 80.0},
    },
}
NORM_TO_METRIC = {
    "zero_norm": "distance_l0",
    "one_norm": "distance_l1",
    "infty_norm": "distance_linf",
}


# Distance
def _dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    return True


def _normalize_actionability(value: object) -> str:
    normalized = str(value).lower()
    return "none" if normalized == "same" else normalized


def build_mace_wrapper(dataset: DatasetObject) -> MaceDatasetWrapper:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    if not _dataset_has_attr(dataset, "mace_encoded_attr_type"):
        raise ValueError("MACE normalized distance requires mace-encoded datasets")

    bounds_raw = dataset.attr("mace_encoded_bounds")
    feature_type = dataset.attr("mace_encoded_attr_type")
    mutability = dataset.attr("encoded_feature_mutability")
    actionability = dataset.attr("encoded_feature_actionability")
    encoded_parent = dataset.attr("mace_encoded_parent")
    dataset_name = (
        str(dataset.attr("name")) if _dataset_has_attr(dataset, "name") else ""
    )

    return MaceDatasetWrapper(
        dataset_name=dataset_name,
        feature_names=feature_names,
        feature_types={key: str(value).lower() for key, value in feature_type.items()},
        bounds={
            key: tuple(map(float, value))
            for key, value in bounds_raw.items()
            if key in feature_names
        },
        mutability={key: bool(value) for key, value in mutability.items()},
        actionability={
            key: _normalize_actionability(value) for key, value in actionability.items()
        },
        encoded_parent={key: str(value) for key, value in encoded_parent.items()},
    )


@register("mace_normalized_distance")
class MaceNormalizedDistanceEvaluation(EvaluationObject):
    @staticmethod
    def _resolve_metrics(metrics: list[str] | None) -> list[str]:
        aliases = {
            "l0": "zero_norm",
            "l1": "one_norm",
            "l2": "two_norm",
            "linf": "infty_norm",
        }
        resolved = []
        for metric in metrics or ["zero_norm", "one_norm", "infty_norm"]:
            metric = str(metric).lower()
            resolved.append(aliases.get(metric, metric))
        invalid = [
            metric
            for metric in resolved
            if metric not in {"zero_norm", "one_norm", "two_norm", "infty_norm"}
        ]
        if invalid:
            raise ValueError(f"Unsupported MACE normalized distance metrics: {invalid}")
        return resolved

    def __init__(self, metrics: list[str] | None = None, **kwargs):
        self._metrics = self._resolve_metrics(metrics)

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        selected_mask = evaluation_mask & success_mask
        results: dict[str, float] = {}
        if int(selected_mask.sum()) == 0:
            for metric in self._metrics:
                results[f"mace_distance_{metric}"] = float("nan")
            return pd.DataFrame([results])

        wrapper = build_mace_wrapper(factuals)
        factual_success = factual_features.loc[selected_mask.to_numpy()]
        counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]

        distances = {metric: [] for metric in self._metrics}
        for row_index in factual_success.index:
            factual_sample = wrapper.factual_to_short_dict(
                factual_success.loc[row_index], predicted_label=0
            )
            counterfactual_sample = wrapper.factual_to_short_dict(
                counterfactual_success.loc[row_index], predicted_label=1
            )
            for metric in self._metrics:
                distances[metric].append(
                    float(
                        normalizedDistance.getDistanceBetweenSamples(
                            factual_sample,
                            counterfactual_sample,
                            metric,
                            wrapper,
                        )
                    )
                )

        for metric, values in distances.items():
            results[f"mace_distance_{metric}"] = float(np.mean(values))
        return pd.DataFrame([results])


# End


# PreProcess
def _dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    return True


def _mace_metadata(
    dataset: DatasetObject,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    if _dataset_has_attr(dataset, "mace_feature_type"):
        return (
            dataset.attr("mace_feature_type"),
            dataset.attr("mace_feature_mutability"),
            dataset.attr("mace_feature_actionability"),
        )
    return (
        dataset.attr("raw_feature_type"),
        dataset.attr("raw_feature_mutability"),
        dataset.attr("raw_feature_actionability"),
    )


def _format_int(value: object) -> str:
    value = float(value)
    if not value.is_integer():
        raise ValueError(
            f"MACE categorical/ordinal values must be integer-like: {value}"
        )
    return str(int(value))


def _as_int_categories(series: pd.Series, lower: float, upper: float) -> list[int]:
    lower_int = int(round(float(lower)))
    upper_int = int(round(float(upper)))
    values = sorted({int(round(float(value))) for value in series.dropna().unique()})
    expected = list(range(lower_int, upper_int + 1))
    if values != expected:
        raise ValueError(
            "MACE encoding expects contiguous integer categories from "
            f"{lower_int} to {upper_int}, got {values}"
        )
    return values


@register("mace_encode")
class MaceEncodePreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, **kwargs):
        self._seed = seed

    @staticmethod
    def _build_onehot(
        series: pd.Series,
        feature_name: str,
        categories: list[int],
    ) -> pd.DataFrame:
        values = series.astype("float64").round().astype("int64")
        columns = {}
        for offset, category in enumerate(categories):
            columns[f"{feature_name}_cat_{offset}"] = (values == category).astype(
                "float64"
            )
        return pd.DataFrame(columns, index=series.index)

    @staticmethod
    def _build_thermometer(
        series: pd.Series,
        feature_name: str,
        categories: list[int],
    ) -> pd.DataFrame:
        lower = categories[0]
        values = series.astype("float64").round().astype("int64")
        columns = {}
        for offset, _category in enumerate(categories):
            columns[f"{feature_name}_ord_{offset}"] = (values >= lower + offset).astype(
                "float64"
            )
        return pd.DataFrame(columns, index=series.index)

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "encoding")
            ensure_flag_absent(input, "mace_encoding")

            df = input.snapshot()
            target_column = input.target_column
            feature_type, feature_mutability, feature_actionability = _mace_metadata(
                input
            )
            source_bound_min = {}
            source_bound_max = {}
            if _dataset_has_attr(input, "balanced"):
                balanced = input.attr("balanced")
                if isinstance(balanced, dict):
                    candidate_min = balanced.get("feature_min")
                    candidate_max = balanced.get("feature_max")
                    if isinstance(candidate_min, dict) and isinstance(
                        candidate_max, dict
                    ):
                        source_bound_min = candidate_min
                        source_bound_max = candidate_max

            final_parts: list[pd.DataFrame] = []
            encoded_sources: dict[str, str] = {}
            encoding: dict[str, list[str]] = {}
            encoded_feature_type: dict[str, str] = {}
            encoded_feature_mutability: dict[str, bool] = {}
            encoded_feature_actionability: dict[str, str] = {}
            encoded_attr_type: dict[str, str] = {}
            encoded_parent: dict[str, str] = {}
            encoded_specs: dict[str, dict[str, object]] = {}
            source_bounds: dict[str, tuple[float, float]] = {}
            encoded_bounds: dict[str, tuple[float, float]] = {}

            for column in df.columns:
                if column == target_column:
                    final_parts.append(df.loc[:, [column]].copy(deep=True))
                    continue

                kind = str(feature_type[column]).lower()
                series = df[column]
                lower = float(source_bound_min.get(column, series.min()))
                upper = float(source_bound_max.get(column, series.max()))
                source_bounds[column] = (lower, upper)

                if kind == "categorical":
                    categories = _as_int_categories(series, lower, upper)
                    encoded = self._build_onehot(series, column, categories)
                    output_columns = list(encoded.columns)
                    attr_type = "sub-categorical"
                    encoding_mode = "onehot"
                    encoded_specs[column] = {
                        "encoding": encoding_mode,
                        "categories": categories,
                        "columns": output_columns,
                    }
                elif kind == "ordinal":
                    categories = _as_int_categories(series, lower, upper)
                    encoded = self._build_thermometer(series, column, categories)
                    output_columns = list(encoded.columns)
                    attr_type = "sub-ordinal"
                    encoding_mode = "thermometer"
                    encoded_specs[column] = {
                        "encoding": encoding_mode,
                        "categories": categories,
                        "columns": output_columns,
                    }
                else:
                    encoded = df.loc[:, [column]].copy(deep=True)
                    output_columns = [column]
                    attr_type = kind
                    encoding_mode = "none"
                    encoded_specs[column] = {
                        "encoding": encoding_mode,
                        "categories": None,
                        "columns": output_columns,
                    }

                final_parts.append(encoded)
                encoding[column] = output_columns
                for encoded_column in output_columns:
                    encoded_sources[encoded_column] = column
                    encoded_feature_type[encoded_column] = (
                        "binary" if attr_type.startswith("sub-") else kind
                    )
                    encoded_feature_mutability[encoded_column] = bool(
                        feature_mutability[column]
                    )
                    encoded_feature_actionability[encoded_column] = str(
                        feature_actionability[column]
                    )
                    encoded_attr_type[encoded_column] = attr_type
                    encoded_parent[encoded_column] = (
                        column if attr_type.startswith("sub-") else encoded_column
                    )
                    if attr_type == "sub-categorical":
                        encoded_bounds[encoded_column] = (0.0, 1.0)
                    elif attr_type == "sub-ordinal":
                        offset = int(encoded_column.rsplit("_", 1)[-1])
                        encoded_bounds[encoded_column] = (
                            (1.0, 1.0) if offset == 0 else (0.0, 1.0)
                        )
                    else:
                        encoded_bounds[encoded_column] = (lower, upper)

            final_df = pd.concat(final_parts, axis=1)
            encoded_count = final_df.shape[1] - 1
            input.update("encoding", encoding, df=final_df)
            input.update("mace_encoding", encoded_specs)
            input.update("mace_encoded_sources", encoded_sources)
            input.update("mace_encoded_attr_type", encoded_attr_type)
            input.update("mace_encoded_parent", encoded_parent)
            input.update("mace_source_bounds", source_bounds)
            input.update("mace_encoded_bounds", encoded_bounds)
            input.update("mace_encoded_dim", int(encoded_count))
            input.update("encoded_feature_type", encoded_feature_type)
            input.update("encoded_feature_mutability", encoded_feature_mutability)
            input.update("encoded_feature_actionability", encoded_feature_actionability)
            return input


# End


def _materialize_dataset(dataset_name: str):
    raw = DEFAULTS[dataset_name]["dataset_cls"]()
    current = BalancePreProcess(
        seed=SEED, strategy="downsample", round_to=250, shuffle=True, range=True
    ).transform(raw)
    current = MaceEncodePreProcess(seed=SEED).transform(current)
    trainset, testset = SplitPreProcess(
        seed=SEED, split=int(DEFAULTS[dataset_name]["split"])
    ).transform(current)
    trainset = FinalizePreProcess(seed=SEED).transform(trainset)
    testset = FinalizePreProcess(seed=SEED).transform(testset)
    return trainset, testset


def _subset_dataset(dataset_obj, indices: pd.Index, flag_name: str):
    feature_df = dataset_obj.get(target=False)
    target_df = dataset_obj.get(target=True)
    subset_df = pd.concat([feature_df.loc[indices], target_df.loc[indices]], axis=1)
    subset_df = subset_df.reindex(columns=dataset_obj.ordered_features())
    subset = dataset_obj.clone()
    subset.update(flag_name, True, df=subset_df)
    subset.freeze()
    return subset


def _target_index(target_model, desired_class) -> int:
    class_to_index = target_model.get_class_to_index()
    if desired_class is None:
        if len(class_to_index) != 2:
            raise ValueError("desired_class=None requires binary classification")
        return 1
    return int(class_to_index[desired_class])


def _select_factuals(dataset_obj, target_model, desired_class, num_factuals: int):
    predictions = (
        target_model.predict(dataset_obj, batch_size=max(1, len(dataset_obj)))
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    target_index = _target_index(target_model, desired_class)
    keep_index = dataset_obj.get(target=False).index[predictions != target_index]
    if keep_index.shape[0] < num_factuals:
        raise ValueError(
            f"Requested {num_factuals} factuals but only found {keep_index.shape[0]}"
        )
    return _subset_dataset(dataset_obj, keep_index[:num_factuals], "testset")


def _build_counterfactual_dataset(factuals, counterfactual_features, desired_class):
    target_column = factuals.target_column
    counterfactual_target = pd.DataFrame(
        -1.0,
        index=counterfactual_features.index,
        columns=[target_column],
    )
    counterfactual_df = pd.concat(
        [counterfactual_features, counterfactual_target], axis=1
    ).reindex(columns=factuals.ordered_features())

    output = factuals.clone()
    output.update("counterfactual", True, df=counterfactual_df)
    if desired_class is not None:
        output.update(
            "evaluation_filter",
            pd.DataFrame(
                True,
                index=counterfactual_df.index,
                columns=["evaluation_filter"],
                dtype=bool,
            ),
        )
    output.freeze()
    return output


def _select_observable_pool(dataset_obj, target_model, desired_class):
    predictions = target_model.predict(dataset_obj).argmax(dim=1).detach().cpu().numpy()
    target_index = _target_index(target_model, desired_class)
    pool_index = dataset_obj.get(target=False).index[predictions == target_index]
    pool = dataset_obj.get(target=False).loc[pool_index].copy(deep=True)
    if pool.empty:
        raise ValueError("Observable pool is empty for MO baseline")
    return pool


def _violates_actionability(wrapper, factual_sample, observable_sample) -> bool:
    for attr_name_kurz in wrapper.getInputAttributeNames("kurz"):
        attr_obj = wrapper.attributes_kurz[attr_name_kurz]
        factual_value = factual_sample[attr_name_kurz]
        observable_value = observable_sample[attr_name_kurz]
        if attr_obj.actionability == "none" and factual_value != observable_value:
            return True
        if (
            attr_obj.actionability == "same-or-increase"
            and factual_value > observable_value
        ):
            return True
        if (
            attr_obj.actionability == "same-or-decrease"
            and factual_value < observable_value
        ):
            return True
    return False


def _compute_mo_counterfactuals(method_obj, factuals, observable_pool: pd.DataFrame):
    wrapper = method_obj._dataset_wrapper
    factual_features = factuals.get(target=False)
    factual_labels = method_obj._predict_label(factual_features)
    observable_labels = method_obj._predict_label(observable_pool)

    observable_samples = [
        wrapper.factual_to_short_dict(observable_pool.loc[row_index], int(label))
        for row_index, label in zip(observable_pool.index, observable_labels)
    ]

    rows = []
    norm_type = str(method_obj._norm_type[0])
    for row_position, row_index in enumerate(factual_features.index):
        factual_sample = wrapper.factual_to_short_dict(
            factual_features.loc[row_index],
            int(factual_labels[row_position]),
        )
        best_sample = None
        best_distance = float("inf")

        for observable_sample in observable_samples:
            if observable_sample["y"] == factual_sample["y"]:
                continue
            if _violates_actionability(wrapper, factual_sample, observable_sample):
                continue
            candidate_distance = float(
                normalizedDistance.getDistanceBetweenSamples(
                    factual_sample, observable_sample, norm_type, wrapper
                )
            )
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_sample = observable_sample

        if best_sample is None:
            rows.append(
                pd.Series(np.nan, index=wrapper._feature_names, dtype="float64")
            )
        else:
            rows.append(wrapper.short_dict_to_feature_row(best_sample))

    counterfactual_features = pd.DataFrame(
        rows,
        index=factual_features.index,
        columns=factual_features.columns,
    )
    return _build_counterfactual_dataset(
        factuals,
        counterfactual_features,
        getattr(method_obj, "_desired_class", None),
    )


def _distance_summary(method_obj, factuals, counterfactuals):
    (
        factual_features,
        counterfactual_features,
        evaluation_mask,
        success_mask,
    ) = resolve_evaluation_inputs(factuals, counterfactuals)
    selected_mask = evaluation_mask & success_mask
    distances_by_metric = {metric: [] for metric in NORM_TO_METRIC.values()}
    if int(selected_mask.sum()) == 0:
        return (
            {metric: float("nan") for metric in NORM_TO_METRIC.values()},
            distances_by_metric,
        )

    factual_success = factual_features.loc[selected_mask.to_numpy()]
    counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]
    wrapper = method_obj._dataset_wrapper
    factual_labels = method_obj._predict_label(factual_success)
    counterfactual_labels = method_obj._predict_label(counterfactual_success)

    for row_position, row_index in enumerate(factual_success.index):
        factual_sample = wrapper.factual_to_short_dict(
            factual_success.loc[row_index],
            int(factual_labels[row_position]),
        )
        counterfactual_sample = wrapper.factual_to_short_dict(
            counterfactual_success.loc[row_index],
            int(counterfactual_labels[row_position]),
        )
        for norm_type, metric_name in NORM_TO_METRIC.items():
            distances_by_metric[metric_name].append(
                float(
                    normalizedDistance.getDistanceBetweenSamples(
                        factual_sample, counterfactual_sample, norm_type, wrapper
                    )
                )
            )

    summary = {
        metric_name: float(np.mean(values)) if values else float("nan")
        for metric_name, values in distances_by_metric.items()
    }
    return summary, distances_by_metric


def _validity(factuals, counterfactuals) -> float:
    return float(
        ValidityEvaluation().evaluate(factuals, counterfactuals)["validity"][0]
    )


def _run_single(dataset_name: str, norm_type: str, epsilon: float, num_factuals: int):
    trainset, testset = _materialize_dataset(dataset_name)
    encoded_dim = int(trainset.attr("mace_encoded_dim"))
    target_model = SklearnLogisticRegressionModel(fit_mode="default", device="cpu")
    target_model.fit(trainset)

    factuals = _select_factuals(
        testset,
        target_model,
        desired_class=None,
        num_factuals=num_factuals,
    )
    method_obj = MaceMethod(
        target_model=target_model,
        seed=SEED,
        device="cpu",
        norm_type=norm_type,
        epsilon=epsilon,
    )
    method_obj.fit(trainset)

    start = time.perf_counter()
    mace_counterfactuals = method_obj.predict(
        factuals, batch_size=max(1, len(factuals))
    )
    mace_seconds = time.perf_counter() - start
    mace_validity = _validity(factuals, mace_counterfactuals)
    mace_summary, mace_pointwise = _distance_summary(
        method_obj, factuals, mace_counterfactuals
    )

    observable_pool = _select_observable_pool(testset, target_model, desired_class=None)
    mo_start = time.perf_counter()
    mo_counterfactuals = _compute_mo_counterfactuals(
        method_obj, factuals, observable_pool
    )
    mo_seconds = time.perf_counter() - mo_start
    mo_validity = _validity(factuals, mo_counterfactuals)
    mo_summary, mo_pointwise = _distance_summary(
        method_obj, factuals, mo_counterfactuals
    )

    optimized_metric = NORM_TO_METRIC[norm_type]
    terms = []
    for mace_value, mo_value in zip(
        mace_pointwise[optimized_metric], mo_pointwise[optimized_metric]
    ):
        if mo_value > 0:
            terms.append(1.0 - float(mace_value) / float(mo_value))
    improvement = 100.0 * float(np.mean(terms)) if terms else float("nan")

    return {
        "dataset": dataset_name,
        "norm_type": norm_type,
        "epsilon": float(epsilon),
        "num_factuals": int(len(factuals)),
        "encoded_dim": encoded_dim,
        "mace_seconds": float(mace_seconds),
        "mo_seconds": float(mo_seconds),
        "mace_validity": mace_validity,
        "mo_validity": mo_validity,
        "mace_distances": mace_summary,
        "mo_distances": mo_summary,
        "optimized_metric": optimized_metric,
        "mace_vs_mo_improvement": improvement,
    }


def _assert_smoke(result: dict, requested: int):
    if int(result["num_factuals"]) != int(requested):
        raise AssertionError("factual count mismatch")
    if int(result["encoded_dim"]) != (51 if result["dataset"] == "adult" else 20):
        raise AssertionError(
            f"unexpected encoded dimension for {result['dataset']}: "
            f"{result['encoded_dim']}"
        )
    if float(result["mace_validity"]) <= 0:
        raise AssertionError("MACE did not produce any valid counterfactuals")
    if float(result["mo_validity"]) <= 0:
        raise AssertionError("MO did not produce any valid counterfactuals")
    for distances in [result["mace_distances"], result["mo_distances"]]:
        for value in distances.values():
            if pd.isna(value):
                raise AssertionError("distance summary contains NaN")


def _assert_strict(result: dict):
    expected = DEFAULTS[result["dataset"]]["targets"][result["norm_type"]]
    actual = round(float(result["mace_vs_mo_improvement"]))
    if actual != int(expected):
        raise AssertionError(
            f"Table 3 mismatch for {result['dataset']} {result['norm_type']}: "
            f"expected {expected:.0f}, got {actual}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), default="adult")
    parser.add_argument(
        "--norm",
        choices=sorted(NORM_TO_METRIC),
        action="append",
        default=None,
    )
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    parser.add_argument("--nfactuals", "--num-factuals", type=int, default=500)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--cache-dir", default="./cache/")
    args = parser.parse_args()

    set_cache_dir(args.cache_dir)
    norms = args.norm or ["zero_norm", "one_norm", "infty_norm"]
    results = []
    for norm_type in norms:
        result = _run_single(
            dataset_name=args.dataset,
            norm_type=norm_type,
            epsilon=float(args.epsilon),
            num_factuals=int(args.nfactuals),
        )
        _assert_smoke(result, int(args.nfactuals))
        if args.strict:
            _assert_strict(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        results.append(result)

    rounded = {
        result["norm_type"]: round(float(result["mace_vs_mo_improvement"]))
        for result in results
    }
    print("rounded_table3:", json.dumps(rounded, sort_keys=True))


if __name__ == "__main__":
    main()
