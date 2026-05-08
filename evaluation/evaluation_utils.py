from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject


def dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    return True


def resolve_restore_mode(
    restore_categorical: bool | str = False,
    restore_numerical: bool | str = False,
) -> tuple[bool, str]:
    if isinstance(restore_categorical, bool):
        categorical_mode = "restore" if restore_categorical else "none"
    elif isinstance(restore_categorical, str):
        value = restore_categorical.strip().lower()
        if value == "true":
            categorical_mode = "restore"
        elif value == "false":
            categorical_mode = "none"
        elif value == "binarize":
            categorical_mode = "binarize"
        else:
            raise ValueError(
                "restore_categorical must be one of: true, false, binarize"
            )
    else:
        raise TypeError("restore_categorical must be a bool or str")

    if isinstance(restore_numerical, bool):
        numerical_mode = restore_numerical
    elif isinstance(restore_numerical, str):
        value = restore_numerical.strip().lower()
        if value == "true":
            numerical_mode = True
        elif value == "false":
            numerical_mode = False
        else:
            raise ValueError("restore_numerical must be one of: true, false")
    else:
        raise TypeError("restore_numerical must be a bool or str")

    if categorical_mode in {"restore", "binarize"} and numerical_mode:
        mode = "both"
    elif categorical_mode in {"restore", "binarize"}:
        mode = "categorical"
    elif numerical_mode:
        mode = "numerical"
    else:
        mode = "none"
    return categorical_mode == "binarize", mode


def restore_features(
    *datasets: DatasetObject, mode: str = "categorical"
) -> tuple[pd.DataFrame, ...]:
    if not isinstance(mode, str):
        raise TypeError("mode must be a str")
    mode = mode.strip().lower()
    if mode not in {"categorical", "numerical", "both"}:
        raise ValueError("mode must be one of: categorical, numerical, both")

    restored_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        raw_feature_type = (
            dataset.attr("raw_feature_type")
            if dataset_has_attr(dataset, "raw_feature_type")
            else {}
        )
        if not isinstance(raw_feature_type, dict):
            raise TypeError("raw_feature_type must be a dict[str, str]")
        raw_feature_type = {
            str(feature): str(feature_type)
            for feature, feature_type in raw_feature_type.items()
        }

        if getattr(dataset, "_freeze", False):
            frame = dataset.get(target=False)
        else:
            snapshot = dataset.snapshot()
            frame = snapshot.loc[:, snapshot.columns != dataset.target_column]

        restored = frame.copy(deep=True)
        if mode in {"categorical", "both"}:
            encoding = (
                dataset.attr("encoding")
                if dataset_has_attr(dataset, "encoding")
                else {}
            )
            value_mapping = (
                dataset.attr("encoded_value_mapping")
                if dataset_has_attr(dataset, "encoded_value_mapping")
                else {}
            )
            if not isinstance(encoding, dict):
                raise TypeError("encoding must be a dict[str, list[str]]")
            if not isinstance(value_mapping, dict):
                raise TypeError(
                    "encoded_value_mapping must be a dict[str, dict[int, object]]"
                )

            encoding = {
                str(source): [str(column) for column in columns]
                for source, columns in encoding.items()
            }
            resolved_mapping = {}
            for source, mapping in value_mapping.items():
                if not isinstance(mapping, dict):
                    raise TypeError(
                        "encoded_value_mapping entries must be dict[int, object] values"
                    )
                resolved_mapping[str(source)] = {
                    int(code): value for code, value in mapping.items()
                }
            value_mapping = resolved_mapping
            for feature in set(encoding) | set(value_mapping):
                raw_feature_type[feature] = "categorical"
            reverse_encoding = {
                column: source
                for source, columns in encoding.items()
                for column in columns
            }

            parts: list[pd.DataFrame] = []
            seen: set[str] = set()
            for column in frame.columns:
                source = reverse_encoding.get(str(column), str(column))
                if source in seen:
                    continue
                seen.add(source)

                if source not in encoding:
                    parts.append(frame.loc[:, [column]].copy(deep=True))
                    continue

                columns = [
                    encoded for encoded in encoding[source] if encoded in frame.columns
                ]
                if len(columns) != len(encoding[source]):
                    raise ValueError(f"Incomplete encoded feature group for '{source}'")

                if len(columns) == 1 and source not in value_mapping:
                    encoded = columns[0]
                    if encoded == source or (
                        "_cat_" not in encoded and "_therm_" not in encoded
                    ):
                        parts.append(frame.loc[:, columns].copy(deep=True))
                        continue

                if source in value_mapping:
                    valid_codes = value_mapping[source]
                    if not valid_codes:
                        raise ValueError(
                            "encoded_value_mapping must contain at least one valid code"
                        )
                    code_values = np.asarray(
                        sorted(valid_codes.keys()), dtype="float64"
                    )
                    values = frame.loc[:, columns[0]].to_numpy(
                        dtype="float64", copy=True
                    )
                    nearest = np.abs(values[:, None] - code_values[None, :]).argmin(
                        axis=1
                    )
                    restored_column = code_values[nearest].astype("float64")
                else:
                    values = frame.loc[:, columns].to_numpy(dtype="float64", copy=True)
                    if all("_cat_" in encoded for encoded in columns):
                        restored_column = values.argmax(axis=1).astype("float64")
                    elif all("_therm_" in encoded for encoded in columns):
                        rounded = np.where(values >= 0.5, 1.0, 0.0)
                        restored_column = np.clip(
                            rounded.sum(axis=1) - 1.0, 0.0, float(len(columns) - 1)
                        ).astype("float64")
                    else:
                        raise ValueError(
                            f"Unsupported encoded feature group: {columns}"
                        )
                parts.append(
                    pd.Series(restored_column, index=frame.index).to_frame(name=source)
                )

            restored = pd.concat(parts, axis=1)

        if mode in {"numerical", "both"}:
            if not dataset_has_attr(dataset, "scaling"):
                restored_frames.append(restored)
                continue
            if not dataset_has_attr(dataset, "scaling_stats"):
                raise ValueError("restore_numerical requires scaling_stats metadata")

            scaling = dataset.attr("scaling")
            scaling_stats = dataset.attr("scaling_stats")
            if not isinstance(scaling, dict):
                raise TypeError("scaling must be a dict[str, str]")
            if not isinstance(scaling_stats, dict):
                raise TypeError("scaling_stats must be a dict[str, dict[str, float]]")

            for column, mode_name in scaling.items():
                column = str(column)
                mode_name = str(mode_name).lower()
                if column not in restored.columns:
                    raise ValueError(
                        f"Missing scaled feature in restored frame: {column}"
                    )
                if raw_feature_type.get(column, "").lower() != "numerical":
                    raise ValueError(f"Scaled feature must be numerical: {column}")

                stats = scaling_stats.get(column)
                if not isinstance(stats, dict):
                    raise ValueError(f"Missing scaling_stats for feature: {column}")
                stats_mode = str(stats.get("mode", "")).lower()
                if stats_mode != mode_name:
                    raise ValueError(
                        f"Scaling metadata mismatch for feature '{column}': "
                        f"{mode_name} vs {stats_mode}"
                    )

                series = restored[column].astype("float64")
                if mode_name == "none":
                    continue
                if mode_name == "normalize":
                    if "min" not in stats or "max" not in stats:
                        raise ValueError(
                            f"normalize scaling_stats must contain min/max for '{column}'"
                        )
                    min_value = float(stats["min"])
                    max_value = float(stats["max"])
                    restored[column] = series * (max_value - min_value) + min_value
                elif mode_name == "standardize":
                    if "mean" not in stats or "std" not in stats:
                        raise ValueError(
                            f"standardize scaling_stats must contain mean/std for '{column}'"
                        )
                    mean_value = float(stats["mean"])
                    std_value = float(stats["std"])
                    restored[column] = series * std_value + mean_value
                else:
                    raise ValueError(
                        f"Unsupported scaling mode for feature '{column}': {mode_name}"
                    )

        restored.attrs["raw_feature_type"] = raw_feature_type
        restored_frames.append(restored)
    return tuple(restored_frames)


def distance(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    metric: str,
    binarize_list: list = [],
) -> torch.Tensor:
    metric = metric.lower()
    if metric not in {"l0", "l1", "l2", "linf"}:
        raise ValueError(f"Unsupported distance metric: {metric}")
    if list(factuals.columns) != list(counterfactuals.columns):
        raise ValueError("factuals and counterfactuals must have the same columns")
    if factuals.shape[0] != counterfactuals.shape[0]:
        raise ValueError("factuals and counterfactuals must have the same length")

    left = factuals.copy(deep=True)
    right = counterfactuals.copy(deep=True)
    if binarize_list:
        for column in binarize_list:
            if column not in left.columns:
                raise ValueError(f"Unknown binarize feature: {column}")
            same = [
                (pd.isna(left_value) and pd.isna(right_value))
                or left_value == right_value
                for left_value, right_value in zip(
                    left.loc[:, column].to_numpy(dtype=object),
                    right.loc[:, column].to_numpy(dtype=object),
                )
            ]
            left[column] = 0.0
            right[column] = np.where(same, 0.0, 1.0)

    diff = torch.abs(to_float_tensor(right) - to_float_tensor(left))
    if metric == "l0":
        return (
            (~torch.isclose(diff, torch.zeros(1, dtype=diff.dtype)))
            .sum(dim=1)
            .to(dtype=torch.float32)
        )
    if metric == "l1":
        return diff.sum(dim=1)
    if metric == "l2":
        return torch.linalg.vector_norm(diff, ord=2, dim=1)
    return diff.max(dim=1).values


def resolve_evaluation_inputs(
    factuals: DatasetObject, counterfactuals: DatasetObject
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if getattr(factuals, "counterfactual", False):
        raise ValueError("factuals must not be marked as counterfactual")
    if not getattr(counterfactuals, "counterfactual", False):
        raise ValueError("counterfactuals must be marked as counterfactual")
    if factuals.target_column != counterfactuals.target_column:
        raise ValueError(
            "factuals and counterfactuals must share the same target column"
        )

    factual_features = factuals.get(target=False)
    counterfactual_features = counterfactuals.get(target=False)
    if factual_features.shape != counterfactual_features.shape:
        raise ValueError(
            "factuals and counterfactuals must have the same feature shape"
        )
    if list(factual_features.columns) != list(counterfactual_features.columns):
        raise ValueError(
            "factuals and counterfactuals must have the same feature columns"
        )
    counterfactual_features = counterfactual_features.loc[factual_features.index]

    evaluation_mask = pd.Series(True, index=factual_features.index, dtype=bool)
    if not hasattr(counterfactuals, "evaluation_filter"):
        success_mask = ~counterfactual_features.isna().any(axis=1)
        return factual_features, counterfactual_features, evaluation_mask, success_mask

    raw_filter = counterfactuals.attr("evaluation_filter")
    if isinstance(raw_filter, pd.Series):
        evaluation_mask = raw_filter.astype(bool)
    elif isinstance(raw_filter, pd.DataFrame):
        if raw_filter.shape[1] == 0:
            raise ValueError("evaluation_filter must contain at least one column")
        if raw_filter.shape[1] == 1:
            evaluation_mask = raw_filter.iloc[:, 0].astype(bool)
        else:
            evaluation_mask = raw_filter.astype(bool).all(axis=1)
    else:
        raise TypeError("evaluation_filter must be a pandas Series or DataFrame")

    if evaluation_mask.shape[0] != counterfactual_features.shape[0]:
        raise ValueError("evaluation_filter length must match counterfactual rows")
    evaluation_mask = evaluation_mask.loc[factual_features.index]

    success_mask = ~counterfactual_features.isna().any(axis=1)
    return factual_features, counterfactual_features, evaluation_mask, success_mask


def resolve_ref_df(refset: DatasetObject) -> pd.DataFrame:
    if getattr(refset, "_freeze", False):
        return pd.concat([refset.get(target=False), refset.get(target=True)], axis=1)
    return refset.snapshot()


def to_float_tensor(df: pd.DataFrame) -> torch.Tensor:
    try:
        values = df.to_numpy(dtype="float32")
    except ValueError as error:
        raise ValueError("Evaluation requires numeric feature values") from error
    return torch.tensor(values, dtype=torch.float32)
