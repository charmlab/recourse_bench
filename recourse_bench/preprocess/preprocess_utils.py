from __future__ import annotations

from recourse_bench.dataset.dataset_object import DatasetObject


def dataset_has_attr(dataset: DatasetObject, flag: str) -> bool:
    try:
        dataset.attr(flag)
    except AttributeError:
        return False
    else:
        return True


def ensure_flag_absent(dataset: DatasetObject, flag: str) -> None:
    if dataset_has_attr(dataset, flag):
        raise ValueError(f"Dataset already contains preprocess flag: {flag}")


def resolve_feature_metadata(
    dataset: DatasetObject,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    if dataset_has_attr(dataset, "encoded_feature_type"):
        feature_type = dataset.attr("encoded_feature_type")
        feature_mutability = dataset.attr("encoded_feature_mutability")
        feature_actionability = dataset.attr("encoded_feature_actionability")
    else:
        feature_type = dataset.attr("raw_feature_type")
        feature_mutability = dataset.attr("raw_feature_mutability")
        feature_actionability = dataset.attr("raw_feature_actionability")
    return feature_type, feature_mutability, feature_actionability
