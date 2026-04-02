from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from method.diverse_dist.support import DiverseDistTrace
from model.mlp.mlp import MlpModel
from utils.logger import setup_logger
from utils.registry import get_registry

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")


class ReferenceDiabetesMlpModel(MlpModel):
    def _build_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        model = super()._build_model(input_dim, output_dim)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        return model


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _build_object(registry_type: str, config: dict, **kwargs):
    registry = get_registry(registry_type)
    item_cfg = deepcopy(config)
    name = item_cfg.pop("name")
    if registry_type.lower() == "model" and name == "mlp":
        cls = ReferenceDiabetesMlpModel
    else:
        cls = registry[name]
    return cls(**item_cfg, **kwargs)


def _materialize_dataset(config: dict):
    ds = _build_object("dataset", config["dataset"])
    for preprocess_cfg in config.get("preprocess", []):
        step = _build_object("preprocess", preprocess_cfg)
        ds = step.transform(ds)
    finalize = get_registry("preprocess")["finalize"]()
    return finalize.transform(ds)


def _clone_with_df(dataset_obj, df: pd.DataFrame, flag: str):
    cloned = dataset_obj.clone()
    cloned.update(flag, True, df=df.copy(deep=True))
    cloned.freeze()
    return cloned


def _split_dataset(dataset_obj, split_seed: int, test_size: float):
    feature_df = dataset_obj.get(target=False)
    target_df = dataset_obj.get(target=True)
    target_series = target_df.iloc[:, 0]
    combined = pd.concat([feature_df, target_df], axis=1)

    train_df, test_df = train_test_split(
        combined,
        test_size=float(test_size),
        random_state=int(split_seed),
        shuffle=True,
        stratify=target_series,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return (
        _clone_with_df(dataset_obj, train_df, "trainset"),
        _clone_with_df(dataset_obj, test_df, "testset"),
    )


def _predict_label_index(model_obj, row: np.ndarray, feature_names: list[str]) -> int:
    df = pd.DataFrame([row], columns=feature_names)
    prediction = model_obj.get_prediction(df, proba=False)
    return int(prediction.detach().cpu().numpy().argmax(axis=1)[0])


def _apply_noise(
    x: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    noise_percent: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    variance = float(noise_percent) * (train_max - train_min) / 100.0
    covariance = np.diag(variance.astype(np.float64, copy=False))
    noisy = rng.multivariate_normal(x.astype(np.float64), covariance)
    return noisy.clip(min=train_min, max=train_max).astype(np.float32, copy=False)


def _compute_distance(x: np.ndarray, y: np.ndarray, norm: int) -> float:
    if int(norm) == 1:
        return float(np.sum(np.abs(x - y)))
    if int(norm) == 2:
        return float(np.linalg.norm(x - y, ord=2))
    raise ValueError("norm must be 1 or 2")


def _check_validity(
    model_obj,
    factual: np.ndarray,
    counterfactuals: list[np.ndarray],
    feature_names: list[str],
) -> bool:
    factual_label = _predict_label_index(model_obj, factual, feature_names)
    for counterfactual in counterfactuals:
        if _predict_label_index(model_obj, counterfactual, feature_names) == factual_label:
            return False
    return True


def _compute_k_distance(
    factual: np.ndarray,
    counterfactuals: list[np.ndarray],
    norm: int,
) -> float:
    return float(
        sum(_compute_distance(factual, cf, norm) for cf in counterfactuals)
        / len(counterfactuals)
    )


def _compute_k_diversity(counterfactuals: list[np.ndarray], norm: int) -> float:
    if len(counterfactuals) < 2:
        return 0.0
    values: list[float] = []
    for i in range(len(counterfactuals)):
        for j in range(i + 1, len(counterfactuals)):
            values.append(_compute_distance(counterfactuals[i], counterfactuals[j], norm))
    return float(sum(values) / len(values))


def _compute_set_distance(
    cfx1: list[np.ndarray],
    cfx2: list[np.ndarray],
    norm: int,
) -> float:
    distances_1 = [
        min(_compute_distance(cf1, cf2, norm) for cf2 in cfx2)
        for cf1 in cfx1
    ]
    distances_2 = [
        min(_compute_distance(cf2, cf1, norm) for cf1 in cfx1)
        for cf2 in cfx2
    ]
    return float(sum(distances_1) / (2 * len(cfx1)) + sum(distances_2) / (2 * len(cfx2)))


def _compute_set_distance_max(
    cfx1: list[np.ndarray],
    cfx2: list[np.ndarray],
    norm: int,
) -> float:
    distances_1 = [
        min(_compute_distance(cf1, cf2, norm) for cf2 in cfx2)
        for cf1 in cfx1
    ]
    distances_2 = [
        min(_compute_distance(cf2, cf1, norm) for cf1 in cfx1)
        for cf2 in cfx2
    ]
    return float(0.5 * (max(distances_1) + max(distances_2)))


def _extract_counterfactual_set(method_obj) -> list[np.ndarray]:
    traces = getattr(method_obj, "_last_explanation_sets", [])
    if len(traces) != 1:
        return []
    trace = traces[0]
    if not isinstance(trace, DiverseDistTrace):
        return []
    return [np.asarray(cf, dtype=np.float32).reshape(-1) for cf in trace.counterfactuals]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}" if np.isfinite(mean) else "nan"


def _run_condition(
    base_config: dict,
    trainset,
    testset,
    model_obj,
    feature_names: list[str],
    train_min: np.ndarray,
    train_max: np.ndarray,
    norm: int,
    opt: bool,
    logger,
) -> dict[str, float | int]:
    method_cfg = deepcopy(base_config["method"])
    method_cfg["norm"] = int(norm)
    method_cfg["opt"] = bool(opt)
    method_obj = _build_object("method", method_cfg, target_model=model_obj)
    method_obj.fit(trainset)

    reproduction_cfg = base_config["reproduction"]
    start_index = int(reproduction_cfg["start_index"])
    num_inputs = int(reproduction_cfg["num_inputs"])
    repeat_times = int(reproduction_cfg["repeat_times"])
    noise_percent = float(reproduction_cfg["noise_percent"])
    noise_seed = int(reproduction_cfg["noise_seed"])
    show_progress = bool(reproduction_cfg.get("show_progress", True))

    test_features = testset.get(target=False).reset_index(drop=True)
    selected = test_features.iloc[start_index : start_index + num_inputs].copy(deep=True)

    rng = np.random.RandomState(noise_seed)
    results: list[dict[str, float]] = []
    skipped_noise_label_mismatch = 0
    evaluated_repeats = 0

    iterator = selected.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=selected.shape[0], desc=f"norm{norm}-opt{int(opt)}")

    for _, factual_row in iterator:
        x = factual_row.to_numpy(dtype=np.float32, copy=True)
        x_label = _predict_label_index(model_obj, x, feature_names)
        for _ in range(repeat_times):
            x_noisy = _apply_noise(
                x=x,
                train_min=train_min,
                train_max=train_max,
                noise_percent=noise_percent,
                rng=rng,
            )
            x_noisy_label = _predict_label_index(model_obj, x_noisy, feature_names)
            if x_noisy_label != x_label:
                skipped_noise_label_mismatch += 1
                continue
            evaluated_repeats += 1

            x_df = pd.DataFrame([x], columns=feature_names)
            x_noisy_df = pd.DataFrame([x_noisy], columns=feature_names)

            start = time.perf_counter()
            method_obj.get_counterfactuals(x_df)
            runtime = time.perf_counter() - start
            cfx = _extract_counterfactual_set(method_obj)

            method_obj.get_counterfactuals(x_noisy_df)
            cfx_noisy = _extract_counterfactual_set(method_obj)

            if len(cfx) == 0 or len(cfx_noisy) == 0:
                continue

            valid_org = _check_validity(model_obj, x, cfx, feature_names)
            valid_noisy = _check_validity(model_obj, x, cfx_noisy, feature_names)
            if not (valid_org and valid_noisy):
                continue

            results.append(
                {
                    "k_distance": _compute_k_distance(x, cfx, norm),
                    "k_diversity": _compute_k_diversity(cfx, norm),
                    "set_distance_sum": _compute_set_distance(cfx, cfx_noisy, norm),
                    "set_distance_max": _compute_set_distance_max(cfx, cfx_noisy, norm),
                    "time": float(runtime),
                }
            )

    validity = (
        100.0 * float(len(results)) / float(evaluated_repeats)
        if evaluated_repeats > 0
        else float("nan")
    )

    k_distance_mean, k_distance_std = _mean_std([item["k_distance"] for item in results])
    k_diversity_mean, k_diversity_std = _mean_std([item["k_diversity"] for item in results])
    set_distance_sum_mean, set_distance_sum_std = _mean_std(
        [item["set_distance_sum"] for item in results]
    )
    set_distance_max_mean, set_distance_max_std = _mean_std(
        [item["set_distance_max"] for item in results]
    )
    time_mean, time_std = _mean_std([item["time"] for item in results])

    logger.warning(
        "Completed norm=%s opt=%s with %s valid repeats and %s skipped-noisy repeats",
        norm,
        opt,
        len(results),
        skipped_noise_label_mismatch,
    )

    return {
        "validity": validity,
        "k_distance_mean": k_distance_mean,
        "k_distance_std": k_distance_std,
        "k_diversity_mean": k_diversity_mean,
        "k_diversity_std": k_diversity_std,
        "set_distance_sum_mean": set_distance_sum_mean,
        "set_distance_sum_std": set_distance_sum_std,
        "set_distance_max_mean": set_distance_max_mean,
        "set_distance_max_std": set_distance_max_std,
        "time_mean": time_mean,
        "time_std": time_std,
        "valid_repeats": len(results),
        "evaluated_repeats": evaluated_repeats,
        "skipped_noise_label_mismatch": skipped_noise_label_mismatch,
    }


def _print_comparison(results: list[dict[str, object]]) -> None:
    rows = []
    for item in results:
        observed = item["observed"]
        target = item["target"]
        rows.append(
            {
                "opt": item["opt"],
                "norm": item["norm"],
                "validity": f"{observed['validity']:.1f}% / {target['validity']:.1f}%",
                "k_distance": f"{_format_mean_std(observed['k_distance_mean'], observed['k_distance_std'])} / {_format_mean_std(target['k_distance_mean'], target['k_distance_std'])}",
                "k_diversity": f"{_format_mean_std(observed['k_diversity_mean'], observed['k_diversity_std'])} / {_format_mean_std(target['k_diversity_mean'], target['k_diversity_std'])}",
                "set_d_sum": f"{_format_mean_std(observed['set_distance_sum_mean'], observed['set_distance_sum_std'])} / {_format_mean_std(target['set_distance_sum_mean'], target['set_distance_sum_std'])}",
                "set_d_max": f"{_format_mean_std(observed['set_distance_max_mean'], observed['set_distance_max_std'])} / {_format_mean_std(target['set_distance_max_mean'], target['set_distance_max_std'])}",
                "time": f"{_format_mean_std(observed['time_mean'], observed['time_std'])} / {_format_mean_std(target['time_mean'], target['time_std'])}",
                "valid_repeats": observed["valid_repeats"],
                "skipped_noise": observed["skipped_noise_label_mismatch"],
            }
        )

    comparison = pd.DataFrame(rows)
    print("Observed / Paper Target")
    print(comparison.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _load_config(config_path)
    configured_device = str(config.get("model", {}).get("device", "cpu")).lower()
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested in config but is not available")
    device = configured_device

    logger = setup_logger(
        level=str(config.get("logger", {}).get("level", "WARNING")),
        path=None,
        name=config.get("name", "diverse_dist_reproduction"),
    )

    dataset_obj = _materialize_dataset(config)
    trainset, testset = _split_dataset(
        dataset_obj,
        split_seed=int(config["reproduction"]["split_seed"]),
        test_size=float(config["reproduction"]["test_size"]),
    )
    model_obj = _build_object("model", config["model"])
    model_obj.fit(trainset)

    train_array = trainset.get(target=False).to_numpy(dtype=np.float32)
    train_min = train_array.min(axis=0)
    train_max = train_array.max(axis=0)
    feature_names = list(trainset.get(target=False).columns)

    results: list[dict[str, object]] = []
    for opt in config["reproduction"]["opt_settings"]:
        target_group = "opt_true" if bool(opt) else "opt_false"
        for norm in config["reproduction"]["norms"]:
            logger.warning("Running diabetes reproduction for norm=%s opt=%s", norm, opt)
            observed = _run_condition(
                base_config=config,
                trainset=trainset,
                testset=testset,
                model_obj=model_obj,
                feature_names=feature_names,
                train_min=train_min,
                train_max=train_max,
                norm=int(norm),
                opt=bool(opt),
                logger=logger,
            )
            target = config["reproduction"]["paper_targets"][target_group][f"l{int(norm)}"]
            results.append(
                {
                    "opt": bool(opt),
                    "norm": int(norm),
                    "observed": observed,
                    "target": target,
                }
            )

    _print_comparison(results)


if __name__ == "__main__":
    main()
