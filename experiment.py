from __future__ import annotations

import argparse
import logging
from copy import deepcopy

import pandas as pd
import yaml

# Trigger Registration
import dataset  # noqa: F401
import evaluation  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from utils.caching import set_cache_dir
from utils.logger import setup_logger
from utils.registry import get_registry


class Experiment:
    _cfg: dict
    _raw_dataset: object
    _preprocess: list
    _target_model: object
    _method: object
    _evaluation: list
    _metrics: pd.DataFrame | None = None

    def __init__(self, config: dict):
        self._cfg = deepcopy(config)
        self._metrics = None

        logger_cfg = self._cfg.get("logger", {})
        self._logger = setup_logger(
            level=logger_cfg.get("level", "INFO"),
            path=logger_cfg.get("path"),
            name=self._cfg.get("name", "benchmark"),
        )

        self._normalize_config()
        self._validate_config()
        self._logger.info(
            "Experiment config:\n%s", yaml.safe_dump(self._cfg, sort_keys=False)
        )

        caching_cfg = self._cfg.get("caching", {})
        set_cache_dir(caching_cfg.get("path", "./cache/"))

        self._raw_dataset = self._build_dataset()
        self._preprocess = self._build_preprocess()
        self._target_model = self._build_model()
        self._method = self._build_method()
        self._evaluation = self._build_evaluation()

    def _fatal(self, message: str) -> None:
        self._logger.fatal(message)
        raise SystemExit(1)

    def _normalize_config(self) -> None:
        preprocess_cfg = list(self._cfg.get("preprocess", []))
        if not any(
            item.get("name", "").lower() == "finalize" for item in preprocess_cfg
        ):
            preprocess_cfg.append({"name": "finalize"})
        self._cfg["preprocess"] = preprocess_cfg

    def _validate_config(self) -> None:
        required_sections = ["dataset", "model", "method", "evaluation"]
        for section in required_sections:
            if section not in self._cfg:
                self._fatal(f"Missing required config section: {section}")

        if not isinstance(self._cfg.get("preprocess", []), list):
            self._fatal("Config section 'preprocess' must be a list")
        if not isinstance(self._cfg.get("evaluation", []), list):
            self._fatal("Config section 'evaluation' must be a list")

        finalize_positions = [
            index
            for index, item in enumerate(self._cfg.get("preprocess", []))
            if item.get("name", "").lower() == "finalize"
        ]
        if (
            finalize_positions
            and finalize_positions[-1] != len(self._cfg["preprocess"]) - 1
        ):
            self._fatal("FinalizePreProcess must be the last preprocess step")

        registries = {
            "dataset": get_registry("Dataset"),
            "preprocess": get_registry("PreProcess"),
            "model": get_registry("TargetModel"),
            "method": get_registry("Method"),
            "evaluation": get_registry("Evaluation"),
        }

        dataset_name = self._cfg["dataset"].get("name")
        model_name = self._cfg["model"].get("name")
        method_name = self._cfg["method"].get("name")
        ensemble_model_cfgs = self._cfg["method"].get("ensemble_models", [])

        if dataset_name not in registries["dataset"]:
            self._fatal(f"Unknown dataset name: {dataset_name}")
        if model_name not in registries["model"]:
            self._fatal(f"Unknown model name: {model_name}")
        if method_name not in registries["method"]:
            self._fatal(f"Unknown method name: {method_name}")
        if not isinstance(ensemble_model_cfgs, list):
            self._fatal("method.ensemble_models must be a list when provided")
        for item in ensemble_model_cfgs:
            if not isinstance(item, dict):
                self._fatal("Each method.ensemble_models item must be a config object")
            ensemble_model_name = item.get("name")
            if ensemble_model_name not in registries["model"]:
                self._fatal(f"Unknown ensemble model name: {ensemble_model_name}")

        for item in self._cfg.get("preprocess", []):
            if item.get("name") not in registries["preprocess"]:
                self._fatal(f"Unknown preprocess name: {item.get('name')}")
        for item in self._cfg.get("evaluation", []):
            if item.get("name") not in registries["evaluation"]:
                self._fatal(f"Unknown evaluation name: {item.get('name')}")

        model_device = self._cfg["model"].get("device", "cpu").lower()
        method_device = self._cfg["method"].get("device", "cpu").lower()
        if model_device != method_device:
            self._fatal("model.device must match method.device")
        for item in ensemble_model_cfgs:
            ensemble_device = item.get("device", model_device).lower()
            if ensemble_device != method_device:
                self._fatal(
                    "All method.ensemble_models[].device values must match method.device"
                )

    def _build_dataset(self):
        cfg = deepcopy(self._cfg["dataset"])
        name = cfg.pop("name")
        dataset_class = get_registry("Dataset")[name]
        return dataset_class(**cfg)

    def _build_preprocess(self) -> list:
        preprocess_objects = []
        registry = get_registry("PreProcess")
        for cfg in self._cfg.get("preprocess", []):
            item_cfg = deepcopy(cfg)
            name = item_cfg.pop("name")
            preprocess_objects.append(registry[name](**item_cfg))
        return preprocess_objects

    def _build_model(self):
        cfg = deepcopy(self._cfg["model"])
        name = cfg.pop("name")
        model_class = get_registry("TargetModel")[name]
        return model_class(**cfg)

    def _build_method(self):
        cfg = deepcopy(self._cfg["method"])
        name = cfg.pop("name")
        ensemble_model_cfgs = list(cfg.pop("ensemble_models", []))
        method_class = get_registry("Method")[name]
        ensemble_models = []
        model_registry = get_registry("TargetModel")
        for item in ensemble_model_cfgs:
            item_cfg = deepcopy(item)
            item_name = item_cfg.pop("name")
            item_cfg.setdefault("device", self._target_model._device)
            ensemble_models.append(model_registry[item_name](**item_cfg))
        return method_class(
            target_model=self._target_model,
            ensemble_models=ensemble_models,
            **cfg,
        )

    def _build_evaluation(self) -> list:
        evaluation_objects = []
        registry = get_registry("Evaluation")
        for cfg in self._cfg.get("evaluation", []):
            item_cfg = deepcopy(cfg)
            name = item_cfg.pop("name")
            evaluation_objects.append(registry[name](**item_cfg))
        return evaluation_objects

    def _resolve_train_test(self, datasets: list):
        trainsets = [
            dataset for dataset in datasets if getattr(dataset, "trainset", False)
        ]
        testsets = [
            dataset for dataset in datasets if getattr(dataset, "testset", False)
        ]

        if len(trainsets) > 1 or len(testsets) > 1:
            self._fatal(
                "Experiment currently expects at most one trainset and one testset"
            )

        if trainsets and testsets:
            return trainsets[0], testsets[0]
        if len(datasets) == 1:
            self._logger.warning(
                "No split preprocess found; using the same frozen dataset for train and test"
            )
            return datasets[0], datasets[0]

        self._fatal("Could not resolve train/test datasets after preprocessing")

    def run(self) -> pd.DataFrame:
        datasets = [self._raw_dataset]

        for preprocess_step in self._preprocess:
            self._logger.info(
                "Starting preprocess: %s", preprocess_step.__class__.__name__
            )
            next_datasets = []
            for current_dataset in datasets:
                transformed = preprocess_step.transform(current_dataset)
                if isinstance(transformed, tuple):
                    next_datasets.extend(list(transformed))
                else:
                    next_datasets.append(transformed)
            datasets = next_datasets
            self._logger.info(
                "Completed preprocess: %s", preprocess_step.__class__.__name__
            )

        trainset, testset = self._resolve_train_test(datasets)

        self._logger.info(
            "Training target model: %s", self._target_model.__class__.__name__
        )
        self._target_model.fit(trainset)
        self._logger.info("Completed target model training")

        self._logger.info(
            "Training recourse method: %s", self._method.__class__.__name__
        )
        self._method.fit(trainset)
        self._logger.info("Completed recourse method training")

        factuals = testset

        self._logger.info("Generating counterfactuals")
        counterfactuals = self._method.predict(factuals)
        self._logger.info("Completed counterfactual generation")

        evaluation_results = []
        for evaluation_step in self._evaluation:
            self._logger.info(
                "Starting evaluation: %s", evaluation_step.__class__.__name__
            )
            evaluation_results.append(
                evaluation_step.evaluate(factuals, counterfactuals)
            )
            self._logger.info(
                "Completed evaluation: %s", evaluation_step.__class__.__name__
            )

        metrics = pd.concat(evaluation_results, axis=1)
        if self._metrics is not None and not self._metrics.equals(metrics):
            changed_columns = [
                column
                for column in metrics.columns
                if column not in self._metrics.columns
                or not self._metrics[column].equals(metrics[column])
            ]
            self._logger.warning("Metrics changed for columns: %s", changed_columns)

        self._metrics = metrics
        return metrics

    def metrics(self, run: bool = True) -> pd.DataFrame | None:
        if self._metrics is not None:
            return self._metrics
        if run:
            return self.run()
        self._logger.warning(
            "Metrics are not available because the experiment has not run yet"
        )
        return None

    def get_config(self) -> dict:
        return deepcopy(self._cfg)


def run_experiment(config_path: str | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default=None)
    args = parser.parse_args()

    if config_path is not None and args.path is not None:
        logging.getLogger(__name__).fatal(
            "Provide config path either via argument or CLI flag, not both"
        )
        raise SystemExit(1)

    resolved_path = config_path or args.path
    if resolved_path is None:
        logging.getLogger(__name__).fatal("Missing experiment config path")
        raise SystemExit(1)

    try:
        with open(resolved_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except Exception as error:
        logging.getLogger(__name__).fatal(
            "Failed to read config from %s: %s", resolved_path, error
        )
        raise SystemExit(1)

    if not isinstance(config, dict):
        logging.getLogger(__name__).fatal("Config file must parse to a dictionary")
        raise SystemExit(1)

    experiment = Experiment(config)
    metrics = experiment.run()
    logging.getLogger(__name__).info("Experiment completed successfully")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
