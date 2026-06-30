from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import logging
import os
import subprocess
from copy import deepcopy
from datetime import datetime, timezone

import pandas as pd
import yaml

# Trigger Registration
import recourse_bench.dataset  # noqa: F401
import recourse_bench.evaluation  # noqa: F401
import recourse_bench.method  # noqa: F401
import recourse_bench.model  # noqa: F401
import recourse_bench.preprocess  # noqa: F401
from recourse_bench.utils.caching import set_cache_dir
from recourse_bench.utils.exceptions import ConfigError, RecourseBenchError
from recourse_bench.utils.logger import setup_logger
from recourse_bench.utils.registry import get_registry


class Experiment:
    """Build and run a full recourse benchmark from a config dictionary.

    Given a config (typically loaded from YAML), an experiment instantiates the
    dataset, preprocessing pipeline, target model, recourse method, and
    evaluation metrics from the registry. :meth:`run` then preprocesses the
    data, resolves train/test splits, trains the model, fits the method,
    generates counterfactuals, and concatenates evaluation outputs into one
    metrics dataframe (carrying provenance under ``metrics.attrs``).

    A single top-level ``seed`` in the config is propagated to every component
    that does not set its own. Configuration errors raise
    :class:`~utils.exceptions.ConfigError` rather than exiting the process.

    Parameters
    ----------
    config : dict
        Experiment configuration. Required sections: ``dataset``, ``model``,
        ``method``, ``evaluation``. Optional: ``preprocess``, ``logger``,
        ``caching``, ``name``, ``seed``.

    Examples
    --------
    >>> import yaml
    >>> from experiments import Experiment
    >>> cfg = yaml.safe_load(open("experiment/toy/smoke_config.yaml"))
    >>> metrics = Experiment(cfg).run()
    """

    _cfg: dict
    _raw_dataset: object
    _preprocess: list
    _target_model: object
    _method: object
    _evaluation: list
    _metrics: pd.DataFrame | None = None
    _trainset: object = None
    _testset: object = None
    _counterfactuals: object = None

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
        raise ConfigError(message)

    def _normalize_config(self) -> None:
        preprocess_cfg = list(self._cfg.get("preprocess", []))
        if not any(
            item.get("name", "").lower() == "finalize" for item in preprocess_cfg
        ):
            preprocess_cfg.append({"name": "finalize"})
        self._cfg["preprocess"] = preprocess_cfg

        self._propagate_seed()

    def _propagate_seed(self) -> None:
        """Fill in a per-component ``seed`` from the top-level ``seed``.

        A single top-level ``seed:`` is the default for every component that
        does not specify its own, so reproducible runs need only one value.
        Per-component ``seed`` always wins when present.
        """
        seed = self._cfg.get("seed")
        if seed is None:
            return
        for section in ("dataset", "model", "method"):
            block = self._cfg.get(section)
            if isinstance(block, dict):
                block.setdefault("seed", seed)
        for item in self._cfg.get("preprocess", []):
            if isinstance(item, dict):
                item.setdefault("seed", seed)

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

        if dataset_name not in registries["dataset"]:
            self._fatal(f"Unknown dataset name: {dataset_name}")
        if model_name not in registries["model"]:
            self._fatal(f"Unknown model name: {model_name}")
        if method_name not in registries["method"]:
            self._fatal(f"Unknown method name: {method_name}")

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
        method_class = get_registry("Method")[name]
        return method_class(target_model=self._target_model, **cfg)

    def _build_evaluation(self) -> list:
        evaluation_objects = []
        registry = get_registry("Evaluation")
        for cfg in self._cfg.get("evaluation", []):
            item_cfg = deepcopy(cfg)
            name = item_cfg.pop("name")
            uses_default_refset = False
            if name == "knn" and "refset" not in item_cfg:
                item_cfg["refset"] = self._raw_dataset
                uses_default_refset = True

            evaluation_object = registry[name](**item_cfg)
            if uses_default_refset:
                setattr(evaluation_object, "_default_refset", True)
            evaluation_objects.append(evaluation_object)
        return evaluation_objects

    def _bind_evaluation_context(self, trainset) -> None:
        for evaluation_step in self._evaluation:
            if not getattr(evaluation_step, "_default_refset", False):
                continue
            if not hasattr(evaluation_step, "set_refset"):
                continue
            evaluation_step.set_refset(trainset)

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
            self._bind_evaluation_context(trainsets[0])
            return trainsets[0], testsets[0]
        if len(datasets) == 1:
            self._logger.warning(
                "No split preprocess found; using the same frozen dataset for train and test"
            )
            self._bind_evaluation_context(datasets[0])
            return datasets[0], datasets[0]

        self._fatal("Could not resolve train/test datasets after preprocessing")

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return the metrics table.

        Runs preprocessing, train/test resolution, model training, method
        fitting, counterfactual generation, and all evaluations. The trained
        model, fitted method, resolved datasets, and counterfactuals are kept
        for inspection via :meth:`target_model`, :meth:`recourse_method`,
        :meth:`trainset`, :meth:`testset`, and :meth:`counterfactuals`.

        Returns
        -------
        pandas.DataFrame
            One-row metrics table. ``metrics.attrs["provenance"]`` records the
            library version, config hash, git revision, timestamp, and seed.
        """
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
        self._trainset = trainset
        self._testset = testset

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
        self._counterfactuals = counterfactuals
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
        metrics.attrs["provenance"] = self._build_provenance()

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

    def _build_provenance(self) -> dict:
        """Self-describing record of how the metrics were produced."""
        config_bytes = yaml.safe_dump(self._cfg, sort_keys=True).encode("utf-8")
        return {
            "library_version": _library_version(),
            "config_hash": hashlib.sha256(config_bytes).hexdigest(),
            "git_revision": _git_revision(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self._cfg.get("seed"),
            "dataset": self._cfg.get("dataset", {}).get("name"),
            "model": self._cfg.get("model", {}).get("name"),
            "method": self._cfg.get("method", {}).get("name"),
        }

    def provenance(self) -> dict | None:
        """Return the provenance attached to the latest metrics, if any."""
        if self._metrics is None:
            return None
        return dict(self._metrics.attrs.get("provenance", {}))

    def target_model(self):
        """Return the (trained, after :meth:`run`) target ModelObject."""
        return self._target_model

    def recourse_method(self):
        """Return the (fitted, after :meth:`run`) recourse MethodObject."""
        return self._method

    def trainset(self):
        """Return the finalized training DatasetObject (available after :meth:`run`)."""
        return self._trainset

    def testset(self):
        """Return the finalized test/factual DatasetObject (available after :meth:`run`)."""
        return self._testset

    def counterfactuals(self):
        """Return the counterfactual DatasetObject produced by :meth:`run`."""
        return self._counterfactuals


def _library_version() -> str:
    try:
        return importlib.metadata.version("recourse_bench")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    revision = result.stdout.strip()
    return revision or None


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

    try:
        experiment = Experiment(config)
        metrics = experiment.run()
    except RecourseBenchError as error:
        logging.getLogger(__name__).fatal("Experiment failed: %s", error)
        raise SystemExit(1)
    logging.getLogger(__name__).info("Experiment completed successfully")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    run_experiment()
