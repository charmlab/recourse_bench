import sys

import benchmark as benchmark
import dataset as dataset
import evaluation as evaluation
import experiments as experiments
import method as method
import model as model
import preprocess as preprocess
import utils as utils
from benchmark.run import run_benchmarks
from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import distance, restore_features
from experiments import Experiment
from method.method_object import MethodObject
from model.model_utils import logits_to_prediction, resolve_device
from model.model_object import ModelObject, process_nan
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import resolve_feature_metadata
from utils.caching import get_cache_dir, set_cache_dir
from utils.exceptions import ConfigError, RecourseBenchError
from utils.logger import setup_logger
from utils.registry import get_registry, register
from utils.seed import seed_context

# Functional API (loaders / discovery / run). Imported after the component
# packages above so the registry is fully populated.
from recourse_bench import api
from recourse_bench.api import (
    list_datasets,
    list_evaluations,
    list_methods,
    list_models,
    list_preprocess,
    run,
    run_config_file,
)

sys.modules[__name__ + ".benchmark"] = benchmark
sys.modules[__name__ + ".dataset"] = dataset
sys.modules[__name__ + ".evaluation"] = evaluation
sys.modules[__name__ + ".experiments"] = experiments
sys.modules[__name__ + ".method"] = method
sys.modules[__name__ + ".model"] = model
sys.modules[__name__ + ".preprocess"] = preprocess
sys.modules[__name__ + ".utils"] = utils

# Canonical named namespaces: rb.methods.wachter, rb.datasets.credit, ...
# Populated dynamically from the registry so they never drift.
datasets = api.build_namespace(
    "dataset", __name__ + ".datasets", "Registered datasets, by name."
)
preprocessors = api.build_namespace(
    "preprocess", __name__ + ".preprocessors", "Registered preprocessing steps, by name."
)
models = api.build_namespace(
    "model", __name__ + ".models", "Registered target models, by name."
)
methods = api.build_namespace(
    "method", __name__ + ".methods", "Registered recourse methods, by name."
)
evaluations = api.build_namespace(
    "evaluation", __name__ + ".evaluations", "Registered evaluation metrics, by name."
)

sys.modules[__name__ + ".datasets"] = datasets
sys.modules[__name__ + ".preprocessors"] = preprocessors
sys.modules[__name__ + ".models"] = models
sys.modules[__name__ + ".methods"] = methods
sys.modules[__name__ + ".evaluations"] = evaluations

__all__ = [
    "benchmark",
    "dataset",
    "DatasetObject",
    "evaluation",
    "EvaluationObject",
    "experiments",
    "Experiment",
    "method",
    "MethodObject",
    "model",
    "ModelObject",
    "preprocess",
    "PreProcessObject",
    "utils",
    "ConfigError",
    "RecourseBenchError",
    # Functional API
    "run",
    "run_config_file",
    "list_datasets",
    "list_preprocess",
    "list_models",
    "list_methods",
    "list_evaluations",
    # Named namespaces
    "datasets",
    "preprocessors",
    "models",
    "methods",
    "evaluations",
    "get_cache_dir",
    "get_registry",
    "distance",
    "logits_to_prediction",
    "process_nan",
    "register",
    "resolve_device",
    "resolve_feature_metadata",
    "restore_features",
    "run_benchmarks",
    "seed_context",
    "set_cache_dir",
    "setup_logger",
]
