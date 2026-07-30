import sys

import recourse_bench.benchmark as benchmark
import recourse_bench.dataset as dataset
import recourse_bench.evaluation as evaluation
import recourse_bench.experiments as experiments
import recourse_bench.method as method
import recourse_bench.model as model
import recourse_bench.preprocess as preprocess
import recourse_bench.utils as utils
from recourse_bench.benchmark.run import run_benchmarks
from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.evaluation.evaluation_object import EvaluationObject
from recourse_bench.evaluation.evaluation_utils import distance, restore_features
from recourse_bench.experiments import Experiment
from recourse_bench.method.method_object import MethodObject
from recourse_bench.model.model_utils import logits_to_prediction, resolve_device
from recourse_bench.model.model_object import ModelObject, process_nan
from recourse_bench.preprocess.preprocess_object import PreProcessObject
from recourse_bench.preprocess.preprocess_utils import resolve_feature_metadata
from recourse_bench.utils.caching import default_cache_dir, get_cache_dir, set_cache_dir
from recourse_bench.utils.dependencies import optional_dependency, require_optional
from recourse_bench.utils.exceptions import (
    ConfigError,
    MissingDependencyError,
    RecourseBenchError,
)
from recourse_bench.utils.logger import setup_logger
from recourse_bench.utils.registry import get_registry, register
from recourse_bench.utils.seed import seed_context

# Functional API (loaders / discovery / run). Imported after the component
# packages above so the registry is fully populated.
from recourse_bench import api
from recourse_bench.api import (
    list_datasets,
    list_evaluations,
    list_methods,
    list_models,
    list_preprocessors,
    run,
    run_config_file,
)

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
    "MissingDependencyError",
    "RecourseBenchError",
    # Functional API
    "run",
    "run_config_file",
    "list_datasets",
    "list_preprocessors",
    "list_models",
    "list_methods",
    "list_evaluations",
    # Named namespaces
    "datasets",
    "preprocessors",
    "models",
    "methods",
    "evaluations",
    "default_cache_dir",
    "get_cache_dir",
    "get_registry",
    "optional_dependency",
    "require_optional",
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
