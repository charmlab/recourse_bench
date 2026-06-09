from __future__ import annotations

import argparse
import copy
import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import urlparse

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = Path(__file__).resolve().parent
DATA_DIR = SITE_ROOT / "data"
BASE_RESULTS_CSV = DATA_DIR / "default_results.csv"
RUNTIME_RESULTS_CSV = DATA_DIR / "runtime_results.csv"

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SITE_ROOT))

# Trigger registration
import dataset  # noqa: F401,E402
import evaluation  # noqa: F401,E402
import method  # noqa: F401,E402
import model  # noqa: F401,E402
import preprocess  # noqa: F401,E402
from experiments import Experiment  # noqa: E402
from sync_data import sync_live_site_data  # noqa: E402
from utils.registry import get_registry  # noqa: E402

JOB_LOCK = Lock()
JOBS: dict[str, dict[str, object]] = {}
EXECUTOR = ThreadPoolExecutor(max_workers=1)

METHOD_DEFAULTS: dict[str, dict[str, object]] = {
    "toy": {
        "seed": 42,
        "device": "cpu",
        "desired_class": 1,
        "max_iterations": 200,
        "step_size": 0.05,
        "lambda_": 0.05,
        "clamp": True,
    },
    "wachter": {
        "seed": 42,
        "device": "cpu",
        "desired_class": 1,
        "lr": 0.01,
        "lambda_": 0.01,
        "n_iter": 300,
        "t_max_min": 0.1,
        "norm": 1,
        "clamp": True,
        "loss_type": "BCE",
    },
    "gs": {
        "seed": 42,
        "device": "cpu",
        "desired_class": 1,
        "n_search_samples": 300,
        "p_norm": 2,
        "step": 0.15,
        "max_iter": 200,
    },
    "face": {
        "seed": 42,
        "device": "cpu",
        "desired_class": 1,
        "mode": "knn",
        "fraction": 0.2,
    },
    "claproar": {
        "seed": 42,
        "device": "cpu",
        "desired_class": 1,
        "individual_cost_lambda": 0.1,
        "external_cost_lambda": 0.1,
        "learning_rate": 0.01,
        "max_iter": 200,
        "tol": 1e-4,
    },
}

MODEL_DEFAULTS: dict[str, dict[str, object]] = {
    "linear": {
        "seed": 42,
        "device": "cpu",
        "epochs": 80,
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer": "adam",
        "criterion": "cross_entropy",
        "output_activation": "softmax",
    },
    "mlp": {
        "seed": 42,
        "device": "cpu",
        "epochs": 120,
        "learning_rate": 0.005,
        "batch_size": 32,
        "layers": [64, 32],
        "optimizer": "adam",
        "criterion": "cross_entropy",
        "output_activation": "softmax",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_method_config(method_name: str) -> dict[str, object]:
    if method_name in METHOD_DEFAULTS:
        return copy.deepcopy(METHOD_DEFAULTS[method_name])
    return {
        "seed": 42,
        "device": "cpu",
        "desired_class": 1,
    }


def sanitize_value(value):
    if isinstance(value, (float, int, str, bool)) or value is None:
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if pd.isna(value):
        return None
    return str(value)


def sanitize_row(row: dict[str, object]) -> dict[str, object]:
    return {key: sanitize_value(value) for key, value in row.items()}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_runtime_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(rows)
    if RUNTIME_RESULTS_CSV.exists():
        existing = pd.read_csv(RUNTIME_RESULTS_CSV)
        combined = pd.concat([existing, incoming], axis=0, ignore_index=True)
    else:
        combined = incoming
    combined.to_csv(RUNTIME_RESULTS_CSV, index=False)


def all_results() -> pd.DataFrame:
    frames = []
    base = read_csv(BASE_RESULTS_CSV)
    if not base.empty:
        frames.append(base)
    runtime = read_csv(RUNTIME_RESULTS_CSV)
    if not runtime.empty:
        frames.append(runtime)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def registry_options() -> dict[str, list[str]]:
    datasets = sorted(get_registry("dataset").keys())
    models = sorted(get_registry("model").keys())
    methods = sorted(get_registry("method").keys())
    evaluations = sorted(get_registry("evaluation").keys())
    return {
        "datasets": datasets,
        "models": models,
        "methods": methods,
        "evaluations": evaluations,
    }


def metrics_to_evaluation(metrics: list[str]) -> list[dict[str, object]]:
    metrics_set = {metric.strip() for metric in metrics if metric and isinstance(metric, str)}
    evaluation_cfg: list[dict[str, object]] = []

    if "validity" in metrics_set:
        evaluation_cfg.append({"name": "validity"})

    distance_metrics = []
    for metric in ["distance_l0", "distance_l1", "distance_l2", "distance_linf"]:
        if metric in metrics_set:
            distance_metrics.append(metric.replace("distance_", ""))
    if distance_metrics:
        evaluation_cfg.append({"name": "distance", "metrics": distance_metrics})

    if "ynn" in metrics_set:
        evaluation_cfg.append({"name": "ynn", "k": 5})
    if "runtime_seconds" in metrics_set:
        evaluation_cfg.append({"name": "runtime"})

    if not evaluation_cfg:
        evaluation_cfg = [
            {"name": "validity"},
            {"name": "distance", "metrics": ["l0", "l1", "l2", "linf"]},
            {"name": "ynn", "k": 5},
            {"name": "runtime"},
        ]

    return evaluation_cfg


def ensure_config_defaults(
    config: dict[str, object],
    run_name: str,
    metrics: list[str] | None,
) -> dict[str, object]:
    cfg = copy.deepcopy(config)

    if "dataset" not in cfg or "model" not in cfg or "method" not in cfg:
        raise ValueError("Config must include dataset, model, and method sections")

    cfg["name"] = cfg.get("name", run_name)
    cfg.setdefault("logger", {
        "level": "INFO",
        "path": f"./logs/live_site/{run_name}.log",
    })
    cfg.setdefault("caching", {"path": "./cache/live_site/"})

    if "preprocess" not in cfg:
        cfg["preprocess"] = [
            {"name": "scale", "seed": 42, "scaling": "standardize", "range": True},
            {"name": "split", "seed": 42, "split": 0.2},
            {"name": "finalize"},
        ]

    if metrics is not None:
        cfg["evaluation"] = metrics_to_evaluation(metrics)
    else:
        cfg.setdefault("evaluation", metrics_to_evaluation([]))

    model_cfg = cfg["model"]
    method_cfg = cfg["method"]
    if not isinstance(model_cfg, dict) or not isinstance(method_cfg, dict):
        raise ValueError("model and method sections must be mappings")

    model_name = str(model_cfg.get("name", ""))
    method_name = str(method_cfg.get("name", ""))
    if not model_name or not method_name:
        raise ValueError("model.name and method.name are required")

    merged_model = copy.deepcopy(MODEL_DEFAULTS.get(model_name, {"seed": 42, "device": "cpu"}))
    merged_model.update(model_cfg)
    cfg["model"] = merged_model

    merged_method = normalize_method_config(method_name)
    merged_method.update(method_cfg)
    cfg["method"] = merged_method

    return cfg


def selection_to_configs(payload: dict[str, object]) -> list[dict[str, object]]:
    dataset_name = str(payload.get("dataset", "")).strip()
    model_name = str(payload.get("model", "")).strip()
    methods = payload.get("methods", [])
    metrics = payload.get("metrics", [])

    if not dataset_name or not model_name:
        raise ValueError("dataset and model are required")
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    configs: list[dict[str, object]] = []

    for index, method_name in enumerate(methods, start=1):
        method_name = str(method_name).strip()
        if not method_name:
            continue
        run_name = f"live_{dataset_name}_{model_name}_{method_name}_{timestamp}_{index:02d}"
        cfg = {
            "name": run_name,
            "dataset": {"name": dataset_name},
            "model": {"name": model_name},
            "method": {"name": method_name},
        }
        configs.append(ensure_config_defaults(cfg, run_name, metrics if isinstance(metrics, list) else []))

    if not configs:
        raise ValueError("No valid methods were provided")
    return configs


def yaml_to_configs(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_yaml = payload.get("yaml_text")
    metrics = payload.get("metrics")
    override_methods = payload.get("methods")
    if not isinstance(raw_yaml, str) or not raw_yaml.strip():
        raise ValueError("yaml_text is required")

    parsed = yaml.safe_load(raw_yaml)
    if not isinstance(parsed, dict):
        raise ValueError("Uploaded YAML must parse to a mapping")

    configs: list[dict[str, object]] = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    base_methods = parsed.get("methods")
    if isinstance(base_methods, list) and base_methods:
        template = copy.deepcopy(parsed)
        template.pop("methods", None)
        method_template = template.pop("method", {})
        for index, method_name in enumerate(base_methods, start=1):
            cfg = copy.deepcopy(template)
            cfg_method = copy.deepcopy(method_template) if isinstance(method_template, dict) else {}
            cfg_method["name"] = str(method_name)
            cfg["method"] = cfg_method
            run_name = f"uploaded_{timestamp}_{index:02d}_{cfg_method['name']}"
            cfg["name"] = cfg.get("name", run_name)
            configs.append(ensure_config_defaults(cfg, run_name, metrics if isinstance(metrics, list) else None))
    elif isinstance(override_methods, list) and override_methods:
        for index, method_name in enumerate(override_methods, start=1):
            cfg = copy.deepcopy(parsed)
            cfg_method = cfg.get("method", {}) if isinstance(cfg.get("method"), dict) else {}
            cfg_method["name"] = str(method_name)
            cfg["method"] = cfg_method
            run_name = f"uploaded_{timestamp}_{index:02d}_{cfg_method['name']}"
            cfg["name"] = cfg.get("name", run_name)
            configs.append(ensure_config_defaults(cfg, run_name, metrics if isinstance(metrics, list) else None))
    else:
        method_name = "method"
        if isinstance(parsed.get("method"), dict) and "name" in parsed["method"]:
            method_name = str(parsed["method"]["name"])
        run_name = f"uploaded_{timestamp}_{method_name}"
        parsed["name"] = parsed.get("name", run_name)
        configs.append(ensure_config_defaults(parsed, run_name, metrics if isinstance(metrics, list) else None))

    return configs


def run_configs(job_id: str, configs: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    with JOB_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"

    for index, config in enumerate(configs, start=1):
        run_name = str(config.get("name", f"run_{index}"))
        method_name = str(config.get("method", {}).get("name", "unknown"))
        dataset_name = str(config.get("dataset", {}).get("name", "unknown"))
        model_name = str(config.get("model", {}).get("name", "unknown"))

        start = perf_counter()
        base = {
            "config_file": f"live://{job_id}/{run_name}.yaml",
            "run_name": run_name,
            "dataset": dataset_name,
            "model": model_name,
            "method": method_name,
        }
        try:
            metrics = Experiment(config).run()
            elapsed = perf_counter() - start
            row = base | {
                "status": "success",
                "elapsed_seconds": elapsed,
            }
            row.update(metrics.iloc[0].to_dict())
        except Exception as error:  # noqa: BLE001
            elapsed = perf_counter() - start
            row = base | {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(error),
                "traceback": traceback.format_exc(limit=1),
            }

        rows.append(sanitize_row(row))

        with JOB_LOCK:
            job = JOBS[job_id]
            progress = job["progress"]
            progress["completed"] = int(progress["completed"]) + 1
            if row["status"] == "failed":
                progress["failed"] = int(progress["failed"]) + 1
            job["rows"] = rows

    write_runtime_rows(rows)

    with JOB_LOCK:
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["finished_at"] = utc_now()


def submit_job(configs: list[dict[str, object]], source: str) -> dict[str, object]:
    job_id = uuid.uuid4().hex[:12]
    job_payload = {
        "id": job_id,
        "source": source,
        "status": "queued",
        "created_at": utc_now(),
        "finished_at": None,
        "progress": {
            "total": len(configs),
            "completed": 0,
            "failed": 0,
        },
        "rows": [],
        "error": None,
    }

    with JOB_LOCK:
        JOBS[job_id] = job_payload

    EXECUTOR.submit(run_configs, job_id, configs)
    return job_payload


class ApiHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_ROOT), **kwargs)

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        payload = self.rfile.read(content_length).decode("utf-8")
        return json.loads(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/options":
            self._send_json(registry_options())
            return
        if path == "/api/results":
            df = all_results()
            rows = [] if df.empty else [sanitize_row(item) for item in df.to_dict(orient="records")]
            self._send_json({"rows": rows, "count": len(rows)})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOB_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "Job not found"}, status=404)
                return
            self._send_json(copy.deepcopy(job))
            return

        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except Exception as error:  # noqa: BLE001
            self._send_json({"error": f"Invalid JSON: {error}"}, status=400)
            return

        try:
            if path == "/api/run-selection":
                configs = selection_to_configs(payload)
                job = submit_job(configs, source="selection")
                self._send_json({"job_id": job["id"], "status": job["status"]}, status=202)
                return
            if path == "/api/run-config":
                configs = yaml_to_configs(payload)
                job = submit_job(configs, source="upload")
                self._send_json({"job_id": job["id"], "status": job["status"]}, status=202)
                return
        except Exception as error:  # noqa: BLE001
            self._send_json({"error": str(error)}, status=400)
            return

        self._send_json({"error": "Unknown endpoint"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    sync_live_site_data()
    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"Serving benchmark live app on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
