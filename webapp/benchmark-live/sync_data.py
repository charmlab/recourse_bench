from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_ROOT = Path(__file__).resolve().parent
DATA_DIR = SITE_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "benchmark" / "configs"
METHOD_CONFIG_DIR = CONFIG_ROOT / "methods"
SOURCE_RESULTS_CSV = PROJECT_ROOT / "benchmark" / "results" / "default_results.csv"
TARGET_RESULTS_CSV = DATA_DIR / "default_results.csv"
TARGET_COMPATIBILITY_JSON = DATA_DIR / "compatibility.json"


def prefer_yml_files(directory: Path) -> list[Path]:
    files_by_stem: dict[str, Path] = {}
    for path in sorted(directory.glob("*")):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        existing = files_by_stem.get(path.stem)
        if existing is None or path.suffix == ".yml":
            files_by_stem[path.stem] = path
    return [files_by_stem[stem] for stem in sorted(files_by_stem)]


def build_compatibility_manifest() -> dict[str, object]:
    methods: dict[str, dict[str, list[str]]] = {}

    for path in prefer_yml_files(METHOD_CONFIG_DIR):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        method_cfg = data.get("method") or {}
        compatibility_cfg = data.get("compatibility") or {}
        method_name = str(method_cfg.get("name") or path.stem).strip()
        if not method_name:
            continue

        allowed_models = [
            str(item).strip()
            for item in compatibility_cfg.get("allowed_models") or []
            if str(item).strip()
        ]
        allowed_datasets = [
            str(item).strip()
            for item in compatibility_cfg.get("allowed_datasets") or []
            if str(item).strip()
        ]

        methods[method_name] = {
            "allowed_models": allowed_models,
            "allowed_datasets": allowed_datasets,
        }

    return {"methods": methods}


def sync_compatibility_json() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_compatibility_manifest()
    TARGET_COMPATIBILITY_JSON.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TARGET_COMPATIBILITY_JSON


def sync_default_results_csv() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing source CSV: {SOURCE_RESULTS_CSV}")
    shutil.copyfile(SOURCE_RESULTS_CSV, TARGET_RESULTS_CSV)
    return TARGET_RESULTS_CSV


def sync_live_site_data() -> dict[str, str]:
    compatibility_path = sync_compatibility_json()
    results_path = sync_default_results_csv()
    return {
        "compatibility_json": str(compatibility_path),
        "default_results_csv": str(results_path),
    }


def main() -> None:
    outputs = sync_live_site_data()
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
