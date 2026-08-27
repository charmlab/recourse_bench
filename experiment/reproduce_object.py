from __future__ import annotations

from abc import ABC, abstractmethod

from pathlib import Path
import torch
import yaml


class ReproductionObject(ABC):
    """
    Abstract Parent Class for reproducing paper claims.
    Ensures a standardized testing pipeline across all methods.
    """
    def __init__(self, config_path: str | Path):
        self._config_path = Path(config_path)
        self._config = self._load_config(self._config_path)
        self._device = self._resolve_runtime_device()
        self._results: dict[str, object] = {}

    def _load_config(self, config_path: Path) -> dict:
        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _resolve_runtime_device(self) -> str:
        model_device = str(self._config["model"]["device"]).lower()
        method_device = str(self._config["method"].get("device", model_device)).lower()
        
        if model_device == "auto":
            model_device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_device != method_device:
            raise ValueError("model.device must match method.device")
        return model_device

    @abstractmethod
    def _run_suite(self) -> dict[str, object]:
        """Executes the reproduction experiment pipeline and returns raw metrics."""
        pass

    def evaluate_reproduction(self) -> dict[str, object]:
        """Public runner that executes the suite and formats comparisons."""
        raw_results = self._run_suite()
        targets = self._config["reproduction"]["targets"]
        
        return {
            "results": raw_results,
            "notebook_comparison": self._compare_against_targets(raw_results, targets.get("notebook", {})),
            "paper_comparison": self._compare_against_targets(raw_results, targets.get("paper", {})),
        }

    def _compare_against_targets(self, results: dict[str, float], targets: dict[str, float]) -> list[tuple[str, float, float, float]]:
        rows = []
        for key, target_value in targets.items():
            if key in results:
                reproduced = float(results[key])
                target = float(target_value)
                rows.append((key, target, reproduced, abs(reproduced - target)))
        return rows