from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class WorkspaceConfig:
    key: str
    name: str
    path: str
    model: str
    provider: str | None = None
    thinking: str | None = None
    tools: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)

    def pi_args(self) -> list[str]:
        args = ["pi", "--mode", "json", "--print", "--no-session"]
        if self.provider:
            args.extend(["--provider", self.provider])
        if self.model:
            args.extend(["--model", self.model])
        if self.thinking:
            args.extend(["--thinking", self.thinking])
        if self.tools:
            args.extend(["--tools", ",".join(self.tools)])
        return args


@dataclass
class FeatureResult:
    feature_id: str
    title: str
    verdict: str  # pass | fail | skip
    reason: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Judgment:
    features: list[FeatureResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    overall: str = "unknown"  # pass | fail | unknown

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": [f.to_dict() for f in self.features],
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "overall": self.overall,
        }


@dataclass
class RunResult:
    model_key: str
    model_name: str
    provider: str | None
    status: str  # completed | failed | timeout | error
    exit_code: int | None
    started_at: str
    ended_at: str
    duration_ms: int
    worktree_path: str
    stdout_path: str
    stderr_path: str
    result_path: str
    error: str | None = None
    judgment: Judgment | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["judgment"] = self.judgment.to_dict() if self.judgment else None
        return data


@dataclass
class BenchSummary:
    run_id: str
    task_dir: str
    started_at: str
    ended_at: str
    duration_ms: int
    selected_models: list[str]
    git_commit: str | None
    results: list[RunResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_dir": self.task_dir,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "selected_models": self.selected_models,
            "git_commit": self.git_commit,
            "results": [r.to_dict() for r in self.results],
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"


def ms_between(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() * 1000)
