from __future__ import annotations

import tomllib
from pathlib import Path

from skill_bench.schemas import WorkspaceConfig


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "skill_bench_runs"
TASK_DIR_DEFAULT = ROOT_DIR / "skill_bench_tasks" / "smoke-tests"
WORKSPACE_CONFIG_NAME = "bench.toml"
TASK_FILE_NAMES = ("TASK.md", "task.md", "PRD.md")


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def load_workspace_configs(task_dir: Path) -> dict[str, WorkspaceConfig]:
    configs: dict[str, WorkspaceConfig] = {}
    for workspace in sorted(task_dir.iterdir()):
        if not workspace.is_dir() or not workspace.name.endswith("-workspace"):
            continue
        config_path = workspace / WORKSPACE_CONFIG_NAME
        if not config_path.exists():
            continue
        parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
        key = workspace.name.removesuffix("-workspace")
        configs[key] = WorkspaceConfig(
            key=key,
            name=parsed.get("name", workspace.name),
            path=str(workspace),
            model=parsed.get("model", ""),
            provider=parsed.get("provider"),
            thinking=parsed.get("thinking"),
            tools=_as_str_list(parsed.get("tools")),
            required_skills=_as_str_list(parsed.get("required_skills")),
        )
    return configs


def resolve_task_file(task_dir: Path) -> Path | None:
    for name in TASK_FILE_NAMES:
        p = task_dir / name
        if p.exists():
            return p
    return None


def skill_available(skill_name: str) -> bool:
    search_roots = [
        ROOT_DIR / ".agents" / "skills",
        ROOT_DIR / ".pi" / "skills",
        Path.home() / ".pi" / "agent" / "skills",
        Path.home() / ".agents" / "skills",
    ]
    for root in search_roots:
        if (root / skill_name).is_dir() or (root / skill_name / "SKILL.md").exists():
            return True
    return False
