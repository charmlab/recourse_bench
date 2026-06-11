from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from skill_bench.config import ROOT_DIR, RUNS_DIR, skill_available
from skill_bench.judge import judge_worktree
from skill_bench.schemas import BenchSummary, RunResult, WorkspaceConfig, ms_between, now_iso


EventCallback = Callable[[dict], Awaitable[None]]
STREAM_LIMIT = 8 * 1024 * 1024


@dataclass
class RunnerOptions:
    prompt: str
    task_dir: Path
    selected_models: list[str]
    timeout_seconds: int = 1800
    max_concurrency: int = 2
    keep_worktrees: bool = False
    run_id: str | None = None


def _git_commit(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(path), capture_output=True, text=True, check=False
    )
    return (proc.stdout or "").strip() or None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.isoformat()


class SkillBenchRunner:
    def __init__(self, workspaces: dict[str, WorkspaceConfig]):
        self.workspaces = workspaces

    def preflight(self, selected_models: list[str]) -> list[str]:
        errors: list[str] = []
        unknown = [m for m in selected_models if m not in self.workspaces]
        if unknown:
            errors.append(f"Unknown model keys: {', '.join(unknown)}")
        if not shutil.which("pi"):
            errors.append("`pi` is not available in PATH.")
        if not shutil.which("git"):
            errors.append("`git` is not available in PATH.")
        for model in selected_models:
            ws = self.workspaces.get(model)
            if ws:
                for skill in ws.required_skills:
                    if not skill_available(skill):
                        errors.append(
                            f"Required skill `{skill}` not found in any skill search path. "
                            f"Place it under .agents/skills/ or ~/.pi/agent/skills/."
                        )
        return errors

    async def run(
        self,
        options: RunnerOptions,
        event_callback: EventCallback | None = None,
    ) -> BenchSummary:
        import uuid

        run_id = options.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "prompt.txt").write_text(options.prompt, encoding="utf-8")

        started = _now()

        sem = asyncio.Semaphore(options.max_concurrency)

        async def run_one(model_key: str) -> RunResult:
            async with sem:
                return await self._run_model(
                    model_key=model_key,
                    workspace=self.workspaces[model_key],
                    options=options,
                    run_dir=run_dir,
                    event_callback=event_callback,
                )

        results = await asyncio.gather(*[run_one(m) for m in options.selected_models])

        ended = _now()
        summary = BenchSummary(
            run_id=run_id,
            task_dir=str(options.task_dir),
            started_at=_iso(started),
            ended_at=_iso(ended),
            duration_ms=ms_between(started, ended),
            selected_models=options.selected_models,
            git_commit=_git_commit(ROOT_DIR),
            results=list(results),
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return summary

    async def _run_model(
        self,
        model_key: str,
        workspace: WorkspaceConfig,
        options: RunnerOptions,
        run_dir: Path,
        event_callback: EventCallback | None,
    ) -> RunResult:
        model_dir = run_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        worktree_path = model_dir / "worktree"
        stdout_path = model_dir / "stdout.log"
        stderr_path = model_dir / "stderr.log"
        result_path = model_dir / "result.json"

        if event_callback:
            await event_callback({"type": "start", "model": model_key, "status": "creating worktree"})

        worktree_created = False
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_path), "HEAD"],
                cwd=str(ROOT_DIR),
                check=True,
                capture_output=True,
            )
            worktree_created = True
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
            return self._error_result(model_key, workspace, model_dir, worktree_path, stdout_path, stderr_path, result_path, f"worktree creation failed: {err}")

        started = _now()
        status = "failed"
        exit_code = None
        error = None

        try:
            cmd = [*workspace.pi_args(), options.prompt]

            if event_callback:
                await event_callback({"type": "run", "model": model_key, "status": "running pi"})

            timeout = options.timeout_seconds if options.timeout_seconds > 0 else None
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(worktree_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=STREAM_LIMIT,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                exit_code = proc.returncode
                status = "completed" if exit_code == 0 else "failed"
            except asyncio.TimeoutError:
                proc.kill()
                stdout_bytes, stderr_bytes = await proc.communicate()
                exit_code = None
                status = "timeout"
                error = f"timed out after {options.timeout_seconds}s"

            stdout_path.write_bytes(stdout_bytes)
            stderr_path.write_bytes(stderr_bytes)

        except Exception as exc:
            status = "error"
            error = str(exc)
            stdout_path.write_bytes(b"")
            stderr_path.write_bytes(b"")

        ended = _now()

        if event_callback:
            await event_callback({"type": "judge", "model": model_key, "status": "judging output"})

        features_file = options.task_dir / "features.yaml"
        judgment = judge_worktree(worktree_path, features_file) if features_file.exists() else None

        result = RunResult(
            model_key=model_key,
            model_name=workspace.model,
            provider=workspace.provider,
            status=status,
            exit_code=exit_code,
            started_at=_iso(started),
            ended_at=_iso(ended),
            duration_ms=ms_between(started, ended),
            worktree_path=str(worktree_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            result_path=str(result_path),
            error=error,
            judgment=judgment,
        )
        result_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")

        if worktree_created and not options.keep_worktrees:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=str(ROOT_DIR),
                check=False,
                capture_output=True,
            )

        if event_callback:
            verdict = judgment.overall if judgment else "unknown"
            await event_callback({"type": "done", "model": model_key, "status": f"{status} | judgment: {verdict}"})

        return result

    def _error_result(
        self,
        model_key: str,
        workspace: WorkspaceConfig,
        model_dir: Path,
        worktree_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        result_path: Path,
        error: str,
    ) -> RunResult:
        import json
        now = _iso(_now())
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        result = RunResult(
            model_key=model_key,
            model_name=workspace.model,
            provider=workspace.provider,
            status="error",
            exit_code=None,
            started_at=now,
            ended_at=now,
            duration_ms=0,
            worktree_path=str(worktree_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            result_path=str(result_path),
            error=error,
            judgment=None,
        )
        result_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        return result
