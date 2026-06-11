from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from skill_bench.config import TASK_DIR_DEFAULT, load_workspace_configs, resolve_task_file
from skill_bench.runner import RunnerOptions, SkillBenchRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Skill-bench: run pi agents against your skill definitions and judge the output"
    )
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Path to task directory containing TASK.md and *-workspace/ subdirs (default: skill_bench_tasks/smoke-tests)",
    )
    parser.add_argument("--models", nargs="+", required=True, help="Model workspace keys to run, e.g. claude gpt gemini")
    parser.add_argument("--prompt", help="Prompt text (overrides TASK.md)")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--keep-worktrees", action="store_true", help="Keep git worktrees after the run (useful for debugging)")
    parser.add_argument("--run-id", default=None, help="Override the auto-generated run ID")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    task_dir = Path(args.task_dir).resolve() if args.task_dir else TASK_DIR_DEFAULT

    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}")
        return 1

    workspaces = load_workspace_configs(task_dir)
    if not workspaces:
        print(f"No *-workspace/ directories with bench.toml found in {task_dir}")
        return 1

    if args.prompt:
        prompt = args.prompt
    else:
        task_file = resolve_task_file(task_dir)
        if not task_file:
            print(f"No TASK.md (or task.md / PRD.md) found in {task_dir}; pass --prompt to override")
            return 1
        prompt = f"Please read {task_file.name} and complete the task exactly as specified."

    runner = SkillBenchRunner(workspaces)
    errors = runner.preflight(args.models)
    if errors:
        for e in errors:
            print(f"[preflight] {e}")
        return 1

    async def on_event(event: dict) -> None:
        kind = event.get("type", "")
        model = event.get("model", "")
        status = event.get("status", "")
        print(f"[{kind}] {model}: {status}")

    options = RunnerOptions(
        prompt=prompt,
        task_dir=task_dir,
        selected_models=args.models,
        timeout_seconds=args.timeout_seconds,
        max_concurrency=args.max_concurrency,
        keep_worktrees=args.keep_worktrees,
        run_id=args.run_id,
    )

    summary = await runner.run(options=options, event_callback=on_event)

    print("\n" + "=" * 60)
    print(f"Run ID : {summary.run_id}")
    print(f"Task   : {summary.task_dir}")
    print(f"Commit : {summary.git_commit}")
    print()
    for result in summary.results:
        j = result.judgment
        verdict = j.overall if j else "no judgment"
        passed = j.passed if j else "-"
        failed = j.failed if j else "-"
        print(f"  {result.model_key:<12} {result.status:<12} judgment={verdict}  features: {passed} pass / {failed} fail")
        if j:
            for feat in j.features:
                mark = "+" if feat.verdict == "pass" else ("~" if feat.verdict == "skip" else "x")
                print(f"    [{mark}] {feat.feature_id}: {feat.reason}")
    print("=" * 60)

    print("\n" + json.dumps(summary.to_dict(), indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
