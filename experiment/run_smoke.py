from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent


def discover_smoke_methods() -> list[str]:
    methods = [path.parent.name for path in EXPERIMENT_DIR.glob("*/smoke.py")]
    return sorted(set(methods))


def run_smoke(method: str) -> int:
    method_dir = EXPERIMENT_DIR / method
    smoke_script = method_dir / "smoke.py"
    log_path = method_dir / f"{method}_smoke_log.txt"

    if not smoke_script.exists():
        raise FileNotFoundError(f"Missing smoke script for method '{method}': {smoke_script}")

    command = [sys.executable, smoke_script.as_posix()]
    completed = subprocess.run(
        command,
        cwd=method_dir,
        capture_output=True,
        text=True,
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        # log_file.write(f"Command: {' '.join(command)}\n")
        # log_file.write(f"Working directory: {method_dir}\n")
        # log_file.write(f"Return code: {completed.returncode}\n\n")
        if completed.stdout:
            log_file.write("--- STDOUT ---\n")
            log_file.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                log_file.write("\n")
        else:
            log_file.write("--- STDOUT ---\nNo standard output captured.\n")

        # if completed.stderr:
        #     log_file.write("\n--- STDERR ---\n")
        #     log_file.write(completed.stderr)
        #     if not completed.stderr.endswith("\n"):
        #         log_file.write("\n")
        # else:
        #     log_file.write("\n--- STDERR ---\nNo standard error captured.\n")

    print(f"[{method}] wrote {log_path.name} (exit {completed.returncode})")
    return completed.returncode


def main() -> int:
    available_methods = discover_smoke_methods()

    parser = argparse.ArgumentParser(
        description="Run smoke tests for experiment methods and store their output logs."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=available_methods,
        help="Subset of smoke methods to run. Defaults to all discovered smoke methods.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print the discovered smoke methods and exit.",
    )
    args = parser.parse_args()

    if args.list_methods:
        for method in available_methods:
            print(method)
        return 0

    selected_methods = args.methods or available_methods
    if not selected_methods:
        print("No smoke methods were discovered.", file=sys.stderr)
        return 1

    exit_code = 0
    for method in selected_methods:
        print(f"Running smoke test for {method}...")
        method_exit_code = run_smoke(method)
        if method_exit_code != 0 and exit_code == 0:
            exit_code = method_exit_code

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())