from __future__ import annotations

from pathlib import Path

global_cache_dir = "./cache/"


def set_cache_dir(path: str) -> None:
    global global_cache_dir
    global_cache_dir = path
    Path(global_cache_dir).mkdir(parents=True, exist_ok=True)


def get_cache_dir(sub: str) -> str:
    path = Path(global_cache_dir) / sub
    path.mkdir(parents=True, exist_ok=True)
    return f"{path.as_posix()}/"
