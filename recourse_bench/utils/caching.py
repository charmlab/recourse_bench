from __future__ import annotations

import os
from pathlib import Path

#: Environment variable overriding the default cache location.
cache_env_var = "RECOURSE_BENCH_CACHE"


def default_cache_dir() -> str:
    """Return the cache directory to use when a config does not name one.

    Caches are derived data, not project files, so they belong in the user's
    cache directory rather than the working directory. Resolution order:

    1. ``$RECOURSE_BENCH_CACHE``, if set.
    2. ``$XDG_CACHE_HOME/recourse_bench``, if ``XDG_CACHE_HOME`` is set.
    3. ``~/.cache/recourse_bench``.

    An experiment config's ``caching.path`` still wins over all of these.

    Returns
    -------
    str
        Path to the default cache directory. Not created by this call.
    """
    override = os.environ.get(cache_env_var)
    if override:
        return override
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return str(base / "recourse_bench")


global_cache_dir = default_cache_dir()


def set_cache_dir(path: str) -> None:
    global global_cache_dir
    global_cache_dir = path
    Path(global_cache_dir).mkdir(parents=True, exist_ok=True)


def get_cache_dir(sub: str) -> str:
    path = Path(global_cache_dir) / sub
    path.mkdir(parents=True, exist_ok=True)
    return f"{path.as_posix()}/"
