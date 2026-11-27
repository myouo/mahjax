"""Small JAX runtime helpers shared by the trainers and evaluation entry points."""

from __future__ import annotations

import os

import jax

_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/jax_mahjong_rl")


def enable_persistent_compilation_cache(cache_dir: str | None = None) -> str:
    """Reuse XLA-compiled executables across processes.

    Compiling the full V6 rollout / Duplicate evaluator can take minutes; without a
    persistent cache every fresh Python process pays that cost again.  The cache is
    keyed on the compiled HLO, so it only helps when shapes and *inputs-not-constants*
    match -- which is exactly the case for the params-as-arguments evaluator.

    Returns the cache directory that was configured (or that was already active).
    """
    active = jax.config.jax_compilation_cache_dir
    if active:
        return active
    path = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR") or _DEFAULT_CACHE_DIR
    os.makedirs(path, exist_ok=True)
    # Older/newer JAX builds may not expose every tuning flag; caching is best-effort.
    for key, value in (
        ("jax_compilation_cache_dir", path),
        ("jax_persistent_cache_min_compile_time_secs", 1.0),
        ("jax_persistent_cache_min_entry_size_bytes", 0),
    ):
        try:
            jax.config.update(key, value)
        except Exception:
            pass
    return path
