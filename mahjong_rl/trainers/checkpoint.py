"""Compact Flax parameter serialization (msgpack)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flax import serialization


def save_params(params: Any, path: Path | str) -> Path:
    """Serialize a Flax parameter pytree to disk as msgpack bytes."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(serialization.to_bytes(params))
    return out


def load_params(target_template: Any, path: Path | str) -> Any:
    """Restore a Flax parameter pytree matching `target_template` from disk."""
    data = Path(path).read_bytes()
    return serialization.from_bytes(target_template, data)
