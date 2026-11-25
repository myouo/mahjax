"""Hugging Face Hub sync & Kaggle operational toolkit for Sovereign-Mahjong V6.

Provides zero-boilerplate reusable helpers for:
- Resolving & verifying `HF_TOKEN` (Kaggle Secrets / env var / cached HF login).
- Uploading & downloading V6 `.npz` shard datasets (`repo_type="dataset"`), with automatic
  `upload_large_folder` switch for large shard counts and content-hash deduplication.
- Uploading & downloading Phase 0/1/2/3 checkpoints (`repo_type="model"`).
- Fast zero-decompression shard health inspection (`inspect_v6_shards`).
- Kaggle GPU / VRAM / disk quota dashboard (`inspect_kaggle_resources`) and safe cleanup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence
import zipfile

from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic


def resolve_hf_token(
    token: Optional[str] = None,
    secret_name: str = "HF_TOKEN",
) -> Optional[str]:
    """Resolve a Hugging Face token from explicit arg, env var, Kaggle Secrets, or local HF cache."""
    if token:
        return token.strip()
    env_tok = os.environ.get(secret_name) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_tok:
        return env_tok.strip()
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        sec = UserSecretsClient().get_secret(secret_name)
        if sec:
            return str(sec).strip()
    except Exception:
        pass
    try:
        from huggingface_hub import get_token

        cached = get_token()
        if cached:
            return str(cached).strip()
    except Exception:
        pass
    return None


def verify_hf_auth(token: Optional[str] = None, secret_name: str = "HF_TOKEN") -> Dict[str, Any]:
    """Verify Hugging Face authentication and print current account & token scopes."""
    from huggingface_hub import HfApi

    tok = resolve_hf_token(token=token, secret_name=secret_name)
    if not tok:
        raise RuntimeError(
            f"No Hugging Face token found! Set Kaggle Secret '{secret_name}' (and toggle it ON), "
            "or set environment variable HF_TOKEN."
        )
    api = HfApi(token=tok)
    info = api.whoami()
    name = info.get("name", "unknown")
    orgs = [o.get("name") for o in info.get("orgs", []) if isinstance(o, dict) and o.get("name")]
    auth_type = info.get("auth", {}).get("type", "unknown")
    print(f"[HF Auth OK] user={name} | auth={auth_type} | orgs={orgs}")
    return info


def _fast_shard_steps(path: Path) -> int:
    with zipfile.ZipFile(path, mode="r") as zf:
        with zf.open("action.npy", mode="r") as fh:
            major, minor = read_magic(fh)
            if (major, minor) == (1, 0):
                shape, _, _ = read_array_header_1_0(fh)
            else:
                shape, _, _ = read_array_header_2_0(fh)
    return int(shape[0])


def inspect_v6_shards(shard_dir: Path | str, verbose: bool = True) -> Dict[str, Any]:
    """Fast header-only integrity & statistics check for a V6 `.npz` shard directory."""
    d = Path(shard_dir)
    if not d.exists():
        res = {
            "exists": False,
            "num_shards": 0,
            "contiguous": True,
            "corrupt_shards": [],
            "tmp_files": [],
            "total_steps": 0,
            "total_mb": 0.0,
            "manifest": None,
            "progress": None,
        }
        if verbose:
            print(f"[Shards] Directory does not exist yet: {d}")
        return res

    shards = sorted(d.glob("shard_*.npz"))
    tmp_files = [str(p.name) for p in list(d.glob(".shard_*.tmp.npz")) + list(d.glob(".*.tmp"))]
    corrupt: List[str] = []
    total_steps = 0
    total_bytes = 0
    expected_idx = 0
    contiguous = True

    for p in shards:
        if p.name != f"shard_{expected_idx:05d}.npz":
            contiguous = False
        expected_idx += 1
        total_bytes += p.stat().st_size
        try:
            n = _fast_shard_steps(p)
            if n <= 0:
                corrupt.append(p.name)
            else:
                total_steps += n
        except Exception:
            corrupt.append(p.name)

    manifest_path = d / "manifest.json"
    progress_path = d / "progress.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None

    status = "EMPTY"
    if manifest is not None and bool(manifest.get("completed", True)):
        status = f"COMPLETE (games={manifest.get('games_processed')}, clean={manifest.get('clean_round_rate', 0.0)*100:.2f}%)"
    elif progress is not None:
        status = f"RESUMABLE_PROGRESS (games={progress.get('games_processed')}, next_game={progress.get('next_start_game')})"
    elif shards:
        status = "PARTIAL_LEGACY (no manifest/progress — auto-resumable via kyoku-fingerprint)"

    res = {
        "exists": True,
        "shard_dir": str(d),
        "status": status,
        "num_shards": len(shards),
        "contiguous": contiguous,
        "corrupt_shards": corrupt,
        "tmp_files": tmp_files,
        "total_steps": total_steps,
        "total_mb": round(total_bytes / (1024 ** 2), 2),
        "manifest": manifest,
        "progress": progress,
    }
    if verbose:
        print(
            f"[Shards] {d}\n"
            f"  status={status}\n"
            f"  shards={len(shards)} | contiguous={contiguous} | corrupt={len(corrupt)} | tmp={len(tmp_files)}\n"
            f"  steps={total_steps:,} | disk={res['total_mb']:.1f} MB "
            f"({(res['total_mb']/max(len(shards), 1)):.1f} MB/shard)"
        )
        if corrupt:
            print(f"  WARNING corrupt shards: {corrupt}")
    return res


def upload_v6_shards_to_hf(
    shard_dir: Path | str,
    repo_id: str,
    *,
    private: bool = True,
    token: Optional[str] = None,
    secret_name: str = "HF_TOKEN",
    use_large_folder_threshold: int = 200,
    commit_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload V6 `.npz` shards (`shard_*.npz`, `manifest.json`, `progress.json`) to a HF Dataset repo."""
    from huggingface_hub import HfApi

    d = Path(shard_dir)
    if not d.exists():
        raise FileNotFoundError(f"Shard directory does not exist: {d}")
    shards = sorted(d.glob("shard_*.npz"))
    if not shards:
        raise ValueError(f"No shard_*.npz files found in {d}")

    tok = resolve_hf_token(token=token, secret_name=secret_name)
    if not tok:
        raise RuntimeError(
            f"HF token required for upload! Bind Kaggle Secret '{secret_name}' or pass token=..."
        )

    api = HfApi(token=tok)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    total_mb = sum(p.stat().st_size for p in shards) / (1024 ** 2)
    patterns = ["shard_*.npz", "manifest.json", "progress.json"]
    msg = commit_message or f"Sync {len(shards)} Sovereign-Mahjong V6 shards ({total_mb:.1f} MB)"

    print(
        f"[HF Upload Shards] {len(shards)} shards ({total_mb:.1f} MB) -> "
        f"https://huggingface.co/datasets/{repo_id} (private={private}) ..."
    )
    if len(shards) >= use_large_folder_threshold and hasattr(api, "upload_large_folder"):
        api.upload_large_folder(
            folder_path=str(d),
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=patterns,
        )
    else:
        api.upload_folder(
            folder_path=str(d),
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=patterns,
            commit_message=msg,
        )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[HF Upload Shards DONE] {url}")
    return {"repo_id": repo_id, "url": url, "num_shards": len(shards), "total_mb": round(total_mb, 2)}


def download_v6_shards_from_hf(
    repo_id: str,
    local_dir: Path | str,
    *,
    token: Optional[str] = None,
    secret_name: str = "HF_TOKEN",
    max_shards: Optional[int] = None,
) -> Dict[str, Any]:
    """Download V6 `.npz` shards from a HF Dataset repo into `local_dir` (skips unchanged files)."""
    from huggingface_hub import snapshot_download

    dst = Path(local_dir)
    dst.mkdir(parents=True, exist_ok=True)
    tok = resolve_hf_token(token=token, secret_name=secret_name)

    if max_shards is not None:
        patterns: List[str] = [f"shard_{i:05d}.npz" for i in range(int(max_shards))] + [
            "manifest.json",
            "progress.json",
        ]
    else:
        patterns = ["shard_*.npz", "manifest.json", "progress.json"]

    print(
        f"[HF Download Shards] https://huggingface.co/datasets/{repo_id} -> {dst} "
        f"(max_shards={max_shards or 'ALL'}) ..."
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dst),
        allow_patterns=patterns,
        token=tok,
    )
    return inspect_v6_shards(dst, verbose=True)


def upload_checkpoints_to_hf(
    ckpt_dir: Path | str,
    repo_id: str,
    *,
    private: bool = True,
    token: Optional[str] = None,
    secret_name: str = "HF_TOKEN",
    commit_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload Sovereign-Mahjong V6 checkpoints & audit reports to a HF Model repo."""
    from huggingface_hub import HfApi

    d = Path(ckpt_dir)
    if not d.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {d}")
    files = [p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise ValueError(f"No checkpoint files found in {d}")

    tok = resolve_hf_token(token=token, secret_name=secret_name)
    if not tok:
        raise RuntimeError(
            f"HF token required for upload! Bind Kaggle Secret '{secret_name}' or pass token=..."
        )

    api = HfApi(token=tok)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    total_mb = sum(p.stat().st_size for p in files) / (1024 ** 2)
    msg = commit_message or f"Sync {len(files)} Sovereign-Mahjong V6 checkpoints/reports ({total_mb:.1f} MB)"

    print(
        f"[HF Upload Models] {len(files)} files ({total_mb:.1f} MB) -> "
        f"https://huggingface.co/{repo_id} (private={private}) ..."
    )
    api.upload_folder(
        folder_path=str(d),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=["*.pkl", "*.json", "*.png", "*.md"],
        commit_message=msg,
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"[HF Upload Models DONE] {url}")
    return {"repo_id": repo_id, "url": url, "num_files": len(files), "total_mb": round(total_mb, 2)}


def download_checkpoints_from_hf(
    repo_id: str,
    local_dir: Path | str,
    *,
    token: Optional[str] = None,
    secret_name: str = "HF_TOKEN",
) -> List[Path]:
    """Download Sovereign-Mahjong V6 checkpoints from a HF Model repo into `local_dir`."""
    from huggingface_hub import snapshot_download

    dst = Path(local_dir)
    dst.mkdir(parents=True, exist_ok=True)
    tok = resolve_hf_token(token=token, secret_name=secret_name)
    print(f"[HF Download Models] https://huggingface.co/{repo_id} -> {dst} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(dst),
        allow_patterns=["*.pkl", "*.json", "*.png", "*.md"],
        token=tok,
    )
    files = sorted(p for p in dst.iterdir() if p.is_file() and not p.name.startswith("."))
    print(f"[HF Download Models DONE] {len(files)} files: {[p.name for p in files]}")
    return files


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for root, _, fnames in os.walk(path):
        for fn in fnames:
            try:
                total += (Path(root) / fn).stat().st_size
            except OSError:
                pass
    return total / (1024 ** 2)


def inspect_kaggle_resources(
    work_root: Path | str = "/kaggle/working",
    tmp_root: Path | str = "/kaggle/tmp",
) -> Dict[str, Any]:
    """Print a compact GPU, VRAM, and disk quota dashboard for Kaggle / local Linux."""
    w = Path(work_root)
    t = Path(tmp_root)
    if not w.exists():
        w = Path.cwd()
    if not t.exists():
        t = w / "tmp"

    gpus: List[str] = []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        gpus = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        gpus = ["nvidia-smi unavailable"]

    w_du = shutil.disk_usage(w)
    t_du = shutil.disk_usage(t if t.exists() else w)

    dirs = {
        "v6_shards_2026": round(_dir_size_mb(w / "v6_shards_2026"), 1),
        "checkpoints": round(_dir_size_mb(w / "checkpoints"), 1),
        "mahjax_repo": round(_dir_size_mb(w / "mahjax"), 1),
        "tmp_parquets": round(_dir_size_mb(t / "tenhou_2026_mjai"), 1),
    }

    print("=== Hardware & Disk Dashboard ===")
    for g in gpus:
        print(f"  GPU: {g}")
    print(
        f"  Disk {w}: free={w_du.free / (1024**3):.2f} GB / total={w_du.total / (1024**3):.2f} GB"
    )
    print(
        f"  Disk {t}: free={t_du.free / (1024**3):.2f} GB / total={t_du.total / (1024**3):.2f} GB"
    )
    print(f"  Folder sizes (MB): {dirs}")
    return {"gpus": gpus, "work_free_gb": round(w_du.free / (1024 ** 3), 2), "dirs_mb": dirs}


def clean_tmp_and_cache(
    shard_dir: Optional[Path | str] = None,
    parquet_dir: Optional[Path | str] = None,
    *,
    remove_parquets: bool = False,
) -> Dict[str, Any]:
    """Remove stray `.tmp` shard files and optionally reclaim raw Parquet space after conversion."""
    removed: List[str] = []
    reclaimed_bytes = 0

    if shard_dir is not None:
        sd = Path(shard_dir)
        if sd.exists():
            for p in list(sd.glob(".shard_*.tmp.npz")) + list(sd.glob(".*.tmp")):
                reclaimed_bytes += p.stat().st_size
                p.unlink(missing_ok=True)
                removed.append(str(p))

    if remove_parquets and parquet_dir is not None:
        pd = Path(parquet_dir)
        if pd.exists():
            for p in sorted(pd.glob("*.parquet")):
                reclaimed_bytes += p.stat().st_size
                p.unlink(missing_ok=True)
                removed.append(str(p))

    reclaimed_mb = round(reclaimed_bytes / (1024 ** 2), 2)
    print(f"[Cleanup] Removed {len(removed)} file(s), reclaimed {reclaimed_mb:.1f} MB")
    return {"removed": removed, "reclaimed_mb": reclaimed_mb}
