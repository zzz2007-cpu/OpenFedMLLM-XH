from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .constants import MANIFEST_FILENAME


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def copy_json(src: Path, dst: Path) -> None:
    write_json(dst, read_json(src))


def norm_posix(path_str: str) -> str:
    return str(path_str).replace("\\", "/")


def join_posix(prefix: str, suffix: str) -> str:
    prefix = norm_posix(prefix).rstrip("/")
    suffix = norm_posix(suffix).lstrip("/")
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}/{suffix}"


def client_sort_key(path: Path):
    m = re.search(r"client_(\d+)", path.stem)
    if m:
        return int(m.group(1))
    return path.stem


def list_client_files(dataset_dir: Path) -> List[Path]:
    return sorted(dataset_dir.glob("client_*.json"), key=client_sort_key)


def load_manifest(dataset_dir: Path) -> Dict[str, Any]:
    manifest_path = dataset_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    return payload


def maybe_relpath(path: Path, start: Path) -> str:
    try:
        return norm_posix(os.path.relpath(str(path), str(start)))
    except Exception:
        return norm_posix(str(path))


def copy_optional_files(src_dir: Path, dst_dir: Path, file_names: Iterable[str]) -> List[str]:
    copied: List[str] = []
    for name in file_names:
        src = src_dir / name
        if src.exists():
            copy_json(src, dst_dir / name)
            copied.append(name)
    return copied

