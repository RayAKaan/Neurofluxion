"""Provenance and environment capture for research experiments.

Every quantitative result produced by this repository must be traceable to an
exact source revision and an exact execution environment. This module provides
the single implementation of that capture so no experiment invents its own
metadata format.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Packages whose versions materially affect numerical or timing results.
TRACKED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "tensorflow",
    "keras",
    "torch",
    "jax",
    "fastapi",
    "pydantic",
    "uvicorn",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip() or None


def git_state(root: Path) -> Dict[str, object]:
    """Return commit, branch, dirty flag and shallow state for ``root``.

    ``dirty`` is computed from ``git status --porcelain`` and includes
    untracked files, because untracked research scripts can change results.
    """
    root = Path(root)
    status = _git(root, "status", "--porcelain")
    return {
        "repository_present": (root / ".git").exists(),
        "commit_sha": _git(root, "rev-parse", "HEAD"),
        "commit_short": (_git(root, "rev-parse", "--short", "HEAD") or None),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit_timestamp": _git(root, "log", "-1", "--format=%cI"),
        "is_dirty": bool(status) if status is not None else None,
        "dirty_path_count": len(status.splitlines()) if status is not None else None,
        "is_shallow": _git(root, "rev-parse", "--is-shallow-repository"),
        "origin_url": _git(root, "config", "--get", "remote.origin.url"),
    }


def _installed_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata

    versions: Dict[str, Optional[str]] = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _node_version() -> Optional[str]:
    node = shutil.which("node")
    if node is None:
        return None
    try:
        return subprocess.check_output([node, "--version"], text=True, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "processor": platform.processor() or None,
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": None,
    }
    try:  # Windows
        import psutil  # type: ignore

        info["physical_cpus"] = psutil.cpu_count(logical=False)
    except Exception:
        pass
    return info


def _memory_info() -> Dict[str, object]:
    total = None
    available = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total = int(vm.total)
        available = int(vm.available)
    except Exception:
        try:  # Windows fallback without psutil
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
        except Exception:
            total = None
    return {"total_bytes": total, "available_bytes_at_capture": available}


@dataclass
class RunProvenance:
    """Immutable provenance record attached to every research result."""

    experiment_id: str
    script: str
    command: List[str]
    seed: Optional[int]
    root: Path
    started_at_utc: str = field(default_factory=utc_now_iso)
    finished_at_utc: Optional[str] = None
    exit_code: Optional[int] = None
    notes: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        root = Path(self.root)
        payload: Dict[str, object] = {
            "experiment_id": self.experiment_id,
            "script": self.script,
            "command": list(self.command),
            "command_line": " ".join(self.command),
            "seed": self.seed,
            "repository_root": str(root),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "exit_code": self.exit_code,
            "git": git_state(root),
            "environment": environment_report(),
            "notes": dict(self.notes),
        }
        return payload


def environment_report() -> Dict[str, object]:
    """Capture OS, runtime, CPU, memory and installed dependency versions."""
    return {
        "captured_at_utc": utc_now_iso(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "full_version": sys.version,
        },
        "node": _node_version(),
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "packages": _installed_versions(),
        "relevant_env_vars": {
            name: os.environ.get(name)
            for name in (
                "PYTHONPATH",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TF_ENABLE_ONEDNN_OPTS",
                "TF_CPP_MIN_LOG_LEVEL",
                "NEUROFLUXION_BENCH_ENV",
            )
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def source_tree_digest(root: Path, patterns: Iterable[str]) -> Dict[str, str]:
    """SHA-256 of every tracked source file matching ``patterns``.

    This lets a result be re-verified even if the git metadata is later lost.
    """
    root = Path(root)
    out: Dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out