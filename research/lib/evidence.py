"""Evidence artifact writer with checksums and an explicit provenance manifest.

Every experiment writes into an immutable directory keyed by the experiment
identifier and the exact source revision that produced it:

    research/evidence/<experiment_id>/<source_sha>/
        raw/        one JSONL/CSV record per observation, never summarised away
        summary/    processed statistics derived from raw/
        tables/     flat CSV tables for manuscript use
        manifest.json  provenance, configuration, checksums, limitations

Overwriting an existing revision directory is refused unless explicitly
requested, so a previously published result cannot be silently replaced.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import SCHEMA_NAMESPACE
from .provenance import RunProvenance, sha256_file, sha256_json, utc_now_iso

EVIDENCE_ROOT = "research/evidence"


class EvidenceError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def write_jsonl(path: Path, records: Iterable[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(_jsonable(record), sort_keys=True, default=str) + "\n"
            )
    return path


def write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(_jsonable(value))
    if value is None:
        return ""
    return value


@dataclass
class EvidenceWriter:
    """Writes one immutable evidence bundle and its manifest."""

    experiment_id: str
    repo_root: Path
    provenance: RunProvenance
    summary_payload: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    interpretation: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    allow_overwrite: bool = False
    extra_raw: Dict[str, Sequence[Any]] = field(default_factory=dict)
    tables: Dict[str, Sequence[Dict[str, Any]]] = field(default_factory=dict)
    files: List[Path] = field(default_factory=list)

    # -- layout ------------------------------------------------------------
    @property
    def source_sha(self) -> str:
        sha = (self.provenance.to_dict().get("git") or {}).get("commit_sha")
        return str(sha or "UNKNOWN_SOURCE_REVISION")

    @property
    def revision_dir(self) -> Path:
        return Path(self.repo_root) / EVIDENCE_ROOT / self.experiment_id / self.source_sha

    def _record(self, relative: str) -> Path:
        path = self.revision_dir / relative
        self.files.append(path)
        return path

    def raw_jsonl(self, name: str, records: Iterable[Any]) -> Path:
        return self._record(str(Path("raw") / f"{name}.jsonl"))

    def raw_json(self, name: str, payload: Any) -> Path:
        return self._record(str(Path("raw") / f"{name}.json"))

    def summary_json(self, name: str, payload: Any) -> Path:
        return self._record(str(Path("summary") / f"{name}.json"))

    def table_csv(self, name: str, rows: Sequence[Dict[str, Any]], fieldnames=None) -> Path:
        return self._record(str(Path("tables") / f"{name}.csv"))

    def text(self, name: str, content: str) -> Path:
        path = self._record(str(Path("summary") / name))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    # -- finalisation ------------------------------------------------------
    def write(self) -> Dict[str, Any]:
        revision = self.revision_dir
        if revision.exists() and any(revision.rglob("*")) and not self.allow_overwrite:
            raise EvidenceError(
                f"evidence directory already exists: {revision}. "
                "Evidence revisions are immutable; use a new source SHA or pass "
                "allow_overwrite=True deliberately."
            )
        revision.mkdir(parents=True, exist_ok=True)

        for name, records in self.extra_raw.items():
            write_jsonl(self.raw_jsonl(name, records), records)
        for name, rows in self.tables.items():
            write_csv(self.table_csv(name, rows), rows)

        summary = dict(self.summary_payload)
        write_json(self._record(str(Path("summary") / "summary.json")), summary)

        checksums = {
            path.relative_to(revision).as_posix(): sha256_file(path)
            for path in sorted(self.files)
            if path.exists()
        }
        manifest: Dict[str, Any] = {
            "schema": f"{SCHEMA_NAMESPACE}.manifest",
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "source_sha": self.source_sha,
            "created_at_utc": utc_now_iso(),
            "provenance": self.provenance.to_dict(),
            "seed": self.seed,
            "configuration": _jsonable(self.configuration),
            "result_summary_sha256": sha256_json(summary),
            "interpretation": _jsonable(self.interpretation),
            "limitations": list(self.limitations),
            "files": checksums,
            "immutability_note": (
                "This directory is keyed by the source SHA that produced it. "
                "Re-running the same experiment on a different revision writes a "
                "sibling directory instead of overwriting this one."
            ),
        }
        manifest_path = self._record("manifest.json")
        write_json(manifest_path, manifest)
        return manifest