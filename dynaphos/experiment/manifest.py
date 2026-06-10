from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return output


def start_manifest(
    path: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
        }
    )
    manifest.setdefault("created_at_utc", utc_now())
    write_manifest(path, manifest)
    return manifest


def complete_manifest(
    path: str | Path,
    manifest: dict[str, Any],
    *,
    report_status: str,
) -> dict[str, Any]:
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "report_status": report_status,
        }
    )
    write_manifest(path, manifest)
    return manifest


def fail_manifest(
    path: str | Path,
    manifest: dict[str, Any],
    *,
    error: BaseException,
    traceback_text: str,
) -> dict[str, Any]:
    manifest.update(
        {
            "status": "failed",
            "completed_at_utc": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback_text,
        }
    )
    write_manifest(path, manifest)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / MANIFEST_NAME
    elif source.name != MANIFEST_NAME:
        raise ValueError(
            f"Expected canonical {MANIFEST_NAME}, got: {source.name}"
        )
    if not source.exists():
        raise FileNotFoundError(f"Missing canonical manifest: {source}")
    with open(source, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must contain a YAML mapping: {source}")
    version = int(manifest.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest schema_version {version} in {source}; "
            f"expected {SCHEMA_VERSION}."
        )
    return manifest
