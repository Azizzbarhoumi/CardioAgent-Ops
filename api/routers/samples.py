"""Read-only demo samples for the "Try a sample" buttons.

Files live in a local, git-ignored folder (SAMPLES_DIR, default ./samples) described by samples.json.
Nothing here touches inference: the frontend downloads a sample and uploads it to /analyze_pcg like any file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


def samples_dir() -> Path:
    default = Path(__file__).resolve().parents[2] / "samples"
    return Path(os.getenv("SAMPLES_DIR", str(default)))


def _catalog() -> list[dict]:
    path = samples_dir() / "samples.json"
    if not path.is_file():
        return []
    items = json.loads(path.read_text(encoding="utf-8")).get("samples", [])
    return [item for item in items if (samples_dir() / item["file"]).is_file()]


@router.get("/samples")
def list_samples() -> dict:
    items = _catalog()
    return {"samples": [{k: v for k, v in item.items()} for item in items],
            "warning": None if items else f"No sample files found in {samples_dir()} (see samples/README.md)."}


@router.get("/samples/{sample_id}")
def get_sample(sample_id: str) -> FileResponse:
    for item in _catalog():
        if item["id"] == sample_id:
            path = (samples_dir() / item["file"]).resolve()
            if samples_dir().resolve() not in path.parents:  # no path traversal through samples.json
                break
            return FileResponse(path, media_type="audio/wav", filename=item["file"])
    raise HTTPException(status_code=404, detail=f"Unknown sample {sample_id!r}")
