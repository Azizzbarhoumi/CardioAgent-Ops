"""/samples is read-only and serves only the files listed in samples.json."""
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers.samples import router


def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SAMPLES_DIR", str(tmp_path))
    app = FastAPI(); app.include_router(router)
    return TestClient(app)


def test_lists_and_serves_existing_files(tmp_path, monkeypatch):
    (tmp_path / "n.wav").write_bytes(b"RIFF....WAVE")
    (tmp_path / "samples.json").write_text(json.dumps({"samples": [
        {"id": "normal", "label": "normal", "file": "n.wav"}, {"id": "abnormal", "label": "abnormal", "file": "missing.wav"}]}))
    c = client(tmp_path, monkeypatch)
    body = c.get("/samples").json()
    assert [s["id"] for s in body["samples"]] == ["normal"]  # missing file is not listed
    r = c.get("/samples/normal")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content == b"RIFF....WAVE"
    assert c.get("/samples/abnormal").status_code == 404


def test_empty_folder_gives_warning(tmp_path, monkeypatch):
    body = client(tmp_path, monkeypatch).get("/samples").json()
    assert body["samples"] == [] and "No sample files" in body["warning"]


def test_no_path_traversal(tmp_path, monkeypatch):
    (tmp_path.parent / "secret.wav").write_bytes(b"x")
    (tmp_path / "samples.json").write_text(json.dumps({"samples": [{"id": "x", "label": "x", "file": "../secret.wav"}]}))
    assert client(tmp_path, monkeypatch).get("/samples/x").status_code == 404
