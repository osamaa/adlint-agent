"""AdLint web app: upload a creative (+ optional logo), pick placements, see what the agent did.

Run locally:  uvicorn webapp.app:app --port 8080
"""

import json
import logging
import os
import re
import shutil
import time
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from adlint.agent import run_image
from adlint.placements import DEFAULT_PLACEMENTS, PLACEMENTS
from adlint.planner import make_planner
from adlint.report import write_outputs

from . import storage

log = logging.getLogger("adlint.web")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

WORK_DIR = os.environ.get("ADLINT_WORK_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs"))
MAX_UPLOAD_BYTES = int(os.environ.get("ADLINT_MAX_UPLOAD_MB", "5")) * 1024 * 1024
MAX_PIXELS = 4096 * 4096
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv")
RUN_ID = re.compile(r"^[0-9a-f]{12}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")

app = FastAPI(title="AdLint Agent")
INDEX = open(os.path.join(os.path.dirname(__file__), "index.html")).read()


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.get("/health")
def health():
    return {"ok": True, "storage": "s3" if storage.uses_s3() else "local", "llm_planner": bool(os.environ.get("OPENROUTER_API_KEY"))}


@app.get("/api/placements")
def placements():
    return {"placements": [{"id": k, "label": v["label"], "default": k in DEFAULT_PLACEMENTS} for k, v in PLACEMENTS.items()],
            "llm_planner": bool(os.environ.get("OPENROUTER_API_KEY"))}


async def _read_upload(upload):
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    if not data:
        raise HTTPException(400, "empty file")
    return data


def _decode_image(data, what):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"{what} is not a readable image")
    if image.shape[0] * image.shape[1] > MAX_PIXELS or min(image.shape[:2]) < 64:
        raise HTTPException(400, f"{what} must be between 64px and 4096x4096")
    return image


@app.post("/api/check")
async def check(creative: UploadFile = File(...), logo: UploadFile | None = File(None),
                placements: str = Form(",".join(DEFAULT_PLACEMENTS)), planner: str = Form("rules")):
    chosen = [p for p in placements.split(",") if p]
    if not chosen or any(p not in PLACEMENTS for p in chosen):
        raise HTTPException(400, "unknown placement")
    if planner not in ("rules", "llm"):
        raise HTTPException(400, "planner must be rules or llm")
    if planner == "llm" and not os.environ.get("OPENROUTER_API_KEY"):
        planner = "rules"

    run_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(WORK_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(creative.filename or "creative")
    data = await _read_upload(creative)
    logo_image = _decode_image(await _read_upload(logo), "logo") if logo and logo.filename else None
    started = time.perf_counter()
    try:
        if name.lower().endswith(VIDEO_TYPES):
            from adlint.video import check_video

            video_path = os.path.join(out_dir, "input" + os.path.splitext(name)[1].lower())
            with open(video_path, "wb") as f:
                f.write(data)
            report = check_video(video_path, chosen, logo_image, make_planner(planner), out_dir)
            os.remove(video_path)
        else:
            image = _decode_image(data, "creative")
            report = write_outputs(run_image(image, chosen, logo_image, make_planner(planner)), out_dir, name)
        with open(os.path.join(out_dir, "trace.json")) as f:
            trace = json.load(f)
        storage.publish(run_id, out_dir)
    except HTTPException:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    except Exception:
        log.exception("run %s failed", run_id)
        shutil.rmtree(out_dir, ignore_errors=True)
        raise HTTPException(500, "the check failed on this file")
    if storage.uses_s3():
        shutil.rmtree(out_dir, ignore_errors=True)  # S3 holds the run now; keep /tmp small
    log.info("run %s file=%s placements=%s planner=%s needs_human=%s seconds=%.1f",
             run_id, name, chosen, planner, report["needs_human"], time.perf_counter() - started)
    return {"run_id": run_id, "base": f"runs/{run_id}/", "report": report, "trace": trace}


@app.get("/runs/{run_id}/{file_name}")
def run_file(run_id: str, file_name: str):
    if not RUN_ID.match(run_id) or not FILE_NAME.match(file_name):
        raise HTTPException(404)
    if storage.uses_s3():
        if file_name.endswith((".html", ".json")):
            media = "text/html" if file_name.endswith(".html") else "application/json"
            try:
                return Response(storage.read(run_id, file_name), media_type=media)
            except Exception:
                raise HTTPException(404)
        return RedirectResponse(storage.presigned_url(run_id, file_name), status_code=302)
    path = os.path.join(WORK_DIR, run_id, file_name)
    if not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)
