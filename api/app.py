"""Reference HTTP API (FastAPI). The backend team can use this directly or
call `retinaai.predict.Predictor` from their own service.

Run (mock mode, no model needed):
    RETINAAI_MOCK=1 uvicorn api.app:app --reload --port 8000
Run with real weights:
    RETINAAI_CHECKPOINT=weights/model_inference.pt uvicorn api.app:app --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from retinaai import DISCLAIMER, __version__

MAX_BYTES = 20 * 1024 * 1024
ALLOWED = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp", "application/octet-stream"}

app = FastAPI(title="RetinaAI Triage API", version=__version__, description=DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        ckpt = os.environ.get("RETINAAI_CHECKPOINT")
        if os.environ.get("RETINAAI_MOCK") == "1" or not ckpt:
            from retinaai.mock import MockPredictor
            _predictor = MockPredictor()
        else:
            from retinaai.predict import Predictor
            _predictor = Predictor(ckpt, mc_samples=int(os.environ.get("RETINAAI_MC_SAMPLES", 8)))
    return _predictor


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "version": __version__,
            "mock": type(p).__name__ == "MockPredictor", "model": p.model_info}


@app.post("/predict")
async def predict(file: UploadFile = File(...), heatmap: bool = Query(True)):
    if file.content_type and file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        return get_predictor().predict(data, return_heatmap=heatmap)
    except ValueError as e:
        raise HTTPException(400, str(e))
