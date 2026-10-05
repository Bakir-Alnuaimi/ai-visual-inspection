"""REST API: upload an image, get the anomaly score back.

Start:  uvicorn api:app --reload
Try it: open http://127.0.0.1:8000/docs in the browser
"""
import io
from functools import cache

from fastapi import FastAPI, HTTPException, UploadFile

import patchcore

app = FastAPI(title="AI Visual Quality Inspection")


@cache  # load each memory bank only ONCE, then keep it in RAM (fast for the next requests)
def get_bank(category: str):
    if not patchcore.bank_file(category).exists():
        raise HTTPException(status_code=404, detail=f"No memory bank for '{category}'")
    return patchcore.load_memory_bank(category)


@app.get("/health")
def health():
    """Is the service running?"""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile, category: str = "bottle"):
    """Inspect one image. The higher the score, the more likely the part is defective."""
    image_bytes = await file.read()
    heatmap, score = patchcore.detect(io.BytesIO(image_bytes), get_bank(category))
    return {"filename": file.filename, "category": category, "anomaly_score": round(score, 2)}
