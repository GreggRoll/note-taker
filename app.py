from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from transcription_engine import TranscriptionEngine


MODELS = ["tiny", "base", "small", "medium"]

app = FastAPI(title="Desktop Audio Whisper Transcriber")
engine = TranscriptionEngine()

base_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")
templates = Jinja2Templates(directory=str(base_dir / "templates"))


class StartPayload(BaseModel):
    model: str = "base"


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "models": MODELS,
            "default_model": "base",
        },
    )


@app.get("/api/state")
def state():
    return engine.snapshot()


@app.post("/api/start")
def start(payload: StartPayload):
    if payload.model not in MODELS:
        raise HTTPException(status_code=400, detail="Unsupported Whisper model")
    if not (engine.snapshot().get("numpy_ok")):
        raise HTTPException(status_code=400, detail='NumPy 2.x detected. Install "numpy<2".')
    engine.start(payload.model)
    return {"ok": True}


@app.post("/api/stop")
def stop():
    engine.stop()
    return {"ok": True}


@app.post("/api/split")
def split():
    engine.split()
    return {"ok": True}


@app.post("/api/clear")
def clear():
    if engine.snapshot()["running"]:
        raise HTTPException(status_code=400, detail="Cannot clear while recording")
    engine.clear()
    return {"ok": True}
