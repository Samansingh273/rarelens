"""RareLens local web server (FastAPI). Start it with  python app.py  from the project folder."""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import DATA, Engine

HERE = Path(__file__).resolve().parent
app = FastAPI(title="RareLens", docs_url="/api/docs", redoc_url=None)
engine: Engine | None = None


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = Engine()
    return engine


class GenRequest(BaseModel):
    rare: str = "DF"
    seed: int = Field(1042, ge=0, le=2**31 - 1)
    n: int = Field(12, ge=1, le=24)


class MorphRequest(GenRequest):
    a: int = Field(0, ge=0, le=23)
    b: int = Field(5, ge=0, le=23)
    t: float = Field(0.5, ge=0, le=1)


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/meta")
def meta():
    e = get_engine()
    m = e.meta
    return {"device": e.device_name, "classes": m["classes"], "class_names": m["class_names"],
            "examples": m["examples"], "rares": m["rares"], "main_model": m["main_model"],
            "models": {k: {x: v[x] for x in ("rare", "config", "label", "seed")} for k, v in m["models"].items()},
            "gan": sorted(e.gans), "copy_check": bool(e.real_feats), "app_model": e.ens_keys, "app_model_name": e.model_name}


@app.get("/api/evidence")
def evidence():
    return JSONResponse(get_engine().evidence)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 25 MB).")
    try:
        img = Engine.load_image(data)
    except Exception:
        raise HTTPException(400, "That file could not be read as an image.")
    return get_engine().predict(img)


@app.get("/api/predict/example/{image_id}")
def predict_example(image_id: str):
    e = get_engine()
    ex = {x["id"]: x for x in e.meta["examples"]}
    if image_id not in ex:
        raise HTTPException(404, "Unknown example.")
    img = Engine.load_image((HERE / ex[image_id]["src"]).read_bytes())
    out = e.predict(img)
    out["truth"] = ex[image_id]["label"]
    return out


def _check_rare(rare: str) -> None:
    if rare not in get_engine().gans:
        raise HTTPException(404, f"No generator for {rare}.")


@app.post("/api/generate")
def generate(req: GenRequest):
    _check_rare(req.rare)
    return get_engine().generate(req.rare, req.seed, req.n)


@app.post("/api/morph")
def morph(req: MorphRequest):
    _check_rare(req.rare)
    if max(req.a, req.b) >= req.n:
        raise HTTPException(400, "a and b must be image positions in the current set.")
    return {"image": get_engine().morph(req.rare, req.seed, req.a, req.b, req.t, req.n)}


@app.get("/api/grid.png")
def grid(rare: str = "DF", seed: int = 1042, n: int = 12):
    _check_rare(rare)
    png = get_engine().grid_png(rare, seed, max(1, min(n, 24)))
    return Response(png, media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="rarelens_synthetic_{rare}_seed{seed}.png"'})


DATA.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/data", StaticFiles(directory=DATA), name="data")


def main(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    import uvicorn

    print("Loading models ...")
    e = get_engine()
    print(f"RareLens ready on {e.device_name}: {len(e.models)} classifiers, GANs for {', '.join(sorted(e.gans)) or 'none'}")
    url = f"http://{host}:{port}"
    print(f"\n  Open  {url}   (press Ctrl+C here to stop)\n")
    if open_browser and not os.environ.get("RARELENS_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
