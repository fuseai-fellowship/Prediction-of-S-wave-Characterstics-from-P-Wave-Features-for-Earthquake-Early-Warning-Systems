import os, sys
# ensure repo root on path for src imports when running via uvicorn
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from obspy import read, UTCDateTime

from api.schemas import PredictResponse, HealthResponse
from api.inference import load_artifacts, predict_from_trace, fetch_trace  # fetch_trace re-export
# NOTE: fetch_trace is available in src.data.fetch_seismogram (import alias below)

# load models at startup
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
try:
    scaler, selector, xgb_model, ann_model = load_artifacts(MODELS_DIR)
    MODELS_LOADED = True
except Exception as e:
    print("Warning: could not load models:", e)
    scaler = selector = xgb_model = ann_model = None
    MODELS_LOADED = False

APP = FastAPI(title="EEW PGA Prediction API")
APP.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@APP.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "models_loaded": MODELS_LOADED}

@APP.post("/predict_upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)):
    if not MODELS_LOADED:
        raise HTTPException(status_code=500, detail="Models not loaded on server")
    contents = await file.read()
    try:
        st = read(io.BytesIO(contents))
        tr = st[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse uploaded file: {e}")
    res = predict_from_trace(tr, scaler, selector, xgb_model, ann_model)
    if res is None:
        raise HTTPException(status_code=422, detail="P-wave not detected or window too small")
    return JSONResponse(content=res)

@APP.post("/predict_station", response_model=PredictResponse)
async def predict_station(network: str = Query("IU"), station: str = Query("ANMO"), channel: str = Query("BHZ"), starttime: str = Query(None)):
    if not MODELS_LOADED:
        raise HTTPException(status_code=500, detail="Models not loaded on server")
    # fetch trace
    from src.data.fetch_seismogram import fetch_trace as _fetch_trace
    try:
        if starttime:
            stime = UTCDateTime(starttime)
            etime = stime + 2*3600
        else:
            etime = UTCDateTime()
            stime = etime - 2*3600
        tr = _fetch_trace(network=network, station=station, channel=channel, starttime=stime, endtime=etime)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch trace: {e}")
    res = predict_from_trace(tr, scaler, selector, xgb_model, ann_model)
    if res is None:
        raise HTTPException(status_code=422, detail="P-wave not detected or window too small")
    return JSONResponse(content=res)
