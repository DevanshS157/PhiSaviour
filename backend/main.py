from pathlib import Path
from typing import Any, Dict
import os
import math

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.model.features import extract_features

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT.parent / 'frontend'
MODEL_PATH = ROOT / 'model' / 'phi_model.pkl'

app = FastAPI(title='PhiSaviour')

# serve static frontend
if FRONTEND.exists():
    # Mount static assets under /static to avoid shadowing API routes
    app.mount('/static', StaticFiles(directory=str(FRONTEND)), name='frontend')


@app.get('/')
def index():
    # Serve the frontend index if present
    index_file = FRONTEND / 'index.html'
    if index_file.exists():
        from fastapi.responses import FileResponse

        return FileResponse(str(index_file))
    return {'message': 'PhiSaviour backend running'}

_model: Any = None
MODEL_FEATURES = None
try:
    # Try joblib first (if available). If joblib is not installed or fails,
    # fall back to pickle to support the simple-pickle package we create in
    # the training script.
    loaded = None
    if MODEL_PATH.exists():
        # skip obviously-bad placeholder files
        if MODEL_PATH.stat().st_size < 32:
            print(f"Model file {MODEL_PATH} is too small; using heuristic fallback")
        else:
            try:
                import joblib

                try:
                    loaded = joblib.load(MODEL_PATH)
                    print(f"Loaded model package via joblib from {MODEL_PATH}")
                except Exception:
                    loaded = None
            except Exception:
                loaded = None

            if loaded is None:
                # try pickle as a fallback
                try:
                    import pickle

                    with open(MODEL_PATH, 'rb') as fh:
                        loaded = pickle.load(fh)
                    print(f"Loaded model package via pickle from {MODEL_PATH}")
                except Exception as e:
                    print(f"Failed to load model package via pickle from {MODEL_PATH}; error: {e}")

            # If we loaded a package dict with 'model' key, extract it
            if isinstance(loaded, dict) and 'model' in loaded:
                _model = loaded['model']
                MODEL_FEATURES = loaded.get('feature_names')
                print(f"Unpacked model package; features: {MODEL_FEATURES}")
            elif loaded is not None:
                # loaded object may be a raw sklearn estimator or similar
                _model = loaded
                print(f"Loaded model object from {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}; error: {e}")

# If a model couldn't be loaded from disk, fall back to a deterministic demo classifier
if _model is None:
    try:
        from backend.model.demo_model import DemoClassifier

        _model = DemoClassifier()
        print("Using DemoClassifier as fallback model")
    except Exception:
        # keep _model as None; API will use heuristic fallback
        _model = None

class PredictIn(BaseModel):
    url: str


@app.get('/health')
def health() -> Dict[str, Any]:
    return {'status': 'ok', 'model_loaded': bool(_model)}


@app.post('/predict')
async def predict(payload: PredictIn):
    url = payload.url
    if not url:
        raise HTTPException(status_code=400, detail='Missing url')

    feats = extract_features(url)

    # Decide threshold from environment (0-1), default 0.5
    try:
        THRESH = float(os.environ.get('PHI_THRESHOLD', '0.6'))
    except Exception:
        THRESH = 0.5

    # If model missing or None, use a simple heuristic score -> convert to probability via sigmoid
    if _model is None:
        raw_score = (feats.get('suspicious_words', 0.0) * 2.0) + (feats.get('num_hyphens', 0.0) * 0.5) + (feats.get('num_subdomains', 0.0) * 0.3)
        raw_score += (50 - min(50, feats.get('url_length', 0.0))) * -0.01  # longer urls slightly more suspicious
        prob = 1.0 / (1.0 + math.exp(-raw_score)) if raw_score is not None else 0.0
        prediction = 'malicious' if prob >= THRESH else 'benign'
        return JSONResponse({'prediction': prediction, 'score': round(prob, 3), 'threshold': THRESH, 'features': feats})

    # Build input row for the model using known feature ordering when available
    try:
        feature_order = MODEL_FEATURES or list(feats.keys())
        row = [float(feats.get(f, 0.0)) for f in feature_order]

        pred = None
        proba = None
        # Helper: provide a tiny DataFrame-like wrapper for models which expect
        # a pandas.DataFrame with iterrows(). This avoids requiring pandas.
        class _DataFrameLike:
            def __init__(self, rows, columns):
                # rows: list of dict or list
                self._rows = []
                for r in rows:
                    if isinstance(r, (list, tuple)):
                        self._rows.append({c: v for c, v in zip(columns, r)})
                    else:
                        self._rows.append(dict(r))

            def iterrows(self):
                for i, r in enumerate(self._rows):
                    yield i, r

        if hasattr(_model, 'predict'):
            try:
                pred_arr = _model.predict([row])
            except Exception:
                # try DataFrame-like wrapper
                try:
                    pred_arr = _model.predict(_DataFrameLike([row], feature_order))
                except Exception:
                    pred_arr = _model.predict(_DataFrameLike([feats], list(feats.keys())))
            pred = pred_arr[0] if hasattr(pred_arr, '__len__') else pred_arr

        if hasattr(_model, 'predict_proba'):
            try:
                proba_arr = _model.predict_proba([row])
            except Exception:
                try:
                    proba_arr = _model.predict_proba(_DataFrameLike([row], feature_order))
                except Exception:
                    proba_arr = _model.predict_proba(_DataFrameLike([feats], list(feats.keys())))
            first = proba_arr[0]
            try:
                proba = list(first)
            except Exception:
                proba = [float(first)]
            # try to infer probability for 'malicious' — assume first entry corresponds to malicious
            prob_malicious = float(proba[0])
        else:
            prob_malicious = 1.0 if str(pred).lower() in ('malicious', 'phish', '1', 'true') else 0.0

        return JSONResponse({'prediction': pred, 'score': round(prob_malicious, 3), 'threshold': THRESH, 'proba': proba, 'features': feats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
