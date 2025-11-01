#!/usr/bin/env python3
"""Minimal, dependency-free HTTP server exposing a /predict endpoint.

This is a small shim for local development when FastAPI isn't available.
It loads the pickled model package (phi_model.pkl) if present and uses
`backend.model.features.extract_features` to compute features.

Usage:
    python backend/simple_server.py

Then POST JSON {"url":"..."} to http://127.0.0.1:8000/predict

Note: This server is for local dev only and intentionally simple.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from pathlib import Path
import pickle
import traceback
from urllib.parse import urlparse

# Use the `backend/` directory as ROOT so relative paths point to backend/*
# (previously this used parents[1] which pointed at the repo root and made
# MODEL_PATH resolve to repo_root/model/... instead of backend/model/...)
ROOT = Path(__file__).resolve().parents[0]
MODEL_PATH = ROOT / 'model' / 'phi_model.pkl'

from backend.model.features import extract_features

# helper to unpickle objects whose classes were defined in __main__ when pickled
class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'SimpleLogistic':
            import backend.model.train as tr
            return getattr(tr, 'SimpleLogistic')
        return super().find_class(module, name)


def load_model_package():
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, 'rb') as fh:
            try:
                pkg = RenamingUnpickler(fh).load()
            except Exception:
                fh.seek(0)
                pkg = pickle.load(fh)
        if isinstance(pkg, dict) and 'model' in pkg:
            return pkg
        else:
            return {'model': pkg, 'feature_names': None, 'hyperparams': None}
    except Exception:
        traceback.print_exc()
        return None

MODEL_PACKAGE = load_model_package()
MODEL = MODEL_PACKAGE['model'] if MODEL_PACKAGE else None
MODEL_FEATURES = MODEL_PACKAGE.get('feature_names') if MODEL_PACKAGE else None


def reload_model():
    """Reload the model package from disk and update module-level globals.

    Returns a dict with status information for use by the /reload endpoint.
    """
    global MODEL_PACKAGE, MODEL, MODEL_FEATURES
    pkg = load_model_package()
    MODEL_PACKAGE = pkg
    MODEL = pkg['model'] if pkg else None
    MODEL_FEATURES = pkg.get('feature_names') if pkg else None
    return {
        'model_loaded': MODEL is not None,
        'model_exists': MODEL_PATH.exists(),
        'model_path': str(MODEL_PATH)
    }

# tiny DataFrame-like wrapper for models that expect iterrows()
class _DataFrameLike:
    def __init__(self, rows, columns):
        self._rows = []
        for r in rows:
            if isinstance(r, (list, tuple)):
                self._rows.append({c: v for c, v in zip(columns, r)})
            else:
                self._rows.append(dict(r))
    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, r


def predict_with_model(feats):
    # Build ordered row
    feature_order = MODEL_FEATURES or list(feats.keys())
    row = [float(feats.get(f, 0.0)) for f in feature_order]

    pred = None
    proba = None
    prob_malicious = 0.0

    if MODEL is None:
        # fallback heuristic (same as server demo)
        raw_score = (feats.get('suspicious_words', 0.0) * 2.0) + (feats.get('num_hyphens', 0.0) * 0.5) + (feats.get('num_subdomains', 0.0) * 0.3)
        raw_score += (50 - min(50, feats.get('url_length', 0.0))) * -0.01
        import math
        prob = 1.0 / (1.0 + math.exp(-raw_score))
        pred = 'malicious' if prob >= 0.6 else 'benign'
        proba = [prob, 1-prob]
        prob_malicious = prob
        return pred, prob_malicious, proba

    try:
        if hasattr(MODEL, 'predict'):
            try:
                pred_arr = MODEL.predict([row])
            except Exception:
                pred_arr = MODEL.predict(_DataFrameLike([row], feature_order))
            pred = pred_arr[0] if hasattr(pred_arr, '__len__') else pred_arr
        if hasattr(MODEL, 'predict_proba'):
            try:
                proba_arr = MODEL.predict_proba([row])
            except Exception:
                proba_arr = MODEL.predict_proba(_DataFrameLike([row], feature_order))
            first = proba_arr[0]
            try:
                proba = list(first)
            except Exception:
                proba = [float(first)]
            prob_malicious = float(proba[0])
        else:
            prob_malicious = 1.0 if str(pred).lower() in ('malicious','phish','1','true') else 0.0
    except Exception:
        traceback.print_exc()
        # fallback heuristic
        raw_score = (feats.get('suspicious_words', 0.0) * 2.0) + (feats.get('num_hyphens', 0.0) * 0.5) + (feats.get('num_subdomains', 0.0) * 0.3)
        raw_score += (50 - min(50, feats.get('url_length', 0.0))) * -0.01
        import math
        prob = 1.0 / (1.0 + math.exp(-raw_score))
        pred = 'malicious' if prob >= 0.6 else 'benign'
        proba = [prob, 1-prob]
        prob_malicious = prob

    return pred, prob_malicious, proba


class Handler(BaseHTTPRequestHandler):
    def _set_json(self, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/health':
            ok = {'status': 'ok', 'model_loaded': MODEL is not None}
            self._set_json(200)
            self.wfile.write(json.dumps(ok).encode())
            return
        if self.path == '/reload':
            info = reload_model()
            resp = {'status': 'ok', **info}
            self._set_json(200)
            self.wfile.write(json.dumps(resp).encode())
            return
        # static serve index.html for convenience
        if self.path == '/' or self.path == '/index.html':
            idx = ROOT.parent / 'frontend' / 'index.html'
            if idx.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(idx.read_bytes())
                return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != '/predict':
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length).decode('utf-8')
        try:
            payload = json.loads(raw)
            url = payload.get('url', '')
            feats = extract_features(url)
            pred, prob_malicious, proba = predict_with_model(feats)
            out = {'prediction': pred, 'score': round(prob_malicious,3), 'threshold': 0.6, 'proba': proba, 'features': feats}
            self._set_json(200)
            self.wfile.write(json.dumps(out).encode('utf-8'))
        except Exception as e:
            self._set_json(500)
            self.wfile.write(json.dumps({'detail': str(e), 'trace': traceback.format_exc()}).encode('utf-8'))


def run(addr='127.0.0.1', port=8000):
    server = HTTPServer((addr, port), Handler)
    print(f"Simple dev server running at http://{addr}:{port}/ (model_loaded={MODEL is not None})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Shutting down')


if __name__ == '__main__':
    run()
