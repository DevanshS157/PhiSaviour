from pathlib import Path
import pickle
import json
from backend.model.features import extract_features

MODEL_PATH = Path(__file__).resolve().parents[1] / 'model' / 'phi_model.pkl'

sample_url = 'http://example-login.test/bad-path?user=1'

if not MODEL_PATH.exists():
    print('Model file not found:', MODEL_PATH)
    raise SystemExit(1)

class RenamingUnpickler(pickle.Unpickler):
    """Map classes pickled as __main__.SimpleLogistic back to the
    real module where the class lives (backend.model.train.SimpleLogistic).
    This handles the case where train.py was executed as a script when
    the model was pickled (its classes were stored under __main__).
    """
    def find_class(self, module, name):
        if module == '__main__' and name == 'SimpleLogistic':
            import backend.model.train as tr

            return getattr(tr, 'SimpleLogistic')
        return super().find_class(module, name)


with open(MODEL_PATH, 'rb') as fh:
    try:
        pkg = RenamingUnpickler(fh).load()
    except Exception:
        # fallback to plain pickle
        fh.seek(0)
        pkg = pickle.load(fh)

# pkg may be a dict with 'model' and metadata, or a raw estimator
if isinstance(pkg, dict) and 'model' in pkg:
    model = pkg['model']
    feature_names = pkg.get('feature_names')
    hyperparams = pkg.get('hyperparams')
else:
    model = pkg
    feature_names = None
    hyperparams = None

feats = extract_features(sample_url)

# build input row according to feature_names if present
if feature_names:
    row = [float(feats.get(fn, 0.0)) for fn in feature_names]
else:
    # fallback: use all features in sorted order
    keys = sorted(feats.keys())
    row = [float(feats[k]) for k in keys]

# model may expect a 2D array-like
try:
    preds = model.predict([row])
except Exception as e:
    preds = None

try:
    proba = model.predict_proba([row])
except Exception:
    proba = None

out = {
    'url': sample_url,
    'features': feats,
    'feature_names_used': feature_names or keys,
    'prediction': preds[0] if preds else None,
    'proba': proba[0] if proba else None,
    'hyperparams': hyperparams,
}

print(json.dumps(out, indent=2))
