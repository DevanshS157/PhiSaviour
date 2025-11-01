"""Train a demo LogisticRegression model on the sample CSV and save it to phi_model.pkl.

This script is intentionally simple and intended for demo / teaching purposes only.
It reads `backend/data/sample_test_urls.csv`, extracts lexical features using
`features.extract_features`, trains a LogisticRegression, and serializes it with joblib.
"""
from pathlib import Path
import csv
import pickle
from typing import List, Tuple

from backend.model.features import extract_features

# Prefer sklearn if available; otherwise fall back to a small pure-Python trainer below.
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    LogisticRegression = None  # type: ignore
    _HAVE_SKLEARN = False


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / 'backend' / 'data' / 'processed_sample.csv'
FALLBACK_CSV = ROOT / 'backend' / 'data' / 'sample_test_urls.csv'
MODEL_PATH = ROOT / 'backend' / 'model' / 'phi_model.pkl'


def build_dataset_from_csv(csv_path: Path) -> Tuple[List[List[float]], List[str], List[str]]:
    """Read a processed CSV (features already present) and return X, y, feature_names.

    Returns X (list of lists), y (list), feature_names (list)
    """
    with csv_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        return [], [], []

    # Determine numeric feature columns (exclude url and label)
    fieldnames = reader.fieldnames or rows[0].keys()
    numeric_fields = [f for f in fieldnames if f not in ('url', 'label')]

    X = []
    y = []
    for r in rows:
        vals = []
        for f in numeric_fields:
            v = r.get(f, '')
            try:
                vals.append(float(v) if v != '' else 0.0)
            except Exception:
                vals.append(0.0)
        X.append(vals)
        y.append(r.get('label', 'malicious'))

    return X, y, numeric_fields


def build_from_raw(csv_path: Path) -> Tuple[List[List[float]], List[str], List[str]]:
    """Fallback: read raw urls and extract features on the fly."""
    with csv_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    X = []
    y = []
    feature_names = None
    for r in rows:
        url = r.get('url', '')
        label = r.get('label', 'malicious')
        feats = extract_features(url)
        if feature_names is None:
            feature_names = list(feats.keys())
        X.append([feats.get(fn, 0.0) for fn in feature_names])
        y.append(label)

    return X, y, feature_names or []


# Simple pure-Python logistic regression (SGD) for binary labels.
class SimpleLogistic:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.coef_ = [0.0] * len(feature_names)
        self.intercept_ = 0.0
        self.classes_ = ['malicious', 'benign']

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            import math

            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def predict_proba(self, Xrows: List[List[float]]):
        probs = []
        for row in Xrows:
            s = sum(w * float(v) for w, v in zip(self.coef_, row)) + self.intercept_
            p_mal = self._sigmoid(s)
            probs.append([p_mal, 1.0 - p_mal])
        return probs

    def predict(self, Xrows: List[List[float]]):
        probs = self.predict_proba(Xrows)
        return ['malicious' if p[0] >= 0.5 else 'benign' for p in probs]

    def fit(self, Xrows: List[List[float]], ylabels: List[str], lr=0.05, epochs=300):
        # Map labels to numeric: non-'benign' -> 1 (malicious), 'benign' -> 0
        ynum = [1 if str(lbl).lower() != 'benign' else 0 for lbl in ylabels]
        import random

        n = len(Xrows)
        m = len(self.coef_)
        for _ in range(epochs):
            idxs = list(range(n))
            random.shuffle(idxs)
            for i in idxs:
                xi = Xrows[i]
                yi = ynum[i]
                s = sum(w * float(v) for w, v in zip(self.coef_, xi)) + self.intercept_
                p = self._sigmoid(s)
                g = p - yi
                for j in range(m):
                    self.coef_[j] -= lr * g * float(xi[j])
                self.intercept_ -= lr * g


def train_and_save():
    if CSV_PATH.exists():
        X, y, feature_names = build_dataset_from_csv(CSV_PATH)
    else:
        X, y, feature_names = build_from_raw(FALLBACK_CSV)

    if not X:
        print('No data to train on.')
        return

    # If sklearn is available, perform a small grid search over C; otherwise
    # tune lr/epochs for the SimpleLogistic fallback. We evaluate on the
    # training set (small demo dataset) and pick the best hyperparams.
    def accuracy(model, Xrows, ylabels):
        preds = model.predict(Xrows)
        correct = sum(1 for p, t in zip(preds, ylabels) if str(p).lower() == str(t).lower())
        return correct / len(ylabels)

    hyperparams = {}
    if _HAVE_SKLEARN:
        best_score = -1.0
        best_clf = None
        best_C = None
        # small grid for demo
        for C in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=500)
            try:
                clf.fit(X, y)
            except Exception:
                continue
            score = accuracy(clf, X, y)
            if score > best_score:
                best_score = score
                best_clf = clf
                best_C = C

        model_obj = best_clf or LogisticRegression(max_iter=200)
        hyperparams = {'method': 'sklearn.LogisticRegression', 'C': best_C, 'train_score': best_score}
    else:
        best_score = -1.0
        best_model = None
        best_params = None
        for lr in (0.1, 0.05, 0.01):
            for epochs in (200, 300, 500):
                m = SimpleLogistic(feature_names or [])
                m.fit(X, y, lr=lr, epochs=epochs)
                score = accuracy(m, X, y)
                if score > best_score:
                    best_score = score
                    best_model = m
                    best_params = {'lr': lr, 'epochs': epochs}

        model_obj = best_model
        hyperparams = {'method': 'SimpleLogistic', **(best_params or {}) , 'train_score': best_score}

    # Serialize model metadata (model + feature names + hyperparams)
    model_package = {
        'model': model_obj,
        'feature_names': feature_names,
        'hyperparams': hyperparams,
    }

    try:
        import joblib

        joblib.dump(model_package, MODEL_PATH)
        print(f"Saved model package to {MODEL_PATH} using joblib")
    except Exception:
        with open(MODEL_PATH, 'wb') as fh:
            pickle.dump(model_package, fh)
        print(f"Saved model package to {MODEL_PATH} using pickle fallback")


if __name__ == '__main__':
    train_and_save()
