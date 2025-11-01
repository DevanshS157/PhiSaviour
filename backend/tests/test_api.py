import sys
from pathlib import Path

# ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert 'status' in r.json()


def test_predict_stub():
    r = client.post('/predict', json={'url': 'https://example.com/login'})
    assert r.status_code == 200
    data = r.json()
    assert 'prediction' in data
    assert 'features' in data
