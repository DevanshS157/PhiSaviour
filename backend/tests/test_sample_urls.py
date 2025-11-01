import csv
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_sample_urls_prediction():
    csv_path = Path(ROOT) / 'backend' / 'data' / 'sample_test_urls.csv'
    assert csv_path.exists(), 'sample_test_urls.csv missing'

    with csv_path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            url = row.get('url')
            label = row.get('label')
            r = client.post('/predict', json={'url': url})
            assert r.status_code == 200, f'POST /predict failed for {url} with {r.status_code} {r.text}'
            data = r.json()
            assert 'prediction' in data, f'No prediction for {url}'
            assert 'features' in data and isinstance(data['features'], dict), f'No features for {url}'
            # prediction should match the expected label for our demo dataset
            assert str(data['prediction']).lower() == str(label).lower(), f"Expected {label} for {url}, got {data['prediction']}"
