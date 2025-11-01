import sys
from pathlib import Path

# ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.model.features import extract_features


def test_extract_empty():
    f = extract_features('')
    assert isinstance(f, dict)
    assert f['url_length'] == 0.0


def test_extract_basic():
    f = extract_features('https://example.com/login')
    assert f['host_length'] > 0
    assert f['suspicious_words'] >= 1
