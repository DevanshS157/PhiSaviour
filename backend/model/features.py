import re
from urllib.parse import urlparse
from typing import Dict


try:
    import tldextract
except Exception:
    tldextract = None

IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def _is_ip(host: str) -> bool:
    return bool(IP_RE.match(host))


def extract_features(url: str) -> Dict[str, float]:
    """Extract simple lexical features from a URL.

    Returns a dict of numeric features suitable for a heuristic or ML model.
    """
    if not url:
        return {
            "url_length": 0.0,
            "host_length": 0.0,
            "path_length": 0.0,
            "num_dots": 0.0,
            "num_hyphens": 0.0,
            "has_ip": 0.0,
            "suspicious_words": 0.0,
            "num_subdomains": 0.0,
        }

    parsed = urlparse(url if '://' in url else 'http://' + url)
    host = parsed.hostname or ''
    path = parsed.path or ''
    query = parsed.query or ''

    subdomain = ''
    # Try to extract subdomain using tldextract when available, otherwise fallback
    if tldextract is not None:
        ext = tldextract.extract(host)
        subdomain = ext.subdomain or ''
    else:
        # naive fallback: everything before the last two labels is treated as subdomain
        parts = host.split('.') if host else []
        if len(parts) > 2:
            subdomain = '.'.join(parts[:-2])
        else:
            subdomain = ''

    suspicious_tokens = ['login', 'signin', 'secure', 'bank', 'update', 'confirm', 'verify']
    s_count = sum(1 for t in suspicious_tokens if t in url.lower())
    has_percent = 1.0 if '%' in url else 0.0

    features = {
        "url_length": float(len(url)),
        "host_length": float(len(host)),
        "path_length": float(len(path) + len(query)),
        "num_dots": float(host.count('.')),
        "num_hyphens": float(host.count('-')),
        "has_ip": 1.0 if _is_ip(host) else 0.0,
        "suspicious_words": float(s_count),
        "has_percent_encoded": float(has_percent),
        "num_subdomains": float(len(subdomain.split('.')) if subdomain else 0.0),
    }

    return features
