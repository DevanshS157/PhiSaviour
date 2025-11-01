"""Prepare and normalize raw URL datasets for training.

Usage examples:
  # normalize and build features from sample_test_urls.csv
  python backend\model\prepare_data.py --input ../data/sample_test_urls.csv --out ../data/processed.csv

This script:
- reads one or more CSVs containing a column named 'url' and optional 'label'
- normalizes URLs (adds scheme if missing, lowercases host)
- deduplicates by hostname (optionally by full url)
- extracts lexical features using model.features.extract_features
- writes a processed CSV ready for training (features + label)

Note: This is an offline utility; it does not fetch remote content or call external services.
"""
from pathlib import Path
import argparse
import csv
import sys
from typing import List

from backend.model.features import extract_features


def normalize_url(u: str) -> str:
    if not u:
        return ''
    u = u.strip()
    # ensure scheme
    if '://' not in u:
        u = 'http://' + u
    return u


def build_dataframe(input_paths: List[Path], dedupe_on: str = 'host') -> List[dict]:
    """Return a list of dict rows with url, label and features. No pandas dependency.

    Deduplication is done on host or full url.
    """
    rows = []
    for p in input_paths:
        with p.open(newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                url = normalize_url(r.get('url', ''))
                label = r.get('label') if 'label' in r else None
                rows.append({'url': url, 'label': label})

    # dedupe
    seen = set()
    out_rows = []
    import urllib.parse

    for r in rows:
        u = r['url']
        key = None
        if dedupe_on == 'host':
            try:
                parsed = urllib.parse.urlparse(u)
                host = (parsed.hostname or '').lower()
            except Exception:
                host = ''
            key = host
        elif dedupe_on == 'url':
            key = u
        else:
            key = None

        if key is None or key == '':
            # keep rows without host when not deduping by host
            out_rows.append(r)
            continue

        if key in seen:
            continue
        seen.add(key)
        out_rows.append(r)

    # extract features and build dict rows
    result = []
    for r in out_rows:
        feats = extract_features(r['url'])
        row = {'url': r['url'], 'label': r.get('label')}
        row.update(feats)
        result.append(row)

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare URL datasets for training')
    parser.add_argument('--input', '-i', required=True, nargs='+', help='Input CSV file(s) with at least a url column')
    parser.add_argument('--out', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--dedupe', choices=['host', 'url', 'none'], default='host', help='Deduplicate by host or full url')

    args = parser.parse_args(argv)
    input_paths = [Path(p) for p in args.input]
    for p in input_paths:
        if not p.exists():
            print(f'Input file not found: {p}', file=sys.stderr)
            sys.exit(2)

    out_path = Path(args.out)
    rows = build_dataframe(input_paths, dedupe_on=args.dedupe)
    if not rows:
        print('No rows to write')
        return
    # determine fieldnames from first row
    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'Wrote processed dataset to {out_path}')


if __name__ == '__main__':
    main()
