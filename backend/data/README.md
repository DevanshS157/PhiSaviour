Data collection notes

Recommended sources (manual download or API where permitted):
- PhishTank (https://www.phishtank.com/) - requires API key for some access
- OpenPhish (https://openphish.com/) - commercial/limited API access; check terms
- Alexa top sites or Tranco list (for benign examples)
- Kaggle public datasets (search for 'phishing URLs')

Suggested workflow
1. Download datasets into `backend/data/raw/` (create the directory).
2. Normalize and deduplicate using `backend/model/prepare_data.py`.
   Example:
     python backend\model\prepare_data.py -i backend\data\raw\phishtank.csv backend\data\raw\benign.csv -o backend\data\processed\train.csv

3. Use `backend/model/train.py` to train a model from the processed CSV.

Privacy and rate limits
- Don't fetch or store raw HTML pages from untrusted domains without sandboxing and explicit consent.
- When using external APIs, secure keys with environment variables and observe rate limits.

CSV format expectations
- Input CSVs should have a `url` column. A `label` column (malicious/benign) is optional; if missing you can add labels manually or use heuristics.

