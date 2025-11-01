# PhiSaviour — Phishing URL Detector (MVP)

This repository is an MVP scaffold for a phishing URL detector with a vanilla frontend and a FastAPI backend.

Quick start (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

Open http://localhost:5500/ and paste a URL to test.

Project layout:
- `frontend/` — static site (index.html, style.css, app.js)
- `backend/` — FastAPI app, feature extractor, placeholder model, tests
- `devops/` — Dockerfile and docker-compose for local dev

Notes:
- The backend uses a simple heuristic when `backend/model/phi_model.pkl` is missing.
- To run tests: install requirements and run `pytest backend`.
- Replace the model placeholder with a real scikit-learn pipeline saved using `joblib.dump` to enable ML predictions.
