# CrewSignal

CrewSignal is an offline-first, synthetic-data prototype for monitoring astronaut health trends and providing explainable decision-support. The application uses a lightweight Random Forest classifier as its primary predictive engine to identify potentially concerning health trends from generated sensor and symptom data. CrewSignal is a demonstration and not a diagnostic or clinical tool.

## Features

- Synthetic astronaut profiles and mission scenarios (baseline, deviation, critical, recovery)
- Hourly synthetic sensor stream: heart rate, blood pressure, SpO2, activity, sleep, stress, symptoms, habitat conditions
- Personal-baseline computation and deviation detection
- Random Forest (scikit-learn) model as the central predictive component
- Explainable outputs: model feature contributions, evidence-backed anomaly flags, and uncertainty/insufficient-evidence states
- Decision-support actions: repeat measurement, log symptom, escalate concern
- Local persistence using SQLite (crew_signal.db) and basic reconciliation of delayed/conflicting records
- Streamlit UI with Plotly visualisations for trend charts and baseline comparison

## Technology stack

- Python
- Streamlit for the web interface
- Plotly for interactive charts
- scikit-learn for Random Forest classifier (n_estimators=50)
- NumPy and pandas for data generation and processing
- SQLite (builtin) for local persistence

## Model and data

- The Random Forest classifier is trained on synthetic data generated inside the repository. The pipeline uses a reproducible random seed (42) and a small, lightweight training set. The model uses 50 decision trees by default and conservative tree depth and leaf settings to remain CPU-friendly.
- Risk categories: LOW, MODERATE, HIGH, CRITICAL. The model returns a predicted risk label, a class probability as a confidence indicator, a short feature-importance explanation, and an evidence state (sufficient / insufficient).
- The system establishes personal baselines from historical synthetic records and compares new readings to detect persistent or combined deviations.
- All data are synthetic. The application does not use, request, or validate real-world medical data.

## Running locally

1. Create a virtual environment (recommended) and activate it.
2. Install dependencies:

    pip install -r requirements.txt

3. Start the Streamlit app:

    streamlit run app.py

4. The UI opens on http://localhost:8501 by default. Use the sidebar to replay scenarios, view trends, and exercise decision-support actions.

## Deployment (Render)

- A `render.yaml` file is provided for deploying the Streamlit app to Render as a web service. The start command uses Streamlit and binds to the port provided by the environment.
- Note: Render instances may not provide durable local filesystem persistence. The included SQLite file (crew_signal.db) is stored on the instance filesystem and may be ephemeral across service restarts or redeploys. For durable storage in production, configure a persistent disk or external database.

## Repository files

- app.py — Streamlit web application and UI
- data.py — Synthetic data generation, local SQLite storage, baselines, and reconciliation
- model.py — Random Forest pipeline and prediction/explanation utilities
- requirements.txt — Minimal Python dependencies
- render.yaml — Render deployment configuration
- .gitignore — Files and local artifacts to ignore

## Limitations and safety

- This project is a prototype for demonstration and educational purposes only. It is not a medical device and does not diagnose or prescribe treatment.
- Model outputs and synthetic scenarios are illustrative. Do not use results for real-world clinical or operational decisions.

## Development notes

- Random seed: 42 for reproducible synthetic generation and model training
- Model training is lightweight and performed at application startup; the trained model is reused during the active session

