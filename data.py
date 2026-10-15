import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().with_name("crew_signal.db")

FEATURE_COLUMNS = [
    "hr",
    "sbp",
    "dbp",
    "spo2",
    "activity",
    "sleep_hours",
    "stress_score",
    "symptom_count",
    "symptom_severity",
    "habitat_temp",
    "humidity",
    "co2",
]

RISK_LABELS = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH",
    3: "CRITICAL",
}


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_store():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS astronauts (
                id TEXT PRIMARY KEY,
                name TEXT,
                mission_start TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                astronaut_id TEXT,
                timestamp TEXT,
                hr REAL,
                sbp REAL,
                dbp REAL,
                spo2 REAL,
                activity REAL,
                sleep_hours REAL,
                stress_score REAL,
                symptom_count REAL,
                symptom_severity REAL,
                habitat_temp REAL,
                humidity REAL,
                co2 REAL,
                status TEXT,
                source TEXT
            )
            """
        )


initialize_store()


def _scenario_signal(scenario, idx, rng):
    if scenario == "baseline":
        hr = 72 + 3 * np.sin(idx / 6.5) + rng.normal(0, 2)
        sbp = 118 + 5 * np.sin(idx / 8.0) + rng.normal(0, 4)
        dbp = 76 + 4 * np.sin(idx / 10.0) + rng.normal(0, 3)
        spo2 = 98.0 + rng.normal(0, 0.5)
        activity = 55 + 18 * np.sin(idx / 5.0) + rng.normal(0, 8)
        sleep_hours = 7.5 + 0.6 * np.cos(idx / 9.0) + rng.normal(0, 0.8)
        stress_score = 30 + 12 * np.sin(idx / 7.0) + rng.normal(0, 6)
        symptom_count = 0
        symptom_severity = 0.5 + rng.normal(0, 0.6)
        habitat_temp = 22.0 + 0.7 * np.sin(idx / 11.0) + rng.normal(0, 1.0)
        humidity = 45 + 5 * np.sin(idx / 9.0) + rng.normal(0, 5)
        co2 = 650 + 80 * np.sin(idx / 10.0) + rng.normal(0, 80)
    elif scenario == "deviation":
        hr = 80 + 8 * np.sin(idx / 6.0) + rng.normal(0, 3)
        sbp = 126 + 8 * np.sin(idx / 7.0) + rng.normal(0, 5)
        dbp = 82 + 6 * np.sin(idx / 8.5) + rng.normal(0, 4)
        spo2 = 96.8 + rng.normal(0, 0.7)
        activity = 68 + 16 * np.sin(idx / 4.2) + rng.normal(0, 10)
        sleep_hours = 6.0 + 0.9 * np.cos(idx / 8.0) + rng.normal(0, 1.0)
        stress_score = 54 + 16 * np.sin(idx / 6.2) + rng.normal(0, 8)
        symptom_count = 1 if idx % 4 == 0 else 0
        symptom_severity = 3.5 + rng.normal(0, 1.5)
        habitat_temp = 23.8 + 1.2 * np.sin(idx / 10.0) + rng.normal(0, 1.1)
        humidity = 58 + 10 * np.sin(idx / 9.0) + rng.normal(0, 6)
        co2 = 820 + 200 * np.sin(idx / 8.5) + rng.normal(0, 140)
    elif scenario == "critical":
        hr = 92 + 10 * np.sin(idx / 5.0) + rng.normal(0, 4)
        sbp = 138 + 12 * np.sin(idx / 6.5) + rng.normal(0, 6)
        dbp = 90 + 10 * np.sin(idx / 7.0) + rng.normal(0, 5)
        spo2 = 94.6 + rng.normal(0, 1.0)
        activity = 80 + 20 * np.sin(idx / 3.5) + rng.normal(0, 12)
        sleep_hours = 4.5 + 0.7 * np.cos(idx / 7.0) + rng.normal(0, 1.2)
        stress_score = 72 + 15 * np.sin(idx / 5.5) + rng.normal(0, 9)
        symptom_count = 2 + (1 if idx % 3 == 0 else 0)
        symptom_severity = 6.5 + rng.normal(0, 2.2)
        habitat_temp = 25.7 + 2.0 * np.sin(idx / 8.0) + rng.normal(0, 1.5)
        humidity = 68 + 12 * np.sin(idx / 6.5) + rng.normal(0, 8)
        co2 = 1100 + 220 * np.sin(idx / 7.5) + rng.normal(0, 160)
    else:
        hr = 75 + 6 * np.sin(idx / 7.0) + rng.normal(0, 3)
        sbp = 120 + 7 * np.sin(idx / 9.0) + rng.normal(0, 5)
        dbp = 78 + 5 * np.sin(idx / 9.5) + rng.normal(0, 4)
        spo2 = 97.5 + rng.normal(0, 0.8)
        activity = 60 + 15 * np.sin(idx / 6.0) + rng.normal(0, 9)
        sleep_hours = 6.8 + 0.8 * np.cos(idx / 8.5) + rng.normal(0, 1.2)
        stress_score = 46 + 13 * np.sin(idx / 6.8) + rng.normal(0, 8)
        symptom_count = 1 if idx % 5 == 0 else 0
        symptom_severity = 2.8 + rng.normal(0, 1.3)
        habitat_temp = 23.2 + 1.0 * np.sin(idx / 10.0) + rng.normal(0, 1.2)
        humidity = 52 + 8 * np.sin(idx / 8.5) + rng.normal(0, 6)
        co2 = 740 + 130 * np.sin(idx / 9.5) + rng.normal(0, 110)

    return {
        "hr": round(float(hr), 2),
        "sbp": round(float(sbp), 2),
        "dbp": round(float(dbp), 2),
        "spo2": round(float(spo2), 2),
        "activity": round(float(activity), 2),
        "sleep_hours": round(float(sleep_hours), 2),
        "stress_score": round(float(stress_score), 2),
        "symptom_count": float(symptom_count),
        "symptom_severity": round(float(max(0.0, symptom_severity)), 2),
        "habitat_temp": round(float(habitat_temp), 2),
        "humidity": round(float(humidity), 2),
        "co2": round(float(co2), 2),
    }


def _generate_synthetic_training_data(n_samples=1500, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        hr = rng.normal(72, 8)
        sbp = rng.normal(118, 10)
        dbp = rng.normal(76, 7)
        spo2 = rng.normal(98, 1.2)
        activity = rng.normal(50, 18)
        sleep_hours = rng.normal(7.2, 1.5)
        stress_score = rng.normal(35, 18)
        symptom_count = max(0, int(rng.poisson(1.1)))
        symptom_severity = max(0.0, rng.normal(2.5, 1.3))
        habitat_temp = rng.normal(22.5, 2.0)
        humidity = rng.normal(50, 12)
        co2 = rng.normal(700, 180)

        risk_score = 0
        if hr > 88 or sbp > 132 or dbp > 90:
            risk_score += 2
        if spo2 < 95.5:
            risk_score += 1
        if activity > 80:
            risk_score += 1
        if sleep_hours < 5.5:
            risk_score += 1
        if stress_score > 70:
            risk_score += 1
        if symptom_count >= 2 or symptom_severity > 5:
            risk_score += 1
        if habitat_temp > 25 or humidity > 68 or co2 > 1000:
            risk_score += 1

        if risk_score >= 6:
            risk_state = 3
        elif risk_score >= 4:
            risk_state = 2
        elif risk_score >= 2:
            risk_state = 1
        else:
            risk_state = 0

        rows.append({
            "hr": float(hr),
            "sbp": float(sbp),
            "dbp": float(dbp),
            "spo2": float(spo2),
            "activity": float(activity),
            "sleep_hours": float(sleep_hours),
            "stress_score": float(stress_score),
            "symptom_count": float(symptom_count),
            "symptom_severity": float(symptom_severity),
            "habitat_temp": float(habitat_temp),
            "humidity": float(humidity),
            "co2": float(co2),
            "risk_state": int(risk_state),
            "risk_label": RISK_LABELS[int(risk_state)],
        })

    return pd.DataFrame(rows)


def generate_astronaut_scenario(name="Casey", days=7, seed=42, scenario="baseline"):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=max(days - 1, 0))
    timestamps = pd.date_range(start=start, periods=days * 24, freq="H")
    rows = []

    for idx, ts in enumerate(timestamps):
        payload = _scenario_signal(scenario, idx, rng)
        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            **payload,
            "status": "observed",
            "source": "synthetic",
        })

    profile = {
        "id": f"astronaut_{name.lower().replace(' ', '_')}_{int(datetime.utcnow().timestamp())}",
        "name": name,
        "mission_start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": scenario,
    }
    return pd.DataFrame(rows), profile


def compute_personal_baseline(records_df):
    if records_df is None or records_df.empty:
        return {}

    baseline = {}
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(records_df[column], errors="coerce").dropna()
        if values.empty:
            baseline[column] = {"mean": 0.0, "std": 0.0}
        else:
            baseline[column] = {"mean": float(values.mean()), "std": float(values.std(ddof=0) or 0.0)}
    return baseline


def load_data_store():
    with _connect() as conn:
        astronauts = [dict(row) for row in conn.execute("SELECT * FROM astronauts ORDER BY created_at ASC")]
        records = [dict(row) for row in conn.execute("SELECT * FROM records ORDER BY timestamp ASC")]
    return {"astronauts": astronauts, "records": records}


def save_to_data_store(profile, records_df):
    if records_df is None:
        return

    if isinstance(records_df, pd.DataFrame):
        items = records_df.to_dict("records")
    else:
        items = records_df

    with _connect() as conn:
        if profile is not None:
            astronaut_id = profile.get("id") or "unknown_astronaut"
            conn.execute(
                """
                INSERT OR REPLACE INTO astronauts (id, name, mission_start, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    astronaut_id,
                    profile.get("name"),
                    profile.get("mission_start"),
                    datetime.utcnow().isoformat(timespec="seconds"),
                ),
            )

        for row in items:
            record_id = row.get("id") or f"{profile.get('id', 'unknown')}_{row.get('timestamp')}_{datetime.utcnow().timestamp()}"
            conn.execute(
                """
                INSERT OR REPLACE INTO records (
                    id, astronaut_id, timestamp, hr, sbp, dbp, spo2, activity, sleep_hours,
                    stress_score, symptom_count, symptom_severity, habitat_temp, humidity, co2,
                    status, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    profile.get("id") if profile else row.get("astronaut_id", "unknown"),
                    row.get("timestamp"),
                    row.get("hr"),
                    row.get("sbp"),
                    row.get("dbp"),
                    row.get("spo2"),
                    row.get("activity"),
                    row.get("sleep_hours"),
                    row.get("stress_score"),
                    row.get("symptom_count"),
                    row.get("symptom_severity"),
                    row.get("habitat_temp"),
                    row.get("humidity"),
                    row.get("co2"),
                    row.get("status", "observed"),
                    row.get("source", "synthetic"),
                ),
            )


def reconcile_data_store():
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM records ORDER BY timestamp ASC").fetchall()
        dedup = {}
        for row in rows:
            key = (row["astronaut_id"], row["timestamp"])
            if key not in dedup:
                dedup[key] = dict(row)
            else:
                prev = dedup[key]
                prev["hr"] = max(prev["hr"], row["hr"])
                prev["sbp"] = max(prev["sbp"], row["sbp"])
                prev["symptom_severity"] = max(prev["symptom_severity"], row["symptom_severity"])
                prev["status"] = "reconciled"
                prev["source"] = "reconciled"
                dedup[key] = prev

        conn.execute("DELETE FROM records")
        for item in dedup.values():
            conn.execute(
                """
                INSERT INTO records (
                    id, astronaut_id, timestamp, hr, sbp, dbp, spo2, activity, sleep_hours,
                    stress_score, symptom_count, symptom_severity, habitat_temp, humidity, co2,
                    status, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["id"],
                    item["astronaut_id"],
                    item["timestamp"],
                    item["hr"],
                    item["sbp"],
                    item["dbp"],
                    item["spo2"],
                    item["activity"],
                    item["sleep_hours"],
                    item["stress_score"],
                    item["symptom_count"],
                    item["symptom_severity"],
                    item["habitat_temp"],
                    item["humidity"],
                    item["co2"],
                    item["status"],
                    item["source"],
                ),
            )


__all__ = [
    "FEATURE_COLUMNS",
    "RISK_LABELS",
    "compute_personal_baseline",
    "generate_astronaut_scenario",
    "initialize_store",
    "load_data_store",
    "save_to_data_store",
    "reconcile_data_store",
    "_generate_synthetic_training_data",
]
