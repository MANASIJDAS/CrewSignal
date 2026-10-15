import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

from data import (
    FEATURE_COLUMNS,
    compute_personal_baseline,
    generate_astronaut_scenario,
    initialize_store,
    load_data_store,
    reconcile_data_store,
    save_to_data_store,
)
from model import load_or_train_model, predict_from_recent

initialize_store()

st.set_page_config(page_title="CrewSignal", layout="wide")


def latest_profile():
    payload = load_data_store()
    astronauts = payload.get("astronauts", [])
    if not astronauts:
        return None
    return astronauts[-1]


def history_records():
    payload = load_data_store()
    records = payload.get("records", [])
    if not records:
        return pd.DataFrame(columns=["timestamp", *FEATURE_COLUMNS, "status", "source"])
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def refresh_scenario(name, scenario):
    records_df, profile = generate_astronaut_scenario(name=name, days=7, seed=42, scenario=scenario)
    save_to_data_store(profile, records_df)
    return profile, history_records()


def add_manual_record(profile, action):
    records = history_records()
    if records.empty:
        return
    latest = records.iloc[-1].copy()
    timestamp = datetime.utcnow()
    new_record = latest.to_dict()
    new_record["id"] = f"{profile['id']}_{action}_{int(timestamp.timestamp())}"
    new_record["timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    new_record["source"] = "manual"
    new_record["status"] = action
    if action == "repeat":
        new_record["hr"] = float(np.clip(latest["hr"] + np.random.normal(0, 2), 50, 120))
        new_record["sbp"] = float(np.clip(latest["sbp"] + np.random.normal(0, 5), 90, 160))
        new_record["spo2"] = float(np.clip(latest["spo2"] + np.random.normal(0, 0.4), 92, 100))
        new_record["stress_score"] = float(np.clip(latest["stress_score"] + np.random.normal(0, 6), 10, 100))
    if action == "symptom":
        new_record["symptom_count"] = float(max(0, latest["symptom_count"] + 1))
        new_record["symptom_severity"] = float(np.clip(latest["symptom_severity"] + 1.5, 0, 10))
        new_record["stress_score"] = float(np.clip(latest["stress_score"] + 8, 10, 100))
    if action == "escalate":
        new_record["status"] = "protocol_escalation"
        new_record["source"] = "protocol"
        new_record["symptom_count"] = float(max(latest["symptom_count"], 2))
        new_record["symptom_severity"] = float(np.clip(latest["symptom_severity"] + 2.0, 0, 10))
        new_record["stress_score"] = float(np.clip(latest["stress_score"] + 12, 10, 100))
    save_to_data_store(profile, pd.DataFrame([new_record]))


if "profile" not in st.session_state:
    default_profile = {
        "id": "astronaut_casey_001",
        "name": "Casey",
        "mission_start": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.profile = default_profile

st.title("CrewSignal")
st.caption("Synthetic astronaut health monitoring · decision support prototype")
st.info("Offline-ready local SQLite storage active.")

with st.sidebar:
    st.subheader("Mission")
    astronaut_name = st.text_input("Astronaut name", value=st.session_state.profile["name"])
    scenario = st.selectbox("Scenario", ["baseline", "deviation", "critical", "recovery"])
    if st.button("Replay scenario"):
        profile, _ = refresh_scenario(astronaut_name, scenario)
        st.session_state.profile = profile
        st.rerun()
    if st.button("Reconcile records"):
        reconcile_data_store()
        st.rerun()
    if st.button("Load latest record set"):
        st.rerun()

profile = latest_profile()
if profile is None:
    profile, _ = refresh_scenario(astronaut_name, scenario)
    st.session_state.profile = profile

records = history_records()
if records.empty:
    profile, records = refresh_scenario(astronaut_name, scenario)
    st.session_state.profile = profile

latest = records.iloc[-1].copy()
baseline = compute_personal_baseline(records)
latest_deviation = {}
for key in FEATURE_COLUMNS:
    mean = baseline.get(key, {}).get("mean", float(latest.get(key, 0.0)))
    std = baseline.get(key, {}).get("std", 1.0)
    value = float(latest.get(key, 0.0))
    latest_deviation[key] = round((value - mean) / max(std, 1.0), 2)

model_bundle = load_or_train_model()
recent = records.tail(12).copy() if len(records) >= 3 else records.copy()
label, confidence, explanation, status = predict_from_recent(model_bundle, recent)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Heart rate", f"{latest['hr']:.1f} bpm", f"{latest_deviation['hr']:+.2f}")
col2.metric("Blood pressure", f"{latest['sbp']:.0f}/{latest['dbp']:.0f} mmHg", f"{latest_deviation['sbp']:+.2f}")
col3.metric("SpO2", f"{latest['spo2']:.1f}%", f"{latest_deviation['spo2']:+.2f}")
col4.metric("Stress", f"{latest['stress_score']:.0f}", f"{latest_deviation['stress_score']:+.2f}")

st.subheader("Assessment")
if label == "INSUFFICIENT EVIDENCE":
    st.warning("INSUFFICIENT EVIDENCE")
else:
    if label in ["HIGH", "CRITICAL"]:
        st.error(f"{label} risk · model confidence {confidence:.2f}")
    elif label == "MODERATE":
        st.warning(f"{label} risk · model confidence {confidence:.2f}")
    else:
        st.success(f"{label} risk · model confidence {confidence:.2f}")
st.write(f"Evidence state: {status.upper()}")

with st.container():
    evidence_cols = st.columns(5)
    evidence_cols[0].metric("Sleep", f"{latest['sleep_hours']:.1f} h", f"{latest_deviation['sleep_hours']:+.2f}")
    evidence_cols[1].metric("Activity", f"{latest['activity']:.0f}", f"{latest_deviation['activity']:+.2f}")
    evidence_cols[2].metric("Symptoms", f"{latest['symptom_count']:.0f}", f"{latest_deviation['symptom_count']:+.2f}")
    evidence_cols[3].metric("Habitat temp", f"{latest['habitat_temp']:.1f} C", f"{latest_deviation['habitat_temp']:+.2f}")
    evidence_cols[4].metric("CO2", f"{latest['co2']:.0f} ppm", f"{latest_deviation['co2']:+.2f}")

st.subheader("Decision support")
workflow = st.columns(3)
with workflow[0]:
    if st.button("Repeat measurement"):
        add_manual_record(profile, "repeat")
        st.rerun()
with workflow[1]:
    if st.button("Log symptom"):
        add_manual_record(profile, "symptom")
        st.rerun()
with workflow[2]:
    if st.button("Escalate concern"):
        add_manual_record(profile, "escalate")
        st.rerun()

trend_metric = st.selectbox("Trend chart", ["hr", "sbp", "dbp", "spo2", "sleep_hours", "stress_score", "activity", "co2"])
trend_fig = px.line(records, x="timestamp", y=trend_metric, markers=True)
trend_fig.update_layout(template="plotly_white", title=f"{trend_metric} over mission time")
st.plotly_chart(trend_fig, use_container_width=True)

st.subheader("Evidence and feature contribution")
if explanation:
    importance_df = pd.DataFrame(
        [{"feature": key, "importance": value} for key, value in explanation.items()]
    ).sort_values("importance", ascending=False)
    importance_fig = px.bar(importance_df, x="feature", y="importance", color="feature")
    importance_fig.update_layout(template="plotly_white", title="Model feature contribution")
    st.plotly_chart(importance_fig, use_container_width=True)
else:
    st.info("No model explanation available.")

st.subheader("Baseline vs latest")
baseline_rows = []
for key in FEATURE_COLUMNS:
    mean = baseline.get(key, {}).get("mean", 0.0)
    std = baseline.get(key, {}).get("std", 1.0)
    value = float(latest.get(key, 0.0))
    baseline_rows.append({"metric": key, "baseline": mean, "latest": value, "deviation": (value - mean) / max(std, 1.0)})
baseline_df = pd.DataFrame(baseline_rows)
summary_fig = px.bar(
    baseline_df,
    x="metric",
    y=["baseline", "latest"],
    barmode="group",
)
summary_fig.update_layout(template="plotly_white", title="Personal baseline comparison")
st.plotly_chart(summary_fig, use_container_width=True)

st.subheader("Mission timeline")
st.dataframe(records[["timestamp", *FEATURE_COLUMNS, "status", "source"]].tail(20), use_container_width=True)

st.caption("All measurements and predictions are synthetic and intended for prototype decision-support only.")
