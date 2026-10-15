import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data import FEATURE_COLUMNS, _generate_synthetic_training_data, compute_personal_baseline


MODEL_PATH = "crew_signal_model.joblib"


def build_random_forest_model():
    df = _generate_synthetic_training_data(n_samples=1500, seed=42)
    features = df[FEATURE_COLUMNS]
    target = df["risk_state"]

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(features, target)
    return model


def load_or_train_model():
    model = build_random_forest_model()
    return model, FEATURE_COLUMNS


def _baseline_deviation(row, baseline):
    deviations = {}
    for key in FEATURE_COLUMNS:
        baseline_value = baseline.get(key, {}).get("mean", 0.0)
        std_value = baseline.get(key, {}).get("std", 1.0)
        value = float(row.get(key, 0.0))
        deviation = (value - baseline_value) / max(std_value, 1.0)
        deviations[key] = deviation
    return deviations


def _severity_state(label):
    mapping = {
        0: "LOW",
        1: "MODERATE",
        2: "HIGH",
        3: "CRITICAL",
    }
    return mapping.get(int(label), "LOW")


def predict_from_recent(model_bundle, recent_df):
    if recent_df is None or recent_df.empty:
        return "INSUFFICIENT EVIDENCE", 0.0, {}, "insufficient"

    if isinstance(model_bundle, tuple):
        model, feature_names = model_bundle
    else:
        model = model_bundle
        feature_names = FEATURE_COLUMNS

    recent = recent_df.copy()
    recent = recent[feature_names].dropna()
    if recent.empty or len(recent) < 3:
        return "INSUFFICIENT EVIDENCE", 0.0, {}, "insufficient"

    baseline = compute_personal_baseline(recent)
    features_for_model = recent[feature_names].tail(1)
    probabilities = model.predict_proba(features_for_model)[0]
    pred_index = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_index])

    if confidence < 0.55:
        return "INSUFFICIENT EVIDENCE", confidence, {}, "insufficient"

    predicted_label = _severity_state(pred_index)
    latest = recent.iloc[-1].to_dict()
    latest_deviation = _baseline_deviation(latest, baseline)

    top_features = dict(
        sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    )

    explanation = {}
    for feature, importance in top_features.items():
        explanation[feature] = round(float(importance), 4)

    return predicted_label, confidence, explanation, "sufficient"


__all__ = [
    "FEATURE_COLUMNS",
    "build_random_forest_model",
    "load_or_train_model",
    "predict_from_recent",
]
