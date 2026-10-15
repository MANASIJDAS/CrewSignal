import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data import FEATURE_COLUMNS, _generate_synthetic_training_data, compute_personal_baseline

MODEL_PATH = "crew_signal_model.joblib"
RISK_LEVELS = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def _risk_state_from_score(score):
    if score >= 6:
        return 3
    if score >= 4:
        return 2
    if score >= 2:
        return 1
    return 0


def build_random_forest_model():
    df = _generate_synthetic_training_data(n_samples=1500, seed=42)
    features = df[FEATURE_COLUMNS]
    target = df["risk_state"]
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        max_depth=8,
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
        baseline_value = baseline.get(key, {}).get("mean", float(row.get(key, 0.0)))
        std_value = baseline.get(key, {}).get("std", 1.0)
        value = float(row.get(key, 0.0))
        deviation = (value - baseline_value) / max(std_value, 1.0)
        deviations[key] = round(float(deviation), 3)
    return deviations


def _label_from_index(index):
    return RISK_LEVELS[int(index)]


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
    recent = recent.tail(12)
    latest = recent.iloc[-1].to_dict()
    feature_row = pd.DataFrame([latest])
    feature_row = feature_row[feature_names]

    probabilities = model.predict_proba(feature_row)[0]
    pred_index = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_index])

    if confidence < 0.52:
        return "INSUFFICIENT EVIDENCE", confidence, {}, "insufficient"

    predicted_label = _label_from_index(pred_index)
    deviation_map = _baseline_deviation(latest, baseline)
    top_features = dict(
        sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    )

    explanation = {feature: round(float(weight), 4) for feature, weight in top_features.items()}
    risk_payload = {
        "confidence": confidence,
        "prediction": predicted_label,
        "deviation": deviation_map,
        "explanation": explanation,
    }

    if predicted_label in ["HIGH", "CRITICAL"]:
        return predicted_label, confidence, explanation, "sufficient"
    if predicted_label == "MODERATE":
        for key, value in deviation_map.items():
            if abs(value) > 1.5:
                return predicted_label, confidence, explanation, "sufficient"
        return "INSUFFICIENT EVIDENCE", confidence, explanation, "insufficient"

    return predicted_label, confidence, explanation, "sufficient"


__all__ = [
    "FEATURE_COLUMNS",
    "RISK_LEVELS",
    "build_random_forest_model",
    "load_or_train_model",
    "predict_from_recent",
]
