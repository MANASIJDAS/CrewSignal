import pandas as pd
from model import load_or_train_model, predict_from_recent
from data import generate_astronaut_scenario


def run_smoke():
    model_bundle = load_or_train_model()
    records, profile = generate_astronaut_scenario(name="SmokeTest", days=2, seed=42, scenario="deviation")
    recent = records.tail(12)
    label, confidence, explanation, status = predict_from_recent(model_bundle, recent)
    print(profile.get("id"))
    print(label)
    print(f"{confidence:.3f}")
    print(status)
    print(explanation)


if __name__ == "__main__":
    run_smoke()
