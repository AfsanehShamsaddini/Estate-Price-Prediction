# src/predict.py
import argparse
import json
import joblib
import pandas as pd
from pathlib import Path

MODEL = Path("models/model.joblib")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON features")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    X = pd.DataFrame([data])

    pipe = joblib.load(MODEL)
    pred = pipe.predict(X)[0]
    print(float(pred))

if __name__ == "__main__":
    main()
