# src/evaluate.py
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = Path("data/processed/final.csv")
MODEL = Path("models/model.joblib")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)

    if "price_lakh" in df.columns:
        y = df["price_lakh"].astype(float)
        X = df.drop(columns=["price_lakh"])
    elif "Price_Lakh" in df.columns:
        y = df["Price_Lakh"].astype(float)
        X = df.drop(columns=["Price_Lakh"])
    elif "price" in df.columns:
        y = df["price"].astype(float)
        X = df.drop(columns=["price"])
    else:
        raise ValueError("Target not found.")

    leak_cols = [c for c in ["Rate_SqFt", "rate_sqft", "price_per_sqft", "Price_per_sqft"] if c in X.columns]
    if leak_cols:
        X = X.drop(columns=leak_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = joblib.load(MODEL)
    pred = pipe.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "n_test": int(len(y_test)),
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Pred vs True
    plt.figure()
    plt.scatter(y_test, pred, s=10)
    plt.xlabel("True price (Lakh)")
    plt.ylabel("Predicted price (Lakh)")
    plt.title("Predicted vs True")
    plt.savefig(FIGS / "pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Residuals
    residuals = y_test - pred
    plt.figure()
    plt.scatter(pred, residuals, s=10)
    plt.axhline(0)
    plt.xlabel("Predicted price (Lakh)")
    plt.ylabel("Residual (true - pred)")
    plt.title("Residual plot")
    plt.savefig(FIGS / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[OK] Saved reports/metrics.json and reports/figures/*.png")
    print(metrics)

if __name__ == "__main__":
    main()
