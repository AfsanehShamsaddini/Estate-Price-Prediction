# src/train.py
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

DATA = Path("data/processed/final.csv")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run: python -m src.make_dataset  (so data/processed/final.csv is created)")

    df = pd.read_csv(DATA)

    # --- 1) Identify the target ---
    # Priority: price_lakh (which is created in make_dataset)
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
        raise ValueError("Couldn't find target. Make sure you have price_lakh or Price_Lakh or price column.")

    # --- 2) Preventing data leakage ---
    # If you have columns like Rate_SqFt, do not use them to predict price
    leak_cols = [c for c in ["Rate_SqFt", "rate_sqft", "price_per_sqft", "Price_per_sqft"] if c in X.columns]
    if leak_cols:
        X = X.drop(columns=leak_cols)

    # --- 3) Feature selection ---
    # Numerical and categorical features are detected automatically
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ])

    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])

    model = HistGradientBoostingRegressor(random_state=42)

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    joblib.dump(pipe, MODELS / "model.joblib")
    print("[OK] Saved models/model.joblib")
    print({"mae": float(mae), "rmse": float(rmse), "r2": float(r2)})

if __name__ == "__main__":
    main()
