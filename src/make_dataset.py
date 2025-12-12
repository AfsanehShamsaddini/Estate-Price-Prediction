# src/make_dataset.py
import pandas as pd
from pathlib import Path
from .utils import norm_str, to_float, parse_price_to_lakh


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) اولویت: اگر Final_dataset.csv داری، همونو به عنوان processed استفاده کن
    final_path = RAW_DIR / "Final_dataset.csv"
    if final_path.exists():
        df = pd.read_csv(final_path)
        out = OUT_DIR / "final.csv"
        df.to_csv(out, index=False)
        print(f"[OK] Saved processed dataset: {out}")
        return

    # 2) اگر Final_dataset نداری، از Raw_Property.csv بساز
    raw_path = RAW_DIR / "Raw_Property.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Put Raw_Property.csv (or Final_dataset.csv) into data/raw/")

    df = pd.read_csv(raw_path)

    # --- اینجا ستون‌ها ممکن است اسمشان فرق کند ---
    # تلاش می‌کنیم چند حالت رایج را پوشش دهیم:
    # price:
    if "price" in df.columns:
        df["price_lakh"] = df["price"].apply(parse_price_to_lakh)
    elif "Price" in df.columns:
        df["price_lakh"] = df["Price"].apply(parse_price_to_lakh)
    elif "Price_Lakh" in df.columns:
        df["price_lakh"] = pd.to_numeric(df["Price_Lakh"], errors="coerce")
    else:
        raise ValueError("I couldn't find a price column. Rename it to Price or price, or add Price_Lakh.")

    # area:
    if "area" in df.columns:
        df["area_sqft"] = pd.to_numeric(df["area"], errors="coerce")
    elif "Area_SqFt" in df.columns:
        df["area_sqft"] = pd.to_numeric(df["Area_SqFt"], errors="coerce")
    else:
        # اگر Area_Tpye داری، فعلاً فقط عددهاشو می‌کشیم بیرون
        if "Area_Tpye" in df.columns:
            df["area_sqft"] = df["Area_Tpye"].astype(str).str.replace(",", "").str.extract(r"([0-9]*\.?[0-9]+)")[0]
            df["area_sqft"] = pd.to_numeric(df["area_sqft"], errors="coerce")

    # bhk/bedroom:
    if "bhk" in df.columns:
        df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
    elif "Bedroom" in df.columns:
        df["bhk"] = pd.to_numeric(df["Bedroom"], errors="coerce")

    # تمیزکاری رشته‌ها
    for c in ["type", "locality", "region", "status", "age", "Location", "Region", "Availability", "Property_Age"]:
        if c in df.columns:
            df[c] = df[c].apply(norm_str)

    # حذف ردیف‌های خراب
    df["area_sqft"] = pd.to_numeric(df.get("area_sqft"), errors="coerce")
    df = df.dropna(subset=["price_lakh", "area_sqft"])
    df = df[(df["price_lakh"] > 0) & (df["area_sqft"] > 50)]

    out = OUT_DIR / "final.csv"
    df.to_csv(out, index=False)
    print(f"[OK] Saved processed dataset: {out} rows={len(df)} cols={len(df.columns)}")

if __name__ == "__main__":
    main()
