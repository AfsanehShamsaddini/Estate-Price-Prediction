# src/utils.py
import re

def norm_str(x):
    if x is None:
        return None
    s = str(x).strip()
    s = " ".join(s.split())
    return s

def to_float(x):
    try:
        if x is None:
            return None
        s = str(x)
        s = s.replace(",", "").strip()
        return float(s)
    except Exception:
        return None

def parse_price_to_lakh(price_str: str):
    """
    تبدیل قیمت‌های شبیه:
    '2.4 Crore' -> 240 (Lakh)
    '95 Lac'    -> 95 (Lakh)
    '1 Cr'      -> 100 (Lakh)
    اگر نتوانست، None برمی‌گرداند.
    """
    if price_str is None:
        return None
    s = str(price_str).lower().replace(",", "").strip()

    m = re.search(r"([0-9]*\.?[0-9]+)\s*(crore|cr|lac|lakh)", s)
    if not m:
        return None

    val = float(m.group(1))
    unit = m.group(2)

    if unit in ("crore", "cr"):
        return val * 100.0  # 1 crore = 100 lakh
    if unit in ("lac", "lakh"):
        return val
    return None
