"""Cleaning + feature engineering, shared by training, validation scoring and the December chart."""
import numpy as np
import pandas as pd

EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
WEIGHT_CAP = 47_500.0     # ceiling observed in the data (1,204 loads sit exactly on it -> clipped values)
QS_NEUTRAL = 2.05         # median quote_signal; used when a row has no quote_signal (December chart)
GEO_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def city_coords(*frames):
    """city -> (lat, lon), built from the *feature* columns of every file that has them.
    Needed because (a) December chart inputs only carry city names and (b) validation has 8 cities
    that never appear in train_test.csv."""
    rows = []
    for f in frames:
        if "pickup_lat" not in f:
            continue
        rows.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        rows.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(rows).groupby("city")[["lat", "lon"]].first()


def clean_frame(df, coords):
    """Row-level cleaning. Returns a new frame with consistent columns."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if "pickup_lat" not in d:                       # December chart rows: look coordinates up by city
        for side in ("pickup", "delivery"):
            d[f"{side}_lat"] = d[side].map(coords["lat"])
            d[f"{side}_lon"] = d[side].map(coords["lon"])
    w = d["weight"].abs()                           # 292 negative weights = sign errors
    d["weight_missing"] = w.isna().astype(int)
    d["weight_clean"] = w.clip(upper=WEIGHT_CAP)
    if "quote_signal" not in d:
        d["quote_signal"] = QS_NEUTRAL
    return d


def make_features(d):
    """Numeric design inputs from a cleaned frame. Deliberately uses NO market_index and NO calendar
    trend: both correlate with rates in-sample but failed forward-in-time validation (see report)."""
    X = pd.DataFrame(index=d.index)
    X["distance"] = d["distance"]
    X["log_distance"] = np.log(d["distance"])
    hv = haversine_miles(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    X["circuity"] = d["distance"] / hv.clip(lower=1)
    X["equipment"] = d["equipment"].map(EQUIP)
    X["weight"] = d["weight_clean"]
    X["weight_missing"] = d["weight_missing"]
    X["quote_signal"] = d["quote_signal"]
    X["dow"] = d["date"].dt.dayofweek
    for c in GEO_COLS:
        X[c] = d[c]
    return X
