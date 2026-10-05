"""Forward-in-time validation of the model + ablations. Writes outputs/cv_results.csv and cv_plot.png."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from features import city_coords, clean_frame
from model import FreightRateModel, OUTLIER_LOG_THRESHOLD

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
train = pd.read_csv("data/train_test.csv"); val = pd.read_csv("data/validation.csv")
coords = city_coords(train, val)
d = clean_frame(train, coords); y = train.posted_rate.values
month = d.date.dt.month.values

# Expanding-window folds, non-overlapping test blocks, mirroring the real task (train Jan-Oct -> predict Nov-Dec)
FOLDS = [("Jan-Apr -> May-Jun", [1, 2, 3, 4], [5, 6]),
         ("Jan-Jun -> Jul-Aug", [1, 2, 3, 4, 5, 6], [7, 8]),
         ("Jan-Aug -> Sep-Oct", [1, 2, 3, 4, 5, 6, 7, 8], [9, 10])]


def metrics(y, p):
    e = p - y
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                MAPE=100 * np.mean(np.abs(e) / y), MedAPE=100 * np.median(np.abs(e) / y))


def run(name, make_pred):
    rows = []
    for label, tr, te in FOLDS:
        a, b = np.isin(month, tr), np.isin(month, te)
        p, flag = make_pred(a, b)
        r = metrics(y[b], p); r2 = metrics(y[b][~flag], p[~flag])
        rows.append(dict(model=name, fold=label, **r, MAE_clean=r2["MAE"], MAPE_clean=r2["MAPE"]))
    return rows


def full_model(a, b):
    m = FreightRateModel().fit(d[a], y[a]); p = m.predict(d[b])
    flag = np.abs(np.log(y[b] / p)) > OUTLIER_LOG_THRESHOLD     # evaluation-only: unpredictable glitch rows
    return p, flag


def naive(a, b):                                                # lane-agnostic: median rate-per-mile by distance band & equipment
    rpm = y / d.distance.values
    band = pd.cut(d.distance, [0, 150, 300, 600, 1000, 1500, 2200, 4000])
    key = band.astype(str) + d.equipment
    med = pd.Series(rpm[a]).groupby(key[a].values).median()
    p = key[b].map(med).values * d.distance.values[b]
    return p, np.abs(np.log(y[b] / p)) > OUTLIER_LOG_THRESHOLD


def with_time_features(use):
    """Full model's design + market_index-based or date-trend feature on top of the ridge stage."""
    dm = pd.concat([train, val]).groupby("date").market_index.median()
    mi = d.date.dt.strftime("%Y-%m-%d").map(dm).values
    day = (d.date - pd.Timestamp("2025-01-01")).dt.days.values / 300
    def f(a, b):
        m = FreightRateModel().fit(d[a], y[a])
        base_tr, base_te = np.log(m.predict(d[a]) / d.distance.values[a]), np.log(m.predict(d[b]) / d.distance.values[b])
        extra = {"market_index": mi, "date_trend": day, "both": np.c_[mi, day]}[use]
        extra = extra.reshape(len(d), -1)
        rt = np.log(y[a] / d.distance.values[a]) - base_tr
        k = m.train_keep_
        r = Ridge(1e-3).fit(extra[a][k], rt[k])
        p = np.exp(base_te + r.predict(extra[b])) * d.distance.values[b]
        return p, np.abs(np.log(y[b] / p)) > OUTLIER_LOG_THRESHOLD
    return f


rows = []
rows += run("Baseline: median $/mile by distance band x equipment", naive)
rows += run("Final model (ridge + GBM corrector)", full_model)
rows += run("Final + market_index", with_time_features("market_index"))
rows += run("Final + date trend", with_time_features("date_trend"))
res = pd.DataFrame(rows)
res.to_csv(OUT / "cv_results.csv", index=False)
summary = res.groupby("model", sort=False)[["MAE", "RMSE", "MAPE", "MedAPE", "MAE_clean", "MAPE_clean"]].mean().round(2)
print(res.round(2).to_string(index=False)); print("\nMEAN OVER FOLDS\n", summary.to_string())
summary.to_csv(OUT / "cv_summary.csv")

# diagnostic plot: monthly mean log-residual (shows non-stationary level) on the forward folds
m = FreightRateModel().fit(d, y); p = m.predict(d)
lres = np.log(y / p); ok = m.train_keep_
mm = pd.Series(lres[ok]).groupby(month[ok]).mean()
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
ax[0].bar(mm.index, mm.values * 100, color="#064A56"); ax[0].axhline(0, color="k", lw=.6)
ax[0].set_title("Mean in-sample residual by month (%)", loc="left", fontsize=10); ax[0].set_xlabel("month 2025")
dm = pd.concat([train, val]).groupby(pd.to_datetime(pd.concat([train, val]).date).dt.month).market_index.median()
ax[1].plot(dm.index, dm.values, marker="o", color="#B34A2B"); ax[1].axvspan(10.5, 12.5, color="#ddd", alpha=.6)
ax[1].set_title("Median market_index by month (grey = validation)", loc="left", fontsize=10); ax[1].set_xlabel("month 2025")
fig.tight_layout(); fig.savefig(OUT / "cv_plot.png"); print("saved plot")
