"""Two-stage freight-rate model.

Target: log(rate per mile).
Stage 1  Ridge on interpretable terms: cubic in log-distance, equipment, weight, quote_signal
         (U-shaped), day-of-week, geography (lat/lon) and pickup/delivery city effects.
Stage 2  Small L1 gradient-boosted-tree corrector on the stage-1 residuals (captures leftover
         non-linear interactions between distance, weight, quote_signal and equipment).
Training rows whose stage-1 residual is > 0.5 log-units (the ~1.4% "x3.5 / x0.28" rate glitches)
are removed iteratively before the final fit; they are unrelated to any feature.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from features import GEO_COLS, QS_NEUTRAL, make_features

OUTLIER_LOG_THRESHOLD = 0.5
CORE_COLS = ["log_distance", "weight", "quote_signal", "equipment", "circuity"]


class FreightRateModel:
    def __init__(self, ridge_alpha=1.0, trim_rounds=2, seed=0):
        self.ridge_alpha, self.trim_rounds, self.seed = ridge_alpha, trim_rounds, seed

    # ---- design matrix -------------------------------------------------------------------
    def _design(self, d, X):
        ld, w = X.log_distance, X.weight.fillna(self.w_fill_) / 47_500
        qc = X.quote_signal - QS_NEUTRAL
        f = pd.DataFrame({
            "ld": ld, "ld2": ld ** 2, "ld3": ld ** 3,
            "flatbed": (X.equipment == 1), "reefer": (X.equipment == 2),
            "w": w, "w2": w ** 2, "w_missing": X.weight_missing,
            "qs_abs": qc.abs(), "qs_abs2": qc.abs() ** 2, "qs": qc,
            "qs_short": qc / (1 + X.distance / 200),
        }).astype(float)
        for k in range(1, 7):
            f[f"dow{k}"] = (X.dow == k).astype(float)
        for c in GEO_COLS:
            f[c] = (X[c] - self.geo_mu_[c]) / self.geo_sd_[c]
        city = pd.concat([
            pd.get_dummies(d["pickup"], prefix="p", dtype=float),
            pd.get_dummies(d["delivery"], prefix="d", dtype=float)], axis=1)
        city = city.reindex(columns=self.city_cols_, fill_value=0.0)   # unseen cities -> all zeros
        return pd.concat([f, city], axis=1)

    # ---- fit / predict -----------------------------------------------------------------
    def fit(self, d, rate):
        X = make_features(d)
        self.w_fill_ = float(X.weight.median())
        self.geo_mu_, self.geo_sd_ = X[GEO_COLS].mean(), X[GEO_COLS].std()
        self.city_cols_ = (["p_" + c for c in sorted(d.pickup.unique())] +
                           ["d_" + c for c in sorted(d.delivery.unique())])
        A = self._design(d, X)
        y = np.log(np.asarray(rate) / d["distance"].values)
        keep = np.ones(len(y), bool)
        for _ in range(self.trim_rounds + 1):
            ridge = Ridge(self.ridge_alpha).fit(A[keep], y[keep])
            resid = y - ridge.predict(A)
            keep = np.abs(resid) < OUTLIER_LOG_THRESHOLD
        self.ridge_ = ridge
        self.train_keep_ = keep
        self.gbm_ = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
            min_samples_leaf=200, random_state=self.seed).fit(X.loc[keep, CORE_COLS], resid[keep])
        return self

    def predict(self, d):
        X = make_features(d)
        log_rpm = self.ridge_.predict(self._design(d, X)) + self.gbm_.predict(X[CORE_COLS])
        return np.exp(log_rpm) * d["distance"].values
