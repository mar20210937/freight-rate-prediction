"""Train on data/train_test.csv, write validation_predictions.csv and the December chart inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
import pandas as pd

from features import city_coords, clean_frame
from model import FreightRateModel

train = pd.read_csv("data/train_test.csv")
val = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation_predictions_template.csv")
dec = pd.read_csv("data/december_chart_inputs.csv")

coords = city_coords(train, val)          # feature columns only; no labels involved
model = FreightRateModel().fit(clean_frame(train, coords), train.posted_rate.values)
print(f"trained on {model.train_keep_.sum():,} of {len(train):,} rows ({(~model.train_keep_).sum()} glitch rows trimmed)")

pred = model.predict(clean_frame(val, coords))
out = pd.DataFrame({"load_id": val.load_id, "predicted_rate": np.round(pred, 2)})
assert set(out.load_id) == set(template.load_id) and len(out) == len(template) == 12_000
out = template[["load_id"]].merge(out, on="load_id", how="left")          # keep template order
assert out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
out.to_csv("validation_predictions.csv", index=False)

# December chart: fixed lane; only the date varies. No quote_signal is supplied -> neutral value is used.
dec["predicted_rate"] = np.round(model.predict(clean_frame(dec, coords)), 2)
dec.to_csv("december_chart_predictions.csv", index=False)
print(dec[["date", "predicted_rate"]].to_string(index=False))
