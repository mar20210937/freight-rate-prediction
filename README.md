# Freight Rate Prediction – Machine Learning Engineer Assessment

Predicts `posted_rate` for 12,000 held-out loads (Nov–Dec 2025) from a model trained on Jan–Oct 2025.

## Run

```bash
python -m pip install -r requirements.txt
# put the provided files in data/ (train_test.csv, validation.csv,
# validation_predictions_template.csv, december_chart_inputs.csv) – data/ is git-ignored
python evaluate.py            # forward-in-time CV + ablations -> outputs/cv_*.csv, cv_plot.png
python train_and_predict.py   # -> validation_predictions.csv, december_chart_predictions.csv
python score.py --predictions validation_predictions.csv \
                --december-predictions december_chart_predictions.csv   # -> scorer_results/candidate_december.png
```

## Layout
| Path | Purpose |
|---|---|
| `src/features.py` | cleaning + feature engineering (shared by train / validation / December chart) |
| `src/model.py` | two-stage model: Ridge backbone + small L1 gradient-boosting corrector |
| `evaluate.py` | expanding-window (forward-in-time) validation, baseline and ablations |
| `train_and_predict.py` | final fit on all labeled data, writes the submission files |
| `score.py` | provided scorer (unchanged) |
| `validation_predictions.csv` | final predictions (`load_id,predicted_rate`) |

## Key decisions
* **Time-based validation.** Train = Jan–Oct, validation = Nov–Dec, so folds train on earlier months and test on later ones. No random splits.
* **Target** = log(rate per mile); predictions are multiplied back by distance.
* **Data quality:** 292 negative weights (sign errors → abs), 300 missing weights (median + flag), 374 missing `market_index`, 1,204 weights clipped at 47,500, 677 (1.4%) rate glitches at ~3.5× or ~0.28× the expected rate (unrelated to any feature → trimmed from training), and 8 cities in validation that never appear in training (handled via lat/lon + zero city effect).
* **`market_index` and calendar trend are not used** – they correlate with rates in-sample but did not hold up consistently when validated forward in time (see report).
* **December chart:** `quote_signal` is not supplied in the chart inputs, so the neutral training median (2.05) is used.
