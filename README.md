# Hotel Booking Cancellation Prediction

Predict whether a hotel booking will be cancelled, using a cleaned Kaggle hotel
bookings dataset. The winning model is **XGBoost** with a test **ROC-AUC of
0.914**.

## Project structure

```
app/                    Streamlit prediction app (app/streamlit_app.py)
src/hotel_booking_cancelation_prediction/
    cleaning.py         Light pandas cleaning
    feature_pipeline.py Feature engineering + encoding + scaling
    preprocessor.py     Persistable sklearn preprocessing pipeline
    train.py            Builds model + preprocessor artifacts
    training_model.py   Winning XGBoost hyperparameters
    predict.py          Inference (CancellationPredictor)
artifacts/              model.joblib, preprocessor.joblib, metadata
data/
    raw/                hotel_bookings.csv
    processed/          train/test processed CSVs
notebooks/              EDA, cleaning, feature engineering, training
```

## Setup

Requires Python >= 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # install project + dev dependencies
```

## Rebuild model artifacts

`artifacts/` and `data/` are git-ignored, so after a fresh clone you must
rebuild them:

```bash
# place data/raw/hotel_bookings.csv first, then:
uv run python -m src.hotel_booking_cancelation_prediction.train
```

## Run the Streamlit app

```bash
uv run streamlit run app/streamlit_app.py
```

Upload booking data as a CSV (any column set is accepted). The app cleans it and
predicts `cancel_probability`, `cancel_prediction`, and `risk_label` for every
row using the saved artifacts.

Column handling:
- **Important columns** (e.g. `lead_time`, `adr`, `deposit_type`) must be present
  or derivable — otherwise the app reports exactly which ones are missing.
- **Optional columns** (e.g. guest history, `agent`) can be absent; they are
  filled with a neutral value and the app tells you it did so.
- Predictions are aligned back to your original rows even when cleaning drops
  some (zero-guest or invalid-ADR) rows.

## Analytics

The exploratory analytics dashboard was built in **Power BI** from the cleaned
dataset (`data/processed/hotel_bookings_cleaned.csv`): KPIs (bookings,
cancellation rate, avg lead time, ADR) plus cancellation-rate charts by month,
market segment, deposit type, customer type, and country, with filters for hotel
type, year, and segment. Power BI connects directly to that CSV, so it is easy
to refresh and share.

## Predict from the command line

```python
from src.hotel_booking_cancelation_prediction.predict import predict_from_csv

results = predict_from_csv("path/to/bookings.csv")
print(results.head())
```

## Documentation & reports

- [Data dictionary](docs/data_dictionary.md) — every column, its meaning, and cleaning decisions
- [EDA report](reports/01_eda_report.md) — dataset findings and preprocessing roadmap
- [Model report](reports/02_model_report.md) — pipeline, hyperparameters, and metrics
- [Model card](reports/03_model_card.md) — intended use, limitations, and fairness notes
- [Notebooks](notebooks/) — full EDA, cleaning, feature engineering, and training walkthrough

