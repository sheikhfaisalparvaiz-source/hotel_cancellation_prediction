# Model Card

## Model overview

| Field | Value |
|---|---|
| Model name | Hotel Booking Cancellation Predictor |
| Model type | Gradient boosted tree ensemble (XGBClassifier) |
| Task | Binary classification — will this booking be cancelled? |
| Version | 0.1.0 (2026-08-05) |
| Developer | Sheikh Faisal |

## Intended use

- **Primary use:** Batch prediction of cancellation probability for new hotel
  bookings, via the Streamlit app (`app/streamlit_app.py`) or the
  `CancellationPredictor` Python class.
- **Input:** raw hotel booking rows (flexible column set — the app reports any
  missing important columns).
- **Output:** `cancel_probability` (0–1), `cancel_prediction` (0/1, threshold
  0.5), and `risk_label` (High/Low Risk).

## Out of scope / limitations

- Trained on **2015–2017** hotel booking data; behavior may not generalize to
  other periods, hotels, or booking platforms.
- `country_freq` and outlier bounds are derived from the training distribution;
  unseen countries map to frequency 0.
- The 0.5 decision threshold is a default; in production you may want to tune it
  to the business cost of false positives vs. false negatives.

## Performance

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.9140 |
| Class balance (test) | 37% cancelled / 63% not |
| Evaluation | 26,169 held-out bookings |

## Data & features

- Source: Kaggle — Hotel Booking Demand (Antonio, Almeida & Nunes, 2019).
- 119,390 raw rows → 87,229 after cleaning → 37 model features.
- Target: `is_canceled`. Leakage columns (`reservation_status`,
  `reservation_status_date`) are dropped.

## Ethics & fairness

- The model predicts cancellations, not guest identity; no PII is used.
- `country` frequency encoding could encode cross-country volume differences —
  treat any country-based business decisions with caution and review for bias.

## Deployment

- App: `uv run streamlit run app/streamlit_app.py`
- Rebuild artifacts: `uv run python -m src.hotel_booking_cancelation_prediction.train`
- Analytics: Power BI dashboard built on `data/processed/hotel_bookings_cleaned.csv`
