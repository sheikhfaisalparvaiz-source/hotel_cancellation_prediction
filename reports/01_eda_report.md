# EDA Report

Source: `notebooks/01_data_understanding_eda.ipynb` and `notebooks/02_data_cleaning.ipynb`.
Raw data: **119,390 bookings** (2015–2017), City Hotel + Resort Hotel.

## Target variable

- **37.0% cancelled**, 63.0% not cancelled — moderately imbalanced.
- Treated with `stratify=y` in the train/test split and `scale_pos_weight` in XGBoost.

## Data quality findings

| Issue | Extent | Action |
|---|---|---|
| Duplicate rows | 31,994 (27%) | Dropped |
| `company` missing | 94,077 (79%) | Replaced with `has_company` flag |
| `agent` missing | 14,435 (12%) | Imputed as `Unknown` |
| `country` missing | 487 | Imputed as `Unknown` |
| `children` missing | 4 | Imputed as 0 |
| Zero-guest rows | 180 | Dropped |
| `adr` outlier | max 5,400 (typo) | Rows with `adr >= 5000` dropped |
| `meal` inconsistency | `Undefined` | Recoded to `SC` |

## Numerical feature findings

- Strong **positive skew** in `lead_time`, `adr`, `days_in_waiting_list`, and
  stay-duration columns → handled with `log1p` before scaling.
- Outliers present in `lead_time`, `adr`, `required_car_parking_spaces`
  (extreme max of 8) → handled with z-score capping (threshold 3.85, chosen by
  Optuna) using **train-only** bounds.

## Categorical feature findings

- `country` is high-cardinality (100+ values, many rare) → **frequency encoding**.
- Low-cardinality features (`hotel`, `meal`, `market_segment`,
  `distribution_channel`, `deposit_type`, `customer_type`, room types, months)
  → **one-hot encoding**.
- `reservation_status` and `reservation_status_date` directly encode the target
  → **dropped as leakage**.

## Key relationships (modeling hypotheses)

- **Deposit type**: `Non Refund` deposits are strongly associated with
  cancellations.
- **Lead time**: longer lead times correlate with higher cancellation rates.
- **Parking spaces & special requests**: bookings requesting these are much less
  likely to cancel (higher commitment).
- **Market segment**: `Online TA` (travel agents) dominates volume and cancels
  more than `Direct` / `Corporate`.

## Preprocessing roadmap (final, all Optuna-validated)

1. Impute: constant (0 for numeric, `Unknown` for categorical).
2. Cap outliers: z-score 3.85, train-only bounds.
3. Construct features: 7 engineered numeric features + `has_company`.
4. Encode: one-hot (low cardinality), frequency (high cardinality).
5. Scale: `log1p` then `StandardScaler`.

## Final dataset after cleaning

- Rows used for modeling: **87,229** (after de-dup, filtering, split).
- Train: 61,060 · Test: 26,169 (70/30 stratified).
- Model input: **37 features** (see `artifacts/selected_features.json`).
