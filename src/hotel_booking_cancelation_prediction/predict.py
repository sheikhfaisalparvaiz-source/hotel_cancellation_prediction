# src/hotel_booking_cancelation_prediction/predict.py

from pathlib import Path

import joblib
import pandas as pd

from .cleaning import prepare_for_feature_pipeline
from .preprocessor import HotelBookingPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Columns user CSV should contain (same as raw Kaggle file)
# is_canceled is optional — only needed if you want to evaluate, not predict
REQUIRED_RAW_COLUMNS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "meal", "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type",
    "assigned_room_type", "booking_changes", "deposit_type", "agent",
    "company", "days_in_waiting_list", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
]

# Columns the model genuinely needs. If one of these is absent from the upload
# AND cannot be derived, prediction cannot happen — we must tell the user.
CRITICAL_COLUMNS = [
    "hotel", "lead_time", "arrival_date_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type", "booking_changes",
    "deposit_type", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
]

# Columns that can be safely defaulted when absent (treated as "unknown").
# Same idea as the imputer's constant fill in the fitted preprocessor.
OPTIONAL_COLUMN_DEFAULTS = {
    "arrival_date_year": 0,
    "arrival_date_week_number": 0,
    "arrival_date_day_of_month": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "agent": 0,
    "days_in_waiting_list": 0,
    "has_company": 0,
}

DROP_BEFORE_MODEL = ["is_canceled", "reservation_status_date"]


def validate_input(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Uploaded file has no rows.")

    missing = [col for col in CRITICAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "The following important column(s) are missing from your file and "
            "cannot be derived from the data:\n  - "
            + "\n  - ".join(missing)
            + "\n\nPlease include them (e.g. in the same format as hotel_bookings.csv)."
        )


class CancellationPredictor:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.preprocessor = HotelBookingPreprocessor.load(self.artifact_dir / "preprocessor.joblib")
        self.model = joblib.load(self.artifact_dir / "model.joblib")
        self.warnings: list[str] = []

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        validate_input(raw_df)

        raw_df = raw_df.copy()
        raw_df["_original_row"] = range(len(raw_df))

        cleaned = prepare_for_feature_pipeline(raw_df)

        dropped_rows = len(raw_df) - len(cleaned)
        if dropped_rows > 0:
            self.warnings.append(
                f"{dropped_rows} row(s) removed by cleaning "
                "(zero guests or adr >= 5000)."
            )

        # Add neutral defaults for optional columns that are absent, so the
        # fitted preprocessor (which expects the full column set) still works.
        for col, default in OPTIONAL_COLUMN_DEFAULTS.items():
            if col not in cleaned.columns:
                cleaned[col] = default
                self.warnings.append(
                    f"Column '{col}' not found — filled with {default} "
                    "(treated as unknown)."
                )

        X = cleaned.drop(columns=DROP_BEFORE_MODEL, errors="ignore")
        X_ready = self.preprocessor.transform(X)

        proba = self.model.predict_proba(X_ready)[:, 1]
        pred = self.model.predict(X_ready)

        results = pd.DataFrame({
            "cancel_probability": proba,
            "cancel_prediction": pred.astype(int),
            "risk_label": ["High Risk" if p >= 0.5 else "Low Risk" for p in proba],
        }, index=cleaned.index)

        if "_original_row" in cleaned.columns:
            results.insert(0, "_original_row", cleaned["_original_row"].values)

        return results


def predict_from_csv(csv_path: str | Path) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)
    predictor = CancellationPredictor()
    return predictor.predict(raw_df)