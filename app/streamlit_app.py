# app/streamlit_app.py

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hotel_booking_cancelation_prediction.predict import (
    CRITICAL_COLUMNS,
    CancellationPredictor,
)

st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide")

st.title("Hotel Booking Cancellation Predictor")
st.write(
    "Upload booking data as a CSV. The app cleans it and predicts which "
    "bookings will be cancelled, using the saved model."
)

metadata_path = PROJECT_ROOT / "artifacts" / "metadata.json"
if metadata_path.exists():
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    st.caption(f"Model test ROC-AUC: {metadata.get('roc_auc_test', 'N/A'):.4f}")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)

    st.subheader("Preview")
    st.dataframe(raw_df.head())

    present_critical = [c for c in CRITICAL_COLUMNS if c in raw_df.columns]
    missing_critical = [c for c in CRITICAL_COLUMNS if c not in raw_df.columns]
    st.caption(
        f"Detected {len(present_critical)}/{len(CRITICAL_COLUMNS)} important "
        f"columns" + (f" — missing: {', '.join(missing_critical)}" if missing_critical else "")
    )

    if st.button("Predict cancellations"):
        try:
            predictor = CancellationPredictor()
            results = predictor.predict(raw_df)

            # Align predictions back to the correct original rows (cleaning may
            # have dropped some rows, so joining by position is NOT safe).
            output = raw_df.reset_index(drop=True).copy()
            pred_lookup = results.set_index("_original_row")[
                ["cancel_probability", "cancel_prediction", "risk_label"]
            ]
            output = output.join(pred_lookup, how="left")

            for warning in predictor.warnings:
                st.warning(warning)

            st.success("Predictions complete.")

            st.subheader("Predicted results")
            st.dataframe(output, hide_index=True)

            csv_bytes = output.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results CSV",
                data=csv_bytes,
                file_name="cancellation_predictions.csv",
                mime="text/csv",
            )
        except ValueError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001 - surface any runtime failure to the user
            st.error(f"Prediction failed: {e}")
