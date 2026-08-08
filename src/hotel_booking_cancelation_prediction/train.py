# src/hotel_booking_cancelation_prediction/train.py

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from .cleaning import prepare_for_feature_pipeline
from .preprocessor import HotelBookingPreprocessor
from .training_model import train_xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv"


def build_artifacts(
    raw_csv: Path = RAW_DATA_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
    test_size: float = 0.3,
    random_state: int = 0,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning raw data...")
    df = prepare_for_feature_pipeline(pd.read_csv(raw_csv))

    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Fitting preprocessor...")
    preprocessor = HotelBookingPreprocessor()
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    print("Training XGBoost...")
    model = train_xgboost(X_train_ready, y_train)

    y_proba = model.predict_proba(X_test_ready)[:, 1]
    y_pred = model.predict(X_test_ready)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    preprocessor.save(artifact_dir / "preprocessor.joblib")
    joblib.dump(model, artifact_dir / "model.joblib")

    with open(artifact_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.selected_features_, f, indent=2)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": "XGBClassifier",
        "roc_auc_test": roc_auc,
        "n_features": len(preprocessor.selected_features_),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": random_state,
        "test_size": test_size,
        "cleaning": "pandas_cleaning + has_company",
        "feature_pipeline": "sklearn impute -> cap outliers -> encode -> scale",
    }
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to: {artifact_dir}")
    return metadata


if __name__ == "__main__":
    build_artifacts()