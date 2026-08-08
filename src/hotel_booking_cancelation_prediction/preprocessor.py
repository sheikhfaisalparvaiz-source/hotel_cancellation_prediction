# src/hotel_booking_cancelation_prediction/preprocessor.py

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_pipeline import (
    CARDINALITY_THRESHOLD,
    INT_COLS,
    OUTLIER_COLS,
    SELECTED_FEATURES,
    ZSCORE_THRESHOLD,
    construct_features,
    get_column_types,
)


class HotelBookingPreprocessor:
    """
    Fitted sklearn feature pipeline for inference.
    Mirrors run_feature_pipeline() but persists all learned state.
    """

    def fit(self, X_train: pd.DataFrame):
        self.num_cols_, self.cat_cols_ = get_column_types(X_train)

        # ---- 1. Impute ----
        self.imputer_ = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='constant', fill_value=0), self.num_cols_),
                ('cat', SimpleImputer(strategy='constant', fill_value='Unknown'), self.cat_cols_),
            ],
            remainder='drop',
        )
        X_imp = self._imputer_to_df(self.imputer_.fit_transform(X_train), X_train.index)

        # ---- 2. Outlier bounds (train only) ----
        self.outlier_bounds_ = {}
        for col in OUTLIER_COLS:
            if col in X_imp.columns:
                mean_val = X_imp[col].mean()
                std_val = X_imp[col].std()
                self.outlier_bounds_[col] = (
                    mean_val - ZSCORE_THRESHOLD * std_val,
                    mean_val + ZSCORE_THRESHOLD * std_val,
                )
        X_capped = self._cap_outliers(X_imp)

        # ---- 3. Feature construction ----
        X_fe, _ = construct_features(X_capped, X_capped)

        # ---- 4. Encode categoricals ----
        cardinality = X_fe[self.cat_cols_].nunique()
        self.low_card_cols_ = cardinality[cardinality <= CARDINALITY_THRESHOLD].index.tolist()
        self.high_card_cols_ = cardinality[cardinality > CARDINALITY_THRESHOLD].index.tolist()

        self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if self.low_card_cols_:
            self.ohe_.fit(X_fe[self.low_card_cols_])
            self.ohe_columns_ = self.ohe_.get_feature_names_out(self.low_card_cols_).tolist()
        else:
            self.ohe_columns_ = []

        self.freq_maps_ = {
            col: X_fe[col].value_counts(normalize=True)
            for col in self.high_card_cols_
        }

        self.engineered_num_cols_ = [
            'total_stay_nights', 'is_weekend_only', 'total_guests',
            'is_family', 'is_solo', 'adr_per_person', 'room_type_changed',
        ]

        X_enc = self._encode(X_fe)

        # ---- 5. Scale ----
        self.continuous_cols_ = [
            c for c in (
                self.num_cols_
                + self.engineered_num_cols_
                + [f'{col}_freq' for col in self.high_card_cols_]
            )
            if c in X_enc.columns
        ]

        X_for_scale = X_enc.copy()
        for col in self.continuous_cols_:
            X_for_scale[col] = np.log1p(np.maximum(0, X_for_scale[col]))

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_for_scale[self.continuous_cols_])

        self.all_feature_columns_ = X_enc.columns.tolist()
        self.selected_features_ = SELECTED_FEATURES.copy()

        missing = [c for c in self.selected_features_ if c not in self.all_feature_columns_]
        if missing:
            raise ValueError(f'Training data missing expected selected features: {missing}')

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = self._imputer_to_df(self.imputer_.transform(X), X.index)
        X_capped = self._cap_outliers(X_imp)
        X_fe, _ = construct_features(X_capped, X_capped)
        X_enc = self._encode(X_fe)

        X_scaled = X_enc.reindex(columns=self.all_feature_columns_, fill_value=0).copy()

        for col in self.continuous_cols_:
            X_scaled[col] = np.log1p(np.maximum(0, X_scaled[col]))

        X_scaled[self.continuous_cols_] = self.scaler_.transform(X_scaled[self.continuous_cols_])
        return X_scaled[self.selected_features_]

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> HotelBookingPreprocessor:
        return joblib.load(path)

    # ---- helpers ----

    def _imputer_to_df(self, array, index) -> pd.DataFrame:
        cols = self.num_cols_ + self.cat_cols_
        df = pd.DataFrame(array, columns=cols, index=index)

        for col in self.num_cols_:
            df[col] = pd.to_numeric(df[col])

        for col in INT_COLS:
            if col in df.columns:
                df[col] = df[col].astype('int64')

        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, (lower, upper) in self.outlier_bounds_.items():
            if col in out.columns:
                out[col] = out[col].clip(lower=lower, upper=upper)
        return out

    def _encode(self, df_fe: pd.DataFrame) -> pd.DataFrame:
        all_num_cols = self.num_cols_ + self.engineered_num_cols_

        if self.low_card_cols_:
            ohe_array = self.ohe_.transform(df_fe[self.low_card_cols_])
            ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_columns_, index=df_fe.index)
        else:
            ohe_df = pd.DataFrame(index=df_fe.index)

        freq_df = pd.DataFrame(index=df_fe.index)
        for col in self.high_card_cols_:
            freq_df[f'{col}_freq'] = df_fe[col].map(self.freq_maps_[col]).fillna(0)

        return pd.concat([df_fe[all_num_cols], ohe_df, freq_df], axis=1)