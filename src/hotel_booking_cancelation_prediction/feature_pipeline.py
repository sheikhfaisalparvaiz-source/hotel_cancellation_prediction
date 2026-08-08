# src/hotel_booking_cancelation_prediction/feature_pipeline.py

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- Proven constants (from Optuna studies — see notebook cells 13, 22, 34) ----
ZSCORE_THRESHOLD = 3.8484811634335334
CARDINALITY_THRESHOLD = 15
LEAKAGE_COLS = ['reservation_status_date', 'reservation_status']

OUTLIER_COLS = [
    'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'booking_changes',
    'days_in_waiting_list', 'adr'
]

INT_COLS = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'required_car_parking_spaces',
    'total_of_special_requests', 'has_company'
]


def get_column_types(df: pd.DataFrame):
    """Split df into numeric / categorical column lists, excluding leakage cols."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in LEAKAGE_COLS]
    return num_cols, cat_cols


def impute_missing_values(x_train, x_test, num_cols, cat_cols):
    """Constant imputation — proven winner (Optuna: AUC 0.8623)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='constant', fill_value=0), num_cols),
            ('cat', SimpleImputer(strategy='constant', fill_value='Unknown'), cat_cols)
        ],
        remainder='drop'
    )

    x_train_imp = pd.DataFrame(
        preprocessor.fit_transform(x_train), columns=num_cols + cat_cols, index=x_train.index
    )
    x_test_imp = pd.DataFrame(
        preprocessor.transform(x_test), columns=num_cols + cat_cols, index=x_test.index
    )

    for col in num_cols:
        x_train_imp[col] = pd.to_numeric(x_train_imp[col])
        x_test_imp[col] = pd.to_numeric(x_test_imp[col])

    for col in INT_COLS:
        if col in x_train_imp.columns:
            x_train_imp[col] = x_train_imp[col].astype('int64')
            x_test_imp[col] = x_test_imp[col].astype('int64')

    return x_train_imp, x_test_imp


def cap_outliers(x_train_imp, x_test_imp, outlier_cols=OUTLIER_COLS, z_thresh=ZSCORE_THRESHOLD):
    """Z-score capping — proven winner (Optuna: AUC 0.8635). Bounds from TRAIN only."""
    x_train_capped = x_train_imp.copy()
    x_test_capped = x_test_imp.copy()

    for col in outlier_cols:
        mean_val = x_train_imp[col].mean()
        std_val = x_train_imp[col].std()
        lower = mean_val - (z_thresh * std_val)
        upper = mean_val + (z_thresh * std_val)

        x_train_capped[col] = x_train_capped[col].clip(lower=lower, upper=upper)
        x_test_capped[col] = x_test_capped[col].clip(lower=lower, upper=upper)

    return x_train_capped, x_test_capped


def construct_features(x_train_capped, x_test_capped):
    """Adds 7 engineered features. FIX: these now actually flow downstream."""
    dfs = []
    for df_in in [x_train_capped, x_test_capped]:
        df_fe = df_in.copy()

        df_fe['total_stay_nights'] = df_fe['stays_in_weekend_nights'] + df_fe['stays_in_week_nights']
        df_fe['is_weekend_only'] = (
            (df_fe['stays_in_weekend_nights'] > 0) & (df_fe['stays_in_week_nights'] == 0)
        ).astype(int)

        df_fe['total_guests'] = df_fe['adults'] + df_fe['children'] + df_fe['babies']
        df_fe['is_family'] = ((df_fe['children'] > 0) | (df_fe['babies'] > 0)).astype(int)
        df_fe['is_solo'] = ((df_fe['adults'] == 1) & (df_fe['total_guests'] == 1)).astype(int)

        df_fe['adr_per_person'] = df_fe['adr'] / np.maximum(df_fe['total_guests'], 1)
        df_fe['room_type_changed'] = (df_fe['reserved_room_type'] != df_fe['assigned_room_type']).astype(int)

        dfs.append(df_fe)

    return dfs[0], dfs[1]


def encode_categoricals(x_train_fe, x_test_fe, num_cols, cat_cols, engineered_num_cols,
                         cardinality_threshold=CARDINALITY_THRESHOLD):
    """One-hot for low cardinality, frequency encoding for high cardinality."""
    cardinality = x_train_fe[cat_cols].nunique()
    low_card_cols = cardinality[cardinality <= cardinality_threshold].index.tolist()
    high_card_cols = cardinality[cardinality > cardinality_threshold].index.tolist()

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    x_train_ohe = pd.DataFrame(index=x_train_fe.index)
    x_test_ohe = pd.DataFrame(index=x_test_fe.index)

    if low_card_cols:
        ohe.fit(x_train_fe[low_card_cols])
        ohe_col_names = ohe.get_feature_names_out(low_card_cols)
        x_train_ohe = pd.DataFrame(
            ohe.transform(x_train_fe[low_card_cols]), columns=ohe_col_names, index=x_train_fe.index
        )
        x_test_ohe = pd.DataFrame(
            ohe.transform(x_test_fe[low_card_cols]), columns=ohe_col_names, index=x_test_fe.index
        )

    freq_maps = {}
    x_train_freq = pd.DataFrame(index=x_train_fe.index)
    x_test_freq = pd.DataFrame(index=x_test_fe.index)

    for col in high_card_cols:
        freq_map = x_train_fe[col].value_counts(normalize=True)
        freq_maps[col] = freq_map
        x_train_freq[col + '_freq'] = x_train_fe[col].map(freq_map)
        x_test_freq[col + '_freq'] = x_test_fe[col].map(freq_map).fillna(0)

    # FIX: include engineered numeric features alongside original num_cols
    all_num_cols = num_cols + engineered_num_cols

    x_train_final = pd.concat([x_train_fe[all_num_cols], x_train_ohe, x_train_freq], axis=1)
    x_test_final = pd.concat([x_test_fe[all_num_cols], x_test_ohe, x_test_freq], axis=1)

    
    return x_train_final, x_test_final


def transform_and_scale(x_train_final, x_test_final, continuous_cols):
    """log1p then StandardScaler — proven winner (Optuna: AUC 0.8441)."""
    x_train_scaled = x_train_final.copy()
    x_test_scaled = x_test_final.copy()

    for col in continuous_cols:
        x_train_scaled[col] = np.log1p(np.maximum(0, x_train_scaled[col]))
        x_test_scaled[col] = np.log1p(np.maximum(0, x_test_scaled[col]))

    scaler = StandardScaler()
    x_train_scaled[continuous_cols] = scaler.fit_transform(x_train_scaled[continuous_cols])
    x_test_scaled[continuous_cols] = scaler.transform(x_test_scaled[continuous_cols])

    return x_train_scaled, x_test_scaled



import os


def save_processed_data(x_train, x_test, y_train, y_test, output_dir='../data/processed'):
    """
    Saves train/test features and the target together in ONE file each
    (not separate X and y files). This makes it impossible for X and y
    to end up misaligned later, which is a bug we ran into before.

    Parameters:
        x_train, x_test: the feature DataFrames (your input columns)
        y_train, y_test: the target Series (the is_canceled column)
        output_dir: folder where the CSV files will be saved
    """
    # Before saving, double check that x_train's rows and y_train's rows
    # are still lined up correctly (same row order, same row identity).
    # If they aren't, we STOP here instead of saving broken data.
    if not (x_train.index == y_train.index).all():
        raise ValueError("x_train and y_train indices don't match — refusing to save!")

    if not (x_test.index == y_test.index).all():
        raise ValueError("x_test and y_test indices don't match — refusing to save!")

    # Create the output folder if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # y_train.name is the column name of your target (e.g. 'is_canceled').
    # If for some reason it doesn't have a name, we default to 'target'.
    target_col = y_train.name if y_train.name else 'target'

    # Combine X and y into a single DataFrame before saving.
    # .values strips away the pandas index and just takes the raw numbers,
    # so they get added in the exact row order x_train is currently in —
    # this is what keeps everything lined up.
    train_combined = x_train.copy()
    train_combined[target_col] = y_train.values

    test_combined = x_test.copy()
    test_combined[target_col] = y_test.values

    train_path = os.path.join(output_dir, 'train_processed.csv')
    test_path = os.path.join(output_dir, 'test_processed.csv')

    train_combined.to_csv(train_path, index=False)
    test_combined.to_csv(test_path, index=False)

    print(f"Saved {train_path} — shape {train_combined.shape}")
    print(f"Saved {test_path} — shape {test_combined.shape}")

    return train_path, test_path


def load_processed_data(output_dir='../data/processed', target_col='is_canceled'):
    """
    Loads the combined train/test files back and splits them apart into
    X (features) and y (target) again. This is the matching "undo" function
    for save_processed_data().

    Parameters:
        output_dir: folder where the CSV files were saved
        target_col: the name of your target column (e.g. 'is_canceled')
    """
    train_combined = pd.read_csv(os.path.join(output_dir, 'train_processed.csv'))
    test_combined = pd.read_csv(os.path.join(output_dir, 'test_processed.csv'))

    # Pull the target column out as y, and drop it from the rest to get X.
    # Because both came from the SAME file, they can never be misaligned.
    y_train = train_combined[target_col]
    x_train = train_combined.drop(columns=[target_col])

    y_test = test_combined[target_col]
    x_test = test_combined.drop(columns=[target_col])

    return x_train, x_test, y_train, y_test


# ---- Proven feature selection result (from notebook consensus analysis) ----
SELECTED_FEATURES = [
   'lead_time', 'required_car_parking_spaces', 'previous_bookings_not_canceled', 
   'market_segment_Offline TA/TO', 'room_type_changed', 'country_freq', 'market_segment_Online TA', 
   'customer_type_Transient', 'deposit_type_Non Refund', 'total_of_special_requests', 'has_company', 
   'customer_type_Transient-Party', 'customer_type_Contract', 'adr_per_person', 'hotel_Resort Hotel', 'adr', 
   'distribution_channel_Direct', 'deposit_type_No Deposit', 'distribution_channel_TA/TO', 'previous_cancellations', 
   'booking_changes', 'market_segment_Corporate', 'meal_SC', 'distribution_channel_Corporate', 'market_segment_Direct',
    'reserved_room_type_G', 'reserved_room_type_A', 'assigned_room_type_G', 'assigned_room_type_A', 'stays_in_week_nights',
    'is_repeated_guest', 'agent', 'total_stay_nights', 'children', 'arrival_date_month_August', 'arrival_date_month_December', 'is_solo'
]

def select_features(x_train_scaled, x_test_scaled):
    """
    Selects the pre-determined feature set from consensus analysis
    (variance threshold + chi2 + mutual info + RFE, run once in the
    feature engineering notebook). Just a column selection — preserves
    the row index automatically, so it can't reintroduce alignment bugs.
    """
    selected = [c for c in SELECTED_FEATURES if c in x_train_scaled.columns]
    missing = [c for c in SELECTED_FEATURES if c not in x_train_scaled.columns]
    if missing:
        raise ValueError(f"Expected columns missing from x_train_scaled: {missing}")

    x_train_fs = x_train_scaled[selected]
    x_test_fs = x_test_scaled[selected]

    return x_train_fs, x_test_fs


def run_feature_pipeline(x_train, x_test):
    """
    Orchestrates the full proven pipeline: impute -> cap outliers -> construct features
    -> encode -> transform & scale. Returns final train/test DataFrames
    """
    num_cols, cat_cols = get_column_types(x_train)

    x_train_imp, x_test_imp = impute_missing_values(x_train, x_test, num_cols, cat_cols)
    x_train_capped, x_test_capped = cap_outliers(x_train_imp, x_test_imp)
    x_train_fe, x_test_fe = construct_features(x_train_capped, x_test_capped)

    engineered_num_cols = [
        'total_stay_nights', 'is_weekend_only', 'total_guests',
        'is_family', 'is_solo', 'adr_per_person', 'room_type_changed'
    ]

    x_train_final, x_test_final = encode_categoricals(
        x_train_fe, x_test_fe, num_cols, cat_cols, engineered_num_cols
    )

    continuous_cols = num_cols + engineered_num_cols + [c for c in x_train_final.columns if c.endswith('_freq')]
    continuous_cols = [c for c in continuous_cols if c in x_train_final.columns]

    x_train_scaled, x_test_scaled = transform_and_scale(x_train_final, x_test_final, continuous_cols)

    # Guard against silent train/test column drift (bit us before — worth always checking)
    if not list(x_train_scaled.columns) == list(x_test_scaled.columns):
        raise AssertionError("Train/test column mismatch after pipeline!")


    return x_train_scaled, x_test_scaled