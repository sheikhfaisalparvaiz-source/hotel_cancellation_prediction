import xgboost as xgb


def train_xgboost(X_train, y_train):
    """
    Trains the XGBoost model using the winning Optuna hyperparameters.
    """
    # Exact best parameters from your Optuna run
    best_params = {
        'n_estimators': 341,
        'max_depth': 9,
        'learning_rate': 0.04962533902377326,
        'subsample': 0.8473942417329839,
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    
    print("Training winning XGBoost model...")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model