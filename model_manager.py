import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, f1_score
import joblib
import os
import logging
from config import MODEL_DIR, PARAM_GRID, VERBOSE_MODE
from feature_processor import prepare_features_for_ai

def train_and_save_model(df: pd.DataFrame, ticker_name: str) -> tuple[RandomForestClassifier, list] | tuple[None, None]:
    """
    Trains a RandomForestClassifier model using GridSearchCV and saves it along with features.
    Uses TimeSeriesSplit for more realistic cross-validation.
    """
    if df is None or df.empty:
        logging.warning(f"[Model Training] DataFrame is empty or None for {ticker_name}. Cannot train model.")
        return None, None

    X, y, features = prepare_features_for_ai(df, for_training=True)

    if X.empty or y is None or len(X) < 10 or y.nunique() < 2:
        logging.warning(f"[Model Training] Not enough valid samples or distinct target labels ({y.nunique() if y is not None else 0}) after feature preparation for meaningful training for {ticker_name}. Training aborted.")
        if VERBOSE_MODE and y is not None:
            logging.info(f"  Target counts: {y.value_counts()}")
        return None, None

    tscv = TimeSeriesSplit(n_splits=5) 
    scorer = make_scorer(f1_score, average='weighted', zero_division=0)

    logging.info(f"\n--- Starting GridSearchCV for Hyperparameter Tuning for {ticker_name} ---")
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=PARAM_GRID,
        scoring=scorer,
        cv=tscv,
        verbose=1,
        n_jobs=-1
    )

    try:
        grid_search.fit(X, y)
    except Exception as e:
        logging.error(f"[Model Training Error] GridSearchCV failed for {ticker_name}: {e}. Skipping model saving for this ticker.")
        return None, None

    logging.info("\n--- GridSearchCV Results ---")
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    logging.info("Model training complete (Best model from Grid Search selected).")

    final_train_idx, final_test_idx = list(tscv.split(X, y))[-1]
    X_train_final, X_test_final = X.iloc[final_train_idx], X.iloc[final_test_idx]
    y_train_final, y_test_final = y.iloc[final_train_idx], y.iloc[final_test_idx]

    if VERBOSE_MODE:
        logging.info(f"\n[Model Training] Final training data shape: {X_train_final.shape}, Final test data shape: {X_test_final.shape}")
        logging.info(f"[Model Training] Final training target distribution:\n{y_train_final.value_counts()}")
        logging.info(f"[Model Training] Final test target distribution:\n{y_test_final.value_counts()}")

    if not X_test_final.empty:
        y_pred = best_model.predict(X_test_final)

        logging.info("\n--- Model Evaluation on Final Test Set (using Best Model) ---")
        logging.info(f"Overall Accuracy: {accuracy_score(y_test_final, y_pred):.4f}")
        logging.info("\nClassification Report (0=Sell, 1=Buy, 2=Hold):")
        target_names_map = {0: 'Sell', 1: 'Buy', 2: 'Hold'}
        all_possible_labels = sorted(y.unique())
        display_target_names = [target_names_map.get(label, f'Label {label}') for label in all_possible_labels]

        logging.info("\n" + classification_report(y_test_final, y_pred, labels=all_possible_labels, target_names=display_target_names, zero_division=0))
        logging.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_test_final, y_pred, labels=all_possible_labels)))
    else:
        logging.warning("\n--- Model Evaluation: Final test set is empty. No evaluation performed. ---")

    if VERBOSE_MODE:
        logging.info(f"\n--- Top 15 Feature Importances for {ticker_name} (Best Model) ---")
        if hasattr(best_model, 'feature_importances_') and len(features) > 0:
            feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            with pd.option_context('display.float_format', '{:.6f}'.format):
                logging.info("\n" + feature_importances.head(15).to_string())
        else:
            logging.info("Feature importances not available or no features.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_filename = os.path.join(MODEL_DIR, f"trading_bot_model_{ticker_name.replace('^', '')}.joblib")
    features_filename = os.path.join(MODEL_DIR, f"trading_bot_model_features_{ticker_name.replace('^', '')}.joblib")

    joblib.dump(best_model, model_filename)
    joblib.dump(features, features_filename)
    logging.info(f"\nModel saved to {model_filename}")
    logging.info(f"Features saved to {features_filename}")

    return best_model, features

def load_models_and_features() -> dict:
    """Loads all trained models and their associated features from the MODEL_DIR."""
    loaded_data = {}
    logging.info(f"Loading models from {MODEL_DIR}...")
    
    from config import STOCK_TICKERS 
    for ticker in STOCK_TICKERS:
        clean_ticker_name = ticker.replace('^', '')
        model_path = os.path.join(MODEL_DIR, f"trading_bot_model_{clean_ticker_name}.joblib")
        features_path = os.path.join(MODEL_DIR, f"trading_bot_model_features_{clean_ticker_name}.joblib")

        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                model = joblib.load(model_path)
                features = joblib.load(features_path)
                loaded_data[ticker] = {'model': model, 'features': features}
                if VERBOSE_MODE:
                    logging.info(f"  - Loaded model and features for {ticker}")
            except Exception as e:
                logging.error(f"  - [Model Load Error] Error loading model/features for {ticker}: {e}")
        else:
            logging.warning(f"  - [Model Load Warning] No trained model found for {ticker} at '{model_path}'. Skipping.")
    return loaded_data

def generate_ai_signal(df: pd.DataFrame, model_info: dict, ticker_name: str, fundamental_data: dict, news_sentiment: dict, chart_patterns: dict) -> str:
    """
    Generates an AI trading signal (BUY, SELL, Hold) for the latest data point,
    incorporating fundamental data, news sentiment, and chart patterns.
    """
    model = model_info.get('model')
    known_features = model_info.get('features')

    if model is None or known_features is None or not known_features:
        if VERBOSE_MODE:
            logging.warning(f"  [AI Signal {ticker_name}] Model or features not loaded/defined. Cannot predict.")
        return 'Hold (AI Model Error)'

    if df.empty or len(df) == 0:
        if VERBOSE_MODE:
            logging.warning(f"  [AI Signal {ticker_name}] Empty DataFrame for prediction. Cannot generate AI signal.")
        return 'Hold (No Data)'

    latest_data_row = df.iloc[[-1]].copy()

    X_pred, _, _ = prepare_features_for_ai(
        latest_data_row,
        fundamental_data=fundamental_data,
        news_sentiment=news_sentiment,
        for_training=False,
        known_features=known_features
    )

    if X_pred.empty or X_pred.shape[1] != len(known_features):
        logging.error(f"  [AI Signal {ticker_name}] Mismatch in feature count for prediction. Expected {len(known_features)}, got {X_pred.shape[1]}. Cannot predict.")
        logging.error(f"  [AI Signal {ticker_name}] Missing features: {set(known_features) - set(X_pred.columns)}")
        logging.error(f"  [AI Signal {ticker_name}] Extra features: {set(X_pred.columns) - set(known_features)}")
        return 'Hold (Feature Mismatch)'

    try:
        prediction = model.predict(X_pred)[0]

        if prediction == 1:
            return 'BUY (AI)'
        elif prediction == 0:
            return 'SELL (AI)'
        else:
            return 'Hold (AI)'
    except Exception as e:
        logging.error(f"  [AI Signal Error {ticker_name}] Unhandled error during AI prediction: {e}")
        return 'Hold (AI Prediction Error)'

