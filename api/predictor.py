# ============================================================
# predictor.py
# Loads all saved models and runs the full inference pipeline
# for a single applicant.
#
# The pipeline order is:
# 1. Convert request to DataFrame
# 2. Run preprocessing (fix anomalies, create features etc.)
# 3. Apply WoE encoding
# 4. Select the 30 IV-selected features
# 5. Calculate credit score using scorecard
# 6. Get default probability from logistic regression
# 7. Get scorecard breakdown
# 8. Return all results
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os

from src.data.preprocessing import (
    fix_anomalies,
    create_new_features,
    create_missing_indicators,
    drop_high_missing_columns,
    winsorize_outliers
)


# ============================================================
# Model loading
# All models are loaded once when the API starts up
# Not on every request — that would be very slow
# ============================================================

# Get the absolute path to the models directory
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_all_models():
    """
    Load all saved model artifacts from the models directory.
    Returns a dictionary with all loaded objects.
    """
    print("Loading models...")

    models = {
        'woe_encoder'    : joblib.load(os.path.join(MODELS_DIR, 'woe_encoder.pkl')),
        'iv_selector'    : joblib.load(os.path.join(MODELS_DIR, 'iv_selector.pkl')),
        'logistic_model' : joblib.load(os.path.join(MODELS_DIR, 'logistic_model.pkl')),
        'scaler'         : joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl')),
        'scorecard'      : joblib.load(os.path.join(MODELS_DIR, 'scorecard.pkl')),
        'shap_explainer' : joblib.load(os.path.join(MODELS_DIR, 'shap_explainer.pkl'))
    }

    print("All models loaded successfully")
    return models


# ============================================================
# Preprocessing for a single applicant
# This mirrors the training pipeline but adapted for
# a single row instead of the full dataset
# ============================================================

def preprocess_single_applicant(applicant_df: pd.DataFrame,
                                  models: dict) -> pd.DataFrame:
    """
    Run preprocessing on a single applicant's data.

    Steps:
    1. Fix anomalies
    2. Create new features
    3. Create missing indicators
    4. Fill missing values with sensible defaults
    5. Apply WoE encoding on selected features
    """

    df = applicant_df.copy()

    # Step 1 — Fix anomalies
    df = fix_anomalies(df)

    # Step 2 — Create new features
    df = create_new_features(df)

    # Step 3 — Create missing indicators
    df = create_missing_indicators(df)

    # Step 4 — Fill any remaining missing values with 0
    df = df.fillna(0)

    # Step 5 — Get the selected features list
    selected_features = models['iv_selector'].get_selected_features()

    # Step 6 — Keep only columns that exist in df and are in selected features
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0

    df_selected = df[selected_features]

    # Step 7 — Apply WoE encoding
    # For a single row pd.cut fails because min == max
    # We use the stored WoE tables directly instead of pd.cut
    woe_encoder = models['woe_encoder']
    result = {}

    for col in df_selected.columns:
        if col not in woe_encoder.woe_tables:
            result[col] = 0.0
            continue

        woe_table    = woe_encoder.woe_tables[col]
        feature_type = woe_encoder.feature_types[col]
        value        = df_selected[col].values[0]

        if feature_type == 'numerical':
            # Find which bin this value falls into
            edges = woe_encoder.bin_edges[col].copy()

            matched_woe = None
            for i in range(len(edges) - 1):
                lower = edges[i]
                upper = edges[i + 1]
                # First bin is inclusive on left
                if i == 0:
                    if value <= upper:
                        matched_woe = woe_table.iloc[i]['WoE']
                        break
                else:
                    if lower < value <= upper:
                        matched_woe = woe_table.iloc[i]['WoE'] if i < len(woe_table) else woe_table.iloc[-1]['WoE']
                        break

            if matched_woe is None:
                matched_woe = woe_table['WoE'].median()

            result[col] = matched_woe

        else:
            # Categorical — direct lookup
            woe_map     = dict(zip(woe_table['bin'].astype(str), woe_table['WoE']))
            matched_woe = woe_map.get(str(value), 0.0)
            result[col] = matched_woe

    df_woe = pd.DataFrame([result])

    return df_woe


# ============================================================
# Main prediction function
# This is what the API endpoint calls
# ============================================================

def run_prediction(applicant_data: dict, models: dict) -> dict:
    """
    Run the full prediction pipeline for one applicant.

    applicant_data : dictionary from the API request
    models         : dictionary of loaded model objects

    Returns a dictionary with all prediction results.
    """

    # Convert the request dictionary to a single row DataFrame
    applicant_df = pd.DataFrame([applicant_data])

    # Run preprocessing and WoE encoding
    applicant_woe = preprocess_single_applicant(applicant_df, models)

    # Get default probability from logistic regression
    X_scaled = models['scaler'].transform(applicant_woe)
    default_prob = models['logistic_model'].predict_proba(X_scaled)[0][1]

    # Calculate credit score
    credit_score = models['scorecard'].calculate_score(applicant_woe)[0]

    # Get risk category
    risk_category = models['scorecard'].get_risk_category(credit_score)

    # Get scorecard breakdown
    breakdown_df = models['scorecard'].get_score_breakdown(
        applicant_woe,
        feature_names = applicant_woe.columns.tolist()
    )

    # Convert breakdown to list of dicts for JSON response
    scorecard_breakdown = []
    for _, row in breakdown_df.iterrows():
        scorecard_breakdown.append({
            'feature'     : str(row['Feature']),
            'woe_value'   : float(row['WoE Value']) if row['WoE Value'] is not None else None,
            'coefficient' : float(row['Coefficient']),
            'points'      : float(row['Points'])
        })

    # Get SHAP explanation
    shap_explanation_df = models['shap_explainer'].explain_single_applicant(applicant_woe)

    # Convert SHAP explanation to list of dicts
    shap_explanation = []
    for _, row in shap_explanation_df.iterrows():
        shap_explanation.append({
            'feature'    : str(row['Feature']),
            'woe_value'  : float(row['WoE_Value']),
            'shap_value' : float(row['SHAP_Value']),
            'impact'     : str(row['Impact'])
        })

    # Get top 3 risk factors (highest positive SHAP values)
    top_risk_factors = [
        row['feature'] for row in shap_explanation
        if row['shap_value'] > 0
    ][:3]

    # Get top 3 strengths (most negative SHAP values)
    top_strengths = [
        row['feature'] for row in reversed(shap_explanation)
        if row['shap_value'] < 0
    ][:3]

    return {
        'credit_score'        : int(credit_score),
        'risk_category'       : risk_category,
        'default_probability' : round(float(default_prob), 4),
        'scorecard_breakdown' : scorecard_breakdown,
        'shap_explanation'    : shap_explanation,
        'top_risk_factors'    : top_risk_factors,
        'top_strengths'       : top_strengths
    }