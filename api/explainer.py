# ============================================================
# explainer.py
# Generates a human readable explanation for a single
# applicant's credit score prediction.
#
# This is separate from predictor.py because explanation
# logic is independent of prediction logic.
# The API can call this separately if only an explanation
# is needed without rerunning the full prediction.
# ============================================================

import pandas as pd


def generate_plain_english_explanation(prediction_result: dict) -> str:
    """
    Convert the raw prediction result into a plain English
    paragraph that can be shown to the applicant or a loan officer.

    This is important for regulatory compliance — banks are
    required to explain why a loan was approved or rejected.

    prediction_result : the dictionary returned by run_prediction()
    """

    score       = prediction_result['credit_score']
    risk        = prediction_result['risk_category']
    prob        = prediction_result['default_probability']
    strengths   = prediction_result['top_strengths']
    risk_factors = prediction_result['top_risk_factors']

    # Opening sentence based on risk category
    if risk == 'Excellent':
        opening = (f"This applicant has an excellent credit profile with a score of {score}. "
                   f"The estimated probability of default is very low at {prob*100:.1f}%.")
    elif risk == 'Good':
        opening = (f"This applicant has a good credit profile with a score of {score}. "
                   f"The estimated probability of default is low at {prob*100:.1f}%.")
    elif risk == 'Fair':
        opening = (f"This applicant has a fair credit profile with a score of {score}. "
                   f"The estimated probability of default is moderate at {prob*100:.1f}%.")
    elif risk == 'Poor':
        opening = (f"This applicant has a poor credit profile with a score of {score}. "
                   f"The estimated probability of default is elevated at {prob*100:.1f}%.")
    else:
        opening = (f"This applicant has a very poor credit profile with a score of {score}. "
                   f"The estimated probability of default is high at {prob*100:.1f}%.")

    # Strengths sentence
    if strengths:
        strengths_text = ", ".join(strengths)
        strengths_sentence = (f"Factors working in the applicant's favor include: "
                               f"{strengths_text}.")
    else:
        strengths_sentence = "No significant positive factors were identified."

    # Risk factors sentence
    if risk_factors:
        risk_text = ", ".join(risk_factors)
        risk_sentence = (f"Factors that increase the credit risk include: "
                          f"{risk_text}.")
    else:
        risk_sentence = "No significant risk factors were identified."

    # Recommendation sentence
    if risk in ['Excellent', 'Good']:
        recommendation = "Recommendation: Approve the loan application."
    elif risk == 'Fair':
        recommendation = "Recommendation: Approve with standard terms and monitoring."
    elif risk == 'Poor':
        recommendation = "Recommendation: Consider with higher interest rate or collateral."
    else:
        recommendation = "Recommendation: Decline or require significant collateral."

    # Combine all sentences into one explanation
    explanation = (f"{opening} {strengths_sentence} "
                   f"{risk_sentence} {recommendation}")

    return explanation


def generate_score_summary(prediction_result: dict) -> dict:
    """
    Generate a concise summary of the prediction result
    that is easy to display in a dashboard or mobile app.
    """

    score = prediction_result['credit_score']
    risk  = prediction_result['risk_category']
    prob  = prediction_result['default_probability']

    # Score percentile approximation based on our distribution
    # Min=365, Max=833, Mean=604
    score_percentile = min(99, max(1, int((score - 365) / (833 - 365) * 100)))

    return {
        'credit_score'      : score,
        'risk_category'     : risk,
        'default_probability' : prob,
        'score_percentile'  : score_percentile,
        'plain_explanation' : generate_plain_english_explanation(prediction_result),
        'top_risk_factors'  : prediction_result['top_risk_factors'],
        'top_strengths'     : prediction_result['top_strengths']
    }