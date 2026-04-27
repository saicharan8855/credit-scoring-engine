import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)


def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Gini coefficient.
    Gini = 2 * AUC - 1
    Higher is better. Above 0.4 is acceptable for a bank scorecard.
    """
    auc  = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    return gini


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate KS (Kolmogorov-Smirnov) statistic.

    KS measures the maximum difference between the cumulative
    distribution of defaulters and non-defaulters across all
    probability thresholds.

    Higher is better. Above 30 is considered good for credit models.
    """

    # Sort by predicted probability descending
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)

    # Total number of defaulters and non-defaulters
    total_defaulters     = (df['y_true'] == 1).sum()
    total_non_defaulters = (df['y_true'] == 0).sum()

    # Calculate cumulative percentage of defaulters and non-defaulters
    # at each threshold
    df['cum_defaulters']     = (df['y_true'] == 1).cumsum() / total_defaulters
    df['cum_non_defaulters'] = (df['y_true'] == 0).cumsum() / total_non_defaulters

    # KS is the maximum difference between the two cumulative distributions
    df['ks_diff'] = abs(df['cum_defaulters'] - df['cum_non_defaulters'])
    ks_statistic  = df['ks_diff'].max() * 100  # expressed as percentage

    return ks_statistic


def calculate_psi(expected: np.ndarray,
                  actual: np.ndarray,
                  bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI measures whether the distribution of predicted probabilities
    has shifted between the training set (expected) and test set (actual).

    PSI < 0.1  — Stable, no action needed
    PSI 0.1-0.25 — Slight shift, monitor closely
    PSI > 0.25 — Significant shift, model needs retraining
    """

    # Create bins based on the expected (train) distribution
    breakpoints = np.linspace(0, 1, bins + 1)
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    # Count observations in each bin
    expected_counts = pd.cut(expected, bins=breakpoints).value_counts().sort_index()
    actual_counts   = pd.cut(actual,   bins=breakpoints).value_counts().sort_index()

    # Convert to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct   = actual_counts   / len(actual)

    # Replace zeros to avoid log(0)
    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct   = actual_pct.replace(0, 0.0001)

    # PSI formula
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi        = psi_values.sum()

    return psi


def evaluate_model(model,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   threshold: float = 0.5,
                   model_name: str = "Model") -> dict:
    """
    Run full evaluation of a trained model.
    Returns a dictionary with all metrics.
    Prints a formatted summary report.
    """

    # Get predicted probabilities
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs  = model.predict_proba(X_test)[:, 1]

    # Get predicted classes using the threshold
    train_preds = (train_probs >= threshold).astype(int)
    test_preds  = (test_probs  >= threshold).astype(int)

    # Calculate all metrics
    train_gini = calculate_gini(y_train, train_probs)
    test_gini  = calculate_gini(y_test,  test_probs)

    train_ks   = calculate_ks_statistic(y_train, train_probs)
    test_ks    = calculate_ks_statistic(y_test,  test_probs)

    train_auc  = roc_auc_score(y_train, train_probs)
    test_auc   = roc_auc_score(y_test,  test_probs)

    test_mcc   = matthews_corrcoef(y_test, test_preds)
    test_psi   = calculate_psi(train_probs, test_probs)

    # Print formatted report
    print("=" * 60)
    print(f"EVALUATION REPORT — {model_name}")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Train':>10} {'Test':>10}")
    print("-" * 50)
    print(f"{'AUC':<30} {train_auc:>10.4f} {test_auc:>10.4f}")
    print(f"{'Gini Coefficient':<30} {train_gini:>10.4f} {test_gini:>10.4f}")
    print(f"{'KS Statistic':<30} {train_ks:>10.2f} {test_ks:>10.2f}")
    print(f"\n{'MCC (Test only)':<30} {test_mcc:>10.4f}")
    print(f"{'PSI (Train vs Test)':<30} {test_psi:>10.4f}")

    # PSI interpretation
    if test_psi < 0.1:
        psi_status = "Stable"
    elif test_psi < 0.25:
        psi_status = "Slight shift — monitor"
    else:
        psi_status = "Significant shift — retrain"
    print(f"{'PSI Status':<30} {psi_status:>10}")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    print(f"\nConfusion Matrix (Test Set, threshold={threshold}):")
    print(f"  True Negatives  (Repaid, predicted Repaid)    : {cm[0][0]:,}")
    print(f"  False Positives (Repaid, predicted Default)   : {cm[0][1]:,}")
    print(f"  False Negatives (Default, predicted Repaid)   : {cm[1][0]:,}")
    print(f"  True Positives  (Default, predicted Default)  : {cm[1][1]:,}")

    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=['Repaid', 'Default']))

    # Return all metrics as a dictionary for MLflow logging
    metrics = {
        'train_auc'  : train_auc,
        'test_auc'   : test_auc,
        'train_gini' : train_gini,
        'test_gini'  : test_gini,
        'train_ks'   : train_ks,
        'test_ks'    : test_ks,
        'test_mcc'   : test_mcc,
        'test_psi'   : test_psi
    }

    return metrics


def plot_roc_curve(model,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   model_name: str = "Model",
                   save_path: str = None):
    """
    Plot the ROC curve for a trained model.
    """

    test_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc = roc_auc_score(y_test, test_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', linewidth=2,
             label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")

    plt.show()