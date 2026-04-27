import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.models.evaluate import evaluate_model


def train_logistic_regression(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series,
                               save_path: str = None) -> tuple:
    """
    Train a Logistic Regression model on WoE transformed features.

    WoE transformed features are already on a similar scale
    so StandardScaler is applied just to be safe.

    Returns the trained model and scaler.
    """

    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)

    # Scale the features
    # Even though WoE values are already normalized,
    # scaling helps logistic regression converge faster
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns,  index=X_test.index)

    # Train Logistic Regression
    # class_weight='balanced' handles the class imbalance automatically
    # C=0.1 is slight regularization to prevent overfitting
    # max_iter=1000 ensures convergence
    model = LogisticRegression(
        C            = 0.1,
        class_weight = 'balanced',
        max_iter     = 1000,
        random_state = 42,
        solver       = 'lbfgs'
    )

    model.fit(X_train_scaled, y_train)
    print("Logistic Regression trained successfully")

    # Evaluate the model
    metrics = evaluate_model(
        model      = model,
        X_train    = X_train_scaled,
        X_test     = X_test_scaled,
        y_train    = y_train,
        y_test     = y_test,
        threshold  = 0.5,
        model_name = "Logistic Regression"
    )

    # Save model and scaler if save path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(model,  os.path.join(save_path, 'logistic_model.pkl'))
        joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
        print(f"\nModel saved to {save_path}/logistic_model.pkl")
        print(f"Scaler saved to {save_path}/scaler.pkl")

    return model, scaler, metrics, X_train_scaled, X_test_scaled


def train_xgboost(X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series) -> tuple:
    """
    Train an XGBoost model for comparison with Logistic Regression.
    XGBoost is not used for the final scorecard but gives us
    an upper bound on what is achievable with this dataset.
    """

    print("\n" + "="*60)
    print("TRAINING XGBOOST (for comparison)")
    print("="*60)

    # Calculate scale_pos_weight to handle class imbalance
    # scale_pos_weight = number of negatives / number of positives
    n_negatives = (y_train == 0).sum()
    n_positives = (y_train == 1).sum()
    scale_pos_weight = n_negatives / n_positives

    print(f"scale_pos_weight set to : {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        random_state     = 42,
        eval_metric      = 'auc',
        verbosity        = 0
    )

    model.fit(X_train, y_train)
    print("XGBoost trained successfully")

    metrics = evaluate_model(
        model      = model,
        X_train    = X_train,
        X_test     = X_test,
        y_train    = y_train,
        y_test     = y_test,
        threshold  = 0.5,
        model_name = "XGBoost"
    )

    return model, metrics


def train_lightgbm(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series) -> tuple:
    """
    Train a LightGBM model for comparison with Logistic Regression.
    LightGBM is faster than XGBoost and often performs similarly.
    """

    print("\n" + "="*60)
    print("TRAINING LIGHTGBM (for comparison)")
    print("="*60)

    # Calculate class weight for imbalance
    n_negatives = (y_train == 0).sum()
    n_positives = (y_train == 1).sum()
    scale_pos_weight = n_negatives / n_positives

    model = LGBMClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        random_state     = 42,
        verbosity        = -1
    )

    model.fit(X_train, y_train)
    print("LightGBM trained successfully")

    metrics = evaluate_model(
        model      = model,
        X_train    = X_train,
        X_test     = X_test,
        y_train    = y_train,
        y_test     = y_test,
        threshold  = 0.5,
        model_name = "LightGBM"
    )

    return model, metrics


def log_to_mlflow(model_name: str,
                  params: dict,
                  metrics: dict,
                  model,
                  experiment_name: str = "credit_scoring"):
    """
    Log model parameters, metrics, and the model itself to MLflow.
    MLflow keeps track of all experiments so we can compare runs.
    """

    # Set the experiment name
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log the model itself
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        print(f"\nLogged to MLflow — experiment: {experiment_name}, run: {model_name}")