import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib


class SHAPExplainer:
    """
    SHAP Explainer wrapper for the credit scoring model.

    Generates both global explanations (which features matter most
    overall) and local explanations (why did THIS applicant get
    this score).

    Usage:
        explainer = SHAPExplainer(model, scaler, feature_names)
        explainer.fit(X_train_woe)
        shap_values = explainer.get_shap_values(X_test_woe)
        explainer.plot_global_importance()
        explainer.explain_single_applicant(X_single)
    """

    def __init__(self,
                 model,
                 scaler,
                 feature_names: list):
        """
        model         : fitted Logistic Regression model
        scaler        : fitted StandardScaler
        feature_names : list of feature names in the same order
                        as the columns in X_woe
        """
        self.model         = model
        self.scaler        = scaler
        self.feature_names = feature_names
        self.explainer     = None
        self.is_fitted     = False


    def fit(self, X_train_woe: pd.DataFrame):
        """
        Initialize the SHAP explainer using training data.

        We use shap.LinearExplainer because our model is
        Logistic Regression which is a linear model.
        LinearExplainer is exact and fast for linear models —
        no approximation needed.

        X_train_woe is used to calculate the background
        distribution (the average prediction baseline).
        """

        print("Fitting SHAP explainer...")

        # Scale the training data the same way as during model training
        X_train_scaled = self.scaler.transform(X_train_woe)

        # Convert to DataFrame to keep feature names
        X_train_scaled_df = pd.DataFrame(
            X_train_scaled,
            columns = self.feature_names
        )

        # Initialize LinearExplainer with the model and training data
        # The training data is used as the background dataset
        # to calculate the expected value (average prediction)
        self.explainer = shap.LinearExplainer(
            self.model,
            X_train_scaled_df
        )

        self.is_fitted = True
        print("SHAP explainer fitted successfully")
        print(f"Expected value (base rate) : {self.explainer.expected_value:.4f}")

        return self


    def get_shap_values(self, X_woe: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for a set of applicants.

        SHAP value for feature i of applicant j tells us:
        How much did feature i push the prediction of applicant j
        away from the average prediction?

        Positive SHAP = pushed prediction higher (more risky)
        Negative SHAP = pushed prediction lower (less risky)

        Returns a numpy array of shape (n_applicants, n_features)
        """

        if not self.is_fitted:
            raise Exception("Call fit() before get_shap_values().")

        # Scale the input data
        X_scaled = self.scaler.transform(X_woe)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_scaled_df)

        return shap_values


    def plot_global_importance(self,
                                X_woe: pd.DataFrame,
                                save_path: str = None):
        """
        Plot global feature importance using SHAP values.

        Global importance = mean absolute SHAP value across all applicants.
        This tells us which features matter most on average.

        Two plots are shown:
        1. Bar plot of mean absolute SHAP values
        2. Beeswarm plot showing distribution of SHAP values per feature
        """

        if not self.is_fitted:
            raise Exception("Call fit() first.")

        shap_values = self.get_shap_values(X_woe)

        X_scaled = self.scaler.transform(X_woe)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        # --- Plot 1 — Bar plot of mean absolute SHAP values ---
        mean_abs_shap = pd.DataFrame({
            'Feature'          : self.feature_names,
            'Mean_Abs_SHAP'    : np.abs(shap_values).mean(axis=0)
        }).sort_values('Mean_Abs_SHAP', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.barh(mean_abs_shap['Feature'],
                 mean_abs_shap['Mean_Abs_SHAP'],
                 color='steelblue', edgecolor='black')
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title('Global Feature Importance — Mean Absolute SHAP Values',
                  fontsize=13, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path.replace('.png', '_bar.png'), bbox_inches='tight')
            print(f"Bar plot saved to {save_path.replace('.png', '_bar.png')}")
        plt.show()

        # --- Plot 2 — Beeswarm plot ---
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_scaled_df,
            feature_names = self.feature_names,
            show          = False,
            plot_type     = 'dot'
        )
        plt.title('SHAP Beeswarm Plot — Feature Impact Distribution',
                  fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path.replace('.png', '_beeswarm.png'), bbox_inches='tight')
            print(f"Beeswarm plot saved to {save_path.replace('.png', '_beeswarm.png')}")
        plt.show()


    def explain_single_applicant(self,
                                  X_single: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a plain-English SHAP explanation for one applicant.

        Returns a DataFrame showing each feature, its value,
        its SHAP value, and a plain-English interpretation.

        This is what gets returned by the API for each prediction.
        """

        if not self.is_fitted:
            raise Exception("Call fit() first.")

        # Get SHAP values for this single applicant
        shap_vals = self.get_shap_values(X_single)[0]

        explanation_rows = []

        for i, feature in enumerate(self.feature_names):

            woe_value  = X_single.iloc[0][feature]
            shap_value = shap_vals[i]

            # Generate plain English interpretation
            if shap_value > 0.05:
                direction = "increases default risk"
            elif shap_value < -0.05:
                direction = "decreases default risk"
            else:
                direction = "neutral impact"

            explanation_rows.append({
                'Feature'      : feature,
                'WoE_Value'    : round(woe_value, 4),
                'SHAP_Value'   : round(shap_value, 4),
                'Impact'       : direction
            })

        explanation_df = pd.DataFrame(explanation_rows)
        explanation_df = explanation_df.sort_values(
            'SHAP_Value', ascending=False, key=abs
        ).reset_index(drop=True)

        return explanation_df


    def save(self, path: str):
        """Save the SHAP explainer to disk."""
        joblib.dump(self, path)
        print(f"SHAP explainer saved to {path}")


    @staticmethod
    def load(path: str):
        """Load a saved SHAP explainer from disk."""
        explainer = joblib.load(path)
        print(f"SHAP explainer loaded from {path}")
        return explainer