import pandas as pd
import numpy as np
import joblib


class CreditScorecard:
    """
    Converts Logistic Regression probability output into
    a credit score scaled between 300 and 900.

    Industry standard scaling parameters:
    - Target score  : 720 (score at target odds)
    - Target odds   : 50  (50 good for every 1 bad at target score)
    - PDO           : 20  (score points to double the odds)
    - Score range   : 300 to 900

    Usage:
        scorecard = CreditScorecard(model, scaler)
        score     = scorecard.calculate_score(X_woe)
        breakdown = scorecard.get_score_breakdown(X_woe)
    """

    def __init__(self,
                 model,
                 scaler,
                 target_score : int   = 720,
                 target_odds  : float = 50.0,
                 pdo          : int   = 20,
                 min_score    : int   = 300,
                 max_score    : int   = 900):
        """
        model        : fitted Logistic Regression model
        scaler       : fitted StandardScaler used during training
        target_score : score assigned at target_odds (default 720)
        target_odds  : good to bad odds ratio at target score (default 50)
        pdo          : points to double the odds (default 20)
        min_score    : minimum possible score (default 300)
        max_score    : maximum possible score (default 900)
        """

        self.model        = model
        self.scaler       = scaler
        self.target_score = target_score
        self.target_odds  = target_odds
        self.pdo          = pdo
        self.min_score    = min_score
        self.max_score    = max_score

        # Calculate scaling factor and offset using anchor points
        # Factor = PDO / ln(2)
        # Offset = Target_Score - Factor * ln(Target_Odds)
        self.factor = pdo / np.log(2)
        self.offset = target_score - self.factor * np.log(target_odds)

        print(f"Scorecard initialized with:")
        print(f"  Target Score : {target_score}")
        print(f"  Target Odds  : {target_odds}")
        print(f"  PDO          : {pdo}")
        print(f"  Factor       : {self.factor:.4f}")
        print(f"  Offset       : {self.offset:.4f}")
        print(f"  Score Range  : {min_score} to {max_score}")


    def _get_log_odds(self, X_woe: pd.DataFrame) -> np.ndarray:
        """
        Calculate the log-odds for each applicant.

        Log-odds = intercept + sum of (coefficient * scaled WoE value)

        This is what Logistic Regression computes internally.
        We extract it here so we can scale it to a score.
        """

        # Scale the WoE features using the same scaler used during training
        X_scaled = self.scaler.transform(X_woe)

        # Log-odds = X * coefficients + intercept
        # This is the raw output of logistic regression before sigmoid
        log_odds = X_scaled.dot(self.model.coef_[0]) + self.model.intercept_[0]

        return log_odds


    def calculate_score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """
        Calculate the final credit score for each applicant.

        Steps:
        1. Get log-odds from the model
        2. Convert log-odds to score using scaling formula
        3. Flip the sign — higher log-odds of defaulting = lower score
        4. Clip score to min_score and max_score range

        Score = Offset + Factor * (-log_odds)
        We negate log_odds because:
        - Higher log_odds means higher probability of DEFAULT
        - Higher score should mean LOWER risk (safer borrower)
        """

        log_odds = self._get_log_odds(X_woe)

        # Negate because higher log-odds = higher default risk = lower score
        score = self.offset + self.factor * (-log_odds)

        # Clip to valid score range
        score = np.clip(score, self.min_score, self.max_score)

        return score.astype(int)


    def get_risk_category(self, score: int) -> str:
        """
        Convert a numeric score into a risk category label.

        These thresholds are based on standard industry practice:
        800+ : Excellent  — very low risk, best interest rates
        720+ : Good       — low risk, favorable terms
        650+ : Fair       — moderate risk, standard terms
        580+ : Poor       — high risk, limited credit
        580-  : Very Poor — very high risk, likely rejection
        """

        if score >= 800:
            return "Excellent"
        elif score >= 720:
            return "Good"
        elif score >= 650:
            return "Fair"
        elif score >= 580:
            return "Poor"
        else:
            return "Very Poor"


    def get_score_breakdown(self,
                             X_woe: pd.DataFrame,
                             feature_names: list = None) -> pd.DataFrame:
        """
        Calculate how much each feature contributes to the final score.

        Each feature contributes:
        Points = Factor * (-coefficient * scaled_WoE_value)

        This is the per-feature scorecard breakdown that makes
        the credit score explainable to the applicant and auditors.

        Returns a DataFrame with feature name, WoE value,
        coefficient, and points contributed for each feature.
        """

        # Scale the features
        X_scaled = self.scaler.transform(X_woe)

        if feature_names is None:
            feature_names = X_woe.columns.tolist()

        results = []

        for i, feature in enumerate(feature_names):

            # Raw WoE value for this feature
            woe_value = X_woe.iloc[0][feature]

            # Scaled WoE value
            scaled_value = X_scaled[0][i]

            # Coefficient from the logistic regression model
            coefficient = self.model.coef_[0][i]

            # Points contributed by this feature
            # Negative because higher log-odds = lower score
            points = self.factor * (-coefficient * scaled_value)

            results.append({
                'Feature'     : feature,
                'WoE Value'   : round(woe_value, 4),
                'Coefficient' : round(coefficient, 4),
                'Points'      : round(points, 2)
            })

        # Add the intercept contribution
        intercept_points = self.factor * (-self.model.intercept_[0]) + self.offset
        results.append({
            'Feature'     : 'Base Score (Intercept)',
            'WoE Value'   : None,
            'Coefficient' : round(self.model.intercept_[0], 4),
            'Points'      : round(intercept_points, 2)
        })

        breakdown_df = pd.DataFrame(results)
        breakdown_df = breakdown_df.sort_values('Points', ascending=False).reset_index(drop=True)

        return breakdown_df


    def score_dataframe(self, X_woe: pd.DataFrame) -> pd.DataFrame:
        """
        Score an entire dataframe of applicants.
        Returns a dataframe with score and risk category for each applicant.
        """

        scores = self.calculate_score(X_woe)

        result = pd.DataFrame({
            'Score'         : scores,
            'Risk_Category' : [self.get_risk_category(s) for s in scores]
        }, index=X_woe.index)

        return result


    def save(self, path: str):
        """Save the scorecard object to disk."""
        joblib.dump(self, path)
        print(f"Scorecard saved to {path}")


    @staticmethod
    def load(path: str):
        """Load a saved scorecard object from disk."""
        scorecard = joblib.load(path)
        print(f"Scorecard loaded from {path}")
        return scorecard