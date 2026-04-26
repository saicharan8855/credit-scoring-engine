import pandas as pd


class IVSelector:
    """
    Information Value based Feature Selector.

    Selects features based on their IV score from WoEEncoder.
    Drops useless features (IV < min_iv) and suspicious ones (IV > max_iv).

    IV Interpretation:
        < 0.02       — Useless
        0.02 – 0.1   — Weak
        0.1  – 0.3   — Medium
        0.3  – 0.5   — Strong
        > 0.5        — Suspicious (possible data leakage)

    Usage:
        selector = IVSelector(min_iv=0.02, max_iv=0.5)
        selector.fit(iv_dataframe)
        selected_features = selector.get_selected_features()
        X_train_selected  = selector.transform(X_train_woe)
    """

    def __init__(self, min_iv: float = 0.02, max_iv: float = 0.5):
        self.min_iv            = min_iv
        self.max_iv            = max_iv
        self.iv_dataframe      = None
        self.selected_features = []
        self.dropped_low_iv    = []
        self.flagged_high_iv   = []
        self.is_fitted         = False


    def fit(self, iv_dataframe: pd.DataFrame):
        """
        Read IV values and decide which features to keep.
        iv_dataframe must have 'feature' and 'IV' columns.
        Comes from WoEEncoder.get_all_iv_values().
        """
        self.iv_dataframe = iv_dataframe.copy()

        low_iv_mask  = self.iv_dataframe['IV'] < self.min_iv
        high_iv_mask = self.iv_dataframe['IV'] > self.max_iv
        good_iv_mask = (~low_iv_mask) & (~high_iv_mask)

        self.dropped_low_iv    = self.iv_dataframe[low_iv_mask]['feature'].tolist()
        self.flagged_high_iv   = self.iv_dataframe[high_iv_mask]['feature'].tolist()
        self.selected_features = self.iv_dataframe[good_iv_mask]['feature'].tolist()

        self.is_fitted = True

        print("IV FEATURE SELECTION SUMMARY")
        print(f"Total features evaluated  : {len(self.iv_dataframe)}")
        print(f"Features selected         : {len(self.selected_features)}")
        print(f"Features dropped (low IV) : {len(self.dropped_low_iv)}")
        print(f"Features flagged (high IV): {len(self.flagged_high_iv)}")

        print(f"\nSelected Features (IV between {self.min_iv} and {self.max_iv}):")
        for _, row in self.iv_dataframe[good_iv_mask].iterrows():
            iv_val   = row['IV']
            strength = "Weak" if iv_val < 0.1 else "Medium" if iv_val < 0.3 else "Strong"
            print(f"  {row['feature']:<45} IV={iv_val:.4f}  ({strength})")

        if self.flagged_high_iv:
            print(f"\nFlagged Features (IV > {self.max_iv} — check for leakage):")
            for feature in self.flagged_high_iv:
                iv_val = self.iv_dataframe[self.iv_dataframe['feature'] == feature]['IV'].values[0]
                print(f"  {feature:<45} IV={iv_val:.4f}")

        print(f"\nDropped Features (IV < {self.min_iv}): {len(self.dropped_low_iv)} features removed")

        return self


    def get_selected_features(self) -> list:
        """Return list of features that passed the IV filter."""
        if not self.is_fitted:
            raise Exception("Call fit() before get_selected_features().")
        return self.selected_features


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep only selected features in X, drop everything else."""
        if not self.is_fitted:
            raise Exception("Call fit() before transform().")

        cols_to_keep = [col for col in self.selected_features if col in X.columns]
        print(f"Keeping {len(cols_to_keep)} selected features out of {X.shape[1]} total")

        return X[cols_to_keep]


    def get_iv_table(self) -> pd.DataFrame:
        """Return the full IV table for all features."""
        if not self.is_fitted:
            raise Exception("Call fit() first.")
        return self.iv_dataframe.copy()