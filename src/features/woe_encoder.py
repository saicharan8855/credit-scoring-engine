import pandas as pd
import numpy as np


class WoEEncoder:
    """
    Weight of Evidence Encoder.

    Calculates WoE for each bin of each feature and stores the
    mapping so it can be applied to new data during transform.

    Usage:
        encoder = WoEEncoder()
        encoder.fit(X_train, y_train)
        X_train_woe = encoder.transform(X_train)
        X_test_woe  = encoder.transform(X_test)
    """

    def __init__(self, max_bins: int = 10):
        self.max_bins      = max_bins
        self.woe_tables    = {}
        self.feature_types = {}
        self.is_fitted     = False


    def _calculate_woe_for_one_feature(self,
                                        feature_values: pd.Series,
                                        target: pd.Series,
                                        feature_name: str,
                                        feature_type: str) -> pd.DataFrame:
        """
        Calculate WoE and IV for each bin of a single feature.
        Returns a table with bin, WoE, IV contribution, and counts.
        """
        temp_df = pd.DataFrame({'feature': feature_values, 'target': target})

        total_defaulters     = (target == 1).sum()
        total_non_defaulters = (target == 0).sum()

        grouped = temp_df.groupby('feature', observed=True)['target'].agg(
            total_count='count',
            defaulters='sum'
        ).reset_index()

        grouped['non_defaulters']     = grouped['total_count'] - grouped['defaulters']
        grouped['pct_defaulters']     = grouped['defaulters'] / total_defaulters
        grouped['pct_non_defaulters'] = grouped['non_defaulters'] / total_non_defaulters

        # Avoid log(0)
        grouped['pct_defaulters']     = grouped['pct_defaulters'].replace(0, 0.0001)
        grouped['pct_non_defaulters'] = grouped['pct_non_defaulters'].replace(0, 0.0001)

        grouped['WoE']   = np.log(grouped['pct_defaulters'] / grouped['pct_non_defaulters'])
        grouped['IV_bin'] = (grouped['pct_defaulters'] - grouped['pct_non_defaulters']) * grouped['WoE']

        grouped['feature_name'] = feature_name
        grouped['feature_type'] = feature_type
        grouped = grouped.rename(columns={'feature': 'bin'})

        return grouped


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Compute WoE tables for all features in X.
        Must be called on training data only to prevent data leakage.
        """
        print(f"Fitting WoE Encoder on {X.shape[1]} features...")

        for col in X.columns:

            feature_type = 'categorical' if X[col].dtype == 'object' else 'numerical'
            self.feature_types[col] = feature_type

            if feature_type == 'numerical':
                try:
                    binned = pd.qcut(X[col], q=self.max_bins, duplicates='drop')
                except Exception:
                    # Fall back to equal-width bins if qcut fails
                    binned = pd.cut(X[col], bins=self.max_bins, duplicates='drop')
            else:
                binned = X[col].astype(str)

            self.woe_tables[col] = self._calculate_woe_for_one_feature(
                feature_values=binned,
                target=y,
                feature_name=col,
                feature_type=feature_type
            )

        self.is_fitted = True
        print(f"WoE Encoder fitted on {len(self.woe_tables)} features")

        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace each feature value with its WoE score.
        Uses the tables computed during fit().
        """
        if not self.is_fitted:
            raise Exception("Call fit() before transform().")

        X_woe = X.copy()

        for col in X.columns:

            if col not in self.woe_tables:
                continue

            woe_table    = self.woe_tables[col]
            feature_type = self.feature_types[col]

            if feature_type == 'numerical':
                bin_boundaries = woe_table['bin'].values
                woe_values = []

                for value in X[col]:
                    matched_woe = None

                    for i, interval in enumerate(bin_boundaries):
                        try:
                            if value in interval:
                                matched_woe = woe_table.iloc[i]['WoE']
                                break
                        except Exception:
                            continue

                    # If value is outside all bins use the nearest boundary
                    if matched_woe is None:
                        if value <= bin_boundaries[0].left:
                            matched_woe = woe_table.iloc[0]['WoE']
                        else:
                            matched_woe = woe_table.iloc[-1]['WoE']

                    woe_values.append(matched_woe)

                X_woe[col] = woe_values

            else:
                woe_map = dict(zip(woe_table['bin'].astype(str), woe_table['WoE']))
                # Unknown categories get WoE = 0 (neutral)
                X_woe[col] = X[col].astype(str).map(woe_map).fillna(0)

        return X_woe


    def get_woe_table(self, feature_name: str) -> pd.DataFrame:
        """Return the WoE table for a specific feature."""
        if feature_name not in self.woe_tables:
            raise Exception(f"Feature '{feature_name}' not found.")
        return self.woe_tables[feature_name]


    def get_all_iv_values(self) -> pd.DataFrame:
        """
        Return total IV for every feature, sorted highest to lowest.
        Used for feature selection in iv_selector.py.
        """
        iv_summary = [
            {'feature': col, 'IV': table['IV_bin'].sum()}
            for col, table in self.woe_tables.items()
        ]

        return (pd.DataFrame(iv_summary)
                  .sort_values('IV', ascending=False)
                  .reset_index(drop=True))