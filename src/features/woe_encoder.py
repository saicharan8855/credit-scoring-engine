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
        Replace each value in X with its corresponding WoE value.
        The WoE tables calculated during fit() are used here.
        This can be applied to both train and test data.

        This version uses pandas cut/qcut with the stored bin edges
        for fast vectorized transformation instead of row by row loops.
        """

        if not self.is_fitted:
            raise Exception("WoEEncoder must be fitted before calling transform. Call fit() first.")

        X_woe = X.copy()

        for col in X.columns:

            if col not in self.woe_tables:
                continue

            woe_table    = self.woe_tables[col]
            feature_type = self.feature_types[col]

            if feature_type == 'numerical':
                # Get the bin intervals stored during fit
                bins = woe_table['bin'].values

                # Extract left and right edges from the intervals
                # to reconstruct the bin boundaries
                left_edges  = [b.left for b in bins]
                right_edges = [b.right for b in bins]

                # Build a single sorted list of all bin edges
                all_edges = sorted(set(left_edges + right_edges))

                # Build a mapping from bin interval to WoE value
                woe_map = dict(zip(woe_table['bin'], woe_table['WoE']))

                # Use pd.cut with the same edges to assign bins vectorized
                binned = pd.cut(
                    X[col],
                    bins=all_edges,
                    include_lowest=True
                )

                # Map each bin to its WoE value
                # For values outside the bin range use the nearest WoE
                woe_series = binned.map(woe_map)

                # Fill any unmapped values with the median WoE of this feature
                median_woe = woe_table['WoE'].median()
                woe_series = woe_series.fillna(median_woe)

                X_woe[col] = woe_series

            else:
                # For categorical features — direct lookup by category name
                woe_map = dict(zip(
                    woe_table['bin'].astype(str),
                    woe_table['WoE']
                ))

                # Replace category with WoE value
                # Unknown categories get WoE of 0 (neutral)
                X_woe[col] = X[col].astype(str).map(woe_map).fillna(0)

        return X_woe