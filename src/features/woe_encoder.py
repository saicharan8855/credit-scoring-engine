import pandas as pd
import numpy as np


class WoEEncoder:
    """
    Weight of Evidence Encoder.

    Fits WoE tables on training data and transforms
    both train and test data using those tables.

    Usage:
        encoder = WoEEncoder()
        encoder.fit(X_train, y_train)
        X_train_woe = encoder.transform(X_train)
        X_test_woe  = encoder.transform(X_test)
    """

    def __init__(self, max_bins: int = 10):
        self.max_bins      = max_bins
        self.woe_tables    = {}
        self.bin_edges     = {}   # stores bin edges for numerical features
        self.feature_types = {}
        self.is_fitted     = False


    def _calculate_woe_table(self,
                              binned_series: pd.Series,
                              target: pd.Series) -> pd.DataFrame:
        """
        Given a binned series and the target, calculate WoE and IV
        for each bin using fully vectorized pandas operations.
        """

        total_defaulters     = (target == 1).sum()
        total_non_defaulters = (target == 0).sum()

        # Build a dataframe and group by bin in one shot
        temp = pd.DataFrame({'bin': binned_series, 'target': target})

        grouped = temp.groupby('bin', observed=True)['target'].agg(
            total_count='count',
            defaulters='sum'
        )
        grouped['non_defaulters'] = grouped['total_count'] - grouped['defaulters']

        # Percentage of all defaulters / non-defaulters in each bin
        grouped['pct_defaulters']     = grouped['defaulters'] / total_defaulters
        grouped['pct_non_defaulters'] = grouped['non_defaulters'] / total_non_defaulters

        # Replace zeros to avoid log(0)
        grouped['pct_defaulters']     = grouped['pct_defaulters'].clip(lower=0.0001)
        grouped['pct_non_defaulters'] = grouped['pct_non_defaulters'].clip(lower=0.0001)

        # WoE and IV — fully vectorized
        grouped['WoE']    = np.log(grouped['pct_defaulters'] / grouped['pct_non_defaulters'])
        grouped['IV_bin'] = (grouped['pct_defaulters'] - grouped['pct_non_defaulters']) * grouped['WoE']

        return grouped.reset_index()


    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit WoE tables for all features.
        Uses a sample of 50k rows for binning to speed up fitting.
        Full data is still used for WoE calculation.
        """

        print(f"Fitting WoE Encoder on {X.shape[1]} features...")

        # Use a sample for finding bin edges on numerical features
        # This is much faster and bin edges are stable with 50k rows
        sample_size = min(50000, len(X))
        X_sample    = X.sample(n=sample_size, random_state=42)
        y_sample    = y.loc[X_sample.index]

        for col in X.columns:

            if X[col].dtype == 'object':
                feature_type = 'categorical'
            else:
                feature_type = 'numerical'

            self.feature_types[col] = feature_type

            if feature_type == 'numerical':
                # Find bin edges using sample for speed
                try:
                    _, bin_edges = pd.qcut(
                        X_sample[col],
                        q=self.max_bins,
                        retbins=True,
                        duplicates='drop'
                    )
                except Exception:
                    _, bin_edges = pd.cut(
                        X_sample[col],
                        bins=self.max_bins,
                        retbins=True,
                        duplicates='drop'
                    )

                # Extend edges slightly to cover min and max of full data
                bin_edges[0]  = -np.inf
                bin_edges[-1] =  np.inf

                self.bin_edges[col] = bin_edges

                # Now bin the full training data using these edges
                binned = pd.cut(
                    X[col],
                    bins=bin_edges,
                    include_lowest=True
                )

            else:
                binned = X[col].astype(str)

            # Calculate WoE table using full training data
            woe_table = self._calculate_woe_table(binned, y)
            woe_table['feature_name'] = col
            woe_table['feature_type'] = feature_type
            self.woe_tables[col] = woe_table

        self.is_fitted = True
        print(f"WoE Encoder fitted successfully on {len(self.woe_tables)} features")
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform X by replacing each value with its WoE value.
        Fully vectorized — no row by row loops.
        """

        if not self.is_fitted:
            raise Exception("Call fit() before transform().")

        X_woe = pd.DataFrame(index=X.index)

        for col in X.columns:

            if col not in self.woe_tables:
                X_woe[col] = X[col]
                continue

            woe_table    = self.woe_tables[col]
            feature_type = self.feature_types[col]

            # Build a mapping from bin to WoE value
            woe_map = woe_table.set_index('bin')['WoE'].to_dict()

            if feature_type == 'numerical':

                # Use numpy digitize instead of pd.cut
                # It is much faster and does not hang on inf edges
                edges = self.bin_edges[col]

                # Replace inf edges with actual min/max of column
                finite_edges = edges.copy()
                finite_edges[0]  = X[col].min() - 1
                finite_edges[-1] = X[col].max() + 1

                # digitize returns index of bin each value falls into
                bin_indices = np.digitize(X[col].values, bins=finite_edges[1:-1])

                # Get the WoE values in order of bins
                woe_values_ordered = woe_table['WoE'].values

                # Map each bin index to WoE value
                # clip to handle any out of range indices
                bin_indices_clipped = np.clip(bin_indices, 0, len(woe_values_ordered) - 1)
                woe_array = woe_values_ordered[bin_indices_clipped]

                X_woe[col] = woe_array

            else:
                woe_series = X[col].astype(str).map(woe_map)
                median_woe = woe_table['WoE'].median()
                X_woe[col] = woe_series.fillna(median_woe)

        return X_woe


    def get_woe_table(self, feature_name: str) -> pd.DataFrame:
        """Return WoE table for a specific feature."""
        if feature_name not in self.woe_tables:
            raise Exception(f"Feature '{feature_name}' not found.")
        return self.woe_tables[feature_name]


    def get_all_iv_values(self) -> pd.DataFrame:
        """
        Return a dataframe with total IV for every feature,
        sorted from highest to lowest.
        """
        iv_summary = []
        for col, woe_table in self.woe_tables.items():
            total_iv = woe_table['IV_bin'].sum()
            iv_summary.append({'feature': col, 'IV': total_iv})

        iv_df = pd.DataFrame(iv_summary)
        iv_df = iv_df.sort_values('IV', ascending=False).reset_index(drop=True)
        return iv_df