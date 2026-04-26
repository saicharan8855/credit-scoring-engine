import pandas as pd
import numpy as np


def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known data anomalies discovered during EDA.

    DAYS_EMPLOYED = 365243 means unemployed/retired — replace with NaN.
    CODE_GENDER = 'XNA' means gender not recorded — replace with NaN.
    """
    df = df.copy()

    anomaly_count = (df['DAYS_EMPLOYED'] == 365243).sum()
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    print(f"Fixed DAYS_EMPLOYED anomaly — {anomaly_count} rows replaced with NaN")

    xna_count = (df['CODE_GENDER'] == 'XNA').sum()
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', np.nan)
    print(f"Fixed CODE_GENDER anomaly  — {xna_count} rows replaced with NaN")

    return df


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from existing raw columns.
    These capture relationships more meaningful for credit risk prediction.
    """
    df = df.copy()

    df['AGE_YEARS']               = (-df['DAYS_BIRTH'] / 365).astype(int)
    df['YEARS_EMPLOYED']          = -df['DAYS_EMPLOYED'] / 365
    df['CREDIT_TO_INCOME_RATIO']  = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['EMPLOYMENT_TO_AGE_RATIO'] = df['YEARS_EMPLOYED'] / df['AGE_YEARS']

    print("Created 6 new features:")
    for col in ['AGE_YEARS', 'YEARS_EMPLOYED', 'CREDIT_TO_INCOME_RATIO',
                'ANNUITY_TO_INCOME_RATIO', 'CREDIT_TO_ANNUITY_RATIO', 'EMPLOYMENT_TO_AGE_RATIO']:
        print(f"  - {col}")

    return df


def create_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    For columns where missingness correlates with default rate,
    create a binary flag column (1 = missing, 0 = present).
    This preserves the signal before imputation removes it.
    """
    df = df.copy()

    columns_to_flag = [
        'EXT_SOURCE_1',
        'EXT_SOURCE_3',
        'OWN_CAR_AGE',
        'OCCUPATION_TYPE',
        'DAYS_EMPLOYED'
    ]

    for col in columns_to_flag:
        indicator_col_name = col + '_IS_MISSING'
        df[indicator_col_name] = df[col].isnull().astype(int)
        print(f"Created missing indicator — {indicator_col_name}")

    return df


def drop_high_missing_columns(df: pd.DataFrame,
                               threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns where more than `threshold` fraction of values are missing.
    Default threshold is 50%. Missing indicators were already created
    for important columns so no signal is lost.
    """
    df = df.copy()

    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)

    print(f"Dropped {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing values")
    print(f"Remaining columns: {df.shape[1]}")

    return df, cols_to_drop


def impute_missing_values(df_train: pd.DataFrame,
                           df_test: pd.DataFrame):
    """
    Impute missing values — numerical with median, categorical with mode.
    Statistics are computed from train set only to prevent data leakage.
    """
    df_train = df_train.copy()
    df_test  = df_test.copy()

    numerical_cols   = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()

    num_imputed = 0
    for col in numerical_cols:
        if df_train[col].isnull().sum() > 0:
            median_value = df_train[col].median()
            df_train[col] = df_train[col].fillna(median_value)
            df_test[col]  = df_test[col].fillna(median_value)
            num_imputed += 1

    print(f"Imputed {num_imputed} numerical columns with median")

    cat_imputed = 0
    for col in categorical_cols:
        if df_train[col].isnull().sum() > 0:
            mode_value = df_train[col].mode()[0]
            df_train[col] = df_train[col].fillna(mode_value)
            df_test[col]  = df_test[col].fillna(mode_value)
            cat_imputed += 1

    print(f"Imputed {cat_imputed} categorical columns with mode")

    return df_train, df_test


def winsorize_outliers(df_train: pd.DataFrame,
                       df_test: pd.DataFrame,
                       lower_percentile: float = 0.01,
                       upper_percentile: float = 0.99):
    """
    Cap extreme values at the 1st and 99th percentile.
    Bounds are computed from train set only and applied to both sets.
    """
    df_train = df_train.copy()
    df_test  = df_test.copy()

    numerical_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    winsor_stats = {}

    for col in numerical_cols:
        lower_bound = df_train[col].quantile(lower_percentile)
        upper_bound = df_train[col].quantile(upper_percentile)
        winsor_stats[col] = {'lower': lower_bound, 'upper': upper_bound}

        df_train[col] = df_train[col].clip(lower=lower_bound, upper=upper_bound)
        df_test[col]  = df_test[col].clip(lower=lower_bound, upper=upper_bound)

    print(f"Winsorized {len(numerical_cols)} numerical columns at 1st and 99th percentile")

    return df_train, df_test, winsor_stats


def run_preprocessing_pipeline(df_train: pd.DataFrame,
                                df_test: pd.DataFrame):
    """
    Run all preprocessing steps in the correct order.

    Order matters:
    1. Fix anomalies        — before anything else
    2. Create new features  — before dropping or imputing
    3. Missing indicators   — before imputing (imputing removes missingness signal)
    4. Drop high-missing    — reduce noise early
    5. Impute               — fill remaining gaps
    6. Winsorize            — after imputation so no NaNs remain
    """
    print("\n" + "="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)

    print("\n--- Step 1: Fix Anomalies ---")
    df_train = fix_anomalies(df_train)
    df_test  = fix_anomalies(df_test)

    print("\n--- Step 2: Create New Features ---")
    df_train = create_new_features(df_train)
    df_test  = create_new_features(df_test)

    print("\n--- Step 3: Create Missing Indicators ---")
    df_train = create_missing_indicators(df_train)
    df_test  = create_missing_indicators(df_test)

    print("\n--- Step 4: Drop High Missing Columns ---")
    df_train, dropped_cols = drop_high_missing_columns(df_train, threshold=0.5)
    cols_to_drop_from_test = [col for col in dropped_cols if col in df_test.columns]
    df_test = df_test.drop(columns=cols_to_drop_from_test)
    print(f"Dropped same {len(cols_to_drop_from_test)} columns from test set")

    print("\n--- Step 5: Impute Missing Values ---")
    df_train, df_test = impute_missing_values(df_train, df_test)

    print("\n--- Step 6: Winsorize Outliers ---")
    df_train, df_test, winsor_stats = winsorize_outliers(df_train, df_test)

    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print(f"Train shape : {df_train.shape}")
    print(f"Test shape  : {df_test.shape}")
    print("="*60)

    return df_train, df_test