import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV data from the given filepath.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully — Shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame,
               target_col: str = 'TARGET',
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Split dataframe into train and test sets.
    Stratified split to preserve class imbalance ratio.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y       # preserve imbalance ratio in both splits
    )

    print(f"Train set : {X_train.shape[0]:,} rows")
    print(f"Test set  : {X_test.shape[0]:,} rows")
    print(f"Default rate in train : {y_train.mean()*100:.2f}%")
    print(f"Default rate in test  : {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test