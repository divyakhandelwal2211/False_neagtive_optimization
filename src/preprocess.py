# src/preprocessing.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Split dataset and scale numeric features
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encoding for categorical
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler