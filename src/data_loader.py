from pathlib import Path
from typing import Tuple

import pandas as pd


def load_train(path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    train_data = pd.read_csv(path)
    if target not in train_data.columns:
        raise ValueError(f"Target column '{target}' not found in train data.")
    X = train_data.drop(columns=[target])
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])
    y = train_data[target]
    return X, y


def load_test(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
