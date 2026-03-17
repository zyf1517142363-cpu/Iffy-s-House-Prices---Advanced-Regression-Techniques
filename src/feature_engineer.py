from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Lightweight, deterministic feature engineering for housing data."""

    def __init__(self, add_features: bool = True) -> None:
        self.add_features = add_features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.add_features:
            return X

        X_out = X.copy()

        def col(name: str, default: float | int | np.ndarray = np.nan) -> pd.Series:
            if name in X_out.columns:
                return X_out[name]
            return pd.Series(default, index=X_out.index)

        total_bsmt = col("TotalBsmtSF", 0)
        first_flr = col("1stFlrSF", 0)
        second_flr = col("2ndFlrSF", 0)
        X_out["TotalSF"] = total_bsmt + first_flr + second_flr

        full_bath = col("FullBath", 0)
        half_bath = col("HalfBath", 0)
        bsmt_full = col("BsmtFullBath", 0)
        bsmt_half = col("BsmtHalfBath", 0)
        X_out["TotalBathrooms"] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

        yr_sold = col("YrSold")
        X_out["HouseAge"] = yr_sold - col("YearBuilt")
        X_out["RemodAge"] = yr_sold - col("YearRemodAdd")
        X_out["GarageAge"] = yr_sold - col("GarageYrBlt")

        X_out["TotalPorchSF"] = (
            col("OpenPorchSF", 0)
            + col("EnclosedPorch", 0)
            + col("3SsnPorch", 0)
            + col("ScreenPorch", 0)
            + col("WoodDeckSF", 0)
        )

        pool_area = col("PoolArea", 0)
        X_out["HasPool"] = (pool_area > 0).astype(int)
        X_out["HasFireplace"] = (col("Fireplaces", 0) > 0).astype(int)
        X_out["HasGarage"] = (
            (col("GarageArea", 0) > 0) | (col("GarageCars", 0) > 0)
        ).astype(int)
        X_out["HasBasement"] = (total_bsmt > 0).astype(int)

        lot_area = col("LotArea")
        safe_lot_area = lot_area.where(lot_area != 0, np.nan)
        X_out["LivLotRatio"] = col("GrLivArea") / safe_lot_area

        return X_out


def add_engineered_features(
    X: pd.DataFrame, add_features: bool = True
) -> pd.DataFrame:
    """Convenience wrapper for notebooks or quick experiments."""
    return FeatureEngineer(add_features=add_features).fit_transform(X)

