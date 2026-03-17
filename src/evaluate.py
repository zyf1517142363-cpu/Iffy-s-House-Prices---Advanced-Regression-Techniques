from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import KFold, cross_val_score


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def cross_val_rmsle(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> Tuple[float, float]:
    scorer = make_scorer(rmsle, greater_is_better=False)
    cv = (
        KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        if random_state is not None
        else cv_folds
    )
    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
    )
    rmsle_scores = -scores
    return rmsle_scores.mean(), rmsle_scores.std()
