from typing import Dict

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


def build_model(random_state: int, n_estimators: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def get_candidate_models(random_state: int, n_estimators: int) -> Dict[str, object]:
    models = {
        "random_forest": build_model(random_state, n_estimators),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state),
    }
    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    return models
