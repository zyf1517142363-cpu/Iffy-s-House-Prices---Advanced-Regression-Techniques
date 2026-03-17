import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from config import TrainConfig
from data_loader import load_train
from evaluate import cross_val_rmsle, rmsle
from model import build_model, get_candidate_models
from feature_engineer import FeatureEngineer
from preprocess import build_preprocessor
from utils import ensure_parent_dir, get_env_info, resolve_path, save_json, set_seed


def build_pipeline(X: pd.DataFrame, model: object) -> Pipeline:
    feature_engineer = FeatureEngineer()
    X_fe = feature_engineer.fit_transform(X)
    preprocessor = build_preprocessor(X_fe)
    return Pipeline(
        steps=[
            ("feature_engineer", feature_engineer),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def build_estimator(X: pd.DataFrame, model: object, log_target: bool):
    pipeline = build_pipeline(X, model)
    if not log_target:
        return pipeline
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_estimators: int,
    cv_folds: int,
    log_target: bool,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_score = float("inf")
    for name, model in get_candidate_models(random_state, n_estimators).items():
        estimator = build_estimator(X, model, log_target)
        mean_score, std_score = cross_val_rmsle(
            estimator, X, y, cv_folds, random_state=random_state
        )
        results[name] = {"rmsle_mean": mean_score, "rmsle_std": std_score}
        if mean_score < best_score:
            best_score = mean_score
            best_name = name
    return best_name, results


def run_random_search(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_estimators: int,
    cv_folds: int,
    n_iter: int,
    log_target: bool,
) -> Tuple[object, Dict[str, object], float, float]:
    base_model = build_model(random_state=random_state, n_estimators=n_estimators)
    estimator = build_estimator(X, base_model, log_target)
    param_prefix = "regressor__model__" if log_target else "model__"
    param_distributions = {
        f"{param_prefix}n_estimators": [200, 300, 500, 800],
        f"{param_prefix}max_depth": [None, 6, 10, 14, 20],
        f"{param_prefix}min_samples_split": [2, 5, 10],
        f"{param_prefix}min_samples_leaf": [1, 2, 4],
        f"{param_prefix}max_features": ["sqrt", "log2", 0.5],
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=make_scorer(rmsle, greater_is_better=False),
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    best_rmsle = -search.best_score_
    best_params = search.best_params_
    best_estimator = search.best_estimator_
    cv_mean, cv_std = cross_val_rmsle(
        best_estimator, X, y, cv_folds, random_state=random_state
    )
    return best_estimator, best_params, cv_mean, cv_std


def train_model(
    train_path: Path,
    target: str,
    model_path: Path,
    model_meta_path: Path,
    metrics_path: Path,
    run_path: Path,
    random_state: int,
    n_estimators: int,
    cv_folds: int,
    compare: bool,
    search: bool,
    search_iter: int,
    log_target: bool,
) -> Tuple[object, float, float]:
    X, y = load_train(train_path, target)
    model_name = "random_forest"
    compare_results = None
    best_params = None

    if search:
        logging.info("Running randomized search.")
        estimator, best_params, cv_mean, cv_std = run_random_search(
            X=X,
            y=y,
            random_state=random_state,
            n_estimators=n_estimators,
            cv_folds=cv_folds,
            n_iter=search_iter,
            log_target=log_target,
        )
        model_name = "random_forest_search"
    else:
        if compare:
            model_name, compare_results = compare_models(
                X=X,
                y=y,
                random_state=random_state,
                n_estimators=n_estimators,
                cv_folds=cv_folds,
                log_target=log_target,
            )
            logging.info("Model comparison results: %s", compare_results)
        model = get_candidate_models(random_state, n_estimators).get(
            model_name, build_model(random_state, n_estimators)
        )
        estimator = build_estimator(X, model, log_target)
        cv_mean, cv_std = cross_val_rmsle(
            estimator, X, y, cv_folds, random_state=random_state
        )
        estimator.fit(X, y)

    ensure_parent_dir(model_path)
    joblib.dump(estimator, model_path)

    feature_columns = list(X.columns)
    metrics = {
        "cv_rmsle_mean": cv_mean,
        "cv_rmsle_std": cv_std,
        "cv_folds": cv_folds,
        "model_name": model_name,
        "best_params": best_params,
        "compare_results": compare_results,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "log_target": log_target,
        "train_rows": int(X.shape[0]),
        "train_cols": int(X.shape[1]),
        "feature_columns": feature_columns,
    }
    save_json(metrics, metrics_path)

    model_meta = {
        "target": target,
        "feature_columns": feature_columns,
        "model_name": model_name,
        "log_target": log_target,
    }
    save_json(model_meta, model_meta_path)

    run_info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "train_path": str(train_path),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "model_meta_path": str(model_meta_path),
        "env": get_env_info(),
        "metrics": metrics,
    }
    save_json(run_info, run_path)
    return estimator, cv_mean, cv_std


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train model and save artifacts.")
    parser.add_argument("--train", default=defaults.train_path, help="Training CSV path.")
    parser.add_argument("--target", default=defaults.target, help="Target column name.")
    parser.add_argument("--model", default=defaults.model_path, help="Model output path.")
    parser.add_argument(
        "--model-meta",
        default=defaults.model_meta_path,
        help="Model metadata output path.",
    )
    parser.add_argument(
        "--metrics",
        default=defaults.metrics_path,
        help="Metrics output path.",
    )
    parser.add_argument(
        "--run-info",
        default=defaults.run_path,
        help="Run info output path.",
    )
    parser.add_argument(
        "--log",
        default=defaults.log_path,
        help="Training log path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=defaults.random_state,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=defaults.n_estimators,
        help="Number of trees.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=defaults.cv_folds,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models and select the best.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run randomized hyperparameter search for Random Forest.",
    )
    parser.add_argument(
        "--search-iter",
        type=int,
        default=defaults.search_iter,
        help="Random search iterations.",
    )
    parser.add_argument(
        "--no-log-target",
        action="store_true",
        help="Disable log1p transform of the target (not recommended for Kaggle).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    train_path = resolve_path(args.train, base_dir)
    model_path = resolve_path(args.model, base_dir)
    model_meta_path = resolve_path(args.model_meta, base_dir)
    metrics_path = resolve_path(args.metrics, base_dir)
    run_path = resolve_path(args.run_info, base_dir)
    log_path = resolve_path(args.log, base_dir)

    ensure_parent_dir(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )

    set_seed(args.random_state)
    _, cv_mean, cv_std = train_model(
        train_path=train_path,
        target=args.target,
        model_path=model_path,
        model_meta_path=model_meta_path,
        metrics_path=metrics_path,
        run_path=run_path,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        cv_folds=args.cv_folds,
        compare=args.compare,
        search=args.search,
        search_iter=args.search_iter,
        log_target=not args.no_log_target,
    )

    print(f"5-fold CV RMSLE: {cv_mean:,.5f} (std: {cv_std:,.5f})")
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved run info to {run_path}")


if __name__ == "__main__":
    main()
