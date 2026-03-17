import argparse
import logging
from pathlib import Path

from config import PredictConfig, TrainConfig
from predict import predict
from train import train_model
from utils import ensure_parent_dir, resolve_path, set_seed


def main() -> None:
    train_defaults = TrainConfig()
    predict_defaults = PredictConfig()

    parser = argparse.ArgumentParser(
        description="Train model and write predictions (legacy entrypoint)."
    )
    parser.add_argument("--train", default=train_defaults.train_path, help="Train CSV.")
    parser.add_argument("--test", default=predict_defaults.test_path, help="Test CSV.")
    parser.add_argument(
        "--output",
        default=predict_defaults.output_path,
        help="Output submission CSV path.",
    )
    parser.add_argument("--target", default=train_defaults.target, help="Target column.")
    parser.add_argument(
        "--model",
        default=train_defaults.model_path,
        help="Model output path.",
    )
    parser.add_argument(
        "--model-meta",
        default=train_defaults.model_meta_path,
        help="Model metadata output path.",
    )
    parser.add_argument(
        "--metrics",
        default=train_defaults.metrics_path,
        help="Metrics output path.",
    )
    parser.add_argument(
        "--run-info",
        default=train_defaults.run_path,
        help="Run info output path.",
    )
    parser.add_argument(
        "--log",
        default=train_defaults.log_path,
        help="Training log path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=train_defaults.random_state,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=train_defaults.n_estimators,
        help="Number of trees.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=train_defaults.cv_folds,
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
        default=train_defaults.search_iter,
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
    test_path = resolve_path(args.test, base_dir)
    output_path = resolve_path(args.output, base_dir)
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

    saved_path = predict(
        model_path=model_path,
        test_path=test_path,
        output_path=output_path,
        target=args.target,
    )
    print(f"Saved predictions to {saved_path}")


if __name__ == "__main__":
    main()
