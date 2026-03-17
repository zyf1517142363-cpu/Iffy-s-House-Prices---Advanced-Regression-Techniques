from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    train_path: str = "../data/train.csv"
    target: str = "SalePrice"
    model_path: str = "../models/model.joblib"
    model_meta_path: str = "../models/model_meta.json"
    metrics_path: str = "../reports/metrics.json"
    run_path: str = "../reports/run.json"
    log_path: str = "../reports/train.log"
    random_state: int = 1
    n_estimators: int = 300
    cv_folds: int = 5
    search_iter: int = 20
    log_target: bool = True


@dataclass(frozen=True)
class PredictConfig:
    test_path: str = "../data/test.csv"
    model_path: str = "../models/model.joblib"
    output_path: str = "../data/submission.csv"
    target: str = "SalePrice"
