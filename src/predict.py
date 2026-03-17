import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import PredictConfig
from data_loader import load_test
from utils import ensure_parent_dir, resolve_path


def predict(
    model_path: Path,
    test_path: Path,
    output_path: Path,
    target: str,
) -> Path:
    model = joblib.load(model_path)
    test_data = load_test(test_path)

    if "Id" not in test_data.columns:
        raise ValueError("Test data must include an 'Id' column.")

    ids = test_data["Id"]
    features = test_data.drop(columns=["Id"])
    predictions = model.predict(features)
    predictions = np.maximum(predictions, 0)
    output = pd.DataFrame({"Id": ids, target: predictions})

    ensure_parent_dir(output_path)
    output.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    defaults = PredictConfig()
    parser = argparse.ArgumentParser(description="Generate predictions from a saved model.")
    parser.add_argument("--test", default=defaults.test_path, help="Test CSV path.")
    parser.add_argument("--model", default=defaults.model_path, help="Model path.")
    parser.add_argument("--output", default=defaults.output_path, help="Output CSV path.")
    parser.add_argument("--target", default=defaults.target, help="Target column name.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    test_path = resolve_path(args.test, base_dir)
    model_path = resolve_path(args.model, base_dir)
    output_path = resolve_path(args.output, base_dir)

    saved_path = predict(
        model_path=model_path,
        test_path=test_path,
        output_path=output_path,
        target=args.target,
    )
    print(f"Saved predictions to {saved_path}")


if __name__ == "__main__":
    main()
