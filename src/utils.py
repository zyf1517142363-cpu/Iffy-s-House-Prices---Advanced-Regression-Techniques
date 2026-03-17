import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def get_env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn_version,
        "joblib": None if joblib is None else joblib.__version__,
    }
