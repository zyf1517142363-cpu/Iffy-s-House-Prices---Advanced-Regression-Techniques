from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from utils import resolve_path

app = FastAPI(title="House Price Predictor", version="1.0")


def _default_model_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    return resolve_path(os.getenv("MODEL_PATH", "../models/model.joblib"), base_dir)


def _default_meta_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    return resolve_path(os.getenv("MODEL_META_PATH", "../models/model_meta.json"), base_dir)


def _load_feature_columns(meta_path: Path) -> Optional[List[str]]:
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    feature_cols = meta.get("feature_columns")
    if isinstance(feature_cols, list):
        return feature_cols
    return None


def _infer_feature_columns(model: Any) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "regressor") and hasattr(model.regressor, "feature_names_in_"):
        return list(model.regressor.feature_names_in_)
    return None


def _prepare_frame(records: List[Dict[str, Any]], feature_columns: Optional[List[str]]) -> pd.DataFrame:
    if not records:
        raise HTTPException(status_code=400, detail="No input records provided.")

    df = pd.DataFrame.from_records(records)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_columns]

    return df


@app.on_event("startup")
def load_artifacts() -> None:
    model_path = _default_model_path()
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    app.state.model = joblib.load(model_path)
    meta_path = _default_meta_path()
    feature_columns = _load_feature_columns(meta_path)
    if feature_columns is None:
        feature_columns = _infer_feature_columns(app.state.model)
    app.state.feature_columns = feature_columns


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Any = Body(...)) -> Dict[str, Any]:
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, dict) and "record" in payload:
        records = [payload["record"]]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise HTTPException(status_code=400, detail="Unsupported payload format.")

    df = _prepare_frame(records, app.state.feature_columns)
    predictions = app.state.model.predict(df)
    predictions = np.maximum(predictions, 0)
    return {"predictions": predictions.tolist()}
