"""
tests/integration/test_data_pipeline.py
────────────────────────────────────────
Integration tests for DataPipeline E2E execution and preprocessor serialization.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.orchestration.data_pipeline import DataPipeline
from src.config_loader import ConfigLoader


def test_data_pipeline_e2e(cfg, sample_df, tmp_path) -> None:
    """Verify end-to-end DataPipeline flow and preprocessor loading/saving."""
    # Write sample_df to a temporary CSV raw path
    raw_path = tmp_path / "telco_churn_raw.csv"
    sample_df.to_csv(raw_path, index=False)

    # Configure temporary directories for output splits and models
    splits_path = tmp_path / "splits"
    models_dir = tmp_path / "models"
    
    # Override settings in ConfigLoader dict via fixture (if mutable, or update config values)
    # Since config values are fetched from the singleton cfg, let's update cfg._config
    cfg._config["data"]["raw_path"] = str(raw_path)
    cfg._config["data"]["splits_path"] = str(splits_path)
    cfg._config["model"]["models_dir"] = str(models_dir)
    cfg._config["split"]["save_format"] = "csv"

    # Instantiate and run pipeline
    pipeline = DataPipeline(cfg)
    train_df, val_df, test_df = pipeline.run(raw_path=str(raw_path))

    # Assert train, val, and test splits exist and have valid shape
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert train_df.shape[1] == val_df.shape[1] == test_df.shape[1]

    # Verify target column is present and encoded/mapped
    target_col = cfg.get("data.target_col", "Churn")
    assert target_col in train_df.columns
    assert set(train_df[target_col].unique()).issubset({0, 1})

    # Verify that splits exist on disk
    assert (splits_path / "train.csv").exists()
    assert (splits_path / "val.csv").exists()
    assert (splits_path / "test.csv").exists()

    # Verify preprocessor serialization
    preprocessor_path = models_dir / "preprocessor.pkl"
    assert preprocessor_path.exists()

    # Load preprocessor using static factory method load
    from src.data_processing.preprocessor_state import PreprocessorState
    loaded_preprocessor = DataPipeline.load(preprocessor_path)
    assert isinstance(loaded_preprocessor, PreprocessorState)

    # Test transform on raw unseen row
    raw_row = sample_df.iloc[[0]].copy()
    # If the target is present, let's ensure transform is clean
    # Remove Target to simulate serving scenario
    if target_col in raw_row.columns:
        raw_row = raw_row.drop(columns=[target_col])
    
    transformed_row = loaded_preprocessor.transform(raw_row)
    
    # Assert transformed features match training schema (excluding Churn target column)
    expected_cols = [col for col in train_df.columns if col != target_col]
    # Check that OHE columns are present
    assert any(col.startswith("gender_") for col in transformed_row.columns)
    # TotalCharges is coerced and transformed
    assert "TotalCharges" in transformed_row.columns
