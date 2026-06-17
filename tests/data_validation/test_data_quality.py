"""
tests/data_validation/test_data_quality.py
──────────────────────────────────────────
Data validation tests verifying schema expectations, null rates, and class balance.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from src.data_processing.data_ingestion import DataIngestor
from src.config_loader import ConfigLoader


def test_data_ingestor_schema_and_quality(cfg, sample_df, tmp_path) -> None:
    """Verify DataIngestor schema checks, numeric coercion, and quality statistics."""
    # Write sample_df to a temporary CSV raw path
    raw_path = tmp_path / "telco_churn_raw.csv"
    sample_df.to_csv(raw_path, index=False)

    # Configure ConfigLoader
    cfg._config["data"]["raw_path"] = str(raw_path)

    # Instantiate DataIngestor
    ingestor = DataIngestor(cfg)
    df = ingestor.load()

    # 1. Assert Schema Expectations
    assert not df.empty
    assert len(df) == len(sample_df)
    
    # Check that expected columns exist
    expected_cols = cfg.get("data.schema.expected_columns")
    for col in expected_cols:
        assert col in df.columns

    # 2. Assert Coercion and Mapping
    # TotalCharges must be numeric (floats)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])

    # Binary columns must be 0/1 mapped
    binary_cols = cfg.get("data.schema.binary_yes_no_cols", [])
    for col in binary_cols:
        assert set(df[col].dropna().unique()).issubset({0, 1})


def test_target_class_balance(cfg, sample_df, tmp_path) -> None:
    """Verify that class labels in the target column are balanced and valid."""
    # Write sample_df to a temporary CSV raw path
    raw_path = tmp_path / "telco_churn_raw.csv"
    sample_df.to_csv(raw_path, index=False)
    cfg._config["data"]["raw_path"] = str(raw_path)

    ingestor = DataIngestor(cfg)
    df = ingestor.load()

    target_col = cfg.get("data.target_col", "Churn")
    assert target_col in df.columns

    # Verify minority class is represented at least 10%
    churn_rate = df[target_col].mean()
    assert 0.10 <= churn_rate <= 0.40, f"Churn rate {churn_rate:.2f} is out of expected [10%, 40%] bounds."


def test_missing_values_post_imputation(cfg, sample_df, tmp_path) -> None:
    """Verify that critical fields contain no missing values after loading."""
    raw_path = tmp_path / "telco_churn_raw.csv"
    sample_df.to_csv(raw_path, index=False)
    cfg._config["data"]["raw_path"] = str(raw_path)

    ingestor = DataIngestor(cfg)
    df = ingestor.load()

    # Check for nulls in critical columns: customerID, tenure, Churn
    id_col = cfg.get("data.customer_id_col", "customerID")
    target_col = cfg.get("data.target_col", "Churn")

    assert df[id_col].isna().sum() == 0, f"Found null customer IDs: {df[id_col].isna().sum()}"
    assert df[target_col].isna().sum() == 0, f"Found null Churn labels: {df[target_col].isna().sum()}"
    assert df["tenure"].isna().sum() == 0, f"Found null tenures: {df['tenure'].isna().sum()}"
