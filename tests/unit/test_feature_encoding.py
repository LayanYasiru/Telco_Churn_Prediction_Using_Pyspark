"""
tests/unit/test_feature_encoding.py
───────────────────────────────────
Unit tests for categorical feature encoders: OneHotEncoderWrapper,
LabelEncoderWrapper, and TargetEncoderWrapper.
"""

from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from src.data_processing.feature_encoding import (
    OneHotEncoderWrapper,
    LabelEncoderWrapper,
    TargetEncoderWrapper,
    FeatureEncoder,
)


def test_one_hot_encoder_drop_first() -> None:
    """Verify OneHotEncoderWrapper correctly encodes and drops first category."""
    df = pd.DataFrame({"color": ["Red", "Blue", "Green", "Red"]})
    encoder = OneHotEncoderWrapper(drop_first=True)
    
    # Fit & Transform
    encoded = encoder.fit_transform(df, ["color"])
    
    # Expected: "color_Green", "color_Red" exist; "color_Blue" (first alphabetically) is dropped
    assert "color_Blue" not in encoded.columns
    assert "color_Green" in encoded.columns
    assert "color_Red" in encoded.columns
    assert encoded["color_Green"].tolist() == [0, 0, 1, 0]
    assert encoded["color_Red"].tolist() == [1, 0, 0, 1]


def test_one_hot_encoder_keep_all() -> None:
    """Verify OneHotEncoderWrapper keeps all categories when drop_first=False."""
    df = pd.DataFrame({"color": ["Red", "Blue", "Green", "Red"]})
    encoder = OneHotEncoderWrapper(drop_first=False)
    
    encoded = encoder.fit_transform(df, ["color"])
    
    assert "color_Red" in encoded.columns
    assert "color_Blue" in encoded.columns
    assert "color_Green" in encoded.columns
    assert encoded["color_Red"].tolist() == [1, 0, 0, 1]


def test_label_encoder_wrapper() -> None:
    """Verify LabelEncoderWrapper converts categories to integer sequences."""
    df = pd.DataFrame({"size": ["S", "M", "L", "S"]})
    encoder = LabelEncoderWrapper()
    
    encoded = encoder.fit_transform(df, ["size"])
    
    assert encoded["size"].dtype in (np.int32, np.int64, int)
    assert encoded["size"].nunique() == 3


def test_target_encoder_wrapper() -> None:
    """Verify TargetEncoderWrapper replaces categories with target mean value."""
    df = pd.DataFrame({"city": ["NY", "LA", "NY", "SF", "LA"]})
    target = pd.Series([1, 0, 1, 0, 1])  # NY has 1.0 mean, LA has 0.5 mean, SF has 0.0 mean, global mean = 3/5 = 0.6
    
    encoder = TargetEncoderWrapper()
    encoded = encoder.fit_transform(df, ["city"], target=target)
    
    assert encoded["city"].iloc[0] == 1.0  # NY
    assert encoded["city"].iloc[1] == 0.5  # LA
    assert encoded["city"].iloc[3] == 0.0  # SF
    
    # Test unseen category fallback to global mean
    unseen_df = pd.DataFrame({"city": ["Chicago"]})
    transformed = encoder.transform(unseen_df, ["city"])
    assert transformed["city"].iloc[0] == 0.6  # global mean


def test_feature_encoder_orchestrator(cfg) -> None:
    """Verify orchestrator FeatureEncoder loads configuration and runs successfully."""
    # Let's mock a simple DataFrame with config's categorical cols
    df = pd.DataFrame({
        "gender": ["Male", "Female", "Male"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "No", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
    })
    
    fe = FeatureEncoder(cfg)
    encoded_df = fe.fit_transform(df)
    
    # Verify gender has been transformed (original gender col removed)
    assert "gender" not in encoded_df.columns
    # Verify that new OHE columns have been created
    assert any(col.startswith("gender_") for col in encoded_df.columns)
