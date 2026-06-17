"""
tests/conftest.py
-----------------
Shared pytest fixtures for the Telco Customer Churn Prediction test suite.

Fixtures
--------
sample_df       : 200-row synthetic DataFrame matching IBM Telco schema.
cfg             : ConfigLoader instance (falls back to stub dict on FileNotFoundError).
small_X_y       : Tuple[np.ndarray, np.ndarray] suitable for quick model unit tests.
reset_config    : autouse fixture that resets ConfigLoader singleton before every test.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBM Telco column catalogue
# ---------------------------------------------------------------------------
_GENDER = ["Male", "Female"]
_YES_NO = ["Yes", "No"]
_MULTIPLE_LINES = ["No", "Yes", "No phone service"]
_INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
_ONLINE_SECURITY = ["No", "Yes", "No internet service"]
_ONLINE_BACKUP = ["No", "Yes", "No internet service"]
_DEVICE_PROTECTION = ["No", "Yes", "No internet service"]
_TECH_SUPPORT = ["No", "Yes", "No internet service"]
_STREAMING_TV = ["No", "Yes", "No internet service"]
_STREAMING_MOVIES = ["No", "Yes", "No internet service"]
_CONTRACT = ["Month-to-month", "One year", "Two year"]
_PAYMENT_METHOD = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_N_ROWS = 200
_SEED = 42


# ---------------------------------------------------------------------------
# Autouse: reset ConfigLoader singleton before every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_config() -> None:
    """
    Reset the ConfigLoader singleton before each test to avoid state leakage.

    This is critical for tests that instantiate ConfigLoader with different
    paths or mock parameters.
    """
    try:
        from src.config_loader import ConfigLoader  # type: ignore

        ConfigLoader.reset()
    except ImportError:
        pass  # ConfigLoader not available; skip reset silently
    yield
    try:
        from src.config_loader import ConfigLoader  # type: ignore

        ConfigLoader.reset()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# cfg fixture — returns ConfigLoader or a stub dict-like object
# ---------------------------------------------------------------------------


class _StubConfig:
    """
    Minimal stub that mimics ConfigLoader.get() for tests running without
    a real config.yaml on the filesystem.

    Attributes
    ----------
    _data : Dict[str, Any]
        Flat dictionary of dot-path -> value mappings.
    """

    _data: Dict[str, Any] = {
        "data.raw_path": "data/raw/telco_churn.csv",
        "data.processed_path": "data/processed",
        "data.splits_path": "data/splits",
        "data.target_col": "Churn",
        "data.customer_id_col": "customerID",
        "data.schema.expected_columns": [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
        ],
        "data.schema.numerical_cols": ["tenure", "MonthlyCharges", "TotalCharges"],
        "data.schema.categorical_cols": [
            "gender", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaymentMethod",
        ],
        "data.schema.binary_yes_no_cols": [
            "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
        ],
        "data.schema.coerce_to_numeric": ["TotalCharges"],
        "preprocessing.missing_values.strategy": "median",
        "preprocessing.missing_values.knn_neighbors": 5,
        "preprocessing.missing_values.numerical_impute_cols": ["TotalCharges"],
        "preprocessing.outlier.method": "iqr",
        "preprocessing.outlier.iqr_factor": 1.5,
        "preprocessing.outlier.zscore_threshold": 3.0,
        "preprocessing.outlier.action": "cap",
        "preprocessing.outlier.columns_to_check": ["tenure", "MonthlyCharges", "TotalCharges"],
        "preprocessing.encoding.method": "onehot",
        "preprocessing.encoding.drop_first": True,
        "preprocessing.scaling.method": "standard",
        "preprocessing.scaling.columns_to_scale": [
            "tenure", "MonthlyCharges", "TotalCharges",
            "service_adoption_score", "avg_monthly_charge_per_service",
            "charge_to_tenure_ratio",
        ],
        "feature_engineering.tenure_bins": [0, 12, 24, 48, 60, 73],
        "feature_engineering.tenure_labels": ["New", "Developing", "Established", "Loyal", "Champion"],
        "feature_engineering.service_cols": [
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies",
        ],
        "feature_engineering.monthly_charge_col": "MonthlyCharges",
        "feature_engineering.total_charge_col": "TotalCharges",
        "feature_engineering.tenure_col": "tenure",
        "feature_engineering.auto_payment_methods": [
            "Bank transfer (automatic)", "Credit card (automatic)"
        ],
        "feature_engineering.payment_method_col": "PaymentMethod",
        "feature_engineering.contract_col": "Contract",
        "feature_engineering.internet_service_col": "InternetService",
        "feature_engineering.derived_cols": {
            "tenure_category": "tenure_category",
            "service_adoption_score": "service_adoption_score",
            "avg_monthly_charge_per_service": "avg_monthly_charge_per_service",
            "charge_to_tenure_ratio": "charge_to_tenure_ratio",
            "payment_reliability": "payment_reliability",
            "intl_usage_flag": "intl_usage_flag",
            "has_internet": "has_internet",
            "is_long_term_contract": "is_long_term_contract",
            "num_streaming_services": "num_streaming_services",
            "num_security_services": "num_security_services",
        },
        "split.test_size": 0.20,
        "split.val_size": 0.125,
        "split.stratify": True,
        "split.random_state": 42,
        "model.models_to_train": [
            "logistic_regression", "decision_tree", "random_forest", "xgboost", "catboost"
        ],
        "model.champion_metric": "roc_auc",
        "model.models_dir": "models",
        "model.champion_dir": "models/champion",
        "model.champion_metadata_file": "models/champion/champion_metadata.json",
        "model.cv_folds": 5,
        "model.imbalance_strategy": "smote",
        "model.smote_k_neighbors": 5,
        "evaluation.threshold_metric": "f1",
        "evaluation.threshold_range.start": 0.10,
        "evaluation.threshold_range.stop": 0.90,
        "evaluation.threshold_range.step": 0.05,
        "evaluation.cost_matrix.fp_cost": 10.0,
        "evaluation.cost_matrix.fn_cost": 500.0,
        "reporting.figures_dir": "reports/figures",
        "reporting.high_risk_threshold": 0.70,
        "reporting.medium_risk_threshold": 0.40,
        "reporting.business.avg_monthly_revenue_per_customer": 65.0,
        "reporting.business.avg_customer_lifetime_months": 32.0,
        "reporting.business.retention_offer_cost": 50.0,
        "reporting.business.retention_success_rate": 0.35,
        "project.random_state": 42,
        "project.log_level": "INFO",
    }

    def get(self, dot_path: str, default: Any = None) -> Any:
        """Return stub value or default."""
        return self._data.get(dot_path, default)

    def get_section(self, section: str) -> dict:
        """Return a mock section dict."""
        return {
            k.split(".", 1)[1]: v
            for k, v in self._data.items()
            if k.startswith(f"{section}.")
        }

    @property
    def random_state(self) -> int:
        """Stub random state."""
        return 42

    def __repr__(self) -> str:
        return "StubConfig()"


@pytest.fixture
def cfg() -> Any:
    """
    Return a ConfigLoader instance, falling back to a _StubConfig if
    the real config.yaml is not found on disk.

    Yields
    ------
    ConfigLoader or _StubConfig
        The configuration accessor for the current test session.
    """
    _config_paths = [
        Path("config.yaml"),
        Path("../config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
    ]
    try:
        from src.config_loader import ConfigLoader  # type: ignore

        for p in _config_paths:
            if p.exists():
                return ConfigLoader.get_instance(str(p))
        logger.warning("config.yaml not found; using _StubConfig for tests.")
        return _StubConfig()
    except (ImportError, Exception):
        logger.warning("ConfigLoader unavailable; using _StubConfig for tests.")
        return _StubConfig()


# ---------------------------------------------------------------------------
# sample_df fixture — 200-row synthetic IBM Telco DataFrame
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Generate a 200-row synthetic DataFrame matching the IBM Telco schema.

    Random seed is fixed at 42 for full reproducibility across test runs.

    Returns
    -------
    pd.DataFrame
        Synthetic Telco dataset with all 21 expected columns and ~26% churn rate.
    """
    rng = np.random.default_rng(_SEED)
    n = _N_ROWS

    # Numerical columns
    tenure = rng.integers(1, 73, size=n).astype(int)
    monthly_charges = np.clip(rng.normal(65.0, 20.0, size=n), 18.0, 120.0)
    noise = rng.uniform(-50.0, 50.0, size=n)
    total_charges = np.round(tenure * monthly_charges + noise, 2)
    total_charges = np.clip(total_charges, 0.0, None)

    # Inject ~5 TotalCharges as string blanks to mimic IBM dataset quirk
    tc_series = total_charges.astype(object)
    blank_idx = rng.choice(n, size=5, replace=False)
    for idx in blank_idx:
        tc_series[idx] = " "

    # Build churn labels (~26% rate)
    churn_prob = rng.uniform(0.0, 1.0, size=n)
    churn = np.where(churn_prob < 0.26, "Yes", "No")

    data = {
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "gender": rng.choice(_GENDER, size=n),
        "SeniorCitizen": rng.choice([0, 1], size=n, p=[0.84, 0.16]).astype(int),
        "Partner": rng.choice(_YES_NO, size=n),
        "Dependents": rng.choice(_YES_NO, size=n, p=[0.70, 0.30]),
        "tenure": tenure,
        "PhoneService": rng.choice(_YES_NO, size=n, p=[0.10, 0.90]),
        "MultipleLines": rng.choice(_MULTIPLE_LINES, size=n),
        "InternetService": rng.choice(_INTERNET_SERVICE, size=n),
        "OnlineSecurity": rng.choice(_ONLINE_SECURITY, size=n),
        "OnlineBackup": rng.choice(_ONLINE_BACKUP, size=n),
        "DeviceProtection": rng.choice(_DEVICE_PROTECTION, size=n),
        "TechSupport": rng.choice(_TECH_SUPPORT, size=n),
        "StreamingTV": rng.choice(_STREAMING_TV, size=n),
        "StreamingMovies": rng.choice(_STREAMING_MOVIES, size=n),
        "Contract": rng.choice(_CONTRACT, size=n, p=[0.55, 0.25, 0.20]),
        "PaperlessBilling": rng.choice(_YES_NO, size=n),
        "PaymentMethod": rng.choice(_PAYMENT_METHOD, size=n),
        "MonthlyCharges": np.round(monthly_charges, 2),
        "TotalCharges": tc_series,
        "Churn": churn,
    }

    df = pd.DataFrame(data)
    logger.debug(
        "sample_df created: shape=%s, churn_rate=%.3f",
        df.shape,
        (df["Churn"] == "Yes").mean(),
    )
    return df


# ---------------------------------------------------------------------------
# small_X_y fixture — minimal numeric arrays for model unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_X_y() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a small (120 x 6) feature matrix and binary target vector for
    lightweight model smoke tests.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X : shape (120, 6)  -- standardised random floats.
        y : shape (120,)    -- binary labels with ~26% positive rate.
    """
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((120, 6)).astype(np.float32)
    y = (rng.uniform(0.0, 1.0, size=120) < 0.26).astype(int)
    logger.debug("small_X_y: X.shape=%s, positive_rate=%.3f", X.shape, y.mean())
    return X, y
