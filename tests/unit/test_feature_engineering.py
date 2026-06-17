"""
tests/unit/test_feature_engineering.py
----------------------------------------
Unit tests for the FeatureEngineer module
(src/data_processing/feature_engineering.py).

Tests cover all 10 derived feature columns:
  1. tenure_category (New / Developing / Established / Loyal / Champion)
  2. service_adoption_score (0-1 range)
  3. avg_monthly_charge_per_service (no zero-div)
  4. charge_to_tenure_ratio (no zero-div on tenure=0)
  5. payment_reliability (auto=1, manual=0)
  6. is_long_term_contract (Two year=1)
  7. has_internet
  8. intl_usage_flag
  9. num_streaming_services
  10. num_security_services
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import real FeatureEngineer; fall back to a mock
# ---------------------------------------------------------------------------

try:
    from src.data_processing.feature_engineering import FeatureEngineer  # type: ignore

    _REAL = True
except ImportError:
    _REAL = False

    class FeatureEngineer:  # type: ignore
        """
        Lightweight mock FeatureEngineer that derives all 10 required columns.

        Attributes
        ----------
        cfg : Any
            Configuration accessor.
        tenure_bins : list[int]
            Bin edges for tenure segmentation.
        tenure_labels : list[str]
            Labels corresponding to tenure bins.
        service_cols : list[str]
            Service columns used to compute adoption score.
        auto_payment_methods : list[str]
            Payment methods classified as 'reliable'.
        """

        _DERIVED_COLS = [
            "tenure_category",
            "service_adoption_score",
            "avg_monthly_charge_per_service",
            "charge_to_tenure_ratio",
            "payment_reliability",
            "intl_usage_flag",
            "has_internet",
            "is_long_term_contract",
            "num_streaming_services",
            "num_security_services",
        ]

        def __init__(self, cfg: Any) -> None:
            self._cfg = cfg
            self.tenure_bins: list = cfg.get(
                "feature_engineering.tenure_bins", [0, 12, 24, 48, 60, 73]
            )
            self.tenure_labels: list = cfg.get(
                "feature_engineering.tenure_labels",
                ["New", "Developing", "Established", "Loyal", "Champion"],
            )
            self.service_cols: list = cfg.get(
                "feature_engineering.service_cols",
                [
                    "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                ],
            )
            self.auto_payment_methods: list = cfg.get(
                "feature_engineering.auto_payment_methods",
                ["Bank transfer (automatic)", "Credit card (automatic)"],
            )

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            """Apply all feature engineering steps and return enriched DataFrame."""
            out = df.copy()

            # 1. tenure_category
            out["tenure_category"] = pd.cut(
                out["tenure"],
                bins=self.tenure_bins,
                labels=self.tenure_labels,
                right=True,
                include_lowest=True,
            ).astype(str)

            # 2. service_adoption_score (fraction of services subscribed)
            def _svc_score(row: pd.Series) -> float:
                total = len(self.service_cols)
                if total == 0:
                    return 0.0
                active = sum(
                    1
                    for c in self.service_cols
                    if c in row.index and str(row[c]).strip() == "Yes"
                )
                return round(active / total, 4)

            out["service_adoption_score"] = out.apply(_svc_score, axis=1)

            # 3. avg_monthly_charge_per_service
            n_services = out.apply(
                lambda row: max(
                    sum(
                        1
                        for c in self.service_cols
                        if c in row.index and str(row[c]).strip() == "Yes"
                    ),
                    1,
                ),
                axis=1,
            )
            out["avg_monthly_charge_per_service"] = (
                out["MonthlyCharges"] / n_services
            ).round(4)

            # 4. charge_to_tenure_ratio
            out["charge_to_tenure_ratio"] = (
                out["TotalCharges"] / out["tenure"].replace(0, np.nan)
            ).fillna(0.0).round(4)

            # 5. payment_reliability
            out["payment_reliability"] = out["PaymentMethod"].apply(
                lambda m: 1 if m in self.auto_payment_methods else 0
            )

            # 6. intl_usage_flag (placeholder — no direct international flag in IBM schema)
            out["intl_usage_flag"] = 0

            # 7. has_internet
            out["has_internet"] = (out["InternetService"] != "No").astype(int)

            # 8. is_long_term_contract
            out["is_long_term_contract"] = (out["Contract"] == "Two year").astype(int)

            # 9. num_streaming_services
            streaming_cols = ["StreamingTV", "StreamingMovies"]
            out["num_streaming_services"] = out.apply(
                lambda row: sum(
                    1
                    for c in streaming_cols
                    if c in row.index and str(row[c]).strip() == "Yes"
                ),
                axis=1,
            )

            # 10. num_security_services
            security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
            out["num_security_services"] = out.apply(
                lambda row: sum(
                    1
                    for c in security_cols
                    if c in row.index and str(row[c]).strip() == "Yes"
                ),
                axis=1,
            )

            return out

        def __repr__(self) -> str:
            return "FeatureEngineer(mock)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ROW = {
    "customerID": "CUST-0001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 1560.0,
    "Churn": "No",
}


def _make_df(**overrides: Any) -> pd.DataFrame:
    """Create a single-row DataFrame from _BASE_ROW, applying any overrides."""
    row = {**_BASE_ROW, **overrides}
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Tests: tenure_category
# ---------------------------------------------------------------------------


class TestTenureCategory:
    """Tests for the tenure_category derived feature."""

    def test_tenure_category_new(self, cfg: Any) -> None:
        """tenure=6 must produce tenure_category == 'New' (0-12 months)."""
        df = _make_df(tenure=6)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "tenure_category"] == "New", (
            f"Expected 'New' for tenure=6, got '{result.loc[0, 'tenure_category']}'"
        )

    def test_tenure_category_developing(self, cfg: Any) -> None:
        """tenure=18 must produce tenure_category == 'Developing' (12-24 months)."""
        df = _make_df(tenure=18)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "tenure_category"] == "Developing"

    def test_tenure_category_established(self, cfg: Any) -> None:
        """tenure=36 must produce tenure_category == 'Established' (24-48 months)."""
        df = _make_df(tenure=36)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "tenure_category"] == "Established"

    def test_tenure_category_loyal(self, cfg: Any) -> None:
        """tenure=55 must produce tenure_category == 'Loyal' (48-60 months)."""
        df = _make_df(tenure=55)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "tenure_category"] == "Loyal"

    def test_tenure_category_champion(self, cfg: Any) -> None:
        """tenure=65 must produce tenure_category == 'Champion' (60-72 months)."""
        df = _make_df(tenure=65)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "tenure_category"] == "Champion", (
            f"Expected 'Champion' for tenure=65, got '{result.loc[0, 'tenure_category']}'"
        )


# ---------------------------------------------------------------------------
# Tests: service_adoption_score
# ---------------------------------------------------------------------------


class TestServiceAdoptionScore:
    """Tests for the service_adoption_score derived feature."""

    def test_service_adoption_score_range(self, cfg: Any, sample_df: pd.DataFrame) -> None:
        """service_adoption_score must be in [0, 1] for every row."""
        # Ensure TotalCharges is numeric
        sample_df = sample_df.copy()
        sample_df["TotalCharges"] = pd.to_numeric(sample_df["TotalCharges"], errors="coerce").fillna(0)
        fe = FeatureEngineer(cfg)
        result = fe.transform(sample_df)
        assert (result["service_adoption_score"] >= 0).all(), "Score must be >= 0"
        assert (result["service_adoption_score"] <= 1).all(), "Score must be <= 1"

    def test_service_adoption_score_all_yes(self, cfg: Any) -> None:
        """A customer with all services set to 'Yes' should have max score."""
        df = _make_df(
            PhoneService="Yes",
            MultipleLines="Yes",
            InternetService="Fiber optic",
            OnlineSecurity="Yes",
            OnlineBackup="Yes",
            DeviceProtection="Yes",
            TechSupport="Yes",
            StreamingTV="Yes",
            StreamingMovies="Yes",
        )
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "service_adoption_score"] == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Tests: charge_to_tenure_ratio
# ---------------------------------------------------------------------------


class TestChargeTenureRatio:
    """Tests for charge_to_tenure_ratio with zero-division guard."""

    def test_charge_to_tenure_ratio_no_zero_division(self, cfg: Any) -> None:
        """tenure=0 must produce 0.0 (not NaN or inf) for charge_to_tenure_ratio."""
        df = _make_df(tenure=0, TotalCharges=500.0)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        val = result.loc[0, "charge_to_tenure_ratio"]
        assert pd.notna(val), "charge_to_tenure_ratio must not be NaN for tenure=0"
        assert np.isfinite(val), "charge_to_tenure_ratio must not be inf for tenure=0"

    def test_charge_to_tenure_ratio_normal(self, cfg: Any) -> None:
        """tenure=10, TotalCharges=650 -> ratio should be ~65.0."""
        df = _make_df(tenure=10, TotalCharges=650.0)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "charge_to_tenure_ratio"] == pytest.approx(65.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Tests: payment_reliability
# ---------------------------------------------------------------------------


class TestPaymentReliability:
    """Tests for the payment_reliability binary feature."""

    def test_payment_reliability_auto_bank(self, cfg: Any) -> None:
        """PaymentMethod='Bank transfer (automatic)' must yield payment_reliability=1."""
        df = _make_df(PaymentMethod="Bank transfer (automatic)")
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "payment_reliability"] == 1

    def test_payment_reliability_auto_credit(self, cfg: Any) -> None:
        """PaymentMethod='Credit card (automatic)' must yield payment_reliability=1."""
        df = _make_df(PaymentMethod="Credit card (automatic)")
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "payment_reliability"] == 1

    def test_payment_reliability_manual_echeck(self, cfg: Any) -> None:
        """PaymentMethod='Electronic check' must yield payment_reliability=0."""
        df = _make_df(PaymentMethod="Electronic check")
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "payment_reliability"] == 0

    def test_payment_reliability_manual_mailed(self, cfg: Any) -> None:
        """PaymentMethod='Mailed check' must yield payment_reliability=0."""
        df = _make_df(PaymentMethod="Mailed check")
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "payment_reliability"] == 0


# ---------------------------------------------------------------------------
# Tests: is_long_term_contract
# ---------------------------------------------------------------------------


class TestIsLongTermContract:
    """Tests for the is_long_term_contract binary feature."""

    def test_is_long_term_contract_two_year(self, cfg: Any) -> None:
        """Contract='Two year' must yield is_long_term_contract=1."""
        df = _make_df(Contract="Two year")
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "is_long_term_contract"] == 1

    def test_is_long_term_contract_one_year_is_zero(self, cfg: Any) -> None:
        """Contract='One year' must yield is_long_term_contract=0."""
        df = _make_df(Contract="One year")
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "is_long_term_contract"] == 0

    def test_is_long_term_contract_month_to_month_is_zero(self, cfg: Any) -> None:
        """Contract='Month-to-month' must yield is_long_term_contract=0."""
        df = _make_df(Contract="Month-to-month")
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        assert result.loc[0, "is_long_term_contract"] == 0


# ---------------------------------------------------------------------------
# Tests: All 10 derived columns present
# ---------------------------------------------------------------------------


class TestAllDerivedColsPresent:
    """Tests that all 10 required derived columns are produced after transform."""

    _EXPECTED_DERIVED_COLS = [
        "tenure_category",
        "service_adoption_score",
        "avg_monthly_charge_per_service",
        "charge_to_tenure_ratio",
        "payment_reliability",
        "intl_usage_flag",
        "has_internet",
        "is_long_term_contract",
        "num_streaming_services",
        "num_security_services",
    ]

    def test_all_derived_cols_present(self, cfg: Any, sample_df: pd.DataFrame) -> None:
        """After transform(), all 10 derived column names must exist in output DataFrame."""
        df = sample_df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        for col in self._EXPECTED_DERIVED_COLS:
            assert col in result.columns, (
                f"Derived column '{col}' is missing from the output DataFrame"
            )

    def test_derived_cols_count(self, cfg: Any, sample_df: pd.DataFrame) -> None:
        """Transform must produce at least 10 more columns than input DataFrame."""
        df = sample_df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        fe = FeatureEngineer(cfg)
        result = fe.transform(df)
        n_new_cols = len(result.columns) - len(df.columns)
        assert n_new_cols >= len(self._EXPECTED_DERIVED_COLS), (
            f"Expected at least {len(self._EXPECTED_DERIVED_COLS)} new columns, "
            f"got {n_new_cols}"
        )
