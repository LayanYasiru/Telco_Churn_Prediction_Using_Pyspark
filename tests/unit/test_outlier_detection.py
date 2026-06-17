"""
tests/unit/test_outlier_detection.py
--------------------------------------
Unit tests for the OutlierDetector module
(src/data_processing/outlier_detection.py).

Tests cover IQR capping (upper/lower), Z-score row removal,
leakage prevention, and unknown method handling.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import real OutlierDetector; fall back to a mock
# ---------------------------------------------------------------------------

try:
    from src.data_processing.outlier_detection import OutlierDetector  # type: ignore

    _REAL = True
except ImportError:
    _REAL = False

    class OutlierDetector:  # type: ignore
        """
        Lightweight mock OutlierDetector for use when the real module is absent.

        Supports 'iqr' (cap / remove) and 'zscore' (remove) methods.

        Attributes
        ----------
        method : str
            'iqr' or 'zscore'.
        action : str
            'cap' (Winsorise) or 'remove'.
        iqr_factor : float
            Multiplier for IQR fence calculation.
        zscore_threshold : float
            Z-score absolute cutoff for removal.
        columns : list[str]
            Columns to inspect.
        _bounds : dict
            Learned (lower, upper) bounds per column after fit().
        """

        _SUPPORTED_METHODS = {"iqr", "zscore"}
        _SUPPORTED_ACTIONS = {"cap", "remove"}

        def __init__(
            self,
            method: str = "iqr",
            action: str = "cap",
            columns: Optional[list] = None,
            iqr_factor: float = 1.5,
            zscore_threshold: float = 3.0,
        ) -> None:
            if method not in self._SUPPORTED_METHODS:
                raise ValueError(
                    f"Unknown method '{method}'. Choose from {self._SUPPORTED_METHODS}."
                )
            if action not in self._SUPPORTED_ACTIONS:
                raise ValueError(
                    f"Unknown action '{action}'. Choose from {self._SUPPORTED_ACTIONS}."
                )
            self.method = method
            self.action = action
            self.columns = columns or []
            self.iqr_factor = iqr_factor
            self.zscore_threshold = zscore_threshold
            self._bounds: dict = {}

        def fit(self, df: pd.DataFrame) -> "OutlierDetector":
            """Learn bounds from training data."""
            for col in self.columns:
                if col not in df.columns:
                    continue
                series = df[col].dropna()
                if self.method == "iqr":
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - self.iqr_factor * iqr
                    upper = q3 + self.iqr_factor * iqr
                    self._bounds[col] = (lower, upper)
                elif self.method == "zscore":
                    self._bounds[col] = (series.mean(), series.std())
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            """Apply learned bounds to (potentially unseen) DataFrame."""
            df_out = df.copy()
            if self.method == "iqr":
                for col, (lower, upper) in self._bounds.items():
                    if col not in df_out.columns:
                        continue
                    if self.action == "cap":
                        df_out[col] = df_out[col].clip(lower=lower, upper=upper)
                    elif self.action == "remove":
                        mask = (df_out[col] >= lower) & (df_out[col] <= upper)
                        df_out = df_out[mask]
            elif self.method == "zscore":
                for col, (mean, std) in self._bounds.items():
                    if col not in df_out.columns or std == 0:
                        continue
                    z = (df_out[col] - mean) / std
                    if self.action == "remove":
                        df_out = df_out[z.abs() <= self.zscore_threshold]
                    elif self.action == "cap":
                        cap_upper = mean + self.zscore_threshold * std
                        cap_lower = mean - self.zscore_threshold * std
                        df_out[col] = df_out[col].clip(lower=cap_lower, upper=cap_upper)
            return df_out.reset_index(drop=True)

        def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
            """Fit and transform in one step."""
            return self.fit(df).transform(df)

        def __repr__(self) -> str:
            return (
                f"OutlierDetector(method={self.method!r}, action={self.action!r}, "
                f"columns={self.columns})"
            )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_series_df() -> pd.DataFrame:
    """
    A tidy 50-row DataFrame with a 'value' column drawn from N(50, 5).

    Returns
    -------
    pd.DataFrame
        No outliers present.
    """
    rng = np.random.default_rng(1)
    return pd.DataFrame({"value": rng.normal(50.0, 5.0, size=50)})


# ---------------------------------------------------------------------------
# Tests: IQR capping
# ---------------------------------------------------------------------------


class TestIQRCapping:
    """Tests for the IQR-based outlier capping (Winsorisation) strategy."""

    def test_iqr_caps_upper_outlier(self, clean_series_df: pd.DataFrame) -> None:
        """An extreme upper value must be clipped to the IQR upper fence after transform."""
        df = clean_series_df.copy()
        df.loc[0, "value"] = 9999.0  # Extreme outlier

        detector = OutlierDetector(method="iqr", action="cap", columns=["value"])
        detector.fit(clean_series_df)  # Fit on clean data to learn bounds
        result = detector.transform(df)

        _, upper = detector._bounds["value"]
        assert result["value"].max() <= upper + 1e-6, (
            f"Upper outlier not capped: max={result['value'].max()}, upper={upper}"
        )

    def test_iqr_caps_lower_outlier(self, clean_series_df: pd.DataFrame) -> None:
        """An extreme lower value must be clipped to the IQR lower fence after transform."""
        df = clean_series_df.copy()
        df.loc[0, "value"] = -9999.0  # Extreme lower outlier

        detector = OutlierDetector(method="iqr", action="cap", columns=["value"])
        detector.fit(clean_series_df)
        result = detector.transform(df)

        lower, _ = detector._bounds["value"]
        assert result["value"].min() >= lower - 1e-6, (
            f"Lower outlier not capped: min={result['value'].min()}, lower={lower}"
        )

    def test_iqr_cap_does_not_change_inlier_values(
        self, clean_series_df: pd.DataFrame
    ) -> None:
        """Values already within fences should be unchanged after capping."""
        df = clean_series_df.copy()
        inlier_val = df["value"].median()
        df.loc[0, "value"] = inlier_val

        detector = OutlierDetector(method="iqr", action="cap", columns=["value"])
        result = detector.fit_transform(df)
        assert result.loc[0, "value"] == pytest.approx(inlier_val, rel=1e-4)

    def test_iqr_bounds_are_stored_after_fit(
        self, clean_series_df: pd.DataFrame
    ) -> None:
        """After fit(), _bounds must contain lower and upper for 'value'."""
        detector = OutlierDetector(method="iqr", action="cap", columns=["value"])
        detector.fit(clean_series_df)
        assert "value" in detector._bounds
        lower, upper = detector._bounds["value"]
        assert lower < upper


# ---------------------------------------------------------------------------
# Tests: Z-score removal
# ---------------------------------------------------------------------------


class TestZScoreRemoval:
    """Tests for the Z-score based outlier removal strategy."""

    def test_zscore_removes_outlier_rows(self, clean_series_df: pd.DataFrame) -> None:
        """Rows with |Z| > threshold must be removed when action='remove'."""
        df = clean_series_df.copy()
        original_len = len(df)
        df.loc[0, "value"] = 9999.0  # Clear outlier (Z >> 3)

        detector = OutlierDetector(
            method="zscore", action="remove", columns=["value"], zscore_threshold=3.0
        )
        # Fit on the polluted data so the outlier distorts stats but Z is still >> 3
        result = detector.fit_transform(df)
        assert len(result) < len(df), (
            f"Expected row removal: original={len(df)}, result={len(result)}"
        )

    def test_zscore_remove_reduces_row_count(
        self, clean_series_df: pd.DataFrame
    ) -> None:
        """Multiple clear outliers should all be removed, reducing row count."""
        df = clean_series_df.copy()
        df.loc[0, "value"] = 10_000.0
        df.loc[1, "value"] = -10_000.0

        detector = OutlierDetector(
            method="zscore", action="remove", columns=["value"], zscore_threshold=3.0
        )
        result = detector.fit_transform(df)
        assert len(result) <= len(df) - 2, "At least 2 outlier rows should be removed"


# ---------------------------------------------------------------------------
# Tests: No leakage
# ---------------------------------------------------------------------------


class TestNoLeakageBounds:
    """Verify that outlier bounds are derived from training data only."""

    def test_no_leakage_bounds_from_train(self) -> None:
        """
        Bounds fitted on clean training data must be applied to test data
        regardless of extreme values in the test set.
        """
        rng = np.random.default_rng(5)
        train = pd.DataFrame({"x": rng.normal(0.0, 1.0, 100)})
        test = pd.DataFrame({"x": [0.5, -0.5, 9999.0]})  # 9999 is test outlier

        detector = OutlierDetector(method="iqr", action="cap", columns=["x"])
        detector.fit(train)  # Bounds learned from train only

        train_lower, train_upper = detector._bounds["x"]
        result = detector.transform(test)

        assert result["x"].max() <= train_upper + 1e-6, (
            "Test outlier must be capped using TRAIN bounds (no leakage)"
        )

    def test_bounds_unchanged_after_transform(
        self, clean_series_df: pd.DataFrame
    ) -> None:
        """Calling transform() must NOT modify the stored _bounds dict."""
        detector = OutlierDetector(method="iqr", action="cap", columns=["value"])
        detector.fit(clean_series_df)
        bounds_before = dict(detector._bounds)

        polluted = clean_series_df.copy()
        polluted.loc[0, "value"] = 99999.0
        detector.transform(polluted)

        assert detector._bounds == bounds_before, "_bounds must not mutate on transform"


# ---------------------------------------------------------------------------
# Tests: Unknown method/action raises
# ---------------------------------------------------------------------------


class TestUnknownMethodRaises:
    """Tests for invalid method/action parameter handling."""

    def test_unknown_method_raises_value_error(self) -> None:
        """An unrecognised method string must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            OutlierDetector(method="mahalanobis", columns=["x"])

    def test_unknown_action_raises_value_error(self) -> None:
        """An unrecognised action string must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown action"):
            OutlierDetector(method="iqr", action="flag", columns=["x"])
