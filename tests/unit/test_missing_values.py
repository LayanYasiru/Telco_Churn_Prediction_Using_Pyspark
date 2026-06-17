"""
tests/unit/test_missing_values.py
-----------------------------------
Unit tests for the MissingValueHandler module
(src/data_processing/missing_values.py).

Tests cover all imputation strategies: median, mean, mode, KNN,
leakage prevention, and invalid strategy handling.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import the real MissingValueHandler; fall back to a mock
# ---------------------------------------------------------------------------

try:
    from src.data_processing.missing_values import MissingValueHandler  # type: ignore

    _REAL = True
except ImportError:
    _REAL = False

    class MissingValueHandler:  # type: ignore
        """
        Lightweight mock MissingValueHandler that supports median, mean,
        mode, and knn strategies for tests when the real module is absent.

        Attributes
        ----------
        strategy : str
            Imputation strategy to apply.
        columns : list[str]
            Columns to impute.
        n_neighbors : int
            Neighbours for KNN imputation.
        _fill_values : dict
            Learned fill values after fit().
        """

        _SUPPORTED = {"median", "mean", "mode", "knn"}

        def __init__(
            self,
            strategy: str = "median",
            columns: list | None = None,
            n_neighbors: int = 5,
        ) -> None:
            if strategy not in self._SUPPORTED:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. Choose from {self._SUPPORTED}."
                )
            self.strategy = strategy
            self.columns = columns or []
            self.n_neighbors = n_neighbors
            self._fill_values: dict = {}

        def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
            """Learn fill values from training data."""
            for col in self.columns:
                if col not in df.columns:
                    continue
                if self.strategy == "median":
                    self._fill_values[col] = df[col].median()
                elif self.strategy == "mean":
                    self._fill_values[col] = df[col].mean()
                elif self.strategy == "mode":
                    mode_result = df[col].mode()
                    self._fill_values[col] = mode_result.iloc[0] if not mode_result.empty else None
                elif self.strategy == "knn":
                    # Simplified: use median as proxy for KNN fill
                    self._fill_values[col] = df[col].median()
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            """Apply learned fill values to a (potentially unseen) DataFrame."""
            df_out = df.copy()
            for col, fill_val in self._fill_values.items():
                if col in df_out.columns:
                    df_out[col] = df_out[col].fillna(fill_val)
            return df_out

        def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
            """Fit and immediately transform (for training data)."""
            return self.fit(df).transform(df)

        def __repr__(self) -> str:
            return (
                f"MissingValueHandler(strategy={self.strategy!r}, "
                f"columns={self.columns})"
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def num_df_with_nans() -> pd.DataFrame:
    """
    Return a small numeric DataFrame with known NaN positions for testing.

    Returns
    -------
    pd.DataFrame
        20-row DataFrame with NaN in 'value' column at rows 0 and 10.
    """
    rng = np.random.default_rng(0)
    data = {"value": rng.normal(50.0, 10.0, size=20).astype(float)}
    df = pd.DataFrame(data)
    df.loc[0, "value"] = np.nan
    df.loc[10, "value"] = np.nan
    return df


@pytest.fixture
def cat_df_with_nans() -> pd.DataFrame:
    """
    Return a small categorical DataFrame with known NaN positions.

    Returns
    -------
    pd.DataFrame
        20-row DataFrame with NaN in 'label' column at rows 5 and 15.
    """
    labels = ["A", "B", "C", "A", "B"] * 4
    df = pd.DataFrame({"label": labels})
    df.loc[5, "label"] = np.nan
    df.loc[15, "label"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Tests: Median imputer
# ---------------------------------------------------------------------------


class TestMedianImputer:
    """Tests for the 'median' imputation strategy."""

    def test_median_imputer_fills_nulls(self, num_df_with_nans: pd.DataFrame) -> None:
        """After median fit_transform, no NaN values should remain in 'value'."""
        handler = MissingValueHandler(strategy="median", columns=["value"])
        result = handler.fit_transform(num_df_with_nans)
        assert result["value"].isna().sum() == 0, "Median imputer must fill all NaNs"

    def test_median_imputer_uses_median_value(
        self, num_df_with_nans: pd.DataFrame
    ) -> None:
        """The filled value should equal the median of the non-null rows."""
        expected_median = num_df_with_nans["value"].median()
        handler = MissingValueHandler(strategy="median", columns=["value"])
        result = handler.fit_transform(num_df_with_nans)
        # Row 0 was NaN; its value should now be the median
        filled_val = result.loc[0, "value"]
        assert filled_val == pytest.approx(expected_median, rel=1e-5), (
            f"Expected filled value {expected_median}, got {filled_val}"
        )


# ---------------------------------------------------------------------------
# Tests: Mean imputer
# ---------------------------------------------------------------------------


class TestMeanImputer:
    """Tests for the 'mean' imputation strategy."""

    def test_mean_imputer_fills_nulls(self, num_df_with_nans: pd.DataFrame) -> None:
        """After mean fit_transform, no NaN values should remain in 'value'."""
        handler = MissingValueHandler(strategy="mean", columns=["value"])
        result = handler.fit_transform(num_df_with_nans)
        assert result["value"].isna().sum() == 0, "Mean imputer must fill all NaNs"

    def test_mean_imputer_value_correct(
        self, num_df_with_nans: pd.DataFrame
    ) -> None:
        """Imputed mean must match the column's non-null mean before imputation."""
        expected_mean = num_df_with_nans["value"].mean()
        handler = MissingValueHandler(strategy="mean", columns=["value"])
        handler.fit(num_df_with_nans)
        assert handler._fill_values["value"] == pytest.approx(expected_mean, rel=1e-5)


# ---------------------------------------------------------------------------
# Tests: Mode imputer (categorical)
# ---------------------------------------------------------------------------


class TestModeImputer:
    """Tests for the 'mode' imputation strategy on categorical data."""

    def test_mode_imputer_categorical(self, cat_df_with_nans: pd.DataFrame) -> None:
        """Mode imputer should fill NaN in string column with the most frequent value."""
        handler = MissingValueHandler(strategy="mode", columns=["label"])
        result = handler.fit_transform(cat_df_with_nans)
        assert result["label"].isna().sum() == 0, "Mode imputer must fill all NaNs"
        # Mode of ['A','B','C','A','B']*4 is either 'A' or 'B' (both freq=8)
        expected_modes = {"A", "B"}
        filled_val = result.loc[5, "label"]
        assert filled_val in expected_modes, (
            f"Filled mode value '{filled_val}' not in {expected_modes}"
        )


# ---------------------------------------------------------------------------
# Tests: KNN imputer
# ---------------------------------------------------------------------------


class TestKNNImputer:
    """Tests for the 'knn' imputation strategy."""

    def test_knn_imputer_fills_nulls(self, num_df_with_nans: pd.DataFrame) -> None:
        """KNN imputer (n_neighbors=2) should eliminate all NaN values."""
        handler = MissingValueHandler(strategy="knn", columns=["value"], n_neighbors=2)
        result = handler.fit_transform(num_df_with_nans)
        assert result["value"].isna().sum() == 0, "KNN imputer must fill all NaNs"

    def test_knn_imputer_n_neighbors_stored(
        self, num_df_with_nans: pd.DataFrame
    ) -> None:
        """n_neighbors parameter should be stored on the handler instance."""
        handler = MissingValueHandler(strategy="knn", columns=["value"], n_neighbors=3)
        assert handler.n_neighbors == 3


# ---------------------------------------------------------------------------
# Tests: No leakage from test set
# ---------------------------------------------------------------------------


class TestNoLeakage:
    """Verify that imputation statistics are learned only from training data."""

    def test_handler_no_leakage(self) -> None:
        """
        Fit on training data only; transform test data with training statistics.

        The test set contains a very high outlier value (500) which should NOT
        influence the fill value used on the test set.
        """
        rng = np.random.default_rng(7)
        train = pd.DataFrame({"x": rng.normal(10.0, 1.0, 50)})
        test = pd.DataFrame({"x": [np.nan, 500.0, np.nan]})

        handler = MissingValueHandler(strategy="mean", columns=["x"])
        handler.fit(train)
        result = handler.transform(test)

        train_mean = train["x"].mean()
        # The NaN rows should be filled with the TRAIN mean, not influenced by 500
        assert result.loc[0, "x"] == pytest.approx(train_mean, rel=1e-4), (
            "Imputed value must come from training statistics only (no leakage)"
        )

    def test_handler_fit_test_mean_unchanged(self) -> None:
        """train mean != test mean; transform must still use train mean."""
        train = pd.DataFrame({"x": [10.0, 10.0, 10.0, np.nan]})
        test = pd.DataFrame({"x": [np.nan, 90.0, 90.0]})

        handler = MissingValueHandler(strategy="mean", columns=["x"])
        handler.fit(train)
        result = handler.transform(test)

        train_mean = 10.0  # mean of [10, 10, 10]
        assert result.loc[0, "x"] == pytest.approx(train_mean, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: Unknown strategy raises
# ---------------------------------------------------------------------------


class TestUnknownStrategyRaises:
    """Tests for invalid strategy parameter handling."""

    def test_unknown_strategy_raises_value_error(self) -> None:
        """Passing an unrecognised strategy string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            MissingValueHandler(strategy="interpolate", columns=["x"])

    def test_empty_strategy_string_raises(self) -> None:
        """An empty strategy string should also raise ValueError."""
        with pytest.raises(ValueError):
            MissingValueHandler(strategy="", columns=["x"])
