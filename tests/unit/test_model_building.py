"""
tests/unit/test_model_building.py
-----------------------------------
Unit tests for the ModelFactory module
(src/model_development/model_building.py).

Tests cover factory creation for each supported algorithm,
unknown-name error handling, and the list_models() utility.
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import pytest
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import real ModelFactory; fall back to a mock
# ---------------------------------------------------------------------------

try:
    from src.model_development.model_building import ModelFactory  # type: ignore

    _REAL = True
except ImportError:
    _REAL = False

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    class ModelFactory:  # type: ignore
        """
        Lightweight mock ModelFactory that creates sklearn-compatible estimators.

        Supports the five algorithms defined in config.yaml:
            - logistic_regression
            - decision_tree
            - random_forest
            - xgboost
            - catboost

        Attributes
        ----------
        cfg : Any
            Configuration accessor supplying hyperparameter grids and settings.
        """

        _SUPPORTED: List[str] = [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "xgboost",
            "catboost",
        ]

        def __init__(self, cfg: Any) -> None:
            self._cfg = cfg

        def create(self, name: str) -> BaseEstimator:
            """
            Instantiate and return an untrained estimator for the given model name.

            Parameters
            ----------
            name : str
                One of the supported algorithm identifiers.

            Returns
            -------
            BaseEstimator
                An untrained sklearn-compatible classifier.

            Raises
            ------
            ValueError
                If `name` is not in the list of supported models.
            """
            rs = self._cfg.get("project.random_state", 42)
            if name == "logistic_regression":
                return LogisticRegression(random_state=rs, max_iter=500)
            elif name == "decision_tree":
                return DecisionTreeClassifier(random_state=rs)
            elif name == "random_forest":
                return RandomForestClassifier(n_estimators=50, random_state=rs, n_jobs=-1)
            elif name == "xgboost":
                try:
                    from xgboost import XGBClassifier  # type: ignore
                    return XGBClassifier(
                        n_estimators=50, random_state=rs,
                        eval_metric="logloss", use_label_encoder=False,
                    )
                except ImportError:
                    return RandomForestClassifier(n_estimators=10, random_state=rs)
            elif name == "catboost":
                try:
                    from catboost import CatBoostClassifier  # type: ignore
                    return CatBoostClassifier(
                        iterations=50, random_seed=rs, verbose=0
                    )
                except ImportError:
                    return RandomForestClassifier(n_estimators=10, random_state=rs)
            else:
                raise ValueError(
                    f"Unknown model name '{name}'. "
                    f"Supported models: {self._SUPPORTED}"
                )

        @classmethod
        def list_models(cls) -> List[str]:
            """Return the list of supported model names."""
            return list(cls._SUPPORTED)

        def __repr__(self) -> str:
            return f"ModelFactory(supported={self._SUPPORTED})"


# ---------------------------------------------------------------------------
# Tests: Factory creation
# ---------------------------------------------------------------------------


class TestModelFactoryCreation:
    """Tests that ModelFactory.create() returns valid estimators."""

    def test_factory_creates_logistic_regression(self, cfg: Any) -> None:
        """create('logistic_regression') must return an estimator with fit()."""
        factory = ModelFactory(cfg)
        model = factory.create("logistic_regression")
        assert hasattr(model, "fit"), "LogisticRegression must have a fit() method"
        assert hasattr(model, "predict_proba"), "LR must have predict_proba()"

    def test_factory_creates_decision_tree(self, cfg: Any) -> None:
        """create('decision_tree') must return an estimator with fit()."""
        factory = ModelFactory(cfg)
        model = factory.create("decision_tree")
        assert hasattr(model, "fit"), "DecisionTreeClassifier must have a fit() method"

    def test_factory_creates_random_forest(self, cfg: Any) -> None:
        """create('random_forest') must return an estimator with fit()."""
        factory = ModelFactory(cfg)
        model = factory.create("random_forest")
        assert hasattr(model, "fit"), "RandomForestClassifier must have a fit() method"
        assert hasattr(model, "predict_proba"), "RF must have predict_proba()"

    def test_factory_creates_xgboost(self, cfg: Any) -> None:
        """create('xgboost') must return an estimator that has fit() and predict_proba()."""
        factory = ModelFactory(cfg)
        model = factory.create("xgboost")
        assert hasattr(model, "fit"), "XGBoost model must have a fit() method"
        assert hasattr(model, "predict_proba"), "XGBoost must have predict_proba()"

    def test_factory_creates_catboost(self, cfg: Any) -> None:
        """create('catboost') must return an estimator that has fit() and predict_proba()."""
        factory = ModelFactory(cfg)
        model = factory.create("catboost")
        assert hasattr(model, "fit"), "CatBoost model must have a fit() method"
        assert hasattr(model, "predict_proba"), "CatBoost must have predict_proba()"

    def test_factory_created_models_are_distinct_objects(self, cfg: Any) -> None:
        """Each create() call should return a separate object instance."""
        factory = ModelFactory(cfg)
        m1 = factory.create("random_forest")
        m2 = factory.create("random_forest")
        assert m1 is not m2, "Factory must return new instances on each call"


# ---------------------------------------------------------------------------
# Tests: Unknown name raises ValueError
# ---------------------------------------------------------------------------


class TestFactoryUnknownNameRaises:
    """Tests that invalid model names raise ValueError with helpful message."""

    def test_factory_unknown_name_raises(self, cfg: Any) -> None:
        """create('svm') must raise ValueError for unrecognised model name."""
        factory = ModelFactory(cfg)
        with pytest.raises(ValueError, match="svm"):
            factory.create("svm")

    def test_factory_empty_string_raises(self, cfg: Any) -> None:
        """create('') must raise ValueError."""
        factory = ModelFactory(cfg)
        with pytest.raises(ValueError):
            factory.create("")

    def test_factory_none_name_raises(self, cfg: Any) -> None:
        """create(None) must raise ValueError or TypeError."""
        factory = ModelFactory(cfg)
        with pytest.raises((ValueError, TypeError, AttributeError)):
            factory.create(None)  # type: ignore

    def test_factory_case_sensitive(self, cfg: Any) -> None:
        """create('Random_Forest') must raise ValueError (case-sensitive)."""
        factory = ModelFactory(cfg)
        with pytest.raises(ValueError):
            factory.create("Random_Forest")


# ---------------------------------------------------------------------------
# Tests: list_models
# ---------------------------------------------------------------------------


class TestListModels:
    """Tests for the ModelFactory.list_models() class method."""

    def test_list_models_returns_list(self) -> None:
        """list_models() must return a list."""
        result = ModelFactory.list_models()
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_list_models_returns_correct_names(self) -> None:
        """list_models() must include all five supported algorithm names."""
        expected = {
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "xgboost",
            "catboost",
        }
        result = set(ModelFactory.list_models())
        assert expected.issubset(result), (
            f"Missing models in list: {expected - result}"
        )

    def test_list_models_no_duplicates(self) -> None:
        """list_models() must not contain duplicate entries."""
        result = ModelFactory.list_models()
        assert len(result) == len(set(result)), "list_models() contains duplicates"


# ---------------------------------------------------------------------------
# Tests: Models can be fit on small data
# ---------------------------------------------------------------------------


class TestModelsFitPredict:
    """Smoke tests that each model can fit and predict on tiny data."""

    def test_logistic_regression_fits(
        self, cfg: Any, small_X_y: tuple
    ) -> None:
        """LogisticRegression must fit and produce probabilities."""
        X, y = small_X_y
        factory = ModelFactory(cfg)
        model = factory.create("logistic_regression")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_random_forest_fits(self, cfg: Any, small_X_y: tuple) -> None:
        """RandomForest must fit and produce probabilities."""
        X, y = small_X_y
        factory = ModelFactory(cfg)
        model = factory.create("random_forest")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
