"""
src/model_development/model_building.py
────────────────────────────────────────
Factory module for constructing scikit-learn-compatible classification
estimators for the Telco Churn Prediction pipeline.

All hyper-parameter defaults and imbalance strategies are sourced from
``config.yaml`` via :class:`~src.config_loader.ConfigLoader`.

Supported models
----------------
- ``logistic_regression``   → :class:`sklearn.linear_model.LogisticRegression`
- ``decision_tree``         → :class:`sklearn.tree.DecisionTreeClassifier`
- ``random_forest``         → :class:`sklearn.ensemble.RandomForestClassifier`
- ``xgboost``               → :class:`xgboost.XGBClassifier`
- ``catboost``              → :class:`catboost.CatBoostClassifier`

Usage
-----
>>> from src.model_development.model_building import ModelFactory
>>> factory = ModelFactory()
>>> model = factory.create("random_forest")
>>> print(factory.list_models())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for optional heavy dependencies
# ---------------------------------------------------------------------------

def _import_xgboost() -> Any:
    """Lazily import XGBClassifier and raise a clear error if missing."""
    try:
        from xgboost import XGBClassifier  # type: ignore[import]
        return XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Run: pip install xgboost"
        ) from exc


def _import_catboost() -> Any:
    """Lazily import CatBoostClassifier and raise a clear error if missing."""
    try:
        from catboost import CatBoostClassifier  # type: ignore[import]
        return CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is not installed. Run: pip install catboost"
        ) from exc


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Factory that constructs sklearn-compatible estimators from config.

    All model hyper-parameters are read from the ``model.hyperparams``
    section of ``config.yaml``.  When ``model.imbalance_strategy`` is set
    to ``'class_weight'``, models that natively support ``class_weight``
    receive ``class_weight='balanced'`` automatically.

    Attributes:
        _cfg: Singleton :class:`~src.config_loader.ConfigLoader` instance.
        _imbalance_strategy: One of ``'smote'``, ``'class_weight'``,
            ``'none'``.
        _random_state: Global random seed sourced from config.
        _registry: Mapping of model name to builder callable.

    Examples:
        >>> factory = ModelFactory()
        >>> lr = factory.create("logistic_regression")
        >>> print(factory.list_models())
        ['catboost', 'decision_tree', 'logistic_regression',
         'random_forest', 'xgboost']
    """

    # Canonical model keys (must match config.yaml model.hyperparams keys)
    _SUPPORTED_MODELS: List[str] = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "catboost",
    ]

    def __init__(self, cfg: ConfigLoader | None = None) -> None:
        """Initialise the factory, reading strategy and random state from config."""
        if cfg is None:
            self._cfg = ConfigLoader.get_instance()
        else:
            self._cfg = cfg
        self._imbalance_strategy: str = str(
            self._cfg.get("model.imbalance_strategy", "none")
        ).lower()
        self._random_state: int = self._cfg.random_state

        # Build registry mapping model names to their builder methods
        self._registry: Dict[str, Any] = {
            "logistic_regression": self._build_logistic_regression,
            "decision_tree": self._build_decision_tree,
            "random_forest": self._build_random_forest,
            "xgboost": self._build_xgboost,
            "catboost": self._build_catboost,
        }

        logger.info(
            "ModelFactory initialised | imbalance_strategy=%s | random_state=%d",
            self._imbalance_strategy,
            self._random_state,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def create_model(cls, model_name: str, cfg: ConfigLoader | None = None) -> BaseEstimator:
        """Static factory method to construct and return a configured model directly.

        Args:
            model_name: One of the supported model keys.
            cfg: Optional ConfigLoader instance.

        Returns:
            An instantiated, configured scikit-learn-compatible classification estimator.
        """
        factory = cls(cfg)
        return factory.create(model_name)

    def create(self, model_name: str) -> BaseEstimator:
        """Construct and return the requested estimator.

        Args:
            model_name: One of the supported model keys (e.g.
                ``'random_forest'``).  See :meth:`list_models` for the
                full list.

        Returns:
            A freshly instantiated, unfitted sklearn-compatible estimator
            configured with parameters from ``config.yaml``.

        Raises:
            ValueError: If ``model_name`` is not in the supported list.

        Examples:
            >>> factory = ModelFactory()
            >>> model = factory.create("xgboost")
        """
        model_name = model_name.strip()
        if model_name not in self._registry:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Supported models: {self._SUPPORTED_MODELS}"
            )
        logger.info("Creating estimator: %s", model_name)
        estimator: BaseEstimator = self._registry[model_name]()
        logger.debug("Estimator created: %r", estimator)
        return estimator

    @classmethod
    def list_models(cls) -> List[str]:
        """Return a sorted list of supported model names.

        Returns:
            List of model key strings recognised by :meth:`create`.

        Examples:
            >>> factory = ModelFactory()
            >>> factory.list_models()
            ['catboost', 'decision_tree', 'logistic_regression',
             'random_forest', 'xgboost']
        """
        return sorted(cls._SUPPORTED_MODELS)

    # ------------------------------------------------------------------
    # Private builder methods
    # ------------------------------------------------------------------

    def _get_params(self, model_key: str) -> Dict[str, Any]:
        """Retrieve the *first value* of each hyper-parameter list from config.

        During factory construction we need concrete (not grid) defaults.
        The first element in each list serves as the default value.

        Args:
            model_key: Key under ``model.hyperparams`` in config.yaml.

        Returns:
            Dict of parameter name to scalar default value.
        """
        raw: Dict[str, Any] = self._cfg.get(
            f"model.hyperparams.{model_key}", {}
        )
        defaults: Dict[str, Any] = {}
        for param, values in raw.items():
            if isinstance(values, list) and values:
                defaults[param] = values[0]
            else:
                defaults[param] = values
        logger.debug("Default params for %s: %s", model_key, defaults)
        return defaults

    def _use_class_weight(self) -> bool:
        """Return True when the imbalance strategy calls for class_weight.

        Returns:
            True if ``imbalance_strategy == 'class_weight'``.
        """
        return self._imbalance_strategy == "class_weight"

    # ---- Logistic Regression ----

    def _build_logistic_regression(self) -> LogisticRegression:
        """Build a :class:`~sklearn.linear_model.LogisticRegression`.

        Returns:
            Configured :class:`~sklearn.linear_model.LogisticRegression`
            instance.
        """
        params = self._get_params("logistic_regression")
        class_weight = "balanced" if self._use_class_weight() else None
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver=str(params.get("solver", "lbfgs")),
            max_iter=int(params.get("max_iter", 500)),
            penalty=str(params.get("penalty", "l2")),
            class_weight=class_weight,
            random_state=self._random_state,
            n_jobs=-1,
        )
        logger.debug(
            "LogisticRegression | C=%s | solver=%s | class_weight=%s",
            params.get("C"),
            params.get("solver"),
            class_weight,
        )
        return estimator

    # ---- Decision Tree ----

    def _build_decision_tree(self) -> DecisionTreeClassifier:
        """Build a :class:`~sklearn.tree.DecisionTreeClassifier`.

        Returns:
            Configured :class:`~sklearn.tree.DecisionTreeClassifier`
            instance.
        """
        params = self._get_params("decision_tree")
        class_weight = "balanced" if self._use_class_weight() else None
        # max_depth can legitimately be None (unlimited)
        raw_depth = params.get("max_depth", 5)
        max_depth = None if raw_depth is None else int(raw_depth)
        estimator = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            criterion=str(params.get("criterion", "gini")),
            class_weight=class_weight,
            random_state=self._random_state,
        )
        logger.debug(
            "DecisionTreeClassifier | max_depth=%s | criterion=%s | class_weight=%s",
            max_depth,
            params.get("criterion"),
            class_weight,
        )
        return estimator

    # ---- Random Forest ----

    def _build_random_forest(self) -> RandomForestClassifier:
        """Build a :class:`~sklearn.ensemble.RandomForestClassifier`.

        Returns:
            Configured :class:`~sklearn.ensemble.RandomForestClassifier`
            instance.
        """
        params = self._get_params("random_forest")
        class_weight = "balanced" if self._use_class_weight() else None
        raw_depth = params.get("max_depth", None)
        max_depth = None if raw_depth is None else int(raw_depth)
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=max_depth,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=str(params.get("max_features", "sqrt")),
            bootstrap=bool(params.get("bootstrap", True)),
            class_weight=class_weight,
            random_state=self._random_state,
            n_jobs=-1,
        )
        logger.debug(
            "RandomForestClassifier | n_estimators=%s | max_depth=%s | class_weight=%s",
            params.get("n_estimators"),
            max_depth,
            class_weight,
        )
        return estimator

    # ---- XGBoost ----

    def _build_xgboost(self) -> Any:
        """Build an :class:`~xgboost.XGBClassifier`.

        Always sets ``eval_metric='logloss'`` and
        ``use_label_encoder=False`` for compatibility.

        Returns:
            Configured :class:`~xgboost.XGBClassifier` instance.

        Raises:
            ImportError: If xgboost is not installed.
        """
        XGBClassifier = _import_xgboost()
        params = self._get_params("xgboost")
        estimator = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=int(params.get("max_depth", 3)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            gamma=float(params.get("gamma", 0)),
            min_child_weight=int(params.get("min_child_weight", 1)),
            reg_alpha=float(params.get("reg_alpha", 0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=self._random_state,
            n_jobs=-1,
        )
        logger.debug(
            "XGBClassifier | n_estimators=%s | max_depth=%s | lr=%s",
            params.get("n_estimators"),
            params.get("max_depth"),
            params.get("learning_rate"),
        )
        return estimator

    # ---- CatBoost ----

    def _build_catboost(self) -> Any:
        """Build a :class:`~catboost.CatBoostClassifier`.

        Always sets ``verbose=0`` to suppress training output.

        Returns:
            Configured :class:`~catboost.CatBoostClassifier` instance.

        Raises:
            ImportError: If catboost is not installed.
        """
        CatBoostClassifier = _import_catboost()
        params = self._get_params("catboost")
        auto_class_weights = "Balanced" if self._use_class_weight() else None
        estimator = CatBoostClassifier(
            iterations=int(params.get("iterations", 100)),
            depth=int(params.get("depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3)),
            border_count=int(params.get("border_count", 64)),
            bagging_temperature=float(params.get("bagging_temperature", 1.0)),
            auto_class_weights=auto_class_weights,
            verbose=0,
            random_seed=self._random_state,
        )
        logger.debug(
            "CatBoostClassifier | iterations=%s | depth=%s | lr=%s | class_weights=%s",
            params.get("iterations"),
            params.get("depth"),
            params.get("learning_rate"),
            auto_class_weights,
        )
        return estimator

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return an informative string representation of the factory.

        Returns:
            String showing supported models and active imbalance strategy.
        """
        return (
            f"ModelFactory("
            f"models={self._SUPPORTED_MODELS}, "
            f"imbalance_strategy='{self._imbalance_strategy}', "
            f"random_state={self._random_state})"
        )
