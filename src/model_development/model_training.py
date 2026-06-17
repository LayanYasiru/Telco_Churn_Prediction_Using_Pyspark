"""
src/model_development/model_training.py
─────────────────────────────────────────
ModelTrainer: StratifiedKFold cross-validation, Grid/RandomizedSearchCV,
SMOTE or class-weight imbalance handling, and joblib model persistence.

All hyperparameters and search settings are sourced from config.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from src.config_loader import ConfigLoader
from src.model_development.model_building import ModelFactory

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains ML models using hyperparameter search over StratifiedKFold CV,
    with optional SMOTE oversampling or class-weight balancing.

    Flow
    ----
    1. Build base estimator via ModelFactory.
    2. Wrap in imblearn Pipeline if SMOTE is enabled.
    3. Run GridSearchCV or RandomizedSearchCV (from config).
    4. Fit best estimator on full training data.
    5. Evaluate via StratifiedKFold cross-val scores.
    6. Save fitted model artifact to models/{model_name}.pkl.
    7. Return (best_estimator, results_dict).

    Parameters
    ----------
    cfg : ConfigLoader
        Singleton config instance.
    """

    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        self.factory = ModelFactory(cfg)

        self.cv_folds: int = int(cfg.get("model.cv_folds", 5))
        self.scoring_metric: str = cfg.get("model.scoring_metric", "roc_auc")
        self.search_type: str = cfg.get("model.hyperparameter_search", "random")
        self.n_iter: int = int(cfg.get("model.n_iter_random_search", 20))
        self.search_cv_folds: int = int(cfg.get("model.search_cv_folds", 3))
        self.n_jobs: int = int(cfg.get("model.search_n_jobs", -1))
        self.imbalance_strategy: str = cfg.get("model.imbalance_strategy", "smote")
        self.smote_k: int = int(cfg.get("model.smote_k_neighbors", 5))
        self.random_state: int = cfg.random_state
        self.models_dir: Path = Path(cfg.get("model.models_dir", "models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ModelTrainer init | search=%s, cv_folds=%d, scoring=%s, "
            "imbalance=%s",
            self.search_type,
            self.cv_folds,
            self.scoring_metric,
            self.imbalance_strategy,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        estimator: Optional[BaseEstimator] = None,
    ) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter search and CV evaluation.

        Parameters
        ----------
        model_name : str
            Key matching a model in config (e.g. 'random_forest').
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training labels (binary 0/1).
        estimator : Optional[BaseEstimator]
            Optional pre-instantiated base estimator to train.

        Returns
        -------
        Dict[str, Any]
            Results dict containing: model_name, estimator, best_params, cv_mean,
            cv_std, best_cv_score, cv_score (mean CV score).
        """
        logger.info("═══ Training: %s ═══", model_name.upper())

        # ── 1. Build base model ───────────────────────────────────────────
        base_model = estimator if estimator is not None else self.factory.create(model_name)

        # ── 2. Build pipeline (with or without SMOTE) ────────────────────
        pipeline, param_grid = self._build_pipeline_and_grid(
            model_name, base_model
        )

        # ── 3. Hyperparameter search ──────────────────────────────────────
        inner_cv = StratifiedKFold(
            n_splits=self.search_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        searcher = self._build_searcher(pipeline, param_grid, inner_cv)
        logger.info(
            "Starting %s search | n_iter=%s, cv=%d, scoring=%s",
            self.search_type,
            self.n_iter if self.search_type == "random" else "full-grid",
            self.search_cv_folds,
            self.scoring_metric,
        )
        searcher.fit(X_train, y_train)
        best_estimator: BaseEstimator = searcher.best_estimator_
        best_params: Dict[str, Any] = searcher.best_params_
        best_search_score: float = float(searcher.best_score_)
        logger.info(
            "Search complete | best_%s=%.4f | best_params=%s",
            self.scoring_metric,
            best_search_score,
            best_params,
        )

        # ── 4. Outer StratifiedKFold cross-validation ────────────────────
        outer_cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        cv_scores: np.ndarray = cross_val_score(
            best_estimator,
            X_train,
            y_train,
            cv=outer_cv,
            scoring=self.scoring_metric,
            n_jobs=self.n_jobs,
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        logger.info(
            "Outer CV (%d-fold) | %s: %.4f ± %.4f",
            self.cv_folds,
            self.scoring_metric,
            cv_mean,
            cv_std,
        )

        # ── 5. Refit on full training data ────────────────────────────────
        best_estimator.fit(X_train, y_train)

        # ── 6. Save artifact ──────────────────────────────────────────────
        model_path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(best_estimator, model_path)
        logger.info("Model saved → %s", model_path)

        results: Dict[str, Any] = {
            "model_name": model_name,
            "estimator": best_estimator,
            "best_params": best_params,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_score": cv_mean,
            "best_cv_score": best_search_score,
            "scoring_metric": self.scoring_metric,
        }
        return results

    def load_model(self, model_name: str) -> BaseEstimator:
        """Load a previously saved model from disk."""
        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        estimator = joblib.load(model_path)
        logger.info("Model loaded ← %s", model_path)
        return estimator

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_pipeline_and_grid(
        self,
        model_name: str,
        base_model: BaseEstimator,
    ) -> Tuple[Any, Dict[str, List[Any]]]:
        """
        Wrap model in imblearn Pipeline if SMOTE enabled, and prefix
        param grid keys accordingly.
        """
        raw_grid: Dict[str, List[Any]] = self.cfg.get(
            f"model.hyperparams.{model_name}", {}
        )

        if self.imbalance_strategy == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.pipeline import Pipeline as ImbPipeline

                smote = SMOTE(
                    k_neighbors=self.smote_k,
                    random_state=self.random_state,
                )
                pipeline = ImbPipeline(
                    steps=[("smote", smote), ("model", base_model)]
                )
                # Prefix keys with 'model__' for pipeline compatibility
                param_grid = {f"model__{k}": v for k, v in raw_grid.items()}
                logger.info("SMOTE pipeline constructed for %s", model_name)
            except ImportError:
                logger.warning(
                    "imbalanced-learn not installed. Falling back to no SMOTE."
                )
                pipeline = base_model
                param_grid = raw_grid
        else:
            pipeline = base_model
            param_grid = raw_grid

        return pipeline, param_grid

    def _build_searcher(
        self,
        pipeline: Any,
        param_grid: Dict[str, List[Any]],
        cv: StratifiedKFold,
    ) -> Any:
        """Build GridSearchCV or RandomizedSearchCV from config."""
        common_kwargs = dict(
            estimator=pipeline,
            param_distributions=param_grid,
            scoring=self.scoring_metric,
            cv=cv,
            n_jobs=self.n_jobs,
            refit=True,
            verbose=1,
            random_state=self.random_state,
        )

        if self.search_type == "grid":
            # GridSearchCV doesn't accept param_distributions or random_state
            return GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=self.scoring_metric,
                cv=cv,
                n_jobs=self.n_jobs,
                refit=True,
                verbose=1,
            )
        else:
            return RandomizedSearchCV(
                **common_kwargs,
                n_iter=self.n_iter,
            )

    def __repr__(self) -> str:
        return (
            f"ModelTrainer("
            f"search={self.search_type}, "
            f"cv_folds={self.cv_folds}, "
            f"scoring={self.scoring_metric}, "
            f"imbalance={self.imbalance_strategy})"
        )
