"""
src/orchestration/training_pipeline.py
───────────────────────────────────────
Orchestrates the full model training, evaluation, and champion-selection
workflow for the Telco Churn Prediction project.

Pipeline stages (in order):
    1. Load processed train / val / test splits from disk
    2. Separate features (X) from target (y) for all three splits
    3. For each model in ``model.models_to_train``:
         a. Instantiate via ModelFactory
         b. Train with cross-validation + hyper-parameter search (ModelTrainer)
         c. Evaluate on the validation set (ModelEvaluator)
         d. Optimise decision threshold on the validation set
    4. Compare all trained models — elect the champion by ``model.champion_metric``
    5. Evaluate the champion on the held-out test set (final unbiased metrics)
    6. Save the champion estimator to ``models/champion/champion_model.pkl``
    7. Save champion metadata JSON (name, params, val metrics, test metrics,
       optimal threshold, timestamp)
    8. Generate all configured evaluation plots (ROC, PR, confusion matrix,
       feature importance)

Usage (CLI):
    python -m src.orchestration.training_pipeline --config config.yaml

Usage (library):
    from src.orchestration.training_pipeline import TrainingPipeline
    from src.config_loader import ConfigLoader
    cfg = ConfigLoader.get_instance()
    pipeline = TrainingPipeline(cfg)
    results = pipeline.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.config_loader import ConfigLoader, setup_logging
from src.model_development.model_building import ModelFactory
from src.model_development.model_training import ModelTrainer
from src.model_development.model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end model training orchestrator for the Telco Churn pipeline.

    Manages the lifecycle of every candidate model: instantiation, training,
    validation-set evaluation, threshold optimisation, champion election, final
    test evaluation, and artefact persistence.

    Attributes:
        cfg: The singleton :class:`~src.config_loader.ConfigLoader` instance.
        factory: Creates model instances by name.
        trainer: Handles CV, hyper-parameter search, and SMOTE resampling.
        evaluator: Computes metrics and generates evaluation plots.
        _results: Accumulated per-model result dictionaries.

    Example:
        >>> from src.config_loader import ConfigLoader
        >>> from src.orchestration.training_pipeline import TrainingPipeline
        >>> cfg = ConfigLoader.get_instance("config.yaml")
        >>> pipeline = TrainingPipeline(cfg)
        >>> results = pipeline.run()
        >>> print(results["champion_model_name"])
    """

    def __init__(self, cfg: ConfigLoader) -> None:
        """Initialise all modelling sub-components from the shared config.

        Args:
            cfg: Loaded :class:`~src.config_loader.ConfigLoader` singleton.

        Raises:
            TypeError: If *cfg* is not a :class:`~src.config_loader.ConfigLoader`.
        """
        if not isinstance(cfg, ConfigLoader):
            raise TypeError(
                f"cfg must be a ConfigLoader instance, got {type(cfg).__name__!r}."
            )

        self.cfg: ConfigLoader = cfg
        self._results: List[Dict[str, Any]] = []

        logger.info("Initialising TrainingPipeline sub-components ...")

        self.factory: ModelFactory = ModelFactory(cfg)
        self.trainer: ModelTrainer = ModelTrainer(cfg)
        self.evaluator: ModelEvaluator = ModelEvaluator(cfg)

        logger.info("TrainingPipeline ready.")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full model training and evaluation pipeline.

        Returns:
            A result dictionary containing:
                - ``model_results``: list of per-model dicts with metrics.
                - ``champion_model_name``: name of the elected champion.
                - ``champion_val_metrics``: validation metrics for champion.
                - ``champion_test_metrics``: held-out test metrics for champion.
                - ``champion_threshold``: optimal decision threshold.
                - ``champion_path``: path to saved champion ``.pkl`` file.
                - ``metadata_path``: path to saved champion metadata JSON.

        Raises:
            FileNotFoundError: If the split CSVs cannot be found.
            ValueError: If ``models_to_train`` is empty.
            RuntimeError: If no model produces a valid champion metric score.
        """
        t0 = time.perf_counter()
        logger.info("=" * 70)
        logger.info("TrainingPipeline.run()  started at %s", datetime.now(timezone.utc).isoformat())
        logger.info("=" * 70)

        # Step 1 & 2: Load splits and separate X / y
        train_df, val_df, test_df = self._load_splits()
        X_train, y_train = self._split_features_target(train_df, "train")
        X_val, y_val = self._split_features_target(val_df, "val")
        X_test, y_test = self._split_features_target(test_df, "test")

        models_to_train: List[str] = self.cfg.get("model.models_to_train")
        if not models_to_train:
            raise ValueError(
                "model.models_to_train is empty — nothing to train. "
                "Update config.yaml."
            )
        logger.info("Models to train: %s", models_to_train)

        # Steps 3 & 4: Train, evaluate, and threshold-optimise each model
        for model_name in models_to_train:
            logger.info("-" * 60)
            logger.info("Processing model: %s", model_name)
            model_result = self._train_and_evaluate_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
            self._results.append(model_result)

        # Step 5: Elect champion
        champion_result = self._elect_champion(self._results)
        champion_name: str = champion_result["model_name"]
        logger.info("=" * 60)
        logger.info("Champion elected: %s", champion_name)
        logger.info("=" * 60)

        # Step 6: Evaluate champion on test set
        champion_estimator = champion_result["estimator"]
        test_metrics = self.evaluator.evaluate(
            estimator=champion_estimator,
            X=X_test,
            y_true=y_test,
            split_name="test",
            threshold=champion_result["optimal_threshold"],
        )
        logger.info("Champion test metrics: %s", test_metrics)

        # Step 7: Generate all plots
        self._generate_plots(
            estimator=champion_estimator,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=list(X_train.columns),
            model_name=champion_name,
        )

        # Step 8 & 9: Save champion artefacts
        champion_path, metadata_path = self._save_champion(
            estimator=champion_estimator,
            metadata={
                "model_name": champion_name,
                "best_params": champion_result.get("best_params", {}),
                "val_metrics": champion_result["val_metrics"],
                "test_metrics": test_metrics,
                "optimal_threshold": champion_result["optimal_threshold"],
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "champion_metric": self.cfg.get("model.champion_metric", "roc_auc"),
            },
        )

        elapsed = time.perf_counter() - t0
        logger.info("TrainingPipeline.run() complete in %.2f s", elapsed)

        return {
            "model_results": self._results,
            "champion_model_name": champion_name,
            "champion_val_metrics": champion_result["val_metrics"],
            "champion_test_metrics": test_metrics,
            "champion_threshold": champion_result["optimal_threshold"],
            "champion_path": str(champion_path),
            "metadata_path": str(metadata_path),
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed train / val / test DataFrames from disk.

        Reads from the directory configured at ``data.splits_path``.  Supports
        both ``.csv`` and ``.parquet`` formats (auto-detected by extension).

        Returns:
            ``(train_df, val_df, test_df)`` tuple of raw DataFrames.

        Raises:
            FileNotFoundError: If any split file is missing.
        """
        splits_dir = Path(self.cfg.get("data.splits_path", "data/splits"))
        save_fmt: str = self.cfg.get("split.save_format", "csv")
        logger.info("Loading splits from: %s  (format=%s)", splits_dir, save_fmt)

        frames: Dict[str, pd.DataFrame] = {}
        for name in ("train", "val", "test"):
            if save_fmt == "parquet":
                path = splits_dir / f"{name}.parquet"
                if not path.exists():
                    raise FileNotFoundError(f"Split file not found: {path}")
                frames[name] = pd.read_parquet(path)
            else:
                path = splits_dir / f"{name}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"Split file not found: {path}")
                frames[name] = pd.read_csv(path)
            logger.info(
                "   Loaded %s split  path=%s  shape=%s",
                name,
                path,
                frames[name].shape,
            )

        return frames["train"], frames["val"], frames["test"]

    def _split_features_target(
        self, df: pd.DataFrame, split_name: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate feature matrix from target vector.

        Drops non-numeric columns, ``data.customer_id_col``, and ``data.target_col``
        from *df* to produce *X*, and extracts the target column as *y*.

        Args:
            df: Split DataFrame containing both features and target.
            split_name: Name of the split (e.g. ``"train"``), used in logging.

        Returns:
            ``(X, y)`` tuple of feature DataFrame and target Series.

        Raises:
            KeyError: If the target column is missing from *df*.
        """
        target_col: str = self.cfg.get("data.target_col")
        id_col: str = self.cfg.get("data.customer_id_col", "customerID")

        if target_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in {split_name} split. "
                f"Available columns: {list(df.columns)}"
            )

        # Select only numeric columns for features, dropping target and customer id
        X = df.select_dtypes(include=["number"])
        if target_col in X.columns:
            X = X.drop(columns=[target_col])
        if id_col in X.columns:
            X = X.drop(columns=[id_col])
        y = df[target_col]

        logger.info(
            "   %s  X=%s  y=%s  positive_rate=%.3f",
            split_name,
            X.shape,
            y.shape,
            float(y.mean()),
        )
        return X, y

    def _train_and_evaluate_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Train a single model and evaluate it on the validation set.

        Workflow:
            1. Instantiate estimator via :class:`~src.model_development.model_factory.ModelFactory`.
            2. Train with CV + hyper-parameter search via
               :class:`~src.model_development.model_trainer.ModelTrainer`.
            3. Evaluate on validation set via
               :class:`~src.model_development.model_evaluator.ModelEvaluator`.
            4. Optimise decision threshold on validation probabilities.

        Args:
            model_name: Registered model name (e.g. ``"random_forest"``).
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Validation feature matrix.
            y_val: Validation target vector.

        Returns:
            Dictionary containing:
                - ``model_name``: str
                - ``estimator``: fitted sklearn-compatible estimator
                - ``best_params``: dict of best hyper-parameters
                - ``cv_score``: mean CV score on training set
                - ``val_metrics``: dict of validation metrics
                - ``optimal_threshold``: float, threshold maximising ``evaluation.threshold_metric``
        """
        logger.info("  [%s] Step 1/4 — creating estimator ...", model_name)
        estimator = self.factory.create(model_name)

        logger.info("  [%s] Step 2/4 — training (CV + search) ...", model_name)
        train_result: Dict[str, Any] = self.trainer.train(
            estimator=estimator,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
        )
        fitted_estimator = train_result["estimator"]
        best_params = train_result.get("best_params", {})
        cv_score: float = train_result.get("cv_score", float("nan"))

        logger.info(
            "  [%s] Step 3/4 — evaluating on val set  cv_score=%.4f ...",
            model_name,
            cv_score,
        )
        val_metrics: Dict[str, float] = self.evaluator.evaluate(
            estimator=fitted_estimator,
            X=X_val,
            y_true=y_val,
            split_name=f"val_{model_name}",
        )

        logger.info("  [%s] Step 4/4 — optimising threshold ...", model_name)
        optimal_threshold: float = self.evaluator.optimise_threshold(
            estimator=fitted_estimator,
            X=X_val,
            y_true=y_val,
        )

        logger.info(
            "  [%s] Complete  |  val_%s=%.4f  threshold=%.2f",
            model_name,
            self.cfg.get("model.champion_metric", "roc_auc"),
            val_metrics.get(self.cfg.get("model.champion_metric", "roc_auc"), 0.0),
            optimal_threshold,
        )

        return {
            "model_name": model_name,
            "estimator": fitted_estimator,
            "best_params": best_params,
            "cv_score": cv_score,
            "val_metrics": val_metrics,
            "optimal_threshold": optimal_threshold,
        }

    def _elect_champion(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select the champion model based on the configured metric.

        Args:
            results: List of per-model result dictionaries from
                :meth:`_train_and_evaluate_model`.

        Returns:
            Result dictionary of the elected champion.

        Raises:
            RuntimeError: If *results* is empty or no valid metric score is found.
        """
        if not results:
            raise RuntimeError("No model results available — cannot elect champion.")

        champion_metric: str = self.cfg.get("model.champion_metric", "roc_auc")
        logger.info("Electing champion by metric: %s", champion_metric)

        best_result: Optional[Dict[str, Any]] = None
        best_score: float = -float("inf")

        for result in results:
            score: float = result["val_metrics"].get(champion_metric, float("nan"))
            if score != score:  # nan check
                logger.warning(
                    "  [%s] metric '%s' is NaN — skipping",
                    result["model_name"],
                    champion_metric,
                )
                continue
            logger.info(
                "  Candidate: %-25s  %s=%.4f",
                result["model_name"],
                champion_metric,
                score,
            )
            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            raise RuntimeError(
                f"Champion election failed: no model produced a valid "
                f"'{champion_metric}' score."
            )

        logger.info(
            "Champion: %s  |  %s=%.4f",
            best_result["model_name"],
            champion_metric,
            best_score,
        )
        return best_result

    def _generate_plots(
        self,
        estimator: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        model_name: str,
    ) -> None:
        """Generate all configured evaluation plots for the champion model.

        Checks each plot flag in ``evaluation.plots`` and calls the
        corresponding evaluator method when enabled.

        Args:
            estimator: Fitted champion estimator.
            X_val: Validation features.
            y_val: Validation targets.
            X_test: Test features.
            y_test: Test targets.
            feature_names: Ordered list of feature column names.
            model_name: Champion model name (used in plot titles/filenames).
        """
        plots_cfg: Dict[str, bool] = self.cfg.get("evaluation.plots", {})
        logger.info("Generating evaluation plots  config=%s", plots_cfg)

        plot_methods = {
            "roc_curve": lambda: self.evaluator.plot_roc_curve(
                estimator, X_test, y_test, model_name
            ),
            "pr_curve": lambda: self.evaluator.plot_pr_curve(
                estimator, X_test, y_test, model_name
            ),
            "confusion_matrix": lambda: self.evaluator.plot_confusion_matrix(
                estimator, X_test, y_test, model_name
            ),
            "feature_importance": lambda: self.evaluator.plot_feature_importance(
                estimator, feature_names, model_name
            ),
            "learning_curves": lambda: self.evaluator.plot_learning_curves(
                estimator, X_val, y_val, model_name
            ),
            "threshold_analysis": lambda: self.evaluator.plot_threshold_analysis(
                estimator, X_val, y_val, model_name
            ),
            "calibration_curve": lambda: self.evaluator.plot_calibration_curve(
                estimator, X_test, y_test, model_name
            ),
        }

        for plot_key, plot_fn in plot_methods.items():
            if plots_cfg.get(plot_key, False):
                try:
                    plot_fn()
                    logger.info("   OK Plot generated: %s", plot_key)
                except Exception as exc:
                    logger.warning(
                        "   WARN Plot '%s' failed: %s", plot_key, exc
                    )

    def _save_champion(
        self,
        estimator: Any,
        metadata: Dict[str, Any],
    ) -> Tuple[Path, Path]:
        """Persist champion estimator and metadata to disk.

        Saves:
            - ``<champion_dir>/champion_model.pkl``  (joblib-serialised estimator)
            - ``<champion_dir>/champion_metadata.json``  (human-readable metadata)

        Args:
            estimator: Fitted champion estimator (sklearn-compatible).
            metadata: Dictionary to serialise as JSON.

        Returns:
            ``(model_path, metadata_path)`` as :class:`~pathlib.Path` objects.

        Raises:
            OSError: If the champion directory cannot be created.
        """
        champion_dir = Path(self.cfg.get("model.champion_dir", "models/champion"))
        champion_dir.mkdir(parents=True, exist_ok=True)

        model_path = champion_dir / "champion_model.pkl"
        metadata_path = Path(
            self.cfg.get(
                "model.champion_metadata_file",
                "models/champion/champion_metadata.json",
            )
        )

        # Save estimator
        joblib.dump(estimator, model_path)
        logger.info("Champion model saved: %s", model_path)

        # Save metadata
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        logger.info("Champion metadata saved: %s", metadata_path)

        return model_path, metadata_path

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        n_results = len(self._results)
        return (
            f"TrainingPipeline("
            f"models_evaluated={n_results}, "
            f"factory={self.factory!r}, "
            f"trainer={self.trainer!r}, "
            f"evaluator={self.evaluator!r}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`~argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.orchestration.training_pipeline",
        description=(
            "Telco Churn -- Model Training Pipeline\n"
            "Trains all configured models, elects the champion, evaluates on "
            "test set, and saves artefacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    # Bootstrap config + logging
    _cfg = ConfigLoader.get_instance(_args.config)
    setup_logging(_cfg)

    logger.info(
        "Telco Churn | TrainingPipeline CLI | config=%s", _args.config
    )

    _pipeline = TrainingPipeline(_cfg)
    _results = _pipeline.run()

    logger.info(
        "Training complete.\n"
        "  Champion   : %s\n"
        "  Val AUC    : %.4f\n"
        "  Test AUC   : %.4f\n"
        "  Threshold  : %.2f\n"
        "  Model path : %s",
        _results["champion_model_name"],
        _results["champion_val_metrics"].get("roc_auc", float("nan")),
        _results["champion_test_metrics"].get("roc_auc", float("nan")),
        _results["champion_threshold"],
        _results["champion_path"],
    )
