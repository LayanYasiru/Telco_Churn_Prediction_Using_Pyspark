"""
src/model_development/model_inference.py
─────────────────────────────────────────
Production inference module for the Telco Churn Prediction pipeline.

Implements :class:`InferencePipeline`, which loads the serialised champion
model from disk, scores raw customer DataFrames, and returns enriched
predictions with churn probability, binary prediction, and risk tier
classification.

Risk tiers are determined by configurable thresholds from ``config.yaml``:

- **High**   – ``churn_probability >= high_risk_threshold``
- **Medium** – ``churn_probability >= medium_risk_threshold``
- **Low**    – ``churn_probability < medium_risk_threshold``

All paths, thresholds, and settings are sourced from
:class:`~src.config_loader.ConfigLoader`.

Typical usage
-------------
>>> from src.model_development.model_inference import InferencePipeline
>>> pipeline = InferencePipeline()
>>> result_df = pipeline.predict(raw_df)
>>> single_result = pipeline.predict_single(customer_dict)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Load and serve the champion churn-prediction model.

    The pipeline:
    1. Reads champion model path and risk-tier thresholds from config.
    2. Loads the serialised model on demand (:meth:`load_model`).
    3. Accepts raw DataFrames (``predict``) or single dicts
       (``predict_single``) and returns scored outputs.
    4. Reads associated metadata from ``champion_metadata.json``
       (:meth:`get_model_metadata`).

    Attributes:
        _cfg: Singleton :class:`~src.config_loader.ConfigLoader` instance.
        _champion_model_path: :class:`~pathlib.Path` to the serialised
            ``.pkl`` champion model file.
        _metadata_path: :class:`~pathlib.Path` to the JSON metadata file.
        _high_risk_threshold: Probability cutoff for "High" risk tier.
        _medium_risk_threshold: Probability cutoff for "Medium" risk tier.
        _model: The loaded sklearn-compatible estimator (``None`` until
            :meth:`load_model` is called).

    Examples:
        >>> pipeline = InferencePipeline()
        >>> df_scored = pipeline.predict(raw_customer_df)
        >>> print(df_scored[["churn_probability", "prediction", "risk_tier"]])
    """

    # Column names added to the output DataFrame
    _COL_CHURN_PROB: str = "churn_probability"
    _COL_PREDICTION: str = "prediction"
    _COL_RISK_TIER: str = "risk_tier"

    # Risk tier labels
    _TIER_HIGH: str = "High"
    _TIER_MEDIUM: str = "Medium"
    _TIER_LOW: str = "Low"

    def __init__(self, cfg: Optional[ConfigLoader] = None) -> None:
        """Initialise the pipeline by reading all settings from ConfigLoader.

        The model is NOT loaded at construction time.  Call
        :meth:`load_model` explicitly, or rely on the lazy-load behaviour
        in :meth:`predict` / :meth:`predict_single`.
        """
        if cfg is None:
            self._cfg = ConfigLoader.get_instance()
        else:
            self._cfg = cfg

        # Paths
        self._champion_model_path: Path = Path(
            self._cfg.get(
                "model.champion_dir", "models/champion"
            )
        ) / "champion_model.pkl"
        self._metadata_path: Path = Path(
            self._cfg.get(
                "model.champion_metadata_file",
                "models/champion/champion_metadata.json",
            )
        )
        self._preprocessor_path: Path = Path(
            self._cfg.get(
                "model.models_dir", "models"
            )
        ) / "preprocessor.pkl"

        # Risk-tier thresholds
        self._high_risk_threshold: float = float(
            self._cfg.get("reporting.high_risk_threshold", 0.70)
        )
        self._medium_risk_threshold: float = float(
            self._cfg.get("reporting.medium_risk_threshold", 0.40)
        )

        # Model and Preprocessor placeholders (lazily loaded)
        self._model: Optional[BaseEstimator] = None
        self._preprocessor: Optional[Any] = None
        self._model_last_loaded_time: Optional[float] = None
        self._preprocessor_last_loaded_time: Optional[float] = None

        logger.info(
            "InferencePipeline initialised | champion_path=%s | "
            "high_risk=%.2f | medium_risk=%.2f",
            self._champion_model_path,
            self._high_risk_threshold,
            self._medium_risk_threshold,
        )

    # ------------------------------------------------------------------
    # Static Factory Method
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, cfg: Optional[ConfigLoader] = None) -> InferencePipeline:
        """Static factory method to construct and prepare the InferencePipeline.

        Loads both the model and the preprocessor during initialisation.

        Args:
            cfg: Optional ConfigLoader instance.

        Returns:
            A prepared and loaded InferencePipeline instance.
        """
        pipeline = cls(cfg)
        pipeline.load_model()
        return pipeline

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the champion model and preprocessor from disk into memory.

        The model file is expected at the path configured in
        ``model.champion_dir``.  Call this method before invoking
        :meth:`predict` or :meth:`predict_single` to control when I/O
        occurs, or let the predict methods load lazily.

        Raises:
            FileNotFoundError: If the ``.pkl`` file does not exist at the
                configured path.
            RuntimeError: If ``joblib.load`` fails for any reason.

        Examples:
            >>> pipeline = InferencePipeline()
            >>> pipeline.load_model()
        """
        import os

        # Load preprocessor if it exists
        if self._preprocessor_path.exists():
            logger.info("Loading preprocessor from: %s", self._preprocessor_path)
            try:
                from src.orchestration.data_pipeline import DataPipeline
                self._preprocessor = DataPipeline.load(self._preprocessor_path)
                self._preprocessor_last_loaded_time = os.path.getmtime(self._preprocessor_path)
            except Exception as exc:
                logger.exception("Failed to load preprocessor from '%s': %s", self._preprocessor_path, exc)
                raise RuntimeError(f"Could not deserialise preprocessor at '{self._preprocessor_path}'") from exc
        else:
            logger.warning("Preprocessor not found at: %s. Incoming raw data must be preprocessed prior to predict().", self._preprocessor_path)

        if not self._champion_model_path.exists():
            raise FileNotFoundError(
                f"Champion model not found at: "
                f"{self._champion_model_path.resolve()}. "
                "Ensure the training pipeline has been run and the model "
                "has been promoted to champion."
            )
        logger.info("Loading champion model from: %s", self._champion_model_path)
        try:
            self._model = joblib.load(self._champion_model_path)
            self._model_last_loaded_time = os.path.getmtime(self._champion_model_path)
        except Exception as exc:
            logger.exception(
                "Failed to load model from '%s': %s",
                self._champion_model_path,
                exc,
            )
            raise RuntimeError(
                f"Could not deserialise model at "
                f"'{self._champion_model_path}'"
            ) from exc
        logger.info(
            "Champion model loaded successfully: %s",
            type(self._model).__name__,
        )

    def check_and_reload(self) -> None:
        """Check modification times of model & preprocessor on disk, reloading if updated."""
        import os
        model_updated = False
        preprocessor_updated = False

        if self._champion_model_path.exists():
            mtime = os.path.getmtime(self._champion_model_path)
            if self._model_last_loaded_time is None or mtime > self._model_last_loaded_time:
                logger.info("Model update detected on disk. Triggering reload...")
                model_updated = True

        if self._preprocessor_path.exists():
            mtime = os.path.getmtime(self._preprocessor_path)
            if self._preprocessor_last_loaded_time is None or mtime > self._preprocessor_last_loaded_time:
                logger.info("Preprocessor update detected on disk. Triggering reload...")
                preprocessor_updated = True

        if model_updated or preprocessor_updated:
            try:
                self.load_model()
            except Exception as e:
                logger.error("Failed to dynamically reload updated model/preprocessor: %s", e)

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of raw customer records.

        Appends three new columns to a copy of ``raw_df``:
        - ``churn_probability`` – float in [0, 1]
        - ``prediction``        – int (1 = churn, 0 = no churn)
        - ``risk_tier``         – ``'High'``, ``'Medium'``, or ``'Low'``

        The input DataFrame must already be preprocessed and feature-
        engineered to match the training schema expected by the champion
        model.

        Args:
            raw_df: DataFrame of raw (pre-encoded, pre-scaled) customer
                features.  Can be a single row or a full batch.

        Returns:
            A copy of ``raw_df`` with the three inference columns appended.

        Raises:
            ValueError: If ``raw_df`` is empty.
            RuntimeError: If the model cannot be loaded.

        Examples:
            >>> pipeline = InferencePipeline()
            >>> scored = pipeline.predict(customer_df)
            >>> scored[["churn_probability", "risk_tier"]].head()
        """
        if raw_df.empty:
            raise ValueError(
                "Input DataFrame is empty. Provide at least one customer row."
            )

        # Check for model/preprocessor updates and reload if necessary
        self.check_and_reload()

        # Lazy-load model on first call
        self._ensure_model_loaded()

        logger.info(
            "Running batch inference | n_records=%d", len(raw_df)
        )

        # Apply preprocessing if preprocessor is loaded
        if self._preprocessor is not None:
            logger.info("Applying fitted preprocessor to raw input data...")
            X_processed = self._preprocessor.transform(raw_df)
        else:
            X_processed = raw_df

        result_df: pd.DataFrame = raw_df.copy()
        probs: np.ndarray = self._get_probabilities(X_processed)

        result_df[self._COL_CHURN_PROB] = probs.round(6)
        result_df[self._COL_PREDICTION] = (probs >= 0.5).astype(int)
        result_df[self._COL_RISK_TIER] = self._assign_risk_tiers(probs)

        churn_count: int = int(result_df[self._COL_PREDICTION].sum())
        high_count: int = int(
            (result_df[self._COL_RISK_TIER] == self._TIER_HIGH).sum()
        )
        med_count: int = int(
            (result_df[self._COL_RISK_TIER] == self._TIER_MEDIUM).sum()
        )
        logger.info(
            "Inference complete | total=%d | predicted_churn=%d | "
            "high_risk=%d | medium_risk=%d | low_risk=%d",
            len(result_df),
            churn_count,
            high_count,
            med_count,
            len(result_df) - high_count - med_count,
        )
        return result_df

    # ------------------------------------------------------------------
    # Single-record prediction
    # ------------------------------------------------------------------

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single customer record provided as a dict.

        Converts the record to a single-row DataFrame, runs
        :meth:`predict`, and returns the enriched record as a flat dict.

        Args:
            record: Dict mapping feature names to their values for one
                customer.

        Returns:
            A copy of ``record`` with three added keys:
            ``'churn_probability'``, ``'prediction'``, ``'risk_tier'``.

        Raises:
            ValueError: If ``record`` is empty.
            RuntimeError: If the model cannot be loaded.

        Examples:
            >>> pipeline = InferencePipeline()
            >>> result = pipeline.predict_single({
            ...     "tenure": 12, "MonthlyCharges": 55.0, ...
            ... })
            >>> result["risk_tier"]
            'Medium'
        """
        if not record:
            raise ValueError(
                "record dict is empty. Provide at least one feature."
            )

        logger.debug("Single-record inference | keys=%s", list(record.keys()))
        single_df: pd.DataFrame = pd.DataFrame([record])
        scored_df: pd.DataFrame = self.predict(single_df)

        output: Dict[str, Any] = scored_df.iloc[0].to_dict()
        logger.info(
            "Single-record result | churn_prob=%.4f | prediction=%d | "
            "risk_tier=%s",
            output.get(self._COL_CHURN_PROB, float("nan")),
            int(output.get(self._COL_PREDICTION, -1)),
            output.get(self._COL_RISK_TIER, "unknown"),
        )
        return output

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_metadata(self) -> Dict[str, Any]:
        """Read and return the champion model's metadata JSON.

        The metadata file is expected at the path configured in
        ``model.champion_metadata_file`` (default:
        ``models/champion/champion_metadata.json``).

        Returns:
            Parsed JSON dict containing model metadata (e.g.
            ``model_name``, ``champion_metric``, ``score``,
            ``trained_at``).

        Raises:
            FileNotFoundError: If the metadata file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.

        Examples:
            >>> pipeline = InferencePipeline()
            >>> meta = pipeline.get_model_metadata()
            >>> print(meta["model_name"])
        """
        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"Champion metadata file not found at: "
                f"{self._metadata_path.resolve()}"
            )

        logger.info(
            "Reading champion metadata from: %s", self._metadata_path
        )
        with open(self._metadata_path, "r", encoding="utf-8") as fh:
            metadata: Dict[str, Any] = json.load(fh)

        logger.debug("Champion metadata loaded: %s", metadata)
        return metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the champion model if not already in memory.

        Raises:
            RuntimeError: Propagated from :meth:`load_model` on I/O
                failure.
        """
        if self._model is None:
            logger.info("Model not yet loaded — triggering lazy load.")
            self.load_model()

    def _get_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Run model inference and return positive-class probabilities.

        Args:
            X: Feature matrix compatible with the trained model.

        Returns:
            1-D float array of churn probabilities (positive class).

        Raises:
            AttributeError: If the loaded model does not support
                ``predict_proba`` or ``decision_function``.
        """
        assert self._model is not None, (
            "_get_probabilities called before model is loaded."
        )

        # Align features to what the model was trained on
        X_clean = X.copy()
        if hasattr(self._model, "feature_names_in_"):
            feature_names = self._model.feature_names_in_
            X_clean = X_clean[feature_names]
        else:
            # Fallback: keep only numeric columns, dropping ID and target
            target_col = self._cfg.get("data.target_col", "Churn")
            id_col = self._cfg.get("data.customer_id_col", "customerID")
            X_clean = X_clean.select_dtypes(include=[np.number])
            if target_col in X_clean.columns:
                X_clean = X_clean.drop(columns=[target_col])
            if id_col in X_clean.columns:
                X_clean = X_clean.drop(columns=[id_col])

        if hasattr(self._model, "predict_proba"):
            probs: np.ndarray = np.asarray(
                self._model.predict_proba(X_clean)
            )[:, 1]
        elif hasattr(self._model, "decision_function"):
            raw_scores: np.ndarray = np.asarray(
                self._model.decision_function(X_clean)
            )
            s_min, s_max = raw_scores.min(), raw_scores.max()
            probs = (
                (raw_scores - s_min) / (s_max - s_min)
                if (s_max - s_min) > 1e-12
                else np.zeros_like(raw_scores)
            )
        else:
            raise AttributeError(
                f"Loaded model '{type(self._model).__name__}' does not "
                "expose predict_proba or decision_function."
            )
        return probs

    def _assign_risk_tiers(self, probs: np.ndarray) -> np.ndarray:
        """Map probability scores to string risk tier labels.

        Tier assignment:
        - ``'High'``   when ``prob >= high_risk_threshold``
        - ``'Medium'`` when ``medium_risk_threshold <= prob < high_risk_threshold``
        - ``'Low'``    when ``prob < medium_risk_threshold``

        Args:
            probs: 1-D array of churn probabilities in [0, 1].

        Returns:
            1-D object array of risk tier strings.
        """
        tiers: np.ndarray = np.full(len(probs), self._TIER_LOW, dtype=object)
        tiers[probs >= self._medium_risk_threshold] = self._TIER_MEDIUM
        tiers[probs >= self._high_risk_threshold] = self._TIER_HIGH
        return tiers

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return an informative string representation.

        Returns:
            String showing model path, load status, and risk thresholds.
        """
        model_status: str = (
            type(self._model).__name__
            if self._model is not None
            else "not loaded"
        )
        return (
            f"InferencePipeline("
            f"champion_path='{self._champion_model_path}', "
            f"model={model_status}, "
            f"high_risk_threshold={self._high_risk_threshold}, "
            f"medium_risk_threshold={self._medium_risk_threshold})"
        )
