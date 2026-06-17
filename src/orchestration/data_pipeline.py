"""
src/orchestration/data_pipeline.py
────────────────────────────────────
Orchestrates the complete data processing pipeline for the Telco Churn project.

Pipeline stages (in order):
    1. Ingest raw CSV
    2. Impute missing values
    3. Detect and handle outliers
    4. Engineer features
    5. Split into train / validation / test  (split BEFORE encode/scale)
    6. Encode categorical columns (fit on train, transform all)
    7. Scale numerical columns (fit on train, transform all)
    8. Bin columns into ordinal buckets (fit on train, transform all)
    9. Save processed artefacts to disk

The class follows a strict **fit-on-train / transform-on-all** pattern to
prevent data leakage.  All stateful transformers (encoder, scaler, binner)
are fitted exclusively on the training portion of the data and subsequently
applied to the validation and test sets.

Usage (CLI):
    python -m src.orchestration.data_pipeline --config config.yaml
    python -m src.orchestration.data_pipeline --config config.yaml \\
        --raw-path data/raw/telco_churn.csv

Usage (library):
    from src.orchestration.data_pipeline import DataPipeline
    from src.config_loader import ConfigLoader
    cfg = ConfigLoader.get_instance()
    pipeline = DataPipeline(cfg)
    train_df, val_df, test_df = pipeline.run()
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd

from src.config_loader import ConfigLoader, setup_logging
from src.data_processing.data_ingestion import DataIngestor
from src.data_processing.handle_missing_values import MissingValueHandler
from src.data_processing.outlier_detection import OutlierHandler
from src.data_processing.feature_engineering import FeatureEngineer
from src.data_processing.feature_encoding import FeatureEncoder
from src.data_processing.feature_scaling import FeatureScaler
from src.data_processing.feature_binning import FeatureBinner
from src.data_processing.data_splitter import DataSplitter

logger = logging.getLogger(__name__)


class DataPipeline:
    """End-to-end data processing orchestrator for the Telco Churn pipeline.

    Coordinates all data-processing sub-components in the correct sequence,
    enforcing a no-leakage fit/transform split boundary.

    Attributes:
        cfg: The singleton :class:`~src.config_loader.ConfigLoader` instance.
        ingestor: Reads raw CSV from disk and validates schema.
        imputer: Fills missing values using the configured strategy.
        outlier_handler: Caps or removes statistical outliers.
        feature_engineer: Derives domain-specific features.
        encoder: Encodes categorical features (OHE / label / target).
        scaler: Standardises or normalises numerical columns.
        binner: Discretises continuous features into ordinal bins.
        splitter: Produces stratified train / val / test DataFrames.
        _is_fitted: Whether the pipeline transformers have been fitted.

    Example:
        >>> from src.config_loader import ConfigLoader
        >>> from src.orchestration.data_pipeline import DataPipeline
        >>> cfg = ConfigLoader.get_instance("config.yaml")
        >>> pipeline = DataPipeline(cfg)
        >>> train_df, val_df, test_df = pipeline.run()
    """

    def __init__(self, cfg: ConfigLoader) -> None:
        """Initialise all sub-processors from the shared config.

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
        self._is_fitted: bool = False

        logger.info("Initialising DataPipeline sub-components ...")

        self.ingestor: DataIngestor = DataIngestor(cfg)
        self.imputer: MissingValueHandler = MissingValueHandler(cfg)
        self.outlier_handler: OutlierHandler = OutlierHandler(cfg)
        self.feature_engineer: FeatureEngineer = FeatureEngineer(cfg)
        self.encoder: FeatureEncoder = FeatureEncoder(cfg)
        self.scaler: FeatureScaler = FeatureScaler(cfg)
        self.binner: FeatureBinner = FeatureBinner(cfg)
        self.splitter: DataSplitter = DataSplitter(cfg)

        logger.info("DataPipeline initialised with 8 sub-components.")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(
        self,
        raw_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the full data processing pipeline end-to-end.

        Reads raw data, applies all transformations, and saves the resulting
        train / val / test splits to the configured splits directory.

        Args:
            raw_path: Optional override for the raw CSV path.  Defaults to
                ``data.raw_path`` from config.yaml.

        Returns:
            A three-element tuple ``(train_df, val_df, test_df)`` containing
            processed and split DataFrames, each with the target column intact.

        Raises:
            FileNotFoundError: If the raw CSV cannot be found.
            ValueError: If the dataset is empty after ingestion.
            RuntimeError: If any pipeline stage fails unexpectedly.
        """
        t0 = time.perf_counter()
        effective_path: str = raw_path or self.cfg.get("data.raw_path")
        logger.info("=" * 70)
        logger.info("DataPipeline.run()  |  raw_path=%s", effective_path)
        logger.info("=" * 70)

        # Stage 1: Ingest
        df = self._stage_ingest(effective_path)

        # Apply training row limit if specified in config (for continuous training / streaming split)
        limit_rows = self.cfg.get("data.limit_training_rows", None)
        if limit_rows is not None:
            logger.info("Limiting pipeline ingestion to the first %d rows.", limit_rows)
            df = df.iloc[:limit_rows]

        # Stage 2: Impute
        df = self._stage_impute_fit_transform(df)

        # Stage 3: Outlier handling
        df = self._stage_outlier_fit_transform(df)

        # Stage 4: Feature Engineering
        df = self._stage_engineer(df)

        # Stage 5: Split FIRST — before encoding/scaling to prevent leakage
        train_df, val_df, test_df = self._stage_split(df)

        # Stage 6: Encode (fit on train only)
        train_df, val_df, test_df = self._stage_encode(train_df, val_df, test_df)

        # Stage 7: Scale (fit on train only)
        train_df, val_df, test_df = self._stage_scale(train_df, val_df, test_df)

        # Stage 8: Bin (fit on train only)
        train_df, val_df, test_df = self._stage_bin(train_df, val_df, test_df)

        self._is_fitted = True

        # Stage 9: Save
        self._save_splits(train_df, val_df, test_df)
        self._save_preprocessor()

        elapsed = time.perf_counter() - t0
        logger.info(
            "DataPipeline.run() complete in %.2f s  |  "
            "train=%s  val=%s  test=%s",
            elapsed,
            train_df.shape,
            val_df.shape,
            test_df.shape,
        )
        return train_df, val_df, test_df

    def run_transform_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all fitted transformers to new (unseen) data for serving.

        The pipeline must have been fitted via :meth:`run` before calling this
        method.  No fitting is performed here — only transforms are applied in
        the same order as during training.

        Args:
            df: Raw or partially processed DataFrame for inference.

        Returns:
            Fully processed DataFrame ready for model scoring.

        Raises:
            RuntimeError: If the pipeline has not been fitted yet.
            ValueError: If *df* is empty.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "DataPipeline is not fitted.  Call run() first before "
                "calling run_transform_only()."
            )
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        logger.info(
            "DataPipeline.run_transform_only()  |  input shape=%s", df.shape
        )

        df = self._log_stage("impute.transform", df, self.imputer.transform)
        df = self._log_stage(
            "outlier.transform", df, self.outlier_handler.transform
        )
        df = self._log_stage(
            "feature_engineer.transform", df, self.feature_engineer.transform
        )
        df = self._log_stage("encoder.transform", df, self.encoder.transform)
        df = self._log_stage("scaler.transform", df, self.scaler.transform)
        df = self._log_stage("binner.transform", df, self.binner.transform)

        logger.info(
            "run_transform_only() complete  |  output shape=%s  dtypes=%s",
            df.shape,
            df.dtypes.value_counts().to_dict(),
        )
        return df

    @classmethod
    def load(cls, filepath: str | Path) -> PreprocessorState:
        """Static factory method to deserialise and load a fitted PreprocessorState.

        Args:
            filepath: Path to the serialised preprocessor .pkl file.

        Returns:
            A fitted PreprocessorState instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If the deserialised object is not PreprocessorState.
        """
        import joblib
        from src.data_processing.preprocessor_state import PreprocessorState
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at: {path.resolve()}")
        logger.info("Loading fitted preprocessor from: %s", path)
        instance = joblib.load(path)
        if not isinstance(instance, PreprocessorState):
            raise TypeError(
                f"Deserialised object is not an instance of PreprocessorState, got {type(instance).__name__}."
            )
        logger.info("Fitted preprocessor state loaded successfully from disk.")
        return instance

    def _save_preprocessor(self) -> None:
        """Serialise the current fitted preprocessor state container to disk using joblib."""
        import joblib
        from src.data_processing.preprocessor_state import PreprocessorState
        models_dir = Path(self.cfg.get("model.models_dir", "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        preprocessor_path = models_dir / "preprocessor.pkl"
        
        state = PreprocessorState(
            ingestor=self.ingestor,
            imputer=self.imputer,
            outlier_handler=self.outlier_handler,
            feature_engineer=self.feature_engineer,
            encoder=self.encoder,
            scaler=self.scaler,
            binner=self.binner
        )
        
        logger.info("Saving fitted preprocessor to: %s", preprocessor_path)
        joblib.dump(state, preprocessor_path)
        logger.info("Preprocessor saved successfully.")

    # -------------------------------------------------------------------------
    # Private stage helpers
    # -------------------------------------------------------------------------

    def _stage_ingest(self, raw_path: str) -> pd.DataFrame:
        """Ingest raw CSV and validate schema.

        Args:
            raw_path: Path to the raw CSV file.

        Returns:
            Raw :class:`~pandas.DataFrame` with schema validated.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the DataFrame is empty after loading.
        """
        logger.info("-- Stage 1 | Ingest  |  source=%s", raw_path)
        df: pd.DataFrame = self.ingestor.ingest(raw_path)
        if df.empty:
            raise ValueError(f"Ingested DataFrame is empty from path: {raw_path}")
        logger.info(
            "   OK Ingested  shape=%s  dtypes=%s  nulls=%d",
            df.shape,
            df.dtypes.value_counts().to_dict(),
            int(df.isnull().sum().sum()),
        )
        return df

    def _stage_impute_fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit imputer on full dataset then transform.

        Note:
            The imputer is fitted on the entire dataset (before splitting)
            because for simple statistical imputers (median/mean/mode) the
            leakage risk is minimal and the benefit is simpler state management.
            Production systems using KNN imputation may prefer fit-on-train.

        Args:
            df: Ingested raw DataFrame.

        Returns:
            DataFrame with missing values filled.
        """
        logger.info("-- Stage 2 | Impute")
        before_nulls = int(df.isnull().sum().sum())
        df = self.imputer.fit_transform(df)
        after_nulls = int(df.isnull().sum().sum())
        logger.info(
            "   OK Imputed  nulls_before=%d  nulls_after=%d  shape=%s",
            before_nulls,
            after_nulls,
            df.shape,
        )
        return df

    def _stage_outlier_fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit outlier handler then apply capping/removal.

        Args:
            df: Imputed DataFrame.

        Returns:
            DataFrame with outliers handled.
        """
        logger.info("-- Stage 3 | Outlier Handling")
        before_shape = df.shape
        df = self.outlier_handler.fit_transform(df)
        logger.info(
            "   OK Outliers handled  shape_before=%s  shape_after=%s",
            before_shape,
            df.shape,
        )
        return df

    def _stage_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to derive domain-specific columns.

        Args:
            df: Clean (imputed + outlier-handled) DataFrame.

        Returns:
            DataFrame augmented with derived features.
        """
        logger.info("-- Stage 4 | Feature Engineering")
        cols_before = set(df.columns)
        df = self.feature_engineer.fit_transform(df)
        new_cols = sorted(set(df.columns) - cols_before)
        logger.info(
            "   OK Features engineered  new_cols=%s  shape=%s",
            new_cols,
            df.shape,
        )
        return df

    def _stage_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train / val / test split.

        Splitting is performed **before** encoding and scaling to prevent
        any statistics derived from val/test bleeding into transformer fits.

        Args:
            df: Fully engineered DataFrame (still un-encoded, un-scaled).

        Returns:
            ``(train_df, val_df, test_df)`` tuple.
        """
        logger.info("-- Stage 5 | Stratified Split")
        train_df, val_df, test_df = self.splitter.split(df)
        logger.info(
            "   OK Split complete  train=%s  val=%s  test=%s",
            train_df.shape,
            val_df.shape,
            test_df.shape,
        )
        return train_df, val_df, test_df

    def _stage_encode(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit encoder on train, transform all three splits.

        Args:
            train_df: Training split (used to fit the encoder).
            val_df: Validation split (transform-only).
            test_df: Test split (transform-only).

        Returns:
            Encoded ``(train_df, val_df, test_df)``.
        """
        logger.info("-- Stage 6 | Encoding  (fit on train)")
        train_df = self.encoder.fit_transform(train_df)
        val_df = self.encoder.transform(val_df)
        test_df = self.encoder.transform(test_df)
        logger.info(
            "   OK Encoded  train=%s  val=%s  test=%s",
            train_df.shape,
            val_df.shape,
            test_df.shape,
        )
        return train_df, val_df, test_df

    def _stage_scale(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit scaler on train, transform all three splits.

        Args:
            train_df: Training split (used to fit the scaler).
            val_df: Validation split (transform-only).
            test_df: Test split (transform-only).

        Returns:
            Scaled ``(train_df, val_df, test_df)``.
        """
        logger.info("-- Stage 7 | Scaling  (fit on train)")
        train_df = self.scaler.fit_transform(train_df)
        val_df = self.scaler.transform(val_df)
        test_df = self.scaler.transform(test_df)
        logger.info(
            "   OK Scaled  train=%s  val=%s  test=%s",
            train_df.shape,
            val_df.shape,
            test_df.shape,
        )
        return train_df, val_df, test_df

    def _stage_bin(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit binner on train, transform all three splits.

        Args:
            train_df: Training split (used to fit the binner).
            val_df: Validation split (transform-only).
            test_df: Test split (transform-only).

        Returns:
            Binned ``(train_df, val_df, test_df)``.
        """
        logger.info("-- Stage 8 | Binning  (fit on train)")
        if not self.cfg.get("preprocessing.binning.enabled", False):
            logger.info("   >> Binning disabled in config -- skipped.")
            return train_df, val_df, test_df

        train_df = self.binner.fit_transform(train_df)
        val_df = self.binner.transform(val_df)
        test_df = self.binner.transform(test_df)
        logger.info(
            "   OK Binned  train=%s  val=%s  test=%s",
            train_df.shape,
            val_df.shape,
            test_df.shape,
        )
        return train_df, val_df, test_df

    def _save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Persist the three splits to the configured splits directory.

        Args:
            train_df: Processed training split.
            val_df: Processed validation split.
            test_df: Processed test split.

        Raises:
            OSError: If the target directory cannot be created.
        """
        logger.info("-- Stage 9 | Saving splits")
        splits_dir = Path(self.cfg.get("data.splits_path", "data/splits"))
        splits_dir.mkdir(parents=True, exist_ok=True)

        save_fmt: str = self.cfg.get("split.save_format", "csv")
        mapping = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
        for name, split_df in mapping.items():
            if save_fmt == "parquet":
                out_path = splits_dir / f"{name}.parquet"
                split_df.to_parquet(out_path, index=False)
            else:
                out_path = splits_dir / f"{name}.csv"
                split_df.to_csv(out_path, index=False)
            logger.info(
                "   OK Saved %s split  path=%s  shape=%s",
                name,
                out_path,
                split_df.shape,
            )

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _log_stage(
        stage_name: str,
        df: pd.DataFrame,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """Apply a transform function and emit a structured log message.

        Args:
            stage_name: Human-readable name for the stage (used in log output).
            df: Input DataFrame.
            transform_fn: Callable that accepts a DataFrame and returns a
                transformed DataFrame.

        Returns:
            Transformed DataFrame.
        """
        before = df.shape
        df = transform_fn(df)
        logger.debug(
            "   [%s]  shape_before=%s  shape_after=%s",
            stage_name,
            before,
            df.shape,
        )
        return df

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"DataPipeline("
            f"fitted={self._is_fitted}, "
            f"ingestor={self.ingestor!r}, "
            f"imputer={self.imputer!r}, "
            f"encoder={self.encoder!r}, "
            f"scaler={self.scaler!r}"
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
        prog="python -m src.orchestration.data_pipeline",
        description=(
            "Telco Churn -- Data Processing Pipeline\n"
            "Runs the full Ingest->Impute->Outlier->FE->Encode->Scale->Bin"
            "->Split->Save pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        dest="raw_path",
        help=(
            "Override the raw CSV path from config.yaml. "
            "If not set, uses data.raw_path from config."
        ),
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    # Bootstrap config + logging
    _cfg = ConfigLoader.get_instance(_args.config)
    setup_logging(_cfg)

    logger.info(
        "Telco Churn | DataPipeline CLI | config=%s | raw_path=%s",
        _args.config,
        _args.raw_path,
    )

    _pipeline = DataPipeline(_cfg)
    _train_df, _val_df, _test_df = _pipeline.run(raw_path=_args.raw_path)

    logger.info(
        "Pipeline finished successfully.\n"
        "  train : %s\n"
        "  val   : %s\n"
        "  test  : %s",
        _train_df.shape,
        _val_df.shape,
        _test_df.shape,
    )
