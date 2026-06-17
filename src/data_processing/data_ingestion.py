"""
src/data_processing/data_ingestion.py
Data loading, schema validation, and quality reporting.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class SchemaValidationError(Exception):
    pass

class DataIngestor:
    """
    Loads raw CSV data, validates schema against config, coerces types,
    and reports data quality metrics.
    """
    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        self.raw_path = Path(cfg.get('data.raw_path'))
        self.target_col = cfg.get('data.target_col')
        self.customer_id_col = cfg.get('data.customer_id_col')
        self.expected_cols = cfg.get('data.schema.expected_columns')
        self.coerce_to_numeric = cfg.get('data.schema.coerce_to_numeric', [])
        self.binary_map = cfg.get('preprocessing.encoding.binary_map', {'Yes':1,'No':0})
        self.binary_yes_no_cols = cfg.get('data.schema.binary_yes_no_cols', [])

    def load(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load CSV and return cleaned DataFrame."""
        return self.ingest(path)

    def ingest(self, path: str | Path | None = None) -> pd.DataFrame:
        """Ingest CSV, validate schema, coerce types, map binary, and report quality."""
        load_path = Path(path) if path is not None else self.raw_path
        logger.info('Loading raw data from: %s', load_path)
        if not load_path.exists():
            raise FileNotFoundError(f'Raw data not found: {load_path}')
        df = pd.read_csv(load_path)
        logger.info('Raw data shape: %s', df.shape)
        self._validate_schema(df)
        df = self._coerce_types(df)
        df = self._map_binary_cols(df)
        self._report_quality(df)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Raise SchemaValidationError if expected columns are missing."""
        missing = set(self.expected_cols) - set(df.columns)
        extra = set(df.columns) - set(self.expected_cols)
        if missing:
            raise SchemaValidationError(f'Missing columns: {missing}')
        if extra:
            logger.warning('Unexpected extra columns (will be kept): %s', extra)
        logger.info('Schema validation passed. All %d expected columns present.', len(self.expected_cols))

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce columns like TotalCharges from string to numeric."""
        for col in self.coerce_to_numeric:
            if col in df.columns:
                before = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info('Coerced %s: %s -> float64 (NaNs introduced: %d)', col, before, df[col].isna().sum())
        return df

    def _map_binary_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Yes/No binary columns to 1/0."""
        for col in self.binary_yes_no_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].map(self.binary_map)
                logger.info('Binary mapped: %s', col)
        return df

    def _report_quality(self, df: pd.DataFrame) -> None:
        self.report_quality(df)

    def report_quality(self, df: pd.DataFrame) -> None:
        """Log a comprehensive data quality report."""
        logger.info('=== Data Quality Report ===')
        logger.info('Shape: %d rows x %d cols', *df.shape)
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df) * 100).round(2)
        for col in null_counts[null_counts > 0].index:
            logger.warning('NULL | %s: %d (%.2f%%)', col, null_counts[col], null_pct[col])
        dups = df.duplicated().sum()
        if dups > 0:
            logger.warning('Duplicate rows: %d', dups)
        else:
            logger.info('No duplicate rows found.')
        churn_rate = df[self.target_col].mean() * 100
        logger.info('Churn rate: %.2f%%', churn_rate)

    def __repr__(self) -> str:
        return f'DataIngestor(raw_path={self.raw_path})'
