"""
src/data_processing/outlier_detection.py
Strategy Pattern for outlier handling: IQR and Z-Score.
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class OutlierStrategy(ABC):
    """Abstract base for outlier detection strategies."""
    @abstractmethod
    def detect(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """Return {col: (lower_bound, upper_bound)} bounds dict."""
        ...


class IQRDetector(OutlierStrategy):
    """
    Detect outliers using the Interquartile Range method.
    Bounds: Q1 - factor*IQR to Q3 + factor*IQR
    """
    def __init__(self, iqr_factor: float = 1.5) -> None:
        self.iqr_factor = iqr_factor

    def detect(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        for col in columns:
            if col not in df.columns:
                logger.warning('Column %s not found, skipping outlier check.', col)
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            logger.info('IQR | %s: lower=%.3f, upper=%.3f, outliers=%d', col, lower, upper, n_outliers)
            bounds[col] = (lower, upper)
        return bounds

    def __repr__(self) -> str:
        return f'IQRDetector(factor={self.iqr_factor})'


class ZScoreDetector(OutlierStrategy):
    """
    Detect outliers using Z-Score method.
    Bounds: mean +/- threshold * std
    """
    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}

    def detect(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
        for col in columns:
            if col not in df.columns:
                logger.warning('Column %s not found, skipping.', col)
                continue
            mean = df[col].mean()
            std = df[col].std()
            self._means[col] = mean
            self._stds[col] = std
            lower = mean - self.threshold * std
            upper = mean + self.threshold * std
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            logger.info('ZScore | %s: lower=%.3f, upper=%.3f, outliers=%d', col, lower, upper, n_outliers)
            bounds[col] = (lower, upper)
        return bounds

    def __repr__(self) -> str:
        return f'ZScoreDetector(threshold={self.threshold})'


class OutlierDetectorFactory:
    """Factory for producing OutlierStrategy instances."""

    @classmethod
    def create_detector(cls, method_name: str, cfg: ConfigLoader) -> OutlierStrategy:
        """Static factory method to construct an OutlierStrategy.

        Args:
            method_name: Method type ('iqr', 'zscore').
            cfg: ConfigLoader containing parameters.

        Returns:
            Configured OutlierStrategy instance.
        """
        name = method_name.lower()
        if name == 'iqr':
            factor = cfg.get('preprocessing.outlier.iqr_factor', 1.5)
            return IQRDetector(iqr_factor=factor)
        elif name == 'zscore':
            threshold = cfg.get('preprocessing.outlier.zscore_threshold', 3.0)
            return ZScoreDetector(threshold=threshold)
        else:
            raise ValueError(
                f"Unknown outlier method: {method_name}. "
                f"Choose from: ['iqr', 'zscore']"
            )


class OutlierHandler:
    """
    Orchestrates outlier detection and remediation (cap or remove).
    Fits bounds on training data only.
    """
    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        method = cfg.get('preprocessing.outlier.method', 'iqr')
        self.action = cfg.get('preprocessing.outlier.action', 'cap')
        self.columns = cfg.get('preprocessing.outlier.columns_to_check', [])

        self._strategy = OutlierDetectorFactory.create_detector(method, cfg)
        self._bounds: Dict[str, Tuple[float, float]] = {}
        logger.info('OutlierHandler: method=%s, action=%s, cols=%s', method, self.action, self.columns)

    def fit(self, df: pd.DataFrame) -> 'OutlierHandler':
        """Compute outlier bounds from training data."""
        self._bounds = self._strategy.detect(df, self.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cap or remove based on config action."""
        df = df.copy()
        if self.action == 'cap':
            for col, (lower, upper) in self._bounds.items():
                df[col] = df[col].clip(lower=lower, upper=upper)
                logger.info('Capped %s to [%.3f, %.3f]', col, lower, upper)
        elif self.action == 'remove':
            mask = pd.Series([True] * len(df), index=df.index)
            for col, (lower, upper) in self._bounds.items():
                mask &= df[col].between(lower, upper)
            before = len(df)
            df = df[mask].reset_index(drop=True)
            logger.info('Removed outlier rows: %d -> %d (dropped %d)', before, len(df), before - len(df))
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        return f'OutlierHandler(strategy={self._strategy}, action={self.action})'
