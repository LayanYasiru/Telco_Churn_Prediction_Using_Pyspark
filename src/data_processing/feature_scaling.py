"""
src/data_processing/feature_scaling.py
Strategy Pattern for scaling: Standard, MinMax, Robust.
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ScalingStrategy(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'ScalingStrategy': ...
    @abstractmethod
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame: ...
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df, columns)


class StandardScalerWrapper(ScalingStrategy):
    """Wrap sklearn StandardScaler for use in the strategy pattern."""
    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._columns: List[str] = []

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'StandardScalerWrapper':
        self._columns = [c for c in columns if c in df.columns]
        self._scaler.fit(df[self._columns])
        logger.debug('StandardScaler fitted on: %s', self._columns)
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in self._columns if c in df.columns]
        df[cols] = self._scaler.transform(df[cols])
        return df

    def __repr__(self) -> str:
        return f'StandardScalerWrapper(cols={self._columns})'


class MinMaxScalerWrapper(ScalingStrategy):
    """Wrap sklearn MinMaxScaler for use in the strategy pattern."""
    def __init__(self) -> None:
        self._scaler = MinMaxScaler()
        self._columns: List[str] = []

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'MinMaxScalerWrapper':
        self._columns = [c for c in columns if c in df.columns]
        self._scaler.fit(df[self._columns])
        logger.debug('MinMaxScaler fitted on: %s', self._columns)
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in self._columns if c in df.columns]
        df[cols] = self._scaler.transform(df[cols])
        return df

    def __repr__(self) -> str:
        return f'MinMaxScalerWrapper(cols={self._columns})'


class RobustScalerWrapper(ScalingStrategy):
    """Wrap sklearn RobustScaler for use in the strategy pattern."""
    def __init__(self) -> None:
        self._scaler = RobustScaler()
        self._columns: List[str] = []

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'RobustScalerWrapper':
        self._columns = [c for c in columns if c in df.columns]
        self._scaler.fit(df[self._columns])
        logger.debug('RobustScaler fitted on: %s', self._columns)
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in self._columns if c in df.columns]
        df[cols] = self._scaler.transform(df[cols])
        return df

    def __repr__(self) -> str:
        return f'RobustScalerWrapper(cols={self._columns})'


class ScalerFactory:
    """Factory for producing ScalingStrategy instances."""

    @classmethod
    def create_scaler(cls, method_name: str, cfg: ConfigLoader) -> ScalingStrategy:
        """Static factory method to construct a ScalingStrategy.

        Args:
            method_name: Scaling method ('standard', 'minmax', 'robust').
            cfg: ConfigLoader instance.

        Returns:
            Configured ScalingStrategy instance.
        """
        name = method_name.lower()
        if name == 'standard':
            return StandardScalerWrapper()
        elif name == 'minmax':
            return MinMaxScalerWrapper()
        elif name == 'robust':
            return RobustScalerWrapper()
        else:
            raise ValueError(
                f"Unknown scaling method: {method_name}. "
                f"Choose from: ['standard', 'minmax', 'robust']"
            )


class FeatureScaler:
    """
    Orchestrates feature scaling. Fits on train, applies to val/test.
    """
    def __init__(self, cfg: ConfigLoader) -> None:
        method = cfg.get('preprocessing.scaling.method', 'standard')
        self.columns = cfg.get('preprocessing.scaling.columns_to_scale', [])

        self._strategy = ScalerFactory.create_scaler(method, cfg)
        logger.info('FeatureScaler: method=%s, cols=%s', method, self.columns)

    def fit(self, df: pd.DataFrame) -> 'FeatureScaler':
        """Fit scaler on training data."""
        avail = [c for c in self.columns if c in df.columns]
        self._strategy.fit(df, avail)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to data."""
        avail = [c for c in self.columns if c in df.columns]
        return self._strategy.transform(df, avail)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        return f'FeatureScaler(strategy={self._strategy})'
