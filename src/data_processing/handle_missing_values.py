"""
src/data_processing/handle_missing_values.py
Strategy Pattern for imputation: Mean, Median, Mode, KNN.
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ImputationStrategy(ABC):
    """Abstract base class for imputation strategies."""
    @abstractmethod
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'ImputationStrategy':
        ...
    @abstractmethod
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        ...
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df, columns)


class MeanImputer(ImputationStrategy):
    """Impute with column mean."""
    def __init__(self) -> None:
        self._means: dict = {}
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'MeanImputer':
        self._means = {c: df[c].mean() for c in columns if c in df.columns}
        logger.debug('MeanImputer fitted: %s', self._means)
        return self
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col, val in self._means.items():
            df[col] = df[col].fillna(val)
        return df
    def __repr__(self) -> str:
        return f'MeanImputer(columns={list(self._means.keys())})'


class MedianImputer(ImputationStrategy):
    """Impute with column median."""
    def __init__(self) -> None:
        self._medians: dict = {}
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'MedianImputer':
        self._medians = {c: df[c].median() for c in columns if c in df.columns}
        logger.debug('MedianImputer fitted: %s', self._medians)
        return self
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col, val in self._medians.items():
            df[col] = df[col].fillna(val)
        return df
    def __repr__(self) -> str:
        return f'MedianImputer(columns={list(self._medians.keys())})'


class ModeImputer(ImputationStrategy):
    """Impute with column mode."""
    def __init__(self) -> None:
        self._modes: dict = {}
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'ModeImputer':
        self._modes = {c: df[c].mode()[0] for c in columns if c in df.columns and not df[c].mode().empty}
        logger.debug('ModeImputer fitted: %s', self._modes)
        return self
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col, val in self._modes.items():
            df[col] = df[col].fillna(val)
        return df
    def __repr__(self) -> str:
        return f'ModeImputer(columns={list(self._modes.keys())})'


class KNNImputerWrapper(ImputationStrategy):
    """Impute using K-nearest-neighbours."""
    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self._imputer: Optional[KNNImputer] = None
        self._columns: List[str] = []
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'KNNImputerWrapper':
        self._columns = [c for c in columns if c in df.columns]
        self._imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self._imputer.fit(df[self._columns])
        logger.debug('KNNImputer fitted on %d columns with k=%d', len(self._columns), self.n_neighbors)
        return self
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        if self._imputer is not None:
            df[self._columns] = self._imputer.transform(df[self._columns])
        return df
    def __repr__(self) -> str:
        return f'KNNImputerWrapper(k={self.n_neighbors}, columns={self._columns})'


class ImputerFactory:
    """Factory for producing ImputationStrategy instances."""

    @classmethod
    def create_imputer(cls, strategy_name: str, cfg: ConfigLoader) -> ImputationStrategy:
        """Static factory method to construct an ImputationStrategy.

        Args:
            strategy_name: Strategy type ('mean', 'median', 'mode', 'knn').
            cfg: ConfigLoader containing tuning parameters.

        Returns:
            Configured ImputationStrategy instance.
        """
        name = strategy_name.lower()
        if name == 'knn':
            k = cfg.get('preprocessing.missing_values.knn_neighbors', 5)
            return KNNImputerWrapper(n_neighbors=k)
        elif name == 'mean':
            return MeanImputer()
        elif name == 'median':
            return MedianImputer()
        elif name == 'mode':
            return ModeImputer()
        else:
            raise ValueError(
                f"Unknown imputation strategy: {strategy_name}. "
                f"Choose from: ['mean', 'median', 'mode', 'knn']"
            )


class MissingValueHandler:
    """
    Orchestrates imputation using a strategy selected from config.

    Fits on training data only to prevent data leakage.
    """
    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        strategy_name = cfg.get('preprocessing.missing_values.strategy', 'median')
        self.numerical_cols = cfg.get('preprocessing.missing_values.numerical_impute_cols', [])
        self.categorical_cols = cfg.get('preprocessing.missing_values.categorical_impute_cols', [])

        self._numerical_strategy = ImputerFactory.create_imputer(strategy_name, cfg)
        self._categorical_strategy = ImputerFactory.create_imputer('mode', cfg)
        logger.info('MissingValueHandler: numerical strategy=%s, categorical strategy=ModeImputer', strategy_name)

    def fit(self, df: pd.DataFrame) -> 'MissingValueHandler':
        """Fit imputers on training data."""
        if self.numerical_cols:
            self._numerical_strategy.fit(df, self.numerical_cols)
        if self.categorical_cols:
            self._categorical_strategy.fit(df, self.categorical_cols)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputers."""
        df = df.copy()
        before_nulls = df.isnull().sum().sum()
        if self.numerical_cols:
            df = self._numerical_strategy.transform(df, self.numerical_cols)
        if self.categorical_cols:
            df = self._categorical_strategy.transform(df, self.categorical_cols)
        after_nulls = df.isnull().sum().sum()
        logger.info('Imputation: %d -> %d total nulls', before_nulls, after_nulls)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        return f'MissingValueHandler(num_strategy={self._numerical_strategy}, cat_strategy={self._categorical_strategy})'
