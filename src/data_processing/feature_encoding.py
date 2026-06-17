"""
src/data_processing/feature_encoding.py
Strategy Pattern for categorical encoding: OneHot, Label, Target.
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class EncodingStrategy(ABC):
    """Abstract base for encoding strategies."""
    @abstractmethod
    def fit(self, df: pd.DataFrame, columns: List[str], target: Optional[pd.Series] = None) -> 'EncodingStrategy':
        ...
    @abstractmethod
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        ...
    def fit_transform(self, df: pd.DataFrame, columns: List[str], target: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(df, columns, target).transform(df, columns)


class OneHotEncoderWrapper(EncodingStrategy):
    """One-hot encode categorical columns (drop_first configurable)."""
    def __init__(self, drop_first: bool = True) -> None:
        self.drop_first = drop_first
        self._categories: dict = {}
        self._fitted_cols: List[str] = []

    def fit(self, df: pd.DataFrame, columns: List[str], target: Optional[pd.Series] = None) -> 'OneHotEncoderWrapper':
        self._fitted_cols = [c for c in columns if c in df.columns]
        for col in self._fitted_cols:
            self._categories[col] = sorted(df[col].dropna().unique().tolist())
        logger.debug('OneHotEncoder fitted on: %s', self._fitted_cols)
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in self._fitted_cols:
            if col not in df.columns:
                logger.warning('Column %s not in DataFrame during transform.', col)
                continue
            cats = self._categories[col]
            for i, cat in enumerate(cats):
                if self.drop_first and i == 0:
                    continue
                new_col = f'{col}_{cat}'
                df[new_col] = (df[col] == cat).astype(int)
            df.drop(columns=[col], inplace=True)
        return df

    def __repr__(self) -> str:
        return f'OneHotEncoderWrapper(drop_first={self.drop_first}, cols={self._fitted_cols})'


class LabelEncoderWrapper(EncodingStrategy):
    """Label-encode categorical columns."""
    def __init__(self) -> None:
        self._encoders: dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame, columns: List[str], target: Optional[pd.Series] = None) -> 'LabelEncoderWrapper':
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self._encoders[col] = le
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col, le in self._encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        return df

    def __repr__(self) -> str:
        return f'LabelEncoderWrapper(cols={list(self._encoders.keys())})'


class TargetEncoderWrapper(EncodingStrategy):
    """Target-encode categorical columns (mean of target per category)."""
    def __init__(self) -> None:
        self._means: dict[str, dict] = {}
        self._global_mean: float = 0.0

    def fit(self, df: pd.DataFrame, columns: List[str], target: Optional[pd.Series] = None) -> 'TargetEncoderWrapper':
        if target is None:
            raise ValueError('TargetEncoder requires target series.')
        self._global_mean = float(target.mean())
        for col in columns:
            if col in df.columns:
                self._means[col] = target.groupby(df[col]).mean().to_dict()
        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col, means in self._means.items():
            if col in df.columns:
                df[col] = df[col].map(means).fillna(self._global_mean)
        return df

    def __repr__(self) -> str:
        return f'TargetEncoderWrapper(cols={list(self._means.keys())})'


class EncoderFactory:
    """Factory for producing EncodingStrategy instances."""

    @classmethod
    def create_encoder(cls, method_name: str, cfg: ConfigLoader) -> EncodingStrategy:
        """Static factory method to construct an EncodingStrategy.

        Args:
            method_name: Encoder method ('onehot', 'label', 'target').
            cfg: ConfigLoader instance.

        Returns:
            Configured EncodingStrategy instance.
        """
        name = method_name.lower()
        if name == 'onehot':
            drop_first = cfg.get('preprocessing.encoding.drop_first', True)
            return OneHotEncoderWrapper(drop_first=drop_first)
        elif name == 'label':
            return LabelEncoderWrapper()
        elif name == 'target':
            return TargetEncoderWrapper()
        else:
            raise ValueError(
                f"Unknown encoding method: {method_name}. "
                f"Choose from: ['onehot', 'label', 'target']"
            )


class FeatureEncoder:
    """
    Orchestrates encoding, fitting only on training data.
    """
    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        method = cfg.get('preprocessing.encoding.method', 'onehot')
        self.categorical_cols = cfg.get('data.schema.categorical_cols', [])

        self._strategy = EncoderFactory.create_encoder(method, cfg)
        logger.info('FeatureEncoder: method=%s, cols=%s', method, self.categorical_cols)

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureEncoder':
        self._strategy.fit(df, self.categorical_cols, target)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._strategy.transform(df, self.categorical_cols)

    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(df, target).transform(df)

    def __repr__(self) -> str:
        return f'FeatureEncoder(strategy={self._strategy})'
