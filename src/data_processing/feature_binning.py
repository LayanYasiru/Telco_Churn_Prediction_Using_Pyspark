"""
src/data_processing/feature_binning.py
Binning continuous variables into ordinal categories.
"""
from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class BinConfig:
    """Data class for a single column's binning configuration."""
    def __init__(self, col: str, n_bins: int, strategy: str, labels: Optional[List[str]] = None) -> None:
        self.col = col
        self.n_bins = n_bins
        self.strategy = strategy
        self.labels = labels

    def __repr__(self) -> str:
        return f'BinConfig(col={self.col}, n_bins={self.n_bins}, strategy={self.strategy})'


class FeatureBinner:
    """
    Applies quantile or uniform binning to specified continuous columns.
    Adds new `{col}_bin` columns and preserves originals.

    Fits bin edges on training data only to prevent data leakage.
    """
    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        self.enabled = cfg.get('preprocessing.binning.enabled', True)
        raw_bins = cfg.get('preprocessing.binning.columns', [])
        self.bin_configs: List[BinConfig] = [
            BinConfig(
                col=b['col'],
                n_bins=b.get('n_bins', 5),
                strategy=b.get('strategy', 'quantile'),
                labels=b.get('labels'),
            )
            for b in raw_bins
        ]
        self._fitted_edges: dict[str, np.ndarray] = {}
        logger.info('FeatureBinner: enabled=%s, configs=%s', self.enabled, self.bin_configs)

    def fit(self, df: pd.DataFrame) -> 'FeatureBinner':
        """Compute and store bin edges from training data."""
        if not self.enabled:
            return self
        for bc in self.bin_configs:
            if bc.col not in df.columns:
                logger.warning('Binning column %s not found in DataFrame.', bc.col)
                continue
            if bc.strategy == 'uniform':
                _, edges = pd.cut(
                    df[bc.col].dropna(),
                    bins=bc.n_bins,
                    retbins=True,
                    include_lowest=True,
                )
            else:
                _, edges = pd.qcut(
                    df[bc.col].dropna(),
                    q=bc.n_bins,
                    retbins=True,
                    duplicates='drop',
                )
            self._fitted_edges[bc.col] = edges
            logger.debug('Fitted bin edges for %s: %s', bc.col, edges)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted bin edges to produce new `{col}_bin` columns."""
        if not self.enabled:
            return df
        df = df.copy()
        for bc in self.bin_configs:
            if bc.col not in df.columns or bc.col not in self._fitted_edges:
                continue
            edges = self._fitted_edges[bc.col]
            n_actual = len(edges) - 1
            labels = bc.labels[:n_actual] if bc.labels and len(bc.labels) >= n_actual else None
            new_col = f'{bc.col}_bin'
            df[new_col] = pd.cut(
                df[bc.col],
                bins=edges,
                labels=labels,
                include_lowest=True,
            ).astype(str)
            logger.debug('Created bin column: %s', new_col)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        return f'FeatureBinner(enabled={self.enabled}, n_configs={len(self.bin_configs)})'
