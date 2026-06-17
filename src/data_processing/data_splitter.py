"""
src/data_processing/data_splitter.py
─────────────────────────────────────
Stratified train / validation / test split with class distribution logging.
All split ratios and random_state are sourced from config.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Performs a stratified two-step split to produce train / val / test sets.

    Strategy
    --------
    Step 1: Full dataset → (train_val, test)   using test_size from config
    Step 2: train_val    → (train, val)         using val_size from config

    All ratios and random_state are read from config.yaml; nothing is
    hardcoded.

    Parameters
    ----------
    cfg : ConfigLoader
        Singleton config instance.
    """

    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        self.target_col: str = cfg.get("data.target_col")
        self.test_size: float = float(cfg.get("split.test_size", 0.2))
        self.val_size: float = float(cfg.get("split.val_size", 0.125))
        self.stratify: bool = bool(cfg.get("split.stratify", True))
        self.random_state: int = cfg.random_state
        self.save_format: str = cfg.get("split.save_format", "csv")
        self.splits_path: Path = Path(cfg.get("data.splits_path", "data/splits"))

        logger.info(
            "DataSplitter init | test_size=%.2f, val_size=%.3f, "
            "stratify=%s, random_state=%d",
            self.test_size,
            self.val_size,
            self.stratify,
            self.random_state,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train, val, and test DataFrames.

        Parameters
        ----------
        df : pd.DataFrame
            The full, preprocessed dataset (must contain target_col).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train, val, test) DataFrames with reset indices.

        Raises
        ------
        KeyError
            If target_col is not present in the DataFrame.
        ValueError
            If the DataFrame is too small to split.
        """
        if self.target_col not in df.columns:
            raise KeyError(
                f"target_col '{self.target_col}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

        if len(df) < 10:
            raise ValueError(
                f"DataFrame too small to split: {len(df)} rows. Need at least 10."
            )

        logger.info("Splitting dataset | total rows=%d", len(df))

        # ── Step 1: full → train_val + test ──────────────────────────────
        strat_all = df[self.target_col] if self.stratify else None
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=strat_all,
            random_state=self.random_state,
        )

        # ── Step 2: train_val → train + val ──────────────────────────────
        strat_tv = train_val[self.target_col] if self.stratify else None
        train, val = train_test_split(
            train_val,
            test_size=self.val_size,
            stratify=strat_tv,
            random_state=self.random_state,
        )

        # Reset indices for clean downstream usage
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

        self._log_split_info(train, val, test)
        return train, val, test

    def save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """
        Persist splits to disk in the format specified in config.

        Parameters
        ----------
        train, val, test : pd.DataFrame
            The split DataFrames to save.
        """
        self.splits_path.mkdir(parents=True, exist_ok=True)
        splits = {"train": train, "val": val, "test": test}

        for name, split_df in splits.items():
            if self.save_format == "parquet":
                fp = self.splits_path / f"{name}.parquet"
                split_df.to_parquet(fp, index=False)
            else:
                fp = self.splits_path / f"{name}.csv"
                split_df.to_csv(fp, index=False)
            logger.info(
                "Saved %s split → %s | rows=%d", name.upper(), fp, len(split_df)
            )

    def load_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved splits from disk.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train, val, test) DataFrames.

        Raises
        ------
        FileNotFoundError
            If any split file is missing.
        """
        splits = {}
        for name in ("train", "val", "test"):
            if self.save_format == "parquet":
                fp = self.splits_path / f"{name}.parquet"
                splits[name] = pd.read_parquet(fp)
            else:
                fp = self.splits_path / f"{name}.csv"
                if not fp.exists():
                    raise FileNotFoundError(
                        f"Split file not found: {fp}. Run data pipeline first."
                    )
                splits[name] = pd.read_csv(fp)
            logger.info(
                "Loaded %s split ← %s | rows=%d", name.upper(), fp, len(splits[name])
            )
        return splits["train"], splits["val"], splits["test"]

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _log_split_info(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Log size and churn rate for each split."""
        total = len(train) + len(val) + len(test)
        logger.info("═══ Split Summary ═══════════════════════")
        for name, split_df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
            churn_rate = split_df[self.target_col].mean() * 100
            pct_of_total = len(split_df) / total * 100
            logger.info(
                "  %s: %5d rows (%5.1f%% of total) | churn rate=%.2f%%",
                name,
                len(split_df),
                pct_of_total,
                churn_rate,
            )
        logger.info("═════════════════════════════════════════")

    def __repr__(self) -> str:
        return (
            f"DataSplitter("
            f"test={self.test_size}, "
            f"val={self.val_size}, "
            f"stratify={self.stratify}, "
            f"format={self.save_format})"
        )
