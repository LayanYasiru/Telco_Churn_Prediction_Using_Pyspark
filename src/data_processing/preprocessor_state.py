"""
src/data_processing/preprocessor_state.py
─────────────────────────────────────────
Container class for serialising and deserialising the fitted preprocessor state.
"""

from __future__ import annotations

import logging
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessorState:
    """State container for fitted data preprocessing transformers.

    Holds references to stateful transformers (ingestor, imputer, encoder, scaler, binner)
    and stateless feature engineer, providing an E2E transform method for serving.
    """

    def __init__(
        self,
        ingestor: Any,
        imputer: Any,
        outlier_handler: Any,
        feature_engineer: Any,
        encoder: Any,
        scaler: Any,
        binner: Any,
    ) -> None:
        """Store references to the preprocessing pipeline components."""
        self.ingestor = ingestor
        self.imputer = imputer
        self.outlier_handler = outlier_handler
        self.feature_engineer = feature_engineer
        self.encoder = encoder
        self.scaler = scaler
        self.binner = binner

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the complete fitted preprocessing chain in sequence.

        Args:
            df: Input raw DataFrame.

        Returns:
            Fully preprocessed DataFrame matching training schema.
        """
        df = df.copy()
        
        # 1. Initial raw data type coercion and mapping (mirroring training ingestion)
        df = self.ingestor._coerce_types(df)
        df = self.ingestor._map_binary_cols(df)
        
        # 2. Apply transforms sequentially
        df = self.imputer.transform(df)
        df = self.outlier_handler.transform(df)
        df = self.feature_engineer.transform(df)
        df = self.encoder.transform(df)
        df = self.scaler.transform(df)
        df = self.binner.transform(df)
        
        return df
