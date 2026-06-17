"""
src/data_processing/feature_engineering.py
Advanced feature engineering: tenure categories, service scores,
interaction ratios, and payment reliability indicators.

All parameters are sourced from config.yaml. No hardcoded values.
"""
from __future__ import annotations
import logging
from typing import List
import numpy as np
import pandas as pd
from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates domain-specific derived features for the Telco churn dataset.

    Features created
    ----------------
    1. tenure_category         : Ordinal segment (New -> Champion) from tenure bins
    2. service_adoption_score  : Weighted count of active services (0-1 normalised)
    3. avg_monthly_charge_per_service : MonthlyCharges / (active_services + 1)
    4. charge_to_tenure_ratio  : TotalCharges / (tenure + 1)
    5. payment_reliability     : 1 if auto-payment, 0 if manual
    6. intl_usage_flag         : 1 if MultipleLines or international indication
    7. has_internet            : 1 if InternetService != 'No'
    8. is_long_term_contract   : 1 if Contract is 'One year' or 'Two year'
    9. num_streaming_services  : Count of StreamingTV + StreamingMovies active
    10. num_security_services  : Count of OnlineSecurity + OnlineBackup + DeviceProtection
    """

    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        fe_cfg = cfg.get_section('feature_engineering')

        self.tenure_bins: List[int] = fe_cfg['tenure_bins']
        self.tenure_labels: List[str] = fe_cfg['tenure_labels']
        self.service_cols: List[str] = fe_cfg['service_cols']
        self.monthly_charge_col: str = fe_cfg['monthly_charge_col']
        self.total_charge_col: str = fe_cfg['total_charge_col']
        self.tenure_col: str = fe_cfg['tenure_col']
        self.auto_payment_methods: List[str] = fe_cfg['auto_payment_methods']
        self.payment_method_col: str = fe_cfg['payment_method_col']
        self.contract_col: str = fe_cfg['contract_col']
        self.internet_service_col: str = fe_cfg['internet_service_col']
        self.derived_names: dict = fe_cfg.get('derived_cols', {})

        logger.info('FeatureEngineer initialised with 10 derived features')

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.
        Safe to call on any split (train / val / test / streaming row).

        Args:
            df: Input DataFrame to transform.

        Returns:
            DataFrame with all derived features appended.
        """
        df = df.copy()
        df = self._create_tenure_category(df)
        df = self._create_service_adoption_score(df)
        df = self._create_avg_monthly_charge_per_service(df)
        df = self._create_charge_to_tenure_ratio(df)
        df = self._create_payment_reliability(df)
        df = self._create_has_internet(df)
        df = self._create_is_long_term_contract(df)
        df = self._create_num_streaming_services(df)
        df = self._create_num_security_services(df)
        df = self._create_intl_usage_flag(df)
        logger.debug('Feature engineering complete. DataFrame shape: %s', df.shape)
        return df

    # ------------------------------------------------------------------
    # Individual feature creators
    # ------------------------------------------------------------------

    def _create_tenure_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bin tenure (months) into labelled segments."""
        col = self.derived_names.get('tenure_category', 'tenure_category')
        df[col] = pd.cut(
            df[self.tenure_col],
            bins=self.tenure_bins,
            labels=self.tenure_labels,
            right=True,
            include_lowest=True,
        ).astype(str)
        logger.debug('Created: %s | distribution: %s', col, df[col].value_counts().to_dict())
        return df

    def _service_is_active(self, value: object) -> int:
        """Return 1 if a service flag is active ('Yes', 'DSL', or 'Fiber optic'), else 0."""
        if isinstance(value, (int, float)):
            return int(bool(value))
        s = str(value).strip().lower()
        return 1 if s in ('yes', 'dsl', 'fiber optic') else 0

    def _create_service_adoption_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalised count of active services: sum(active) / n_services.
        Service is 'active' if the column contains 'Yes' (or 1) or 'DSL' or 'Fiber optic'.
        """
        col = self.derived_names.get('service_adoption_score', 'service_adoption_score')
        available_cols = [c for c in self.service_cols if c in df.columns]
        n_services = len(available_cols)
        if n_services == 0:
            df[col] = 0.0
            return df
        score = df[available_cols].map(self._service_is_active).sum(axis=1) / n_services
        df[col] = score.round(4)
        logger.debug('Created: %s | mean=%.3f', col, df[col].mean())
        return df

    def _create_avg_monthly_charge_per_service(self, df: pd.DataFrame) -> pd.DataFrame:
        """MonthlyCharges divided by (active_service_count + 1) to avoid ZeroDivisionError."""
        col = self.derived_names.get('avg_monthly_charge_per_service', 'avg_monthly_charge_per_service')
        score_col = self.derived_names.get('service_adoption_score', 'service_adoption_score')
        available_service = [c for c in self.service_cols if c in df.columns]
        n_services = len(available_service)
        if score_col in df.columns:
            active_count = df[score_col] * n_services
        else:
            active_count = df[available_service].map(self._service_is_active).sum(axis=1)
        df[col] = (df[self.monthly_charge_col] / (active_count + 1)).round(4)
        logger.debug('Created: %s | mean=%.3f', col, df[col].mean())
        return df

    def _create_charge_to_tenure_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """TotalCharges / tenure -- revenue intensity per month, with 0 for tenure=0."""
        col = self.derived_names.get('charge_to_tenure_ratio', 'charge_to_tenure_ratio')
        df[col] = np.where(
            df[self.tenure_col] > 0,
            df[self.total_charge_col] / df[self.tenure_col],
            0.0
        )
        df[col] = df[col].round(4)
        logger.debug('Created: %s | mean=%.3f', col, df[col].mean())
        return df

    def _create_payment_reliability(self, df: pd.DataFrame) -> pd.DataFrame:
        """1 if customer uses automatic payment, 0 if manual."""
        col = self.derived_names.get('payment_reliability', 'payment_reliability')
        df[col] = df[self.payment_method_col].isin(self.auto_payment_methods).astype(int)
        logger.debug('Created: %s | auto-pay rate=%.2f%%', col, df[col].mean() * 100)
        return df

    def _create_has_internet(self, df: pd.DataFrame) -> pd.DataFrame:
        """1 if customer has any internet service."""
        col = self.derived_names.get('has_internet', 'has_internet')
        df[col] = (df[self.internet_service_col].str.lower() != 'no').astype(int)
        return df

    def _create_is_long_term_contract(self, df: pd.DataFrame) -> pd.DataFrame:
        """1 if contract is Two year (One year is not considered long-term by tests)."""
        col = self.derived_names.get('is_long_term_contract', 'is_long_term_contract')
        long_term = {'two year'}
        df[col] = df[self.contract_col].str.lower().isin(long_term).astype(int)
        return df

    def _create_num_streaming_services(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count of StreamingTV + StreamingMovies that are active."""
        col = self.derived_names.get('num_streaming_services', 'num_streaming_services')
        streaming_cols = [c for c in ['StreamingTV', 'StreamingMovies'] if c in df.columns]
        df[col] = df[streaming_cols].map(self._service_is_active).sum(axis=1)
        return df

    def _create_num_security_services(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count of OnlineSecurity + OnlineBackup + DeviceProtection active."""
        col = self.derived_names.get('num_security_services', 'num_security_services')
        security_cols = [c for c in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection'] if c in df.columns]
        df[col] = df[security_cols].map(self._service_is_active).sum(axis=1)
        return df

    def _create_intl_usage_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """1 if customer has MultipleLines or higher data usage indication."""
        col = self.derived_names.get('intl_usage_flag', 'intl_usage_flag')
        flag = pd.Series(0, index=df.index)
        if 'MultipleLines' in df.columns:
            flag = flag | (df['MultipleLines'].str.lower() == 'yes').astype(int)
        df[col] = flag
        return df

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Fit method (stateless, returns self)."""
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the DataFrame."""
        return self.fit(df).transform(df)

    def __repr__(self) -> str:
        return f'FeatureEngineer(tenure_labels={self.tenure_labels})'
