"""
src/config_loader.py
────────────────────
Singleton configuration loader for the Telco Churn Prediction pipeline.

All modules import this class to access config.yaml parameters.
Supports dot-path access: cfg.get("model.cv_folds")
"""

from __future__ import annotations

import logging
import os
from functools import reduce
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when required config keys are missing or malformed."""


class ConfigLoader:
    """
    Thread-safe singleton that reads config.yaml once and caches the result.

    Usage
    -----
    >>> from src.config_loader import ConfigLoader
    >>> cfg = ConfigLoader.get_instance("config.yaml")
    >>> folds = cfg.get("model.cv_folds")          # dot-path accessor
    >>> target = cfg.get("data.target_col")
    """

    _instance: Optional["ConfigLoader"] = None
    _config: dict = {}

    # Required top-level keys — raise ConfigError if absent
    _REQUIRED_KEYS: list[str] = [
        "project",
        "data",
        "preprocessing",
        "feature_engineering",
        "split",
        "model",
        "evaluation",
        "reporting",
    ]

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self._config_path = Path(config_path)
        self._config = self._load()
        self._validate()

    # ──────────────────────────────────────────────────────────────────────
    # Singleton factory
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, config_path: str | Path = "config.yaml") -> "ConfigLoader":
        """Return the singleton ConfigLoader, initialising it if necessary."""
        if cls._instance is None:
            cls._instance = cls(config_path)
            logger.info("ConfigLoader initialised from: %s", config_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful in tests)."""
        cls._instance = None
        cls._config = {}

    # ──────────────────────────────────────────────────────────────────────
    # Load & validate
    # ──────────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Parse YAML file from disk."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at: {self._config_path.resolve()}"
            )
        with open(self._config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ConfigError("config.yaml must be a YAML mapping at the top level.")
        logger.debug("Raw config loaded with %d top-level keys.", len(data))
        return data

    def _validate(self) -> None:
        """Assert that all required top-level sections exist."""
        missing = [k for k in self._REQUIRED_KEYS if k not in self._config]
        if missing:
            raise ConfigError(
                f"config.yaml is missing required sections: {missing}"
            )
        logger.debug("Config validation passed.")

    # ──────────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────────

    def get(self, dot_path: str, default: Any = None) -> Any:
        """
        Retrieve a value by dot-separated key path.

        Parameters
        ----------
        dot_path:
            e.g. "model.cv_folds" or "preprocessing.outlier.iqr_factor"
        default:
            Returned if the key does not exist (instead of raising).

        Examples
        --------
        >>> cfg.get("model.cv_folds")
        5
        >>> cfg.get("data.target_col")
        'Churn'
        """
        keys = dot_path.split(".")
        try:
            return reduce(lambda d, k: d[k], keys, self._config)
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise ConfigError(
                f"Config key '{dot_path}' not found and no default supplied."
            )

    def get_section(self, section: str) -> dict:
        """Return an entire top-level section as a dict."""
        if section not in self._config:
            raise ConfigError(f"Config section '{section}' does not exist.")
        return self._config[section]

    @property
    def raw(self) -> dict:
        """Direct access to the underlying dict (read-only by convention)."""
        return self._config

    @property
    def random_state(self) -> int:
        return int(self.get("project.random_state", 42))

    @property
    def log_level(self) -> str:
        return str(self.get("project.log_level", "INFO"))

    # ──────────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────────────────

    def resolve_path(self, dot_path: str) -> Path:
        """Return a Path relative to the project root (CWD)."""
        raw_path = self.get(dot_path)
        return Path(raw_path)

    def __repr__(self) -> str:
        return f"ConfigLoader(path={self._config_path}, keys={list(self._config.keys())})"


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience accessor
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(cfg: ConfigLoader) -> None:
    """
    Configure root logger using settings from config.yaml.

    Call once at the entry-point of each pipeline script.
    """
    log_level_str: str = cfg.get("project.log_level", "INFO")
    log_format: str = cfg.get(
        "project.log_format",
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    log_file: str = cfg.get("project.log_file", "logs/pipeline.log")

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
        force=True,
    )
    logger.info("Logging initialised | level=%s | file=%s", log_level_str, log_file)
