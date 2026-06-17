"""
tests/integration/test_training_pipeline.py
───────────────────────────────────────────
Integration tests for TrainingPipeline E2E execution and champion model promotion.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.orchestration.data_pipeline import DataPipeline
from src.orchestration.training_pipeline import TrainingPipeline
from src.config_loader import ConfigLoader


def test_training_pipeline_e2e(cfg, sample_df, tmp_path) -> None:
    """Verify end-to-end TrainingPipeline flow, model selection, and artifact saving."""
    # Write sample_df to a temporary CSV raw path
    raw_path = tmp_path / "telco_churn_raw.csv"
    sample_df.to_csv(raw_path, index=False)

    # Configure temporary directories for splits, figures, and models
    splits_path = tmp_path / "splits"
    models_dir = tmp_path / "models"
    figures_dir = tmp_path / "figures"
    champion_dir = models_dir / "champion"
    champion_metadata_file = champion_dir / "champion_metadata.json"

    # Configure ConfigLoader to use these temporary paths
    cfg._config["data"]["raw_path"] = str(raw_path)
    cfg._config["data"]["splits_path"] = str(splits_path)
    cfg._config["model"]["models_dir"] = str(models_dir)
    cfg._config["model"]["champion_dir"] = str(champion_dir)
    cfg._config["model"]["champion_metadata_file"] = str(champion_metadata_file)
    cfg._config["reporting"]["figures_dir"] = str(figures_dir)
    cfg._config["split"]["save_format"] = "csv"
    
    # Train only quick models for test speed (e.g. decision_tree and logistic_regression)
    cfg._config["model"]["models_to_train"] = ["logistic_regression", "decision_tree"]

    # 1. Run DataPipeline to generate training/validation/test splits
    dp = DataPipeline(cfg)
    dp.run(raw_path=str(raw_path))

    # 2. Run TrainingPipeline
    tp = TrainingPipeline(cfg)
    results = tp.run()

    # 3. Assert outputs are created
    assert "champion_model_name" in results
    assert "champion_val_metrics" in results
    assert "champion_test_metrics" in results
    assert "champion_threshold" in results

    # Assert model PKL and metadata JSON exist on disk
    champion_pkl = champion_dir / "champion_model.pkl"
    assert champion_pkl.exists()
    assert champion_metadata_file.exists()

    # Read and verify metadata
    with open(champion_metadata_file, "r") as f:
        metadata = json.load(f)
    assert metadata["model_name"] in ["logistic_regression", "decision_tree"]
    assert "val_metrics" in metadata
    assert "test_metrics" in metadata
    assert "optimal_threshold" in metadata

    # Assert at least one figure/plot was generated in figures_dir
    assert figures_dir.exists()
    figs = os.listdir(figures_dir)
    assert len(figs) > 0
    assert any(fig.endswith(".png") for fig in figs)
