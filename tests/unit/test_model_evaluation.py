"""
tests/unit/test_model_evaluation.py
───────────────────────────────────
Unit tests for ModelEvaluator metrics, threshold optimization, and plot generation.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.model_development.model_evaluation import ModelEvaluator


def test_evaluator_metrics(cfg, small_X_y) -> None:
    """Verify ModelEvaluator computes correct metrics on synthetic data."""
    X, y = small_X_y
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    evaluator = ModelEvaluator(cfg)
    results = evaluator.evaluate(
        model_name="test_model",
        estimator=clf,
        X=X,
        y=y,
        threshold=0.5,
    )

    # Assert expected keys in metrics
    assert results["model_name"] == "test_model"
    assert results["threshold"] == 0.5
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "roc_auc" in results
    assert "pr_auc" in results
    assert "confusion_matrix" in results
    assert "classification_report" in results


def test_evaluator_threshold_optimization(cfg, small_X_y) -> None:
    """Verify ModelEvaluator sweeps threshold and finds best threshold correctly."""
    X, y = small_X_y
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    evaluator = ModelEvaluator(cfg)
    best_thresh, best_metrics = evaluator.optimize_threshold(
        estimator=clf,
        X_val=X,
        y_val=y,
    )

    assert 0.0 <= best_thresh <= 1.0
    assert "f1" in best_metrics
    
    # Check optimise_threshold returning only a float
    float_thresh = evaluator.optimise_threshold(clf, X, y)
    assert float_thresh == best_thresh


def test_evaluator_plot_generation(cfg, small_X_y, tmp_path) -> None:
    """Verify ModelEvaluator plotting functions run without errors."""
    X, y = small_X_y
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    # Override figures_dir to a temporary directory for the test
    evaluator = ModelEvaluator(cfg)
    evaluator.figures_dir = tmp_path

    # Check that plotting functions run without throwing exceptions
    # Using small dummy inputs
    evaluator.plot_roc_curve(clf, X, y, "LogisticRegression")
    evaluator.plot_pr_curve(clf, X, y, "LogisticRegression")
    evaluator.plot_confusion_matrix(clf, X, y, "LogisticRegression")
    evaluator.plot_learning_curves(clf, X, y, "LogisticRegression")
    evaluator.plot_threshold_analysis(clf, X, y, "LogisticRegression")
    evaluator.plot_calibration_curve(clf, X, y, "LogisticRegression")

    # Confirm some png files were created
    files = os.listdir(tmp_path)
    assert len(files) > 0
    assert any(f.endswith(".png") for f in files)
