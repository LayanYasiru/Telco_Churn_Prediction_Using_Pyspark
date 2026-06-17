"""
src/model_development/model_evaluation.py
──────────────────────────────────────────
ModelEvaluator: comprehensive metrics, threshold optimisation,
cost-sensitive analysis, champion selection, and automated plots.

All thresholds, cost matrix values, plot flags, and paths are
sourced from config.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server/CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates trained classifiers with a comprehensive suite of metrics,
    threshold optimisation, cost-sensitive analysis, and automated plots.

    Parameters
    ----------
    cfg : ConfigLoader
        Singleton config instance.
    """

    def __init__(self, cfg: ConfigLoader) -> None:
        self.cfg = cfg
        self.champion_metric: str = cfg.get("model.champion_metric", "roc_auc")
        self.threshold_metric: str = cfg.get("evaluation.threshold_metric", "f1")

        # Threshold sweep range
        t_cfg = cfg.get("evaluation.threshold_range", {})
        self.thresh_start: float = float(t_cfg.get("start", 0.10))
        self.thresh_stop: float = float(t_cfg.get("stop", 0.90))
        self.thresh_step: float = float(t_cfg.get("step", 0.05))

        # Business cost matrix
        cost_cfg = cfg.get("evaluation.cost_matrix", {})
        self.fp_cost: float = float(cost_cfg.get("fp_cost", 10.0))
        self.fn_cost: float = float(cost_cfg.get("fn_cost", 500.0))

        # Plot config
        self.figures_dir: Path = Path(cfg.get("reporting.figures_dir", "reports/figures"))
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.plot_cfg: dict = cfg.get("evaluation.plots", {})

        logger.info(
            "ModelEvaluator init | champion_metric=%s, threshold_metric=%s, "
            "fp_cost=%.1f, fn_cost=%.1f",
            self.champion_metric,
            self.threshold_metric,
            self.fp_cost,
            self.fn_cost,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Core evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model_name: str = "model",
        estimator: BaseEstimator = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Compute a comprehensive set of evaluation metrics.

        Parameters
        ----------
        model_name : str
            Human-readable model identifier (or split name).
        estimator : BaseEstimator
            Fitted sklearn-compatible classifier.
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True binary labels.
        threshold : float
            Decision threshold (default 0.5).

        Returns
        -------
        Dict[str, Any]
            Keys: model_name, accuracy, precision, recall, f1, roc_auc,
            pr_auc, confusion_matrix, classification_report, threshold.
        """
        est = kwargs.get("estimator", estimator)
        x_data = kwargs.get("X", X)
        y_data = kwargs.get("y_true", y)
        name = kwargs.get("split_name", model_name)
        thresh = kwargs.get("threshold", threshold)

        y_proba = self._get_probas(est, x_data)
        y_pred = (y_proba >= thresh).astype(int)

        acc = float(accuracy_score(y_data, y_pred))
        prec = float(precision_score(y_data, y_pred, zero_division=0))
        rec = float(recall_score(y_data, y_pred, zero_division=0))
        f1 = float(f1_score(y_data, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_data, y_proba))
        pr_auc = float(average_precision_score(y_data, y_proba))
        cm = confusion_matrix(y_data, y_pred).tolist()
        cls_report = classification_report(y_data, y_pred, output_dict=True)

        results: Dict[str, Any] = {
            "model_name": name,
            "threshold": thresh,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "classification_report": cls_report,
        }

        logger.info(
            "Evaluation | %s | threshold=%.2f | acc=%.4f | prec=%.4f | "
            "rec=%.4f | f1=%.4f | roc_auc=%.4f | pr_auc=%.4f",
            name,
            thresh,
            acc,
            prec,
            rec,
            f1,
            roc_auc,
            pr_auc,
        )
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Threshold optimisation
    # ──────────────────────────────────────────────────────────────────────

    def optimise_threshold(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        y_true: np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Sweep thresholds and return only the float threshold (for training_pipeline).
        """
        best_thresh, _ = self.optimize_threshold(estimator=estimator, X_val=X, y_val=y_true, **kwargs)
        return best_thresh

    def optimize_threshold(
        self,
        estimator: BaseEstimator,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs: Any,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Sweep thresholds over the range defined in config and return the
        threshold that maximises the configured `threshold_metric`.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted classifier.
        X_val, y_val :
            Validation set features and labels.

        Returns
        -------
        Tuple[float, Dict[str, Any]]
            (optimal_threshold, metrics_at_optimal_threshold)
        """
        x_data = X_val if X_val is not None else kwargs.get("X")
        y_data = y_val if y_val is not None else kwargs.get("y_true")

        y_proba = self._get_probas(estimator, x_data)
        thresholds = np.arange(
            self.thresh_start, self.thresh_stop + 1e-9, self.thresh_step
        )

        best_thresh = 0.5
        best_score = -np.inf
        best_metrics: Dict[str, float] = {}

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            score = self._compute_threshold_metric(y_data, y_pred, y_proba)
            if score > best_score:
                best_score = score
                best_thresh = float(thresh)
                best_metrics = {
                    self.threshold_metric: float(score),
                    "threshold": float(thresh),
                    "f1": float(f1_score(y_data, y_pred, zero_division=0)),
                    "precision": float(precision_score(y_data, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_data, y_pred, zero_division=0)),
                }

        logger.info(
            "Threshold optimisation | best_threshold=%.2f | best_%s=%.4f",
            best_thresh,
            self.threshold_metric,
            best_score,
        )
        return best_thresh, best_metrics

    def _compute_threshold_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """Compute the metric used for threshold selection."""
        if self.threshold_metric == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        elif self.threshold_metric == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))
        elif self.threshold_metric == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))
        elif self.threshold_metric == "cost":
            cost = self.cost_sensitive_analysis(y_true, y_pred, threshold=0.5)
            return -cost["total_cost"]  # minimise cost → maximise negative cost
        else:
            return float(f1_score(y_true, y_pred, zero_division=0))

    # ──────────────────────────────────────────────────────────────────────
    # Cost-sensitive analysis
    # ──────────────────────────────────────────────────────────────────────

    def cost_sensitive_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute business costs using the FP/FN cost matrix from config.

        Parameters
        ----------
        y_true, y_pred :
            True and predicted labels.
        threshold : float
            Threshold used to produce y_pred (informational only).

        Returns
        -------
        Dict with: tp, fp, fn, tn, fp_cost_total, fn_cost_total, total_cost.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fp_total = float(fp) * self.fp_cost
        fn_total = float(fn) * self.fn_cost
        total = fp_total + fn_total

        result = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "fp_cost_per_instance": self.fp_cost,
            "fn_cost_per_instance": self.fn_cost,
            "fp_cost_total": fp_total,
            "fn_cost_total": fn_total,
            "total_cost": total,
        }
        logger.info(
            "Cost analysis | threshold=%.2f | FP=%d ($%.0f) | FN=%d ($%.0f) | "
            "Total cost=$%.0f",
            threshold,
            fp,
            fp_total,
            fn,
            fn_total,
            total,
        )
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Champion selection
    # ──────────────────────────────────────────────────────────────────────

    def compare_models(self, results_list: List[Dict[str, Any]]) -> str:
        """
        Select the champion model by the metric defined in config.

        Parameters
        ----------
        results_list : List[Dict]
            Each dict must contain 'model_name' and the champion_metric key.

        Returns
        -------
        str
            Name of the champion model.
        """
        if not results_list:
            raise ValueError("results_list is empty; cannot select champion.")

        metric = self.champion_metric
        best = max(results_list, key=lambda r: r.get(metric, -1.0))
        champion = best["model_name"]

        logger.info("═══ Model Comparison (metric=%s) ═══", metric)
        for r in sorted(results_list, key=lambda x: x.get(metric, 0), reverse=True):
            logger.info(
                "  %s: %s=%.4f",
                r["model_name"],
                metric,
                r.get(metric, float("nan")),
            )
        logger.info("Champion → %s (%.4f)", champion, best.get(metric, 0.0))
        return champion

    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────

    def _align_plot_args(
        self,
        model_name: str,
        estimator: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, BaseEstimator, np.ndarray, np.ndarray]:
        """Align argument order regardless of whether model_name or estimator was passed first."""
        if not isinstance(model_name, str):
            # training_pipeline format: (estimator, X, y, model_name)
            est = model_name
            x_data = estimator
            y_data = X
            name = y if isinstance(y, str) else (args[0] if len(args) > 0 else "model")
        else:
            name = model_name
            est = estimator
            x_data = X
            y_data = y
        return name, est, x_data, y_data

    def plot_roc_curve(
        self,
        model_name: str,
        estimator: BaseEstimator = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot and save ROC curve."""
        if not self.plot_cfg.get("roc_curve", True):
            return
        name, est, x_data, y_data = self._align_plot_args(model_name, estimator, X, y, *args, **kwargs)

        y_proba = self._get_probas(est, x_data)
        fpr, tpr, _ = roc_curve(y_data, y_proba)
        auc = roc_auc_score(y_data, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, lw=2, label=f"ROC AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC Curve — {name}",
        )
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"roc_curve_{name}.png")

    def plot_pr_curve(
        self,
        model_name: str,
        estimator: BaseEstimator = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot and save Precision-Recall curve."""
        if not self.plot_cfg.get("pr_curve", True):
            return
        name, est, x_data, y_data = self._align_plot_args(model_name, estimator, X, y, *args, **kwargs)

        y_proba = self._get_probas(est, x_data)
        precision, recall, _ = precision_recall_curve(y_data, y_proba)
        pr_auc = average_precision_score(y_data, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, lw=2, label=f"PR AUC = {pr_auc:.4f}")
        baseline = y_data.mean()
        ax.axhline(baseline, color="r", linestyle="--", lw=1, label=f"Baseline={baseline:.3f}")
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"Precision-Recall Curve — {name}",
        )
        ax.legend()
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"pr_curve_{name}.png")

    def plot_confusion_matrix(
        self,
        model_name: str,
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot and save normalised confusion matrix heatmap."""
        if not self.plot_cfg.get("confusion_matrix", True):
            return

        if not isinstance(model_name, str):
            # training_pipeline format: (estimator, X_test, y_test, model_name)
            est = model_name
            X = y_true
            y_actual = y_pred
            name = args[0] if len(args) > 0 else kwargs.get("model_name", "model")
            
            y_proba = self._get_probas(est, X)
            y_predicted = (y_proba >= 0.5).astype(int)
        else:
            # Original format: (model_name, y_true, y_pred)
            name = model_name
            y_actual = y_true
            y_predicted = y_pred

        cm = confusion_matrix(y_actual, y_predicted)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, data, title in zip(
            axes,
            [cm, cm_norm],
            ["Counts", "Normalised"],
        ):
            fmt = "d" if title == "Counts" else ".2%"
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                ax=ax,
            )
            ax.set(
                xlabel="Predicted",
                ylabel="Actual",
                title=f"Confusion Matrix ({title}) — {name}",
            )
        self._save_fig(fig, f"confusion_matrix_{name}.png")

    def plot_feature_importance(
        self,
        model_name: str,
        estimator: BaseEstimator = None,
        feature_names: List[str] = None,
        top_n: int = 20,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot and save top-N feature importance bar chart."""
        if not self.plot_cfg.get("feature_importance", True):
            return

        if not isinstance(model_name, str):
            # training_pipeline format: (estimator, feature_names, model_name)
            est = model_name
            feats = estimator
            name = feature_names if isinstance(feature_names, str) else (args[0] if len(args) > 0 else "model")
            n_top = top_n
        else:
            name = model_name
            est = estimator
            feats = feature_names
            n_top = top_n

        # Unwrap imblearn pipeline or sklearn pipeline if needed
        model = est
        if hasattr(est, "named_steps") and "model" in est.named_steps:
            model = est.named_steps["model"]
        elif hasattr(est, "steps"):
            model = est.steps[-1][1]

        importances: Optional[np.ndarray] = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        if importances is None:
            logger.warning(
                "Model %s does not expose feature importances; skipping plot.",
                name,
            )
            return

        n = min(n_top, len(feats), len(importances))
        indices = np.argsort(importances)[::-1][:n]
        top_feats = [feats[i] for i in indices]
        top_vals = importances[indices]

        fig, ax = plt.subplots(figsize=(10, max(6, n // 2)))
        ax.barh(range(n), top_vals[::-1], color="steelblue", edgecolor="white")
        ax.set_yticks(range(n))
        ax.set_yticklabels(top_feats[::-1])
        ax.set(
            xlabel="Importance Score",
            title=f"Top {n} Feature Importances — {name}",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._save_fig(fig, f"feature_importance_{name}.png")

    def plot_learning_curves(
        self,
        model_name: str,
        estimator: BaseEstimator = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot and save learning curves using sklearn learning_curve."""
        if not self.plot_cfg.get("learning_curves", True):
            return
        name, est, x_data, y_data = self._align_plot_args(model_name, estimator, X, y, *args, **kwargs)

        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            est, x_data, y_data, cv=3, scoring=self.champion_metric, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
        ax.plot(train_sizes, val_mean, "o-", color="g", label="Cross-validation score")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")
        ax.set(
            xlabel="Training examples",
            ylabel=self.champion_metric.upper(),
            title=f"Learning Curves ({self.champion_metric}) — {name}",
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"learning_curves_{name}.png")

    def plot_threshold_analysis(
        self,
        model_name: str,
        estimator: BaseEstimator = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot Precision, Recall, and F1 score across decision thresholds."""
        if not self.plot_cfg.get("threshold_analysis", True):
            return
        name, est, x_data, y_data = self._align_plot_args(model_name, estimator, X, y, *args, **kwargs)

        y_proba = self._get_probas(est, x_data)
        thresholds = np.linspace(0.0, 1.0, 100)
        precisions = []
        recalls = []
        f1s = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            precisions.append(precision_score(y_data, y_pred, zero_division=0))
            recalls.append(recall_score(y_data, y_pred, zero_division=0))
            f1s.append(f1_score(y_data, y_pred, zero_division=0))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(thresholds, precisions, label="Precision", color="blue", lw=2)
        ax.plot(thresholds, recalls, label="Recall", color="green", lw=2)
        ax.plot(thresholds, f1s, label="F1-Score", color="red", lw=2)
        ax.set(
            xlabel="Decision Threshold",
            ylabel="Score",
            title=f"Threshold Analysis — {name}",
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"threshold_analysis_{name}.png")

    def plot_calibration_curve(
        self,
        model_name: str,
        estimator: BaseEstimator = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Plot reliability diagram (calibration curve)."""
        if not self.plot_cfg.get("calibration_curve", True):
            return
        name, est, x_data, y_data = self._align_plot_args(model_name, estimator, X, y, *args, **kwargs)

        from sklearn.calibration import calibration_curve
        y_proba = self._get_probas(est, x_data)
        prob_true, prob_pred = calibration_curve(y_data, y_proba, n_bins=10)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, "s-", label=f"{name}", color="blue", lw=2)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.set(
            xlabel="Mean predicted probability",
            ylabel="Fraction of positives",
            title=f"Calibration Curve — {name}",
        )
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        self._save_fig(fig, f"calibration_curve_{name}.png")

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_probas(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Extract positive-class probabilities safely."""
        # Unwrap imblearn pipeline if needed
        model = estimator
        if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
            model = estimator.named_steps["model"]
        elif hasattr(estimator, "steps"):
            model = estimator.steps[-1][1]

        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # Normalise to [0,1]
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            raise AttributeError(
                f"Estimator {type(model).__name__} has neither "
                "`predict_proba` nor `decision_function`."
            )

    def _save_fig(self, fig: plt.Figure, filename: str) -> None:
        """Save a matplotlib figure and close it."""
        fp = self.figures_dir / filename
        fig.savefig(fp, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved → %s", fp)

    def __repr__(self) -> str:
        return (
            f"ModelEvaluator("
            f"champion_metric={self.champion_metric}, "
            f"threshold_metric={self.threshold_metric})"
        )
