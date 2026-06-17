"""
src/orchestration/streaming_inference_pipeline.py
──────────────────────────────────────────────────
Simulates a real-time / streaming inference scenario for the Telco Churn
prediction project by processing incoming customer records in micro-batches.

Workflow:
    1. Load the champion model artefact from disk.
    2. Iterate over rows of the source DataFrame.
    3. Every ``streaming.batch_size`` rows form a micro-batch.
    4. Pass the micro-batch through the :class:`~src.model_development.inference_pipeline.InferencePipeline`
       to obtain churn predictions and probability scores.
    5. Log each prediction result to the console.
    6. Append results (JSONL format) to the configured output file.
    7. Sleep ``streaming.delay_seconds`` between micro-batches to simulate
       stream arrival rate.
    8. Emit a final summary counting High / Medium / Low risk predictions.

Risk tiers (configurable via ``reporting``):
    - **High**   : churn_probability >= ``reporting.high_risk_threshold``  (default 0.70)
    - **Medium** : churn_probability >= ``reporting.medium_risk_threshold`` (default 0.40)
    - **Low**    : churn_probability <  ``reporting.medium_risk_threshold``

Usage (CLI):
    python -m src.orchestration.streaming_inference_pipeline \\
        --input data/raw/telco_churn.csv --config config.yaml

Usage (library):
    from src.orchestration.streaming_inference_pipeline import StreamingInferencePipeline
    from src.config_loader import ConfigLoader
    cfg = ConfigLoader.get_instance()
    pipeline = StreamingInferencePipeline(cfg)
    pipeline.run(source_df)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config_loader import ConfigLoader, setup_logging
from src.model_development.model_inference import InferencePipeline

logger = logging.getLogger(__name__)

# Optional tqdm import — gracefully degrade to manual progress logging
try:
    from tqdm import tqdm as _tqdm_cls
    _TQDM_AVAILABLE = True
    logger.debug("tqdm detected — progress bars enabled.")
except ImportError:  # pragma: no cover
    _TQDM_AVAILABLE = False
    logger.debug("tqdm not installed — using manual progress logging.")


class StreamingInferencePipeline:
    """Simulates real-time micro-batch inference for the Telco Churn model.

    Reads a source DataFrame row-by-row, assembles micro-batches of
    ``streaming.batch_size`` rows, passes them through the
    :class:`~src.model_development.inference_pipeline.InferencePipeline`,
    and emits structured JSONL output at a configurable rate.

    Attributes:
        cfg: The singleton :class:`~src.config_loader.ConfigLoader` instance.
        inference_pipeline: Handles data transformation + model scoring.
        batch_size: Number of rows per micro-batch.
        delay_seconds: Sleep interval between micro-batch submissions.
        log_predictions: Whether to log each prediction record.
        output_file: Path to the JSONL output file.
        high_risk_threshold: Probability threshold for HIGH risk tier.
        medium_risk_threshold: Probability threshold for MEDIUM risk tier.

    Example:
        >>> import pandas as pd
        >>> from src.config_loader import ConfigLoader
        >>> from src.orchestration.streaming_inference_pipeline import StreamingInferencePipeline
        >>> cfg = ConfigLoader.get_instance("config.yaml")
        >>> pipeline = StreamingInferencePipeline(cfg)
        >>> df = pd.read_csv("data/raw/telco_churn.csv")
        >>> pipeline.run(df)
    """

    def __init__(self, cfg: ConfigLoader) -> None:
        """Initialise the streaming pipeline and load the champion model.

        Args:
            cfg: Loaded :class:`~src.config_loader.ConfigLoader` singleton.

        Raises:
            TypeError: If *cfg* is not a :class:`~src.config_loader.ConfigLoader`.
            FileNotFoundError: If the champion model artefact is not found.
        """
        if not isinstance(cfg, ConfigLoader):
            raise TypeError(
                f"cfg must be a ConfigLoader instance, got {type(cfg).__name__!r}."
            )

        self.cfg: ConfigLoader = cfg

        # Streaming parameters from config
        self.batch_size: int = int(cfg.get("streaming.batch_size", 10))
        self.delay_seconds: float = float(cfg.get("streaming.delay_seconds", 0.5))
        self.log_predictions: bool = bool(cfg.get("streaming.log_predictions", True))
        self.output_file: Path = Path(
            cfg.get("streaming.output_file", "logs/streaming_predictions.jsonl")
        )

        # Risk tier thresholds
        self.high_risk_threshold: float = float(
            cfg.get("reporting.high_risk_threshold", 0.70)
        )
        self.medium_risk_threshold: float = float(
            cfg.get("reporting.medium_risk_threshold", 0.40)
        )

        logger.info(
            "StreamingInferencePipeline config  |  batch_size=%d  delay=%.2fs  "
            "output=%s  high_thr=%.2f  med_thr=%.2f",
            self.batch_size,
            self.delay_seconds,
            self.output_file,
            self.high_risk_threshold,
            self.medium_risk_threshold,
        )

        # Initialise inference sub-pipeline (loads model internally)
        self.inference_pipeline: InferencePipeline = InferencePipeline(cfg)

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info("StreamingInferencePipeline ready.")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, source_df: pd.DataFrame) -> None:
        """Stream-process all rows in *source_df* in micro-batches.

        Each micro-batch of ``batch_size`` rows is scored, logged, and
        appended to the JSONL output file.  A configurable sleep pause
        simulates realistic stream arrival rate.  Progress is shown via tqdm
        (if available) or periodic log messages.

        Args:
            source_df: Input DataFrame of raw customer records.  May contain
                the target column (it is dropped before scoring) and the
                customer ID column.

        Raises:
            ValueError: If *source_df* is empty.
            RuntimeError: If the InferencePipeline fails on a batch.
        """
        if source_df.empty:
            raise ValueError(
                "source_df is empty — nothing to score.  "
                "Pass a DataFrame with at least one row."
            )

        n_rows = len(source_df)
        logger.info(
            "StreamingInferencePipeline.run()  |  rows=%d  batch_size=%d  "
            "estimated_batches=%d",
            n_rows,
            self.batch_size,
            (n_rows + self.batch_size - 1) // self.batch_size,
        )

        # Risk counters
        risk_counts: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        total_records_processed: int = 0
        batch_index: int = 0
        t0 = time.perf_counter()

        # Build row index iterator (with optional tqdm progress bar)
        row_indices = range(0, n_rows, self.batch_size)
        if _TQDM_AVAILABLE:
            row_indices = _tqdm_cls(  # type: ignore[assignment]
                row_indices,
                desc="Streaming batches",
                unit="batch",
                total=(n_rows + self.batch_size - 1) // self.batch_size,
                dynamic_ncols=True,
            )

        for start_idx in row_indices:
            batch_index += 1
            end_idx = min(start_idx + self.batch_size, n_rows)
            batch_df = source_df.iloc[start_idx:end_idx].copy()

            logger.debug(
                "Batch %d  |  rows %d-%d  |  shape=%s",
                batch_index,
                start_idx,
                end_idx - 1,
                batch_df.shape,
            )

            # Score the micro-batch
            try:
                prediction_df = self._score_batch(batch_df, batch_index)
            except Exception as exc:
                logger.error(
                    "Batch %d scoring failed: %s — skipping batch.", batch_index, exc
                )
                continue

            # Write to JSONL and accumulate risk counts
            for _, row in prediction_df.iterrows():
                risk_tier = self._classify_risk(float(row.get("churn_probability", 0.0)))
                risk_counts[risk_tier] += 1
                total_records_processed += 1

                record: Dict = {
                    **row.to_dict(),
                    "risk_tier": risk_tier,
                    "batch_index": batch_index,
                }
                self._write_jsonl(record)

                if self.log_predictions:
                    logger.info(
                        "  [Batch %03d | Row %05d]  churn_prob=%.4f  "
                        "prediction=%s  risk=%s",
                        batch_index,
                        total_records_processed,
                        row.get("churn_probability", float("nan")),
                        row.get("churn_prediction", "?"),
                        risk_tier,
                    )

            # Simulate stream arrival delay
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)

        # Final summary
        elapsed = time.perf_counter() - t0
        self._log_summary(
            total_records=total_records_processed,
            risk_counts=risk_counts,
            elapsed=elapsed,
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _score_batch(self, batch_df: pd.DataFrame, batch_index: int) -> pd.DataFrame:
        """Score a single micro-batch through the inference pipeline.

        Drops the target column (if present) before calling
        :meth:`~src.model_development.inference_pipeline.InferencePipeline.predict`
        to avoid passing labels into the model.

        Args:
            batch_df: Micro-batch DataFrame containing raw feature rows.
            batch_index: 1-based batch number (used in logging).

        Returns:
            DataFrame with ``churn_prediction`` and ``churn_probability``
            columns appended.

        Raises:
            RuntimeError: Propagated from the InferencePipeline on failure.
        """
        target_col: str = self.cfg.get("data.target_col", "Churn")
        id_col: str = self.cfg.get("data.customer_id_col", "customerID")

        # Drop target if present (serving scenario — labels are unknown)
        serving_df = batch_df.drop(
            columns=[c for c in [target_col] if c in batch_df.columns],
            errors="ignore",
        )

        try:
            result_df: pd.DataFrame = self.inference_pipeline.predict(serving_df)
        except Exception as exc:
            raise RuntimeError(
                f"InferencePipeline.predict() failed on batch {batch_index}: {exc}"
            ) from exc

        # Re-attach customer ID for traceability if present
        if id_col in batch_df.columns and id_col not in result_df.columns:
            result_df.insert(0, id_col, batch_df[id_col].values)

        return result_df

    def _classify_risk(self, probability: float) -> str:
        """Map a churn probability to a named risk tier.

        Args:
            probability: Predicted churn probability in [0, 1].

        Returns:
            ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"`` based on the configured
            thresholds.
        """
        if probability >= self.high_risk_threshold:
            return "HIGH"
        if probability >= self.medium_risk_threshold:
            return "MEDIUM"
        return "LOW"

    def _write_jsonl(self, record: Dict) -> None:
        """Append a single prediction record to the JSONL output file.

        Each record is serialised as a single JSON object on one line.
        The file is opened in append mode so multiple pipeline runs accumulate.

        Args:
            record: Dictionary to serialise.  Values must be JSON-serialisable
                (converted via ``default=str`` as a fallback).
        """
        try:
            with open(self.output_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.error("Failed to write JSONL record: %s", exc)

    def _log_summary(
        self,
        total_records: int,
        risk_counts: Dict[str, int],
        elapsed: float,
    ) -> None:
        """Emit a structured end-of-run summary to the log.

        Args:
            total_records: Total number of records successfully scored.
            risk_counts: Count of predictions per risk tier.
            elapsed: Wall-clock time elapsed (seconds).
        """
        high = risk_counts.get("HIGH", 0)
        medium = risk_counts.get("MEDIUM", 0)
        low = risk_counts.get("LOW", 0)
        pct_high = 100.0 * high / max(total_records, 1)
        pct_medium = 100.0 * medium / max(total_records, 1)
        pct_low = 100.0 * low / max(total_records, 1)

        logger.info("=" * 60)
        logger.info("STREAMING INFERENCE SUMMARY")
        logger.info("=" * 60)
        logger.info("  Total records processed : %d", total_records)
        logger.info("  Elapsed time            : %.2f s", elapsed)
        logger.info(
            "  Throughput              : %.1f records/s",
            total_records / max(elapsed, 1e-6),
        )
        logger.info("  HIGH  risk (>= %.0f%%)  : %d  (%.1f%%)", self.high_risk_threshold * 100, high, pct_high)
        logger.info("  MEDIUM risk (>= %.0f%%)  : %d  (%.1f%%)", self.medium_risk_threshold * 100, medium, pct_medium)
        logger.info("  LOW   risk (< %.0f%%)   : %d  (%.1f%%)", self.medium_risk_threshold * 100, low, pct_low)
        logger.info("  Output file             : %s", self.output_file)
        logger.info("=" * 60)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"StreamingInferencePipeline("
            f"batch_size={self.batch_size}, "
            f"delay_seconds={self.delay_seconds}, "
            f"output_file={self.output_file}, "
            f"high_risk_threshold={self.high_risk_threshold}, "
            f"medium_risk_threshold={self.medium_risk_threshold}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`~argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.orchestration.streaming_inference_pipeline",
        description=(
            "Telco Churn -- Streaming Inference Pipeline\n"
            "Simulates real-time micro-batch inference using the champion model.\n"
            "Writes predictions to a JSONL file and prints a risk-tier summary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing customer records to score.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        dest="batch_size",
        help=(
            "Override streaming.batch_size from config.yaml. "
            "Number of rows per micro-batch."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        dest="delay_seconds",
        help=(
            "Override streaming.delay_seconds from config.yaml. "
            "Seconds to sleep between batches."
        ),
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    # Bootstrap config + logging
    _cfg = ConfigLoader.get_instance(_args.config)
    setup_logging(_cfg)

    logger.info(
        "Telco Churn | StreamingInferencePipeline CLI | config=%s | input=%s",
        _args.config,
        _args.input,
    )

    # Load source DataFrame
    _input_path = Path(_args.input)
    if not _input_path.exists():
        raise FileNotFoundError(f"Input file not found: {_input_path}")

    _source_df = pd.read_csv(_input_path)
    logger.info("Source file loaded  |  shape=%s  |  path=%s", _source_df.shape, _input_path)

    # Instantiate pipeline (config overrides applied before construction)
    _pipeline = StreamingInferencePipeline(_cfg)

    # Apply CLI overrides (after construction via attribute mutation)
    if _args.batch_size is not None:
        _pipeline.batch_size = _args.batch_size
        logger.info("CLI override: batch_size=%d", _args.batch_size)
    if _args.delay_seconds is not None:
        _pipeline.delay_seconds = _args.delay_seconds
        logger.info("CLI override: delay_seconds=%.2f", _args.delay_seconds)

    _pipeline.run(_source_df)
