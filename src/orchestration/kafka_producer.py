"""
src/orchestration/kafka_producer.py
───────────────────────────────────
Kafka Producer script for streaming Telco Churn customer records.
Loads raw data, skips the first 5000 rows, and publishes 10 rows every 5 seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

from src.config_loader import ConfigLoader, setup_logging

logger = logging.getLogger(__name__)


class TelcoKafkaProducer:
    """Publishes customer records from raw CSV to Kafka topic in micro-batches."""

    def __init__(self, cfg: ConfigLoader) -> None:
        """Initialise producer using settings from config."""
        self.cfg = cfg
        
        # Kafka config
        self.bootstrap_servers = cfg.get("kafka.bootstrap_servers", "localhost:9092")
        self.topic = cfg.get("kafka.topic", "telco-churn-topic")
        
        # Streaming params
        self.delay = float(cfg.get("kafka.producer.delay_seconds", 5.0))
        self.batch_size = int(cfg.get("kafka.producer.batch_size", 10))
        self.skip_rows = int(cfg.get("kafka.producer.skip_rows", 5000))
        
        self.raw_path = Path(cfg.get("data.raw_path", "data/raw/telco_churn.csv"))
        
        logger.info(
            "TelcoKafkaProducer init | servers=%s | topic=%s | delay=%.1fs | batch=%d",
            self.bootstrap_servers,
            self.topic,
            self.delay,
            self.batch_size,
        )

        # Lazy-initialised producer client
        self._producer: KafkaProducer | None = None

    def connect(self) -> None:
        """Create the underlying KafkaProducer client."""
        logger.info("Connecting to Kafka brokers at: %s", self.bootstrap_servers)
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
                api_version=(2, 5, 0),
            )
            logger.info("KafkaProducer connection established.")
        except Exception as exc:
            logger.error("Failed to connect to Kafka broker: %s", exc)
            raise RuntimeError(f"Could not connect to Kafka: {exc}") from exc

    def start_streaming(self) -> None:
        """Stream customer records starting from the designated skip index."""
        if not self._producer:
            self.connect()

        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data CSV not found: {self.raw_path}")

        logger.info("Reading raw data from: %s", self.raw_path)
        df = pd.read_csv(self.raw_path)
        total_rows = len(df)
        
        if total_rows <= self.skip_rows:
            logger.warning(
                "Total rows (%d) is less than or equal to skip_rows (%d). Nothing to stream.",
                total_rows,
                self.skip_rows,
            )
            return

        # Slice data from the offset index onwards
        stream_df = df.iloc[self.skip_rows:].copy()
        n_stream = len(stream_df)
        logger.info(
            "Streaming %d records (out of %d total) starting from index %d.",
            n_stream,
            total_rows,
            self.skip_rows,
        )

        batch_index = 0
        
        # Process streaming data in chunks
        for start_idx in range(0, n_stream, self.batch_size):
            batch_index += 1
            end_idx = min(start_idx + self.batch_size, n_stream)
            batch = stream_df.iloc[start_idx:end_idx]

            logger.info(
                "Publishing Batch %03d | rows %d-%d of %d",
                batch_index,
                self.skip_rows + start_idx,
                self.skip_rows + end_idx - 1,
                total_rows,
            )

            # Send each row in the batch as JSON
            for _, row in batch.iterrows():
                record = row.to_dict()
                # Ensure customer ID exists, make sure formatting is clean
                try:
                    self._producer.send(self.topic, value=record)
                except Exception as e:
                    logger.error("Error sending record %s: %s", record.get("customerID"), e)

            # Flush batch to broker
            self._producer.flush()
            logger.info("Batch %03d flushed successfully.", batch_index)

            # Wait before publishing next batch
            if end_idx < n_stream:
                logger.debug("Sleeping for %.1f seconds...", self.delay)
                time.sleep(self.delay)

        logger.info("Streaming simulation complete. All records sent.")

    def close(self) -> None:
        """Close Kafka producer connection."""
        if self._producer:
            logger.info("Closing KafkaProducer connection...")
            self._producer.close()
            logger.info("KafkaProducer connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco Customer Churn Kafka Producer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = ConfigLoader.get_instance(args.config)
    setup_logging(cfg)

    producer = TelcoKafkaProducer(cfg)
    try:
        producer.start_streaming()
    except KeyboardInterrupt:
        logger.info("Streaming producer stopped by user.")
    finally:
        producer.close()
