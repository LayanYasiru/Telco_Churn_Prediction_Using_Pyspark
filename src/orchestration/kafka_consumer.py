"""
src/orchestration/kafka_consumer.py
───────────────────────────────────
Kafka Consumer script for scoring incoming Telco Churn stream records in real-time.
Features dynamic reloading of the preprocessor state and champion model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from kafka import KafkaConsumer

from src.config_loader import ConfigLoader, setup_logging
from src.model_development.model_inference import InferencePipeline

logger = logging.getLogger(__name__)


class TelcoKafkaConsumer:
    """Consumes customer records from Kafka, runs inference, and writes predictions."""

    def __init__(self, cfg: ConfigLoader, bootstrap_servers_override: str | None = None) -> None:
        """Initialise consumer settings from config."""
        self.cfg = cfg
        
        # Kafka config
        self.bootstrap_servers = bootstrap_servers_override or cfg.get("kafka.bootstrap_servers", "localhost:9092")
        self.topic = cfg.get("kafka.topic", "telco-churn-topic")
        self.group_id = cfg.get("kafka.group_id", "telco-churn-consumer-group")
        
        # Consumer settings
        self.output_file = Path(cfg.get("kafka.consumer.output_file", "logs/kafka_predictions.jsonl"))
        self.poll_timeout = float(cfg.get("kafka.consumer.poll_timeout_seconds", 1.0))
        
        # Micro-batch trigger parameters
        self.batch_size = int(cfg.get("kafka.producer.batch_size", 10))

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "TelcoKafkaConsumer init | servers=%s | topic=%s | group=%s | output=%s",
            self.bootstrap_servers,
            self.topic,
            self.group_id,
            self.output_file,
        )

        # Lazy-initialised model pipeline and Kafka consumer
        self.inference_pipeline: InferencePipeline | None = None
        self._consumer: KafkaConsumer | None = None

    def connect(self) -> None:
        """Connect to Kafka broker and instantiate model InferencePipeline."""
        logger.info("Connecting to Kafka brokers at: %s", self.bootstrap_servers)
        try:
            self._consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                api_version=(2, 5, 0),
            )
            logger.info("KafkaConsumer connection established.")
        except Exception as exc:
            logger.error("Failed to connect KafkaConsumer: %s", exc)
            raise RuntimeError(f"Could not connect consumer to Kafka: {exc}") from exc

        # Instantiate InferencePipeline using Static Factory Method (loads preprocessor & model)
        logger.info("Loading InferencePipeline and model artifacts...")
        try:
            self.inference_pipeline = InferencePipeline.create(self.cfg)
            logger.info("InferencePipeline instantiated successfully.")
        except Exception as exc:
            logger.error("Failed to load model/preprocessor in InferencePipeline: %s", exc)
            # We don't crash, we will retry loading dynamically on prediction if it is not ready
            self.inference_pipeline = InferencePipeline(self.cfg)

    def start_consuming(self) -> None:
        """Consumer poll loop with micro-batch compilation and scoring."""
        if not self._consumer:
            self.connect()

        logger.info("Starting consumption loop on topic: %s", self.topic)
        
        buffer: List[Dict] = []
        
        try:
            while True:
                # Poll for messages
                message_pack = self._consumer.poll(timeout_ms=int(self.poll_timeout * 1000))
                
                for _, messages in message_pack.items():
                    for message in messages:
                        record = message.value
                        buffer.append(record)
                        
                        # Process when buffer reaches batch_size
                        if len(buffer) >= self.batch_size:
                            self._process_batch(buffer)
                            buffer.clear()
                            
                # Periodically flush residual records in buffer even if batch size is not reached
                if buffer and len(message_pack) == 0:
                    logger.debug("Flushing residual buffer records...")
                    self._process_batch(buffer)
                    buffer.clear()
                    
        except KeyboardInterrupt:
            logger.info("Kafka consumer stopped by user request.")
        except Exception as exc:
            logger.exception("Critical error in Kafka consumer poll loop: %s", exc)
        finally:
            self.close()

    def _process_batch(self, raw_records: List[Dict]) -> None:
        """Convert records to DataFrame, score using pipeline, and write to JSONL."""
        n_records = len(raw_records)
        logger.info("Processing stream batch | n_records=%d", n_records)
        
        batch_df = pd.DataFrame(raw_records)
        
        # Clean up identifiers if model doesn't expect them
        target_col = self.cfg.get("data.target_col", "Churn")
        id_col = self.cfg.get("data.customer_id_col", "customerID")
        
        serving_df = batch_df.drop(
            columns=[c for c in [target_col] if c in batch_df.columns],
            errors="ignore",
        )

        try:
            # Inference pipeline automatically checks timestamps on disk and reloads on change!
            result_df = self.inference_pipeline.predict(serving_df)
            
            # Re-attach customerID for output logging
            if id_col in batch_df.columns and id_col not in result_df.columns:
                result_df.insert(0, id_col, batch_df[id_col].values)
                
            # Write results to output file (JSONL format)
            with open(self.output_file, "a", encoding="utf-8") as f:
                for _, row in result_df.iterrows():
                    record = row.to_dict()
                    f.write(json.dumps(record) + "\n")
                    
            logger.info("Batch processed successfully. Predictions appended to %s", self.output_file)
            
        except Exception as exc:
            logger.error("Failed to score streaming batch: %s", exc)

    def close(self) -> None:
        """Close Kafka consumer connection."""
        if self._consumer:
            logger.info("Closing KafkaConsumer connection...")
            self._consumer.close()
            logger.info("KafkaConsumer connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco Customer Churn Kafka Consumer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--bootstrap-servers", type=str, default=None, help="Override bootstrap servers")
    args = parser.parse_args()

    cfg = ConfigLoader.get_instance(args.config)
    setup_logging(cfg)

    consumer = TelcoKafkaConsumer(cfg, bootstrap_servers_override=args.bootstrap_servers)
    consumer.start_consuming()
