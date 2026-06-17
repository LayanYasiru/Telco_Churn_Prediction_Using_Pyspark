# Telco Customer Churn Prediction

> **MLOps Pipeline · Python 3.10+ · scikit-learn · XGBoost · CatBoost**
> Version: 1.0.0 | Status: Production-Ready

---

## Overview

This project implements an end-to-end machine learning pipeline for predicting
customer churn on the IBM Telco Customer dataset. The system is designed with
production-grade engineering practices: a centralised `config.yaml` drives all
parameters, every module is fully type-annotated and tested, and three
distinct orchestration pipelines cover the full MLOps lifecycle — from raw data
ingestion through real-time streaming inference.

Key features:
- **Zero hardcoded values** — all parameters flow from `config.yaml`
- **No data leakage** — encoders/scalers are fit on train and applied to val/test
- **Champion model election** — configurable metric selects the best model
- **Streaming simulation** — micro-batch inference with JSONL audit trail
- **Pluggable models** — add new model families via `ModelFactory` in one line

---

## Directory Structure

```
Telco/
├── config.yaml                          # Central configuration (all params)
├── requirements.txt                     # Pinned Python dependencies
├── README.md                            # This file
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv              # IBM Telco raw dataset
│   ├── processed/                       # Intermediate processed data
│   └── splits/                          # train.csv / val.csv / test.csv
│
├── models/
│   └── champion/
│       ├── champion_model.pkl           # Serialised champion estimator
│       └── champion_metadata.json       # Metrics, params, threshold, timestamp
│
├── logs/
│   ├── pipeline.log                     # Unified rotating log file
│   └── streaming_predictions.jsonl     # JSONL audit trail from streaming
│
├── reports/
│   └── figures/                         # ROC, PR, confusion matrix, FI plots
│
├── src/
│   ├── __init__.py
│   ├── config_loader.py                 # Singleton ConfigLoader + setup_logging
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── data_ingestor.py             # CSV ingestion + schema validation
│   │   ├── missing_value_handler.py     # Median / mean / mode / KNN imputation
│   │   ├── outlier_handler.py           # IQR / Z-score capping or removal
│   │   ├── feature_engineer.py          # Domain feature derivation (10+ features)
│   │   ├── feature_encoder.py           # OHE / label / target encoding
│   │   ├── feature_scaler.py            # Standard / MinMax / Robust scaling
│   │   ├── feature_binner.py            # Quantile / uniform discretisation
│   │   └── data_splitter.py             # Stratified train / val / test split
│   │
│   ├── model_development/
│   │   ├── __init__.py
│   │   ├── model_factory.py             # Creates LR / DT / RF / XGB / CatBoost
│   │   ├── model_trainer.py             # CV + RandomSearch + SMOTE resampling
│   │   ├── model_evaluator.py           # Metrics, threshold optimisation, plots
│   │   └── inference_pipeline.py        # Serving: transform + predict + score
│   │
│   └── orchestration/
│       ├── __init__.py
│       ├── data_pipeline.py             # Full data processing orchestrator
│       ├── training_pipeline.py         # Full model training orchestrator
│       └── streaming_inference_pipeline.py  # Real-time micro-batch inference
│
└── tests/
    ├── conftest.py
    ├── test_data_processing/
    └── test_model_development/
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the data pipeline

Processes raw CSV → imputation → outlier handling → feature engineering →
encode → scale → bin → stratified split → saves `data/splits/{train,val,test}.csv`.

```bash
python -m src.orchestration.data_pipeline --config config.yaml
```

Optional: override the raw data path:

```bash
python -m src.orchestration.data_pipeline \
    --config config.yaml \
    --raw-path data/raw/telco_churn.csv
```

### 3. Run the training pipeline

Trains all configured models, elects the champion, evaluates on the test set,
saves `models/champion/champion_model.pkl` and `champion_metadata.json`.

```bash
python -m src.orchestration.training_pipeline --config config.yaml
```

### 4. Run the streaming inference pipeline

Simulates real-time scoring on a CSV of customer records.  Predictions are
written to `logs/streaming_predictions.jsonl`.

```bash
python -m src.orchestration.streaming_inference_pipeline \
    --input data/raw/telco_churn.csv \
    --config config.yaml
```

Optional overrides:

```bash
python -m src.orchestration.streaming_inference_pipeline \
    --input data/raw/telco_churn.csv \
    --config config.yaml \
    --batch-size 20 \
    --delay 0.2
```

---

## Configuration Guide (`config.yaml`)

All pipeline behaviour is controlled from `config.yaml`.  Below is a summary
of the major sections:

| Section | Key Parameters | Description |
|---|---|---|
| `project` | `random_state`, `log_level`, `log_file` | Global project metadata and logging |
| `data` | `raw_path`, `splits_path`, `target_col`, `customer_id_col` | File paths and column names |
| `data.schema` | `numerical_cols`, `categorical_cols`, `binary_yes_no_cols` | Column type declarations |
| `preprocessing.missing_values` | `strategy`, `knn_neighbors` | Imputation method (`median`, `mean`, `mode`, `knn`) |
| `preprocessing.outlier` | `method`, `iqr_factor`, `action` | Outlier detection and handling |
| `preprocessing.encoding` | `method`, `drop_first` | Categorical encoding strategy |
| `preprocessing.scaling` | `method`, `columns_to_scale` | Feature scaling strategy |
| `preprocessing.binning` | `enabled`, `columns` | Ordinal discretisation config |
| `feature_engineering` | `tenure_bins`, `service_cols`, `derived_cols` | Feature derivation parameters |
| `split` | `test_size`, `val_size`, `stratify`, `save_format` | Stratified split ratios |
| `model` | `models_to_train`, `champion_metric`, `cv_folds` | Which models to train, election metric |
| `model.hyperparams` | per-model grids | Hyper-parameter search spaces |
| `evaluation` | `threshold_metric`, `threshold_range`, `cost_matrix` | Threshold optimisation and business costs |
| `evaluation.plots` | `roc_curve`, `pr_curve`, `confusion_matrix`, etc. | Toggle individual plot generation |
| `reporting` | `figures_dir`, `high_risk_threshold`, `medium_risk_threshold` | Plot output dir and risk tiers |
| `streaming` | `delay_seconds`, `batch_size`, `output_file` | Streaming inference parameters |

---

## Module Descriptions

### `src/config_loader.py`
Thread-safe singleton that reads `config.yaml` once.  All modules access
parameters via `cfg.get("section.key")` dot-path notation.  `setup_logging(cfg)`
configures the root logger (file + console handlers).

### `src/data_processing/`
| Module | Responsibility |
|---|---|
| `data_ingestor.py` | Reads raw CSV, validates schema, coerces types |
| `missing_value_handler.py` | Fills nulls via median / mean / mode / KNN |
| `outlier_handler.py` | IQR or Z-score detection, cap or remove action |
| `feature_engineer.py` | Derives 10+ domain features (tenure segment, service score, etc.) |
| `feature_encoder.py` | One-hot / label / target encodes categorical columns |
| `feature_scaler.py` | StandardScaler / MinMaxScaler / RobustScaler |
| `feature_binner.py` | KBinsDiscretizer with quantile or uniform strategy |
| `data_splitter.py` | Stratified train / val / test split with configurable ratios |

### `src/model_development/`
| Module | Responsibility |
|---|---|
| `model_factory.py` | Instantiates LR, DT, RF, XGBoost, CatBoost by name |
| `model_trainer.py` | Fits models with RandomSearch CV + SMOTE class-balance |
| `model_evaluator.py` | Computes AUC-ROC, PR-AUC, F1, Recall, Precision; optimises threshold |
| `inference_pipeline.py` | Applies fitted transforms then scores with champion model |

### `src/orchestration/`
| Module | Responsibility |
|---|---|
| `data_pipeline.py` | Full 9-stage data processing orchestrator; also exposes `run_transform_only()` for serving |
| `training_pipeline.py` | Multi-model training, champion election, test eval, plot generation, artefact persistence |
| `streaming_inference_pipeline.py` | Micro-batch real-time scoring with tqdm progress, JSONL output, risk-tier summary |

---

## Testing

Run the full test suite with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Run only data processing tests:

```bash
pytest tests/test_data_processing/ -v
```

Run only model development tests:

```bash
pytest tests/test_model_development/ -v
```

Run a single test file:

```bash
pytest tests/test_data_processing/test_feature_engineer.py -v
```

Generate an HTML coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in a browser
```

---

## Logging

All pipelines write structured logs to `logs/pipeline.log` (configured in
`config.yaml` under `project.log_file`).  The log level, format, and file path
are all configurable.  The streaming pipeline additionally produces a
per-prediction JSONL audit trail at `logs/streaming_predictions.jsonl`.

---

## Author & Version

| Field | Value |
|---|---|
| **Project** | Telco Customer Churn Prediction |
| **Version** | 1.0.0 |
| **Python** | 3.10+ |
| **Framework** | scikit-learn + XGBoost + CatBoost |
| **Paradigm** | MLOps — config-driven, no-leakage, reproducible |
