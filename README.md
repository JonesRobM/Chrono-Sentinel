# ThreatSimTS: Synthetic Threat/Event Time-Series ML Pipeline

This repository provides a framework for developing and evaluating machine learning models on time-series data, with a focus on synthetic threat/event sequences. It also includes tools for benchmarking on the Numenta Anomaly Benchmark (NAB) dataset.

## Project Goal

The primary goal of this project is to provide a safe and reproducible environment for modeling defense and security-related event sequences without the need for sensitive or classified data. It aims to demonstrate end-to-end machine learning pipeline skills, from data generation to model evaluation.

## Features

*   **Synthetic Data Generation**: Scripts to generate synthetic threat sequences.
*   **NAB Benchmark**: Tools to download and preprocess the Numenta Anomaly Benchmark (NAB) dataset.
*   **Time-Series Feature Engineering**: Routines for extracting features from time-series data.
*   **Transformer-Based Models**: Implementation of transformer-based models for time-series classification.
*   **Uncertainty-Aware Forecasting**: Methods for uncertainty-aware forecasting.

## Getting Started

### Prerequisites

*   Python 3.6+
*   Git

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/ThreatSimTS.git
    cd ThreatSimTS
    ```

2.  **Install dependencies:**

    This project uses the Numenta Anomaly Benchmark (NAB) library, which is not available on PyPI. Therefore, it needs to be installed from the source.

    ```bash
    # Clone the NAB repository
    git clone https://github.com/numenta/NAB.git NAB_temp

    # Install NAB
    pip install ./NAB_temp

    # Install other dependencies
    pip install -r requirements.txt
    ```

3.  **Install the `threatsim` package:**

    ```bash
    pip install -e .
    ```

### Usage

The project is organized into notebooks for each step of the pipeline.

1.  **Generate Data**: `notebooks/01_generate_data.ipynb`
2.  **Feature Engineering**: `notebooks/02_feature_engineering.ipynb`
3.  **Model Training**: `notebooks/03_model_training.ipynb`

To run the notebooks, you will need to have Jupyter Notebook or JupyterLab installed.

```bash
pip install jupyter
jupyter notebook
```

## Project Structure

```
ThreatSimTS/
├─ README.md
├─ data/
│  ├─ synthetic/           # Scripts to generate synthetic threat sequences
│  └─ nab/                 # Scripts to download and preprocess NAB data
├─ notebooks/
│  ├─ 01_generate_data.ipynb
│  ├─ 02_feature_engineering.ipynb
│  └─ 03_model_training.ipynb
├─ threatsim/
│  ├─ __init__.py
│  ├─ data.py               # Synthetic/NAB data loaders
│  ├─ features.py           # Feature extraction routines
│  ├─ models.py             # Transformer / VAE / uncertainty-aware models
│  └─ utils.py              # Helper functions
├─ scripts/
│  ├─ train.py              # Full pipeline runner
│  └─ evaluate.py           # Metrics & uncertainty calibration
├─ requirements.txt
├─ setup.py
```
