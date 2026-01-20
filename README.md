# Chrono-Sentinel: Time-Series Anomaly Detection with Uncertainty Quantification

This repository demonstrates applied machine learning on time-series anomaly detection, using the **Numenta Anomaly Benchmark (NAB)** dataset. The pipeline features **transformer-based classification** and **Monte Carlo Dropout uncertainty quantification**.

---

## Project Goals

1. Provide a safe, reproducible environment for modelling defence/security-relevant event sequences without sensitive data.
2. Demonstrate end-to-end ML pipeline skills: data loading, feature engineering, model training, evaluation, and uncertainty quantification.
3. Implement modern ML techniques: transformers for time-series and Bayesian uncertainty estimation via MC Dropout.
4. Benchmark performance on a recognised industry dataset: NAB.

---

## Features

* **NAB Data Loading**: Utilities to load and preprocess Numenta Anomaly Benchmark datasets
* **Sliding Window Processing**: Convert time-series to windowed sequences for classification
* **Time-Series Feature Extraction**: Statistical features (mean, std, slope, skewness, kurtosis, etc.)
* **Transformer Architecture**: Attention-based model for temporal pattern recognition
* **MC Dropout Uncertainty**: Monte Carlo Dropout for uncertainty quantification at inference
* **Calibration Analysis**: Metrics and visualisations for uncertainty quality assessment

---

## Getting Started

### Prerequisites

* Python 3.8+
* Git

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Chrono-Sentinel.git
    cd Chrono-Sentinel
    ```

2. **Clone the NAB repository (for data):**

    ```bash
    git clone https://github.com/numenta/NAB.git NAB_temp
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

### Quick Start

**Train the model:**
```bash
python scripts/train.py
```

**Evaluate with MC Dropout uncertainty:**
```bash
python scripts/evaluate.py
```

**Or explore interactively via notebooks:**
```bash
pip install jupyter
jupyter notebook notebooks/
```

---

## Usage

### Training

```bash
python scripts/train.py --epochs 50 --batch-size 32 --lr 1e-3
```

Key arguments:
- `--epochs`: Maximum training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--window-size`: Sliding window size (default: 50)
- `--d-model`: Transformer embedding dimension (default: 64)
- `--dropout`: Dropout probability for MC Dropout (default: 0.2)
- `--patience`: Early stopping patience (default: 10)

### Evaluation

```bash
python scripts/evaluate.py --model-path outputs/best_model.pt --mc-samples 30
```

Key arguments:
- `--model-path`: Path to trained model checkpoint
- `--mc-samples`: Number of MC Dropout forward passes (default: 30)
- `--threshold`: Classification threshold (default: 0.5)

### Outputs

Training and evaluation produce:
- `outputs/best_model.pt` - Trained model checkpoint
- `outputs/training_history.json` - Training metrics
- `outputs/training_history.png` - Loss curves
- `outputs/evaluation_metrics.json` - Test metrics
- `outputs/roc_curve.png` - ROC curve
- `outputs/calibration_curve.png` - Reliability diagram
- `outputs/uncertainty_histogram.png` - Uncertainty analysis
- `outputs/predictions_timeline.png` - Predictions with confidence bands

---

## Project Structure

```
Chrono-Sentinel/
├── README.md
├── requirements.txt
├── setup.py
├── threatsim/
│   ├── __init__.py
│   ├── data.py           # NAB data loading, windowing, DataLoaders
│   ├── features.py       # Statistical feature extraction
│   ├── models.py         # Transformer with MC Dropout
│   └── utils.py          # Training utilities, visualisation helpers
├── scripts/
│   ├── train.py          # Training pipeline
│   └── evaluate.py       # MC Dropout evaluation
├── notebooks/
│   ├── 01_generate_data.ipynb        # Data exploration
│   ├── 02_feature_engineering.ipynb  # Feature analysis
│   └── 03_model_training.ipynb       # Interactive training
├── NAB_temp/             # Cloned NAB repository (data source)
└── outputs/              # Training outputs (created at runtime)
```

---

## Model Architecture

The transformer architecture is designed for time-series anomaly classification:

```
Input Window (batch, seq_len)
    ↓
Linear Projection → (batch, seq_len, d_model=64)
    ↓
Sinusoidal Positional Encoding
    ↓
2× Transformer Encoder Layers (4 heads, dropout=0.2)
    ↓
Mean Pooling
    ↓
Classification Head (Linear → Dropout → Linear → Sigmoid)
    ↓
Output: Anomaly Probability
```

**MC Dropout Uncertainty**: Dropout remains active during inference. Multiple forward passes produce a distribution of predictions; the mean gives the estimate and standard deviation gives uncertainty.

---

## Key Concepts

### Monte Carlo Dropout

MC Dropout treats dropout as approximate Bayesian inference:
1. Keep dropout enabled at inference time
2. Run N forward passes (default: 30)
3. Mean of predictions → point estimate
4. Std of predictions → epistemic uncertainty

This allows the model to express "I don't know" through higher uncertainty, particularly valuable for anomaly detection where novel patterns may not match training data.

### Calibration Metrics

We evaluate uncertainty quality via:
- **Expected Calibration Error (ECE)**: Measures prediction confidence vs actual accuracy
- **Uncertainty-Error Correlation**: Higher uncertainty should correlate with errors
- **Prediction Interval Coverage**: Do confidence intervals contain true values?

---

## Licence

This project is for educational and portfolio demonstration purposes.
