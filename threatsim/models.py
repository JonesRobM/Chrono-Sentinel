"""
Neural network models for time-series anomaly detection.

This module provides a transformer-based classifier with dropout for
Monte Carlo Dropout uncertainty quantification.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies, allowing the model to understand
    the temporal ordering of sequence elements.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model embeddings.
            max_len: Maximum sequence length to pre-compute encodings for.
            dropout: Dropout probability applied after adding positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder for time-series anomaly classification.

    Architecture:
    1. Linear projection from input dimension to model dimension
    2. Sinusoidal positional encoding
    3. Stack of transformer encoder layers
    4. Mean pooling across sequence
    5. Classification head with dropout

    Dropout is placed throughout to enable Monte Carlo Dropout at inference
    for uncertainty quantification.
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        max_seq_len: int = 100,
    ):
        """
        Args:
            input_dim: Dimension of input features at each timestep.
            d_model: Dimension of transformer embeddings.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Dimension of feedforward network in encoder.
            dropout: Dropout probability (used throughout for MC Dropout).
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super().__init__()

        self.d_model = d_model
        self.dropout_p = dropout

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len) or
               (batch_size, seq_len, input_dim).

        Returns:
            Anomaly probability tensor of shape (batch_size,).
        """
        # Handle 2D input (batch_size, seq_len) -> (batch_size, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Project to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Mean pooling across sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        x = self.classifier(x)  # (batch, 1)

        # Squeeze and apply sigmoid for probability
        return torch.sigmoid(x.squeeze(-1))

    def enable_mc_dropout(self):
        """
        Enables dropout for Monte Carlo Dropout inference.

        Call this before running multiple forward passes for uncertainty
        estimation. Note: model.train() achieves the same effect but also
        affects batch normalisation (not used here).
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_mc_dropout(self):
        """
        Disables dropout for standard inference.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.eval()


def mc_dropout_predict(
    model: TimeSeriesTransformer,
    x: torch.Tensor,
    n_samples: int = 30,
) -> tuple:
    """
    Performs Monte Carlo Dropout prediction for uncertainty quantification.

    Runs multiple forward passes with dropout enabled to obtain a distribution
    of predictions. The mean gives the point estimate, and the standard
    deviation provides an uncertainty measure.

    Args:
        model: Trained TimeSeriesTransformer model.
        x: Input tensor of shape (batch_size, seq_len) or
           (batch_size, seq_len, input_dim).
        n_samples: Number of stochastic forward passes.

    Returns:
        Tuple of (mean_predictions, std_predictions) where each has
        shape (batch_size,).
    """
    model.enable_mc_dropout()

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)  # (n_samples, batch_size)

    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    model.disable_mc_dropout()

    return mean, std


def create_model(
    window_size: int = 50,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> TimeSeriesTransformer:
    """
    Factory function to create a TimeSeriesTransformer with sensible defaults.

    Args:
        window_size: Length of input sequences.
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        dropout: Dropout probability.

    Returns:
        Configured TimeSeriesTransformer model.
    """
    return TimeSeriesTransformer(
        input_dim=1,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=d_model * 2,
        dropout=dropout,
        max_seq_len=window_size + 10,  # Small buffer
    )
