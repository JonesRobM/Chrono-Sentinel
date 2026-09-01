"""
Neural network models for time-series anomaly detection.

This module provides a transformer-based classifier with dropout for
Monte Carlo Dropout uncertainty quantification.

The model consumes two views of each window: the per-window z-scored sequence,
which carries shape, and a scaled statistical feature vector, which carries
the level and scale information that z-scoring necessarily discards. On NAB, a
logistic regression on the feature vector alone reaches a materially higher
AUC than one on the z-scored sequence alone, so a sequence-only model is
working with the weaker of the two views.

The forward pass returns **logits**, not probabilities, so training can use
BCEWithLogitsLoss. Use predict_proba where a probability is wanted.
"""

import math

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
    5. Concatenation with an encoded statistical feature vector
    6. Classification head with dropout, emitting a logit

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
        feature_dim: int = 0,
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
            feature_dim: Length of the statistical feature vector. 0 disables
                the feature branch, leaving a sequence-only model.
        """
        super().__init__()

        self.d_model = d_model
        self.dropout_p = dropout
        self.feature_dim = feature_dim

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

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

        # Feature branch: a small MLP so the statistical vector is embedded on
        # a comparable scale to the pooled sequence representation.
        head_dim = d_model
        if feature_dim > 0:
            feature_hidden = max(16, d_model // 2)
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, feature_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_dim = d_model + feature_hidden
        else:
            self.feature_encoder = None

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len) or
               (batch_size, seq_len, input_dim).
            features: Optional statistical features of shape
                (batch_size, feature_dim). Required when feature_dim > 0.

        Returns:
            Logit tensor of shape (batch_size,). Apply a sigmoid for a
            probability, or use predict_proba.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # (batch, d_model)

        if self.feature_encoder is not None:
            if features is None:
                raise ValueError(
                    f"Model was built with feature_dim={self.feature_dim} but "
                    "forward() was called without a features tensor."
                )
            x = torch.cat([x, self.feature_encoder(features)], dim=1)

        return self.classifier(x).squeeze(-1)

    def predict_proba(
        self, x: torch.Tensor, features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Returns anomaly probabilities rather than logits.

        Args:
            x: Input tensor, as for forward.
            features: Optional statistical features, as for forward.

        Returns:
            Probability tensor of shape (batch_size,).
        """
        return torch.sigmoid(self.forward(x, features))

    def enable_mc_dropout(self) -> None:
        """
        Enables dropout for Monte Carlo Dropout inference.

        Flipping the nn.Dropout modules is not sufficient.
        nn.TransformerEncoderLayer takes a fused inference path while it is in
        eval mode, and that kernel never consults its dropout submodules. An
        earlier version of this method set only nn.Dropout to train mode, so
        eight of the twelve dropout sites were switched on and then silently
        ignored: the encoder contributed exactly zero variance and the
        uncertainty came only from the positional encoding, the feature branch
        and the classifier head. The encoder layers and their attention modules
        therefore have to be put in train mode as well.

        (Measured effect on this model: mean sigma 0.10714 -> 0.10776. Small,
        because mean-pooling over 50 timesteps averages most of the encoder
        noise away, but the sampling was not doing what it claimed.)
        """
        self._set_stochastic_modules(train=True)

    def disable_mc_dropout(self) -> None:
        """Disables dropout for standard deterministic inference."""
        self._set_stochastic_modules(train=False)

    def _set_stochastic_modules(self, train: bool) -> None:
        """Puts every module that samples dropout into train or eval mode."""
        for module in self.modules():
            if isinstance(
                module,
                (nn.Dropout, nn.TransformerEncoderLayer, nn.MultiheadAttention),
            ):
                module.train(train)


def mc_dropout_predict(
    model: TimeSeriesTransformer,
    x: torch.Tensor,
    features: torch.Tensor | None = None,
    n_samples: int = 30,
    batched: bool = True,
    manage_mode: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs Monte Carlo Dropout prediction for uncertainty quantification.

    Runs multiple stochastic forward passes to obtain a distribution of
    predictions. The mean is the point estimate; the standard deviation is an
    epistemic uncertainty measure.

    With batched=True the n_samples passes are folded into a single forward
    pass over a batch of replicated inputs. Dropout masks are sampled
    elementwise per row, so the replicas remain independent draws and the
    result is distributionally identical to the sequential loop, but it costs
    one kernel launch sequence instead of n_samples of them.

    Args:
        model: Trained TimeSeriesTransformer model.
        x: Input tensor of shape (batch_size, seq_len) or
           (batch_size, seq_len, input_dim).
        features: Optional statistical features of shape (batch_size, feature_dim).
        n_samples: Number of stochastic forward passes.
        batched: Fold the passes into one batch. Set False for the sequential
            reference implementation.
        manage_mode: Toggle the dropout modules into train mode on entry and
            back on exit. Set False when the caller has already enabled dropout
            and guarantees it stays enabled. Toggling mutates state shared by
            every thread using the model, so a concurrent server that leaves
            this True must serialise its forward passes; one that enables
            dropout once at load can set it False and run passes in parallel.

    Returns:
        Tuple of (mean_probabilities, std_probabilities), each of shape
        (batch_size,).
    """
    was_training = model.training
    if manage_mode:
        model.enable_mc_dropout()

    batch_size = x.shape[0]

    try:
        with torch.no_grad():
            if batched:
                repeated_x = x.repeat_interleave(n_samples, dim=0)
                repeated_features = (
                    features.repeat_interleave(n_samples, dim=0)
                    if features is not None
                    else None
                )
                logits = model(repeated_x, repeated_features)
                # (batch * n_samples,) -> (batch, n_samples)
                predictions = torch.sigmoid(logits).view(batch_size, n_samples)
                mean = predictions.mean(dim=1)
                std = predictions.std(dim=1)
            else:
                samples = [torch.sigmoid(model(x, features)) for _ in range(n_samples)]
                stacked = torch.stack(samples, dim=0)  # (n_samples, batch)
                mean = stacked.mean(dim=0)
                std = stacked.std(dim=0)
    finally:
        if manage_mode and not was_training:
            model.disable_mc_dropout()

    return mean, std


def create_model(
    window_size: int = 50,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    feature_dim: int = 0,
) -> TimeSeriesTransformer:
    """
    Factory function to create a TimeSeriesTransformer with sensible defaults.

    Args:
        window_size: Length of input sequences.
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of encoder layers.
        dropout: Dropout probability.
        feature_dim: Length of the statistical feature vector, or 0 to build a
            sequence-only model.

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
        feature_dim=feature_dim,
    )
