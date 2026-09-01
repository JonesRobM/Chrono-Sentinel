"""
The model's forward pass, in numpy.

The serving container runs this instead of torch. torch is 635 MB of the
image for a model of 76k parameters, and the service only ever needs a
forward pass -- no autograd, no optimiser, no DataLoader. Training still uses
torch; `scripts/export_weights.py` bridges the two.

This is a second implementation of one function, which is a real risk: if it
drifts from `threatsim.models.TimeSeriesTransformer` the service returns
plausible wrong answers. `tests/test_forward_parity.py` is the contract. It
checks deterministic logits agree with torch to 1e-4 and that the Monte Carlo
mean and sigma agree distributionally.

Reproducing `nn.TransformerEncoderLayer` exactly needs three details that are
easy to get wrong:

  * **Packed QKV.** `in_proj_weight` is (3*d, d): rows 0:d are Q, d:2d are K,
    2d:3d are V. One matmul, then split.
  * **Post-norm.** With `norm_first=False` (the default) the layer computes
    `x = norm1(x + sa(x))` then `x = norm2(x + ff(x))`. Pre-norm would give
    different numbers from the same weights.
  * **Seven dropout sites**, not three. Positional encoding, attention
    weights *inside* the attention block, after the attention block, inside
    the feedforward, after the feedforward, the feature branch, and twice in
    the classifier. Missing the attention-weight one leaves MC Dropout
    understating its spread.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_KEY = "__config__"
LAYER_NORM_EPS = 1e-5


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """torch.nn.Linear: x @ W.T + b, with W stored as (out_features, in_features)."""
    return x @ weight.T + bias


def layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = LAYER_NORM_EPS
) -> np.ndarray:
    """torch.nn.LayerNorm over the final axis, with affine parameters."""
    mean = x.mean(axis=-1, keepdims=True)
    # Biased variance, matching torch.
    variance = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit."""
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function, computed without overflowing on large negatives."""
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def dropout(
    x: np.ndarray, p: float, rng: np.random.Generator | None, training: bool
) -> np.ndarray:
    """
    Inverted dropout, matching torch.nn.Dropout in training mode.

    Elements are zeroed with probability p and the survivors scaled by
    1/(1-p), so the expectation is unchanged and inference needs no rescaling.

    Args:
        x: Input array.
        p: Drop probability.
        rng: Source of randomness. Required when training is True.
        training: Whether dropout is active. False makes this the identity.

    Returns:
        The array with dropout applied, or unchanged.
    """
    if not training or p <= 0.0:
        return x
    # float32 masks: the draw is only compared against a scalar, so float64
    # doubles the bytes for nothing. Dropout sampling is ~44% of a prediction.
    keep = rng.random(x.shape, dtype=np.float32) >= p
    return x * keep / (1.0 - p)


def multi_head_attention(
    x: np.ndarray,
    in_proj_weight: np.ndarray,
    in_proj_bias: np.ndarray,
    out_proj_weight: np.ndarray,
    out_proj_bias: np.ndarray,
    num_heads: int,
    p: float,
    rng: np.random.Generator | None,
    training: bool,
) -> np.ndarray:
    """
    Self-attention, matching torch.nn.MultiheadAttention with batch_first=True.

    Args:
        x: Input of shape (batch, seq, d_model).
        in_proj_weight: Packed QKV projection, shape (3*d_model, d_model).
        in_proj_bias: Packed QKV bias, shape (3*d_model,).
        out_proj_weight: Output projection, shape (d_model, d_model).
        out_proj_bias: Output projection bias.
        num_heads: Number of attention heads.
        p: Dropout probability applied to the attention weights.
        rng: Source of randomness.
        training: Whether dropout is active.

    Returns:
        Array of shape (batch, seq, d_model).
    """
    batch, seq, d_model = x.shape
    head_dim = d_model // num_heads

    projected = linear(x, in_proj_weight, in_proj_bias)  # (batch, seq, 3*d_model)
    q, k, v = np.split(projected, 3, axis=-1)

    def to_heads(t: np.ndarray) -> np.ndarray:
        return t.reshape(batch, seq, num_heads, head_dim).transpose(0, 2, 1, 3)

    q, k, v = to_heads(q), to_heads(k), to_heads(v)

    # float(...) makes this a weak Python scalar; a numpy float64 scalar
    # would promote the float32 scores to float64 under NEP 50.
    scale = float(np.sqrt(head_dim))
    scores = q @ k.transpose(0, 1, 3, 2) / scale
    weights = softmax(scores, axis=-1)
    # Dropout on the attention weights themselves. TransformerEncoderLayer
    # passes its `dropout` argument to MultiheadAttention, so this site exists
    # even though nothing in the encoder layer's own code mentions it.
    weights = dropout(weights, p, rng, training)

    attended = weights @ v  # (batch, heads, seq, head_dim)
    merged = attended.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
    return linear(merged, out_proj_weight, out_proj_bias)


def encoder_layer(
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    prefix: str,
    num_heads: int,
    p: float,
    rng: np.random.Generator | None,
    training: bool,
) -> np.ndarray:
    """
    One torch.nn.TransformerEncoderLayer in post-norm form with ReLU.

    Args:
        x: Input of shape (batch, seq, d_model).
        weights: Full parameter dictionary.
        prefix: Key prefix for this layer's parameters.
        num_heads: Attention heads.
        p: Dropout probability.
        rng: Source of randomness.
        training: Whether dropout is active.

    Returns:
        Array of shape (batch, seq, d_model).
    """
    attended = multi_head_attention(
        x,
        weights[f"{prefix}.self_attn.in_proj_weight"],
        weights[f"{prefix}.self_attn.in_proj_bias"],
        weights[f"{prefix}.self_attn.out_proj.weight"],
        weights[f"{prefix}.self_attn.out_proj.bias"],
        num_heads,
        p,
        rng,
        training,
    )
    x = layer_norm(
        x + dropout(attended, p, rng, training),
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"],
    )

    hidden = relu(
        linear(
            x, weights[f"{prefix}.linear1.weight"], weights[f"{prefix}.linear1.bias"]
        )
    )
    hidden = dropout(hidden, p, rng, training)
    projected = linear(
        hidden, weights[f"{prefix}.linear2.weight"], weights[f"{prefix}.linear2.bias"]
    )
    return layer_norm(
        x + dropout(projected, p, rng, training),
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
    )


class NumpyModel:
    """
    A trained TimeSeriesTransformer, evaluated without torch.

    Load with `NumpyModel.from_npz`, which reads the artefact written by
    scripts/export_weights.py.
    """

    def __init__(self, weights: dict[str, np.ndarray], config: dict[str, Any]):
        """
        Args:
            weights: Parameter arrays keyed by torch state-dict name.
            config: Architecture config exported alongside them.
        """
        self.weights = weights
        self.config = config
        self.window_size = int(config["window_size"])
        self.d_model = int(config["d_model"])
        self.num_heads = int(config.get("nhead", 4))
        self.num_layers = int(config["num_layers"])
        self.dropout_p = float(config["dropout"])
        self.feature_dim = int(config.get("feature_dim", 0))
        self.model_version = str(config.get("model_version", ""))

    @classmethod
    def from_npz(cls, path: Path) -> "NumpyModel":
        """Loads the exported artefact."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python scripts/export_weights.py"
            )
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(str(data[CONFIG_KEY]))
            weights = {k: data[k] for k in data.files if k != CONFIG_KEY}
        return cls(weights, config)

    def logits(
        self,
        sequence: np.ndarray,
        features: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        training: bool = False,
    ) -> np.ndarray:
        """
        Runs the forward pass and returns logits.

        Args:
            sequence: Normalised windows, shape (batch, window_size).
            features: Scaled features, shape (batch, feature_dim), when the
                model was built with a feature branch.
            rng: Source of randomness for dropout.
            training: Whether dropout is active (Monte Carlo sampling).

        Returns:
            Logits of shape (batch,).
        """
        w = self.weights
        p = self.dropout_p

        x = np.asarray(sequence, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        x = x[:, :, None]  # (batch, seq, 1)

        x = linear(x, w["input_projection.weight"], w["input_projection.bias"])
        x = x + w["pos_encoder.pe"][: x.shape[1]]
        x = dropout(x, p, rng, training)

        for index in range(self.num_layers):
            x = encoder_layer(
                x,
                w,
                f"transformer_encoder.layers.{index}",
                self.num_heads,
                p,
                rng,
                training,
            )

        pooled = x.mean(axis=1)  # (batch, d_model)

        if self.feature_dim > 0:
            if features is None:
                raise ValueError(
                    f"model has feature_dim={self.feature_dim} but no features given"
                )
            f = np.asarray(features, dtype=np.float32)
            if f.ndim == 1:
                f = f[None, :]
            encoded = relu(
                linear(f, w["feature_encoder.0.weight"], w["feature_encoder.0.bias"])
            )
            encoded = dropout(encoded, p, rng, training)
            pooled = np.concatenate([pooled, encoded], axis=1)

        # classifier: Dropout, Linear, ReLU, Dropout, Linear
        h = dropout(pooled, p, rng, training)
        h = relu(linear(h, w["classifier.1.weight"], w["classifier.1.bias"]))
        h = dropout(h, p, rng, training)
        h = linear(h, w["classifier.4.weight"], w["classifier.4.bias"])
        return h[:, 0]

    def predict_proba(
        self, sequence: np.ndarray, features: np.ndarray | None = None
    ) -> np.ndarray:
        """Deterministic probabilities with dropout disabled."""
        return sigmoid(self.logits(sequence, features, training=False))

    def mc_dropout_predict(
        self,
        sequence: np.ndarray,
        features: np.ndarray | None = None,
        n_samples: int = 30,
        rng: np.random.Generator | None = None,
        batched: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout prediction.

        With batched=True the samples are folded into one forward pass over
        replicated inputs, so cost is sub-linear in n_samples. Dropout masks
        are drawn elementwise per row, so the replicas stay independent and
        the result is distributionally identical to the sequential loop.

        Args:
            sequence: Normalised windows, shape (batch, window_size).
            features: Scaled features, shape (batch, feature_dim).
            n_samples: Stochastic passes.
            rng: Source of randomness.
            batched: Fold the passes into one call. False runs the sequential
                reference implementation, which the benchmark compares against.

        Returns:
            Tuple of (mean probabilities, standard deviations), each shape
            (batch,).
        """
        if rng is None:
            rng = np.random.default_rng()

        x = np.asarray(sequence, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        batch = x.shape[0]

        f = None
        if features is not None:
            f = np.asarray(features, dtype=np.float32)
            if f.ndim == 1:
                f = f[None, :]

        if batched:
            repeated = np.repeat(x, n_samples, axis=0)
            repeated_features = (
                np.repeat(f, n_samples, axis=0) if f is not None else None
            )
            probs = sigmoid(
                self.logits(repeated, repeated_features, rng=rng, training=True)
            ).reshape(batch, n_samples)
        else:
            draws = [
                sigmoid(self.logits(x, f, rng=rng, training=True))
                for _ in range(n_samples)
            ]
            probs = np.stack(draws, axis=1)

        # ddof=1 to match torch.std, which is Bessel-corrected by default.
        return probs.mean(axis=1), probs.std(axis=1, ddof=1)
