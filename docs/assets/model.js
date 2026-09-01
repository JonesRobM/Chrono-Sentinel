// The model's forward pass, in the browser.
//
// This is the third implementation of one function: PyTorch for training,
// numpy for the container, and this for the static demo. That is a hazard,
// so tests/web/parity.test.mjs checks it against golden vectors generated
// from Python by scripts/export_web_model.py.
//
// The three details that are easy to get wrong when reproducing
// nn.TransformerEncoderLayer:
//
//   * in_proj_weight packs Q, K and V into one (3d, d) matrix.
//   * norm_first=false means post-norm: x = norm1(x + sa(x)), then
//     x = norm2(x + ff(x)).
//   * there are twelve dropout sites, including one on the attention weights
//     inside the attention block. Only MC Dropout uses them; the
//     deterministic path skips all twelve.
//
// Everything is plain Float32Array and flat indexing. No dependencies.

import { extractWindowFeatures } from './features.js';

/** Row-major matmul: (rows x inner) @ (inner x cols)^T with bias. */
function linear(input, rows, inner, weight, bias, cols) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    const inBase = r * inner;
    const outBase = r * cols;
    for (let c = 0; c < cols; c += 1) {
      let total = bias ? bias[c] : 0;
      const wBase = c * inner;
      for (let k = 0; k < inner; k += 1) {
        total += input[inBase + k] * weight[wBase + k];
      }
      out[outBase + c] = total;
    }
  }
  return out;
}

/** LayerNorm over the last axis, biased variance, eps 1e-5 to match torch. */
function layerNorm(input, rows, width, weight, bias) {
  const out = new Float32Array(input.length);
  for (let r = 0; r < rows; r += 1) {
    const base = r * width;
    let mu = 0;
    for (let i = 0; i < width; i += 1) mu += input[base + i];
    mu /= width;
    let variance = 0;
    for (let i = 0; i < width; i += 1) {
      const d = input[base + i] - mu;
      variance += d * d;
    }
    variance /= width;
    const inv = 1 / Math.sqrt(variance + 1e-5);
    for (let i = 0; i < width; i += 1) {
      out[base + i] = (input[base + i] - mu) * inv * weight[i] + bias[i];
    }
  }
  return out;
}

/** In-place ReLU. */
function reluInPlace(values) {
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] < 0) values[i] = 0;
  }
  return values;
}

/** Numerically stable logistic function. */
export function sigmoid(x) {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * Inverted dropout. Zeroes with probability p and scales survivors by
 * 1/(1-p), matching torch.nn.Dropout in training mode.
 */
function dropoutInPlace(values, p, random) {
  if (p <= 0) return values;
  const scale = 1 / (1 - p);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = random() >= p ? values[i] * scale : 0;
  }
  return values;
}

/**
 * A small deterministic PRNG (mulberry32).
 *
 * Math.random cannot be seeded, and the demo needs reproducible Monte Carlo
 * draws so the same window gives the same interval twice in a row.
 */
export function makeRandom(seed) {
  let state = seed >>> 0;
  return function next() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class BrowserModel {
  /**
   * @param {object} metadata Parsed model.json.
   * @param {ArrayBuffer} buffer Contents of model.bin.
   */
  constructor(metadata, buffer) {
    this.meta = metadata;
    this.windowSize = metadata.windowSize;
    this.dModel = metadata.dModel;
    this.numHeads = metadata.numHeads;
    this.numLayers = metadata.numLayers;
    this.dropout = metadata.dropout;
    this.featureDim = metadata.featureDim;
    this.modelVersion = metadata.modelVersion;

    const all = new Float32Array(buffer);
    this.weights = {};
    for (const t of metadata.tensors) {
      this.weights[t.name] = all.subarray(t.offset, t.offset + t.length);
    }
  }

  /** Fetches model.json and model.bin from a base URL. */
  static async load(baseUrl = '.') {
    const [metadata, buffer] = await Promise.all([
      fetch(`${baseUrl}/model.json`).then((r) => r.json()),
      fetch(`${baseUrl}/model.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new BrowserModel(metadata, buffer);
  }

  /**
   * Turns a raw window into the two model inputs.
   *
   * Identical contract to AnomalyScorer.preprocess: per-window z-scoring for
   * the sequence, and the statistical features scaled by the exported
   * FeatureScaler.
   */
  preprocess(window) {
    if (window.length !== this.windowSize) {
      throw new Error(`expected ${this.windowSize} values, got ${window.length}`);
    }
    let mu = 0;
    for (let i = 0; i < window.length; i += 1) mu += window[i];
    mu /= window.length;
    let variance = 0;
    for (let i = 0; i < window.length; i += 1) {
      const d = window[i] - mu;
      variance += d * d;
    }
    const sigma = Math.sqrt(variance / window.length);
    const divisor = sigma > 0 ? sigma : 1;

    const sequence = new Float32Array(window.length);
    for (let i = 0; i < window.length; i += 1) sequence[i] = (window[i] - mu) / divisor;

    let features = null;
    if (this.featureDim > 0) {
      const raw = extractWindowFeatures(window);
      features = new Float32Array(this.featureDim);
      const { mean: scalerMean, std: scalerStd } = this.meta.scaler;
      for (let i = 0; i < this.featureDim; i += 1) {
        features[i] = (raw[i] - scalerMean[i]) / scalerStd[i];
      }
    }
    return { sequence, features };
  }

  /** Self-attention over one sequence, matching MultiheadAttention. */
  #attention(x, seq, prefix, p, random) {
    const d = this.dModel;
    const heads = this.numHeads;
    const headDim = d / heads;
    const w = this.weights;

    const qkv = linear(
      x, seq, d, w[`${prefix}.self_attn.in_proj_weight`],
      w[`${prefix}.self_attn.in_proj_bias`], 3 * d,
    );

    const context = new Float32Array(seq * d);
    const scale = 1 / Math.sqrt(headDim);
    const scores = new Float32Array(seq);

    for (let h = 0; h < heads; h += 1) {
      const qOff = h * headDim;
      const kOff = d + h * headDim;
      const vOff = 2 * d + h * headDim;

      for (let i = 0; i < seq; i += 1) {
        let maximum = -Infinity;
        for (let j = 0; j < seq; j += 1) {
          let dot = 0;
          for (let k = 0; k < headDim; k += 1) {
            dot += qkv[i * 3 * d + qOff + k] * qkv[j * 3 * d + kOff + k];
          }
          const s = dot * scale;
          scores[j] = s;
          if (s > maximum) maximum = s;
        }
        let total = 0;
        for (let j = 0; j < seq; j += 1) {
          scores[j] = Math.exp(scores[j] - maximum);
          total += scores[j];
        }
        for (let j = 0; j < seq; j += 1) scores[j] /= total;

        // Dropout on the attention weights themselves. Easy to miss: it comes
        // from the argument TransformerEncoderLayer passes to its attention
        // module, not from anything in the layer's own body.
        if (p > 0) dropoutInPlace(scores, p, random);

        for (let k = 0; k < headDim; k += 1) {
          let acc = 0;
          for (let j = 0; j < seq; j += 1) {
            acc += scores[j] * qkv[j * 3 * d + vOff + k];
          }
          context[i * d + h * headDim + k] = acc;
        }
      }
    }

    return linear(
      context, seq, d, w[`${prefix}.self_attn.out_proj.weight`],
      w[`${prefix}.self_attn.out_proj.bias`], d,
    );
  }

  /** One post-norm TransformerEncoderLayer with ReLU. */
  #encoderLayer(x, seq, prefix, p, random) {
    const d = this.dModel;
    const w = this.weights;
    const ff = w[`${prefix}.linear1.bias`].length;

    let attended = this.#attention(x, seq, prefix, p, random);
    if (p > 0) dropoutInPlace(attended, p, random);
    const residual = new Float32Array(seq * d);
    for (let i = 0; i < residual.length; i += 1) residual[i] = x[i] + attended[i];
    let out = layerNorm(residual, seq, d, w[`${prefix}.norm1.weight`], w[`${prefix}.norm1.bias`]);

    const hidden = reluInPlace(
      linear(out, seq, d, w[`${prefix}.linear1.weight`], w[`${prefix}.linear1.bias`], ff),
    );
    if (p > 0) dropoutInPlace(hidden, p, random);
    const projected = linear(
      hidden, seq, ff, w[`${prefix}.linear2.weight`], w[`${prefix}.linear2.bias`], d,
    );
    if (p > 0) dropoutInPlace(projected, p, random);

    const residual2 = new Float32Array(seq * d);
    for (let i = 0; i < residual2.length; i += 1) residual2[i] = out[i] + projected[i];
    out = layerNorm(residual2, seq, d, w[`${prefix}.norm2.weight`], w[`${prefix}.norm2.bias`]);
    return out;
  }

  /**
   * Forward pass, returning a logit.
   *
   * @param {Float32Array} sequence Normalised window.
   * @param {Float32Array|null} features Scaled feature vector.
   * @param {function|null} random Source of randomness; null for deterministic.
   */
  logit(sequence, features, random = null) {
    const d = this.dModel;
    const seq = sequence.length;
    const w = this.weights;
    const p = random ? this.dropout : 0;

    let x = linear(
      sequence, seq, 1, w['input_projection.weight'], w['input_projection.bias'], d,
    );

    const pe = w['pos_encoder.pe'];
    for (let i = 0; i < seq; i += 1) {
      for (let k = 0; k < d; k += 1) x[i * d + k] += pe[i * d + k];
    }
    if (p > 0) dropoutInPlace(x, p, random);

    for (let layer = 0; layer < this.numLayers; layer += 1) {
      x = this.#encoderLayer(x, seq, `transformer_encoder.layers.${layer}`, p, random);
    }

    // Mean pool across the sequence.
    const pooled = new Float32Array(d);
    for (let i = 0; i < seq; i += 1) {
      for (let k = 0; k < d; k += 1) pooled[k] += x[i * d + k];
    }
    for (let k = 0; k < d; k += 1) pooled[k] /= seq;

    let head = pooled;
    if (this.featureDim > 0) {
      const hidden = w['feature_encoder.0.bias'].length;
      const encoded = reluInPlace(
        linear(features, 1, this.featureDim, w['feature_encoder.0.weight'],
          w['feature_encoder.0.bias'], hidden),
      );
      if (p > 0) dropoutInPlace(encoded, p, random);
      head = new Float32Array(d + hidden);
      head.set(pooled, 0);
      head.set(encoded, d);
    }

    // classifier: Dropout, Linear, ReLU, Dropout, Linear
    const headCopy = Float32Array.from(head);
    if (p > 0) dropoutInPlace(headCopy, p, random);
    const mid = w['classifier.1.bias'].length;
    const h = reluInPlace(
      linear(headCopy, 1, headCopy.length, w['classifier.1.weight'],
        w['classifier.1.bias'], mid),
    );
    if (p > 0) dropoutInPlace(h, p, random);
    const out = linear(h, 1, mid, w['classifier.4.weight'], w['classifier.4.bias'], 1);
    return out[0];
  }

  /** Deterministic probability, dropout off. */
  predictProba(window) {
    const { sequence, features } = this.preprocess(window);
    return sigmoid(this.logit(sequence, features, null));
  }

  /**
   * Monte Carlo Dropout prediction.
   *
   * @returns {{score:number, std:number, lower:number, upper:number, samples:number[]}}
   */
  score(window, nSamples = 30, seed = 42) {
    const { sequence, features } = this.preprocess(window);
    const random = makeRandom(seed);
    const samples = new Array(nSamples);
    for (let i = 0; i < nSamples; i += 1) {
      samples[i] = sigmoid(this.logit(sequence, features, random));
    }
    let total = 0;
    for (const s of samples) total += s;
    const score = total / nSamples;
    let variance = 0;
    for (const s of samples) variance += (s - score) ** 2;
    // ddof=1, matching torch.std and the numpy backend.
    const std = Math.sqrt(variance / (nSamples - 1));
    return {
      score,
      std,
      lower: Math.max(0, score - 2 * std),
      upper: Math.min(1, score + 2 * std),
      samples,
    };
  }
}
