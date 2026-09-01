// The ten statistical window features, ported from threatsim/features.js's
// Python original (threatsim/features.py).
//
// The model consumes two views of a window: the z-scored sequence, which
// carries shape, and these features, which carry the level and scale that
// z-scoring throws away. Both have to be reproduced exactly or the scores
// drift from the Python implementations.
//
// Order matters. It must match get_feature_names():
//   mean, std, min, max, range, slope, skewness, kurtosis,
//   zero_crossings, autocorr_1

export const FEATURE_NAMES = [
  'mean', 'std', 'min', 'max', 'range',
  'slope', 'skewness', 'kurtosis', 'zero_crossings', 'autocorr_1',
];

/** Arithmetic mean. */
export function mean(values) {
  let total = 0;
  for (let i = 0; i < values.length; i += 1) total += values[i];
  return total / values.length;
}

/**
 * Population standard deviation (ddof=0), matching numpy's default.
 * Using the sample form here would shift every scaled feature.
 */
export function std(values, precomputedMean) {
  const mu = precomputedMean === undefined ? mean(values) : precomputedMean;
  let total = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = values[i] - mu;
    total += d * d;
  }
  return Math.sqrt(total / values.length);
}

/** Least-squares slope against an index axis 0..n-1. */
export function slope(values) {
  const n = values.length;
  const xMean = (n - 1) / 2;
  const yMean = mean(values);
  let numerator = 0;
  let denominator = 0;
  for (let i = 0; i < n; i += 1) {
    const dx = i - xMean;
    numerator += dx * (values[i] - yMean);
    denominator += dx * dx;
  }
  return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Sign changes of the mean-centred series.
 *
 * numpy's np.sign gives 0 for an exact zero, and the Python original counts
 * a change whenever consecutive signs differ, so a zero counts as two
 * changes when it sits between a positive and a negative. Reproduced here
 * rather than "cleaned up", because the model was trained on that number.
 */
export function zeroCrossings(values) {
  const mu = mean(values);
  let count = 0;
  let previous = Math.sign(values[0] - mu);
  for (let i = 1; i < values.length; i += 1) {
    const current = Math.sign(values[i] - mu);
    if (current !== previous) count += 1;
    previous = current;
  }
  return count;
}

/** Lag-1 autocorrelation, normalised by the population variance. */
export function autocorrelation(values, lag = 1) {
  const n = values.length;
  if (n <= lag) return 0;
  const mu = mean(values);
  let variance = 0;
  for (let i = 0; i < n; i += 1) {
    const d = values[i] - mu;
    variance += d * d;
  }
  variance /= n;
  if (variance === 0) return 0;

  let total = 0;
  for (let i = 0; i < n - lag; i += 1) {
    total += (values[i] - mu) * (values[i + lag] - mu);
  }
  // The Python takes the mean over the overlapping pairs, i.e. n - lag of them.
  return total / (n - lag) / variance;
}

/**
 * Extracts all ten features from one window.
 *
 * @param {ArrayLike<number>} window Raw values, unnormalised.
 * @returns {Float32Array} Features in FEATURE_NAMES order.
 */
export function extractWindowFeatures(window) {
  const values = Array.from(window, Number);
  const mu = mean(values);
  const sigma = std(values, mu);

  let minimum = Infinity;
  let maximum = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] < minimum) minimum = values[i];
    if (values[i] > maximum) maximum = values[i];
  }

  // A constant window has no shape to describe, so both higher moments are
  // defined as zero rather than dividing by a zero standard deviation.
  let skewness = 0;
  let kurtosis = 0;
  if (sigma > 0) {
    let third = 0;
    let fourth = 0;
    for (let i = 0; i < values.length; i += 1) {
      const d = values[i] - mu;
      third += d * d * d;
      fourth += d * d * d * d;
    }
    third /= values.length;
    fourth /= values.length;
    skewness = third / sigma ** 3;
    kurtosis = fourth / sigma ** 4 - 3.0;
  }

  return Float32Array.from([
    mu,
    sigma,
    minimum,
    maximum,
    maximum - minimum,
    slope(values),
    skewness,
    kurtosis,
    zeroCrossings(values),
    autocorrelation(values, 1),
  ]);
}
