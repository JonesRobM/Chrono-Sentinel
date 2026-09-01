// Interactive demo: draw a window, watch the detector rescore it.
//
// Everything runs client-side. The weights are fetched once as a 297 KB
// ArrayBuffer and the forward pass is docs/assets/model.js. Nothing is
// uploaded anywhere, which is also why there is no drift endpoint here --
// drift is a property of a deployed service watching a traffic stream, not
// of a single page.

import { extractWindowFeatures, FEATURE_NAMES } from './features.js';
import { BrowserModel } from './model.js';

// The F1-optimal threshold chosen on the validation split. Quoted so the
// meter shows where "fires" begins rather than implying 0.5 is meaningful.
const THRESHOLD = 0.925;

const el = (id) => document.getElementById(id);
const canvas = el('chart');
const ctx = canvas.getContext('2d');

let model = null;
let windowValues = null;
let pending = false;

// ---------------------------------------------------------------- presets

const PRESETS = {
  flat: (n) => Array.from({ length: n }, () => 85),
  noisy: (n) => Array.from({ length: n }, () => 85 + (Math.random() - 0.5) * 24),
  ramp: (n) => Array.from({ length: n }, (_, i) => 80 + (i / (n - 1)) * 45),
  step: (n) => Array.from({ length: n }, (_, i) => (i < n / 2 ? 85 : 20)),
  spike: (n) => Array.from({ length: n }, (_, i) => (i === Math.floor(n * 0.6) ? 190 : 85)),
  sawtooth: (n) => Array.from({ length: n }, (_, i) => 85 + Math.abs((i % 10) - 5) * 7),
};

// ------------------------------------------------------------------ chart

// Fixed value range so the y-axis does not rescale while drawing, which would
// make the signal look unchanged as you drag it.
const V_MIN = 0;
const V_MAX = 200;
const PAD = { top: 14, right: 12, bottom: 22, left: 40 };

function plotArea() {
  return {
    x: PAD.left,
    y: PAD.top,
    w: canvas.width - PAD.left - PAD.right,
    h: canvas.height - PAD.top - PAD.bottom,
  };
}

const valueToY = (v, a) => a.y + a.h * (1 - (v - V_MIN) / (V_MAX - V_MIN));
const yToValue = (y, a) => V_MIN + (1 - (y - a.y) / a.h) * (V_MAX - V_MIN);

function draw() {
  const a = plotArea();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = '#191b2a';
  ctx.fillRect(a.x, a.y, a.w, a.h);

  ctx.strokeStyle = '#2f3350';
  ctx.fillStyle = '#8d93b3';
  ctx.font = '11px ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let v = V_MIN; v <= V_MAX; v += 50) {
    const y = valueToY(v, a);
    ctx.beginPath();
    ctx.moveTo(a.x, y);
    ctx.lineTo(a.x + a.w, y);
    ctx.stroke();
    ctx.fillText(String(v), 6, y + 4);
  }

  if (!windowValues) return;
  const step = a.w / (windowValues.length - 1);

  ctx.beginPath();
  ctx.moveTo(a.x, valueToY(windowValues[0], a));
  for (let i = 1; i < windowValues.length; i += 1) {
    ctx.lineTo(a.x + i * step, valueToY(windowValues[i], a));
  }
  ctx.strokeStyle = '#89b4fa';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = '#89b4fa';
  for (let i = 0; i < windowValues.length; i += 1) {
    ctx.beginPath();
    ctx.arc(a.x + i * step, valueToY(windowValues[i], a), 2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = '#8d93b3';
  ctx.fillText(`${windowValues.length} points`, a.x + a.w - 66, a.y + a.h + 16);
}

// ---------------------------------------------------------------- drawing

let drawing = false;
let lastIndex = null;

function pointerToIndexValue(event) {
  const rect = canvas.getBoundingClientRect();
  // The canvas is CSS-scaled, so map through its intrinsic size.
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  const a = plotArea();
  const step = a.w / (windowValues.length - 1);
  const index = Math.round((x - a.x) / step);
  const value = Math.min(V_MAX, Math.max(V_MIN, yToValue(y, a)));
  return { index: Math.min(windowValues.length - 1, Math.max(0, index)), value };
}

function paint(event) {
  if (!windowValues) return;
  const { index, value } = pointerToIndexValue(event);
  if (lastIndex !== null && Math.abs(index - lastIndex) > 1) {
    // Interpolate across a fast drag so the line stays continuous.
    const from = lastIndex;
    const stepDir = index > from ? 1 : -1;
    const fromValue = windowValues[from];
    for (let i = from; i !== index; i += stepDir) {
      const t = Math.abs(i - from) / Math.abs(index - from);
      windowValues[i] = fromValue + (value - fromValue) * t;
    }
  }
  windowValues[index] = value;
  lastIndex = index;
  draw();
  scheduleScore();
}

canvas.addEventListener('pointerdown', (e) => {
  drawing = true;
  lastIndex = null;
  canvas.setPointerCapture(e.pointerId);
  paint(e);
});
canvas.addEventListener('pointermove', (e) => { if (drawing) paint(e); });
canvas.addEventListener('pointerup', () => { drawing = false; lastIndex = null; });
canvas.addEventListener('pointercancel', () => { drawing = false; lastIndex = null; });

// ---------------------------------------------------------------- scoring

function scheduleScore() {
  if (pending) return;
  pending = true;
  // Coalesce to one score per frame while dragging.
  requestAnimationFrame(() => {
    pending = false;
    score();
  });
}

function verdict(value) {
  if (value >= THRESHOLD) return { text: 'anomalous', colour: '#f38ba8' };
  if (value >= 0.5) return { text: 'elevated', colour: '#f9e2af' };
  return { text: 'normal', colour: '#a6e3a1' };
}

function score() {
  if (!model || !windowValues) return;
  const nSamples = Number(el('samples').value);
  const started = performance.now();
  const result = model.score(Float64Array.from(windowValues), nSamples, 42);
  const elapsed = performance.now() - started;

  const { text, colour } = verdict(result.score);
  const scoreEl = el('score');
  scoreEl.textContent = result.score.toFixed(3);
  scoreEl.style.color = colour;
  el('label').textContent = text;
  el('label').style.color = colour;
  el('interval').textContent =
    `±${(2 * result.std).toFixed(3)}  →  [${result.lower.toFixed(3)}, ${result.upper.toFixed(3)}]`;

  el('band').style.left = `${result.lower * 100}%`;
  el('band').style.width = `${Math.max(0.4, (result.upper - result.lower) * 100)}%`;
  el('mark').style.left = `${result.score * 100}%`;
  el('mark').style.background = colour;
  el('threshold').style.left = `${THRESHOLD * 100}%`;

  el('status').className = 'status';
  el('status').textContent =
    `${nSamples} Monte Carlo passes in ${elapsed.toFixed(1)} ms, in this browser.`;

  renderFeatures();
}

function renderFeatures() {
  const raw = extractWindowFeatures(Float64Array.from(windowValues));
  const { features } = model.preprocess(Float64Array.from(windowValues));
  const rows = FEATURE_NAMES.map((name, i) => {
    const scaled = features[i];
    // Flag anything far from the training distribution: that is the same
    // signal the deployed service turns into a PSI drift metric.
    const colour = Math.abs(scaled) > 3 ? ' style="color:#f9e2af"' : '';
    return `<tr><td>${name}</td>`
      + `<td class="num">${raw[i].toFixed(3)}</td>`
      + `<td class="num"${colour}>${scaled.toFixed(2)}</td></tr>`;
  });
  el('features').innerHTML = rows.join('');
}

// ------------------------------------------------------------------ setup

function applyPreset(name) {
  windowValues = PRESETS[name](model.windowSize);
  draw();
  score();
}

for (const button of document.querySelectorAll('[data-preset]')) {
  button.addEventListener('click', () => applyPreset(button.dataset.preset));
}

el('samples').addEventListener('input', (e) => {
  el('samplesOut').textContent = e.target.value;
  scheduleScore();
});

(async function init() {
  try {
    model = await BrowserModel.load('assets');
    el('version').textContent = model.modelVersion;
    applyPreset('step');
  } catch (error) {
    el('status').className = 'status error';
    el('status').textContent = `Could not load the model: ${error.message}`;
    // A file:// open fails CORS on fetch; say so rather than leaving it blank.
    if (window.location.protocol === 'file:') {
      el('status').textContent +=
        ' Opening the page directly from disk blocks fetch — serve it over HTTP'
        + ' (python3 -m http.server) or use the hosted version.';
    }
  }
}());
