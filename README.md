# Chrono-Sentinel

[![CI](https://github.com/JonesRobM/Chrono-Sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/JonesRobM/Chrono-Sentinel/actions/workflows/ci.yml)
[![Release](https://github.com/JonesRobM/Chrono-Sentinel/actions/workflows/release.yml/badge.svg)](https://github.com/JonesRobM/Chrono-Sentinel/actions/workflows/release.yml)
[![Container](https://img.shields.io/badge/ghcr.io-chrono--sentinel-2496ED?logo=docker&logoColor=white)](https://github.com/JonesRobM/Chrono-Sentinel/pkgs/container/chrono-sentinel)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)

Anomaly detection on time series, with an uncertainty estimate, served as a
containerised HTTP API that reports its own latency, throughput and drift.

A transformer trained on the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB)
scores a window of points. Monte Carlo Dropout puts an interval around each
score. `/metrics` tells you whether the traffic still looks like the data the
model was trained on.

The container carries **no deep-learning framework**. Training uses PyTorch;
the service runs the forward pass in numpy from a 282 KB weights file, which
takes the image from 1.35 GB to 355 MB.

Run it without cloning anything:

```bash
docker run --rm -p 7860:7860 ghcr.io/jonesrobm/chrono-sentinel:0.3.0
```

![Scoring demo](docs/demo.gif)

| Payload | Shape | Score |
| --- | --- | ---: |
| `examples/flat_window.json` | constant | 0.16 |
| `examples/noisy_window.json` | high variance, no level shift | 0.31 |
| `examples/step_change.json` | level shift | 0.91 |

| | |
| --- | --- |
| **Model** | Transformer + statistical-feature branch, 76k params, MC Dropout |
| **Test performance** | ROC-AUC 0.78, average precision 0.30 against a 0.033 base rate, three held-out NAB series |
| **Serving** | FastAPI, 6.7 ms p50 single-client, ~360 req/s at concurrency 8 |
| **Image** | 355 MB, no framework, non-root, 0.54 s to ready |
| **Observability** | Prometheus latency, throughput, score distribution, PSI drift |
| **Pipeline** | Locked dependencies, lint + tests + container smoke test in CI, GHCR on tag |

Every number here came out of a script in this repo. Anything not yet measured
says so. There's a [What doesn't work](#what-doesnt-work) section and it isn't
empty — the uncertainty interval, which is the headline feature, doesn't do the
job you'd hope it does.

---

## Quick start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.lock && uv pip install -e . --no-deps

python scripts/fetch_data.py        # ~2.7 MB, only the 12 NAB series used
python scripts/train.py             # reproduces the shipped checkpoint exactly
python scripts/evaluate.py
python scripts/export_weights.py    # best_model.pt -> model.npz for serving
python scripts/build_reference.py

uvicorn threatsim.serving.app:app --port 8077
```

---

## Results

### Latency

Measured with `scripts/loadtest.py` on an Apple M5 Pro (15 cores), macOS 26.6.2,
one uvicorn worker, window size 50, 1000 requests after 100 discarded warm-ups.
Run-to-run variance is around 10%, so read these to one significant figure.

![Load test](docs/loadtest.gif)

What the Monte Carlo sample count costs, at concurrency 8:

| `mc_samples` | p50 | p99 | throughput |
| ---: | ---: | ---: | ---: |
| 2 | 9.4 ms | 15.1 ms | 855 req/s |
| 10 | 11.3 ms | 19.6 ms | 690 req/s |
| 30 *(default)* | 22.2 ms | 26.1 ms | 359 req/s |
| 100 | 65.5 ms | 70.8 ms | 122 req/s |

Concurrency, at `mc_samples=30`, 600 requests after 60 warm-ups:

| concurrency | p50 | p95 | p99 | throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.7 ms | 7.2 ms | 9.0 ms | 148 req/s |
| 4 | 13.3 ms | 14.1 ms | 16.6 ms | 299 req/s |
| 8 | 22.2 ms | 24.7 ms | 27.9 ms | 359 req/s |
| 16 | 44.0 ms | 51.8 ms | 56.7 ms | 358 req/s |

Throughput plateaus at about 360 req/s. numpy's many small operations hold the
GIL more than torch's kernels do, so one worker saturates early. Four workers
measured 475 req/s at ~58 MB resident each, which is affordable now the image
carries no framework — set `CHRONO_WORKERS`. On a 2-vCPU host leave it at 1.

There's no hosted-endpoint row: this project ships a container, not a URL (see
[Deployment](#deployment)). `scripts/loadtest.py --url <host>` gives you the
same table against anything you deploy it to.

### The cost of dropping torch

`scripts/bench_batching.py`, batch 1, single-threaded, 300 iterations after 50
discarded, HTTP excluded:

| | torch | numpy |
| --- | ---: | ---: |
| forward pass, `mc_samples=30` | 4.9 ms | 5.7 ms |
| image | 1.35 GB | **355 MB** |
| cold start to `/readyz` | 1.14 s | **0.54 s** |
| peak throughput | ~715 req/s | ~360 req/s (475 with 4 workers) |

Per-request cost is close to a wash. Throughput is where numpy loses, and image
size and cold start are where it wins. For a project whose deployment target is
a small free host, that trade is worth taking; for a high-throughput service it
would not be.

Two bugs found while optimising the numpy path, both worth knowing about:
`np.sqrt(head_dim)` returns a *strong* float64 numpy scalar, which under NEP 50
promotes the whole attention path to float64; and the dropout RNG defaults to
float64, which was 44% of a prediction. Fixing both took 9.4 ms to 5.7 ms.

Batching the Monte Carlo passes into one forward pass over replicated inputs,
rather than looping, is worth a further 1.6-1.8x:

| `mc_samples` | sequential | batched | speedup |
| ---: | ---: | ---: | ---: |
| 10 | 3.36 ms | 2.04 ms | 1.6x |
| 30 | 10.07 ms | 5.73 ms | 1.8x |
| 100 | 33.28 ms | 18.93 ms | 1.8x |

### Model quality

Test is three NAB series never seen in training: `machine_temperature_system_failure`,
`rds_cpu_utilization_cc0c53`, `elb_request_count_8c0756`. 3063 windows, 3.3%
anomalous. Reproduce with `python scripts/evaluate.py`.

| method | val AP | val AUC | test AP | test AUC |
| --- | ---: | ---: | ---: | ---: |
| **Transformer + MC Dropout** | 0.156 | 0.573 | **0.304** | 0.781 |
| Logistic regression, 10 features | 0.255 | 0.640 | 0.167 | **0.786** |
| Logistic regression, z-scored sequence | 0.038 | 0.442 | 0.032 | 0.517 |
| Always-anomaly | 0.044 | 0.500 | 0.033 | 0.500 |

Average precision is the headline. At a 3.3% base rate, ROC-AUC rewards ranking
across a huge negative majority, while AP tracks how well the few positives get
surfaced. The transformer manages 9.2x the base rate.

At the threshold that maximises F1 on validation (0.925, never tuned on test):
precision 0.92, recall 0.11, F1 0.19. It is a high-precision, low-recall
operating point — it catches a ninth of the anomalies and is almost never wrong
when it fires.

Logistic regression on ten statistical features is genuinely competitive and
still wins on test ROC-AUC. A transformer that couldn't beat it wouldn't be
worth serving.

| | |
| :-: | :-: |
| ![ROC curve](outputs/roc_curve.png) | ![Precision-recall curve](outputs/pr_curve.png) |

---

## What doesn't work

**The uncertainty interval doesn't identify errors.** Using sigma to rank which
predictions are wrong gives an AUC of 0.42 — worse than a coin flip. Mean sigma
is 0.1001 on correct predictions and 0.0934 on incorrect ones, so if anything
the model is marginally *more* confident when it's wrong.

![Uncertainty, correct vs incorrect](outputs/uncertainty_histogram.png)

Sigma does respond to the input (0.03 to 0.26 across the test set), so it's a
real measurement. It measures how much dropout perturbs a prediction, not how
likely that prediction is to be wrong.

**The probabilities are badly calibrated.** Expected Calibration Error is 0.47.
Training uses `pos_weight=28.4` to counter the class imbalance, which inflates
predicted probabilities well above the 3.3% base rate. Threshold the scores;
don't read them as probabilities.

![Reliability diagram](outputs/calibration_curve.png)

**MC Dropout helps on test and hurts on validation.** It's there for the
interval, and its effect on ranking is not dependable:

| split | mode | AUC | AP |
| --- | --- | ---: | ---: |
| val | deterministic | 0.557 | 0.335 |
| val | MC Dropout, n=30 | 0.587 | 0.164 |
| test | deterministic | 0.755 | 0.246 |
| test | MC Dropout, n=30 | 0.783 | 0.291 |

The checkpoint is also selected on *deterministic* validation AP but served with
MC averaging, so selection and deployment aren't quite measuring the same thing.

**Seed variance is large.** Best validation AP across five seeds: 0.212, 0.245,
0.267, 0.335, 0.338. Any single-run difference below about 0.1 AP on this
dataset is noise, including differences quoted above.

**Evaluation is stochastic**, so it's seeded (`--seed`, default 42).

---

## Two bugs worth reading about

### MC Dropout was only sampling a third of the network

Porting the forward pass to numpy made the two implementations disagree on
sigma by 5%. Bisecting the dropout sites found the reason:
`nn.TransformerEncoderLayer` takes a fused inference path while it is in eval
mode, and that kernel **never consults its dropout submodules**.

`enable_mc_dropout()` set twelve dropout sites to train mode. The eight inside
the encoder were switched on and then silently ignored — measured contribution
to variance, exactly zero. The uncertainty came only from the positional
encoding, the feature branch and the classifier head.

The fix is to put `TransformerEncoderLayer` and `MultiheadAttention` in train
mode too, not just `nn.Dropout`. It moved test average precision from 0.254 to
0.304, and moved the error-detection AUC from 0.54 to 0.42 — the detector got
better and the uncertainty got measurably worse.
`tests/test_forward_parity.py::test_all_dropout_sites_contribute` guards it.

### The split wasn't a split

The checkpoint this repo originally shipped scored **ROC-AUC 0.481** — worse
than chance. It called every window an anomaly, and its sigma was a flat 0.021
whatever the input. Five causes, all fixed, all guarded by `tests/test_data.py`:

1. **The split wasn't a split.** `get_dataloaders` concatenated every series
   and *then* split by time. One series was 5.6x longer than the other, so both
   boundaries landed inside it: train and validation were 100% machine
   temperature, test 100% EC2 CPU. It looked like 70/15/15 and was an
   out-of-distribution transfer test. Splits are now **grouped by series**,
   keeping positive rates comparable (3.4% / 4.4% / 3.3%).

2. **The labels were ~43x too narrow.** The old mask marked ±30 wall-clock
   minutes per annotation — ±6 samples at NAB's 5-minute rate. NAB's own
   convention is a window totalling a fraction of the series length, split
   across its anomalies: 567 points for machine temperature. That left roughly
   11 positive training windows to learn from.

3. **Per-window z-scoring erased the signal.** Zeroing each window's mean and
   variance throws away level and scale, which is what a machine-temperature
   failure looks like. It can't just be dropped — the service gets a bare
   window with no series identity, so preprocessing must be a pure function of
   that window, and a global scaler is worse still (AUC 0.125; the series span
   means from 0.1 to 87). The fix keeps z-scoring for the sequence and **adds a
   feature branch** of ten statistical features, scaled by a `FeatureScaler`
   fitted on train and stored with the weights.

4. **`nn.BCELoss` on a sigmoid output** with a hand-applied weight, rather than
   `BCEWithLogitsLoss(pos_weight=...)` on logits.

5. **Selection on validation loss.** At a 3% positive rate the loss is
   dominated by the negative class, so a model that collapses to the prior
   posts a respectable one. Selection is now on validation average precision.

Two smaller findings: `ReduceLROnPlateau` *hurt* (validation AP 0.335 → 0.227 on
seed 42, halving the rate during dips the run recovers from), so it's opt-in;
and training defaults to CPU, because MPS numerics shifted results enough that a
sweep and a training run disagreed on identical settings.

`python scripts/train.py` with no flags reproduces the shipped checkpoint
byte-for-byte (sha256 `711eac756d17…`).

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /score` | Score one window; returns score, interval, model version, inference time |
| `GET /healthz` | Liveness. Doesn't touch the model |
| `GET /readyz` | Readiness, plus model version and expected window size. 503 until loaded |
| `GET /drift` | Population drift against the training reference |
| `GET /metrics` | Prometheus exposition |

Liveness and readiness are separate because loading the weights takes long
enough that a combined probe would report a crash loop during a slow cold start.

`POST /score` takes `values` (exactly the model's window size — check `/readyz`)
and an optional `mc_samples` between 2 and 200. Both multiply into inference
cost, so both are bounded. A wrong-length window gets a 422 rather than being
padded, because silently reshaping the input would return a confident score for
something the caller never asked about.

Payloads for all three example shapes are in [`examples/`](examples/).

---

## Observability

`/metrics` answers the four questions you actually ask of a deployed model:

| Question | Metric |
| --- | --- |
| How much traffic? | `chrono_requests_total{endpoint,status}` |
| How fast? | `chrono_request_latency_seconds`, `chrono_inference_latency_seconds` |
| What's it saying? | `chrono_anomaly_score`, `chrono_uncertainty_std` |
| Is it still valid? | `chrono_drift_psi{feature}`, `chrono_drift_max_psi` |

Buckets are tuned for a CPU forward pass in the single-to-tens of milliseconds;
the `prometheus_client` defaults start at 5 ms and would put nearly everything
in one bucket. No label is client-controlled — an unmatched route reports
`endpoint="unmatched"` rather than letting a caller mint unbounded series.

### Drift

`scripts/build_reference.py` records how the model's inputs and outputs were
distributed **on the training split only**. The service keeps a rolling buffer
of recent requests and reports the Population Stability Index against that
reference, per feature and for the score distribution.

```
PSI < 0.10   stable
0.10 - 0.25  moderate
PSI > 0.25   significant
```

![Drift demo](docs/drift.gif)

Synthetic traffic looks nothing like NAB, so every feature reports
`significant`. That's the detector working.

The reference is never rebuilt from live traffic — it would drift along with
the data and report stability while the model went stale, which is the failure
it exists to catch. PSI stays hidden below 50 observations, where it's just
binning noise.

`tests/test_reference.py` checks both directions: below 0.10 on an unshifted
sample, above 0.25 on a three-sigma shift, on a variance change alone, and on
an output-only collapse with inputs held fixed.

All three recordings render from committed [VHS](https://github.com/charmbracelet/vhs)
scripts (`vhs docs/demo.tape`), so they regenerate instead of rotting the way
screenshots do.

---

## Architecture

```
Raw window (50 points)
    |
    +-- per-window z-score ------> Transformer encoder (2 layers, 4 heads, d=64)
    |                                        |
    |                                   mean pooling
    |                                        |
    +-- 10 statistical features ---> feature MLP
        (scaled by the exported                     |
         FeatureScaler)                    concatenate
                                                    |
                                          classifier -> logit
                                                    |
                                       30x stochastic passes
                                                    |
                                        mean = score, sd = uncertainty
```

76,000 parameters. Both preprocessing paths are pure functions of the incoming
window plus constants exported with the weights, so training and serving apply
identical transforms.

### Two backends, one forward pass

```
train.py (PyTorch) ──► best_model.pt ──► export_weights.py ──► model.npz
                             │                                     │
                             ▼                                     ▼
                   evaluate.py (PyTorch)                 serving/ (numpy only)
```

Training keeps PyTorch, which is where autograd and DataLoaders matter. The
container runs `threatsim/serving/forward.py`, about 150 lines of numpy.

Reproducing `nn.TransformerEncoderLayer` exactly needs three details that are
easy to get wrong: `in_proj_weight` packs Q, K and V into one (3d, d) matrix;
`norm_first=False` means post-norm, so `x = norm1(x + sa(x))`; and there are
twelve dropout sites, including one on the attention weights inside the
attention block.

Two implementations of one function is a real hazard — they could diverge and
the service would return confident wrong answers.
`tests/test_forward_parity.py` is the contract: deterministic logits agree to
2.4e-07, and over 150 seeded repeats the Monte Carlo mean agrees within 0.002
and sigma within 0.3%.

---

## Deployment

Actions builds and pushes to GHCR on every `v*` tag. The published image is the
deliverable: a public, immutable tag anyone can run.

```
git tag v0.3.0  ->  release.yml  ->  ghcr.io/jonesrobm/chrono-sentinel:0.3.0
                                              |
                                docker run -p 7860:7860 <tag>
```

**There's no hosted demo, and that's deliberate.** The plan was a free Hugging
Face Space on the Docker SDK. As of 2026-08-31 that isn't free: Hugging Face
requires PRO for Gradio or Docker Spaces and returns `402 Payment Required`.
Every other free container host I checked has since withdrawn its free compute
tier or wants a card. A public image beats a dead URL.

Four things about the release chain are easy to get wrong; `release.yml` handles
all four:

- **No leading `v` on the published tag.** A git tag of `v0.3.0` publishes
  `0.3.0` — `metadata-action`'s `{{version}}` strips it.
- **No `:latest`.** `metadata-action` adds it by default on a semver push, so
  the workflow sets `flavor: latest=false`. A floating tag would make the
  numbers above unattributable to any particular image.
- **The GHCR package is private by default**, even for a public repo, and only
  exists after a successful push. Check it the way an anonymous puller would:

  ```bash
  curl -s "https://ghcr.io/token?scope=repository:jonesrobm/chrono-sentinel:pull&service=ghcr.io"
  ```

  Public returns `{"token":"..."}`; private or absent returns `DENIED`.
- **`linux/amd64` only.** Pulling on Apple Silicon needs `--platform linux/amd64`.

| image | contents | size |
| --- | --- | ---: |
| v0.2.x | CPU-only torch + runtime deps | 1.35 GB |
| **v0.3.0** | **numpy + FastAPI, no framework** | **355 MB** |

site-packages is 120 MB of that, down from 873 MB. Dropping torch also removed
the CUDA problem entirely: torch's Linux wheel declares seven CUDA requirements
on PyPI, which is why earlier versions had to install from the CPU wheel index.

Cold start from `docker run` to `/readyz` is **0.54 s**.

---

## Layout

```
threatsim/
  data.py           NAB loading, labelling, grouped splits
  features.py       10 statistical window features
  models.py         Transformer + batched MC Dropout (training and evaluation)
  reference.py      Drift profile and PSI
  scaling.py        FeatureScaler, kept free of heavy imports
  utils.py          Seeding, checkpoint IO, plots
  serving/
    forward.py      The forward pass in numpy: what the container runs
    inference.py    Weight loading and the preprocessing contract
    app.py, metrics.py, schemas.py
scripts/
  fetch_data.py         Download only the NAB series used
  train.py / evaluate.py
  export_weights.py     best_model.pt -> model.npz for the container
  build_reference.py    Drift reference from the training split
  loadtest.py           Latency and throughput
  bench_batching.py     Batching speedup and the torch/numpy comparison
  make_examples.py      Regenerates examples/ from the checkpoint
  lock_requirements.sh  Regenerates the dependency locks
examples/           Ready-to-POST request payloads
docs/               VHS tapes and the recordings they render
tests/              84 tests, 9 of them backend parity
.github/            CI (lint, tests, image smoke test), release to GHCR, Dependabot
pyproject.toml      Packaging, ruff and pytest config
requirements.lock   Fully-pinned graph for CI; requirements-runtime.lock for the image
```

## Reproducing everything

```bash
python scripts/fetch_data.py
python scripts/train.py                  # byte-identical checkpoint
python scripts/evaluate.py               # quality table, "what doesn't work"
python scripts/export_weights.py
python scripts/build_reference.py
python scripts/bench_batching.py         # batching and backend comparison

uvicorn threatsim.serving.app:app --port 8077 &
python scripts/loadtest.py --requests 1000 --concurrency 8 --warmup 100 --sweep-mc 2,10,30,100

pytest -q                                # 84 tests
ruff check . && ruff format --check .    # the CI lint gate
```

Latency figures are machine-specific; the conditions are stated so a rerun
elsewhere is comparable rather than merely different.

## Limitations

- NAB is 12 series at one sample every five minutes. Nothing here has been
  tested on higher-frequency or multivariate data.
- Univariate only. `/score` takes one channel.
- One model. No A/B path, no shadow scoring, no automatic retraining when drift
  fires — `/drift` reports, a human decides.
- Drift lives in an in-memory buffer, so it resets when the container restarts.
- Two forward-pass implementations. A change to the architecture has to be made
  twice, and the parity test is the only thing that catches a mismatch.

## Licence

MIT.
