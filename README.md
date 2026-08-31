# Chrono-Sentinel

Time-series anomaly detection with Monte Carlo Dropout uncertainty, served as a
containerised HTTP API that reports its own latency, throughput, score
distribution and population drift.

Train a transformer on the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB),
POST a window of points, get back an anomaly score with an uncertainty
interval, and scrape `/metrics` to see whether the model has gone stale.

Every number in this README was measured by a script in this repository, under
conditions stated beside it. Where something has not been measured yet it says
`TBD` rather than an estimate.

---

## Quick start

```bash
# 1. Environment
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-serve.txt -r requirements-dev.txt
uv pip install -e .

# 2. Data (downloads only the 12 series used, ~2.7 MB, not the 100 MB NAB repo)
python scripts/fetch_data.py

# 3. Train, evaluate, build the drift reference
python scripts/train.py
python scripts/evaluate.py
python scripts/build_reference.py

# 4. Serve
uvicorn threatsim.serving.app:app --port 8077
```

Score a window:

```bash
curl -X POST http://127.0.0.1:8077/score \
  -H 'Content-Type: application/json' \
  -d "{\"values\": $(python -c 'print([85.0]*25 + [20.0]*25)')}"
```

```json
{
  "score": 0.921,
  "uncertainty": {"std": 0.061, "lower": 0.799, "upper": 1.0},
  "mc_samples": 30,
  "model_version": "711eac756d17",
  "inference_ms": 2.93
}
```

That window is a step change, the machine-failure signature; a flat window
scores around 0.17. Identical requests return slightly different scores
because MC Dropout is stochastic — that variation *is* the uncertainty being
reported.

Interactive docs at `/docs`.

---

## Results

### Serving latency

Local, measured with `scripts/loadtest.py`. **Conditions:** Apple M5 Pro
(15 cores), macOS 26.6.2, Python 3.12, torch 2.13.0 CPU, one uvicorn worker,
`CHRONO_TORCH_THREADS=1`, window size 50, 1000 measured requests after 100
discarded warm-up requests, client on the same machine as the server.

Varying the number of Monte Carlo passes, at concurrency 8:

| `mc_samples` | p50 | p99 | throughput |
| ---: | ---: | ---: | ---: |
| 2 | 5.63 ms | 14.93 ms | 1341 req/s |
| 10 | 6.33 ms | 13.03 ms | 1246 req/s |
| 30 *(default)* | 7.05 ms | 12.66 ms | 1058 req/s |
| 100 | 14.36 ms | 17.64 ms | 553 req/s |

Cost is sub-linear in `mc_samples` because the passes are folded into one
batched forward pass rather than looped.

Varying concurrency at `mc_samples=30`, 800 measured requests after 80 warm-up:

| concurrency | p50 | p95 | p99 | max | throughput |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.92 ms | 3.06 ms | 3.19 ms | 3.61 ms | 340 req/s |
| 2 | 3.73 ms | 3.91 ms | 4.01 ms | 4.57 ms | 535 req/s |
| 4 | 4.29 ms | 4.74 ms | 5.15 ms | 30.08 ms | 906 req/s |
| **8** | **7.10 ms** | **10.25 ms** | **12.06 ms** | 13.03 ms | **1082 req/s** |
| 16 | 10.16 ms | 93.76 ms | 193.36 ms | 286.17 ms | 684 req/s |
| 32 | 52.86 ms | 241.44 ms | 412.67 ms | 672.59 ms | 374 req/s |

Throughput peaks at concurrency 8 and degrades beyond it: past the core count
the requests queue, and the tail goes first. p50 at concurrency 32 is still
53 ms while p99 is 413 ms, which is what saturation looks like from the
outside.

**Deployed (Hugging Face Spaces, free-tier shared CPU): TBD.** Not yet
deployed. Those numbers will be reported in a separate row, not merged with
these, because a request through the Spaces proxy from a laptop measures the
network and a shared vCPU as much as the model.

### MC Dropout batching

`scripts/bench_batching.py`, single-threaded, 300 timed iterations after 50
discarded, HTTP excluded:

| `mc_samples` | sequential loop | batched | speedup |
| ---: | ---: | ---: | ---: |
| 10 | 1.81 ms | 0.77 ms | 2.4x |
| 30 | 5.46 ms | 2.00 ms | 2.7x |
| 100 | 18.11 ms | 6.29 ms | 2.9x |

The two forms draw independent dropout masks, so they cannot agree
sample-for-sample; they agree on what those samples estimate. Over 300 seeded
draws at `n=30` the mean differs by 0.0003 absolute and sigma by 1.1%
relative, which is what makes the batched form a substitute rather than a
different computation. `tests/test_models.py` pins this.

### Model quality

Test is three NAB series never seen in training: `machine_temperature_system_failure`,
`rds_cpu_utilization_cc0c53`, `elb_request_count_8c0756`. 3063 windows, 3.30%
anomalous. Reproduce with `python scripts/evaluate.py`.

| method | val AP | val AUC | test AP | test AUC |
| --- | ---: | ---: | ---: | ---: |
| **Transformer + MC Dropout** | 0.160 | 0.599 | **0.254** | 0.773 |
| Logistic regression, 10 features | 0.255 | 0.640 | 0.167 | **0.786** |
| Logistic regression, z-scored sequence | 0.038 | 0.442 | 0.032 | 0.517 |
| Always-anomaly (degenerate baseline) | 0.044 | 0.500 | 0.033 | 0.500 |

Average precision is the headline: at a 3.3% base rate, ROC-AUC rewards
ranking across a vast negative majority, while AP tracks how well the few
positives are surfaced. The transformer reaches **7.7x the base-rate AP**.

At the threshold that maximises F1 on validation (0.660, never tuned on test):
precision 0.101, recall 0.604, F1 0.174.

The baselines are in the table because they are the only thing that makes the
model's number meaningful. Logistic regression on ten statistical features is
a genuinely competitive baseline here — it wins on test ROC-AUC — and a
transformer that could not beat it would not be worth serving.

---

## What does not work

A results table that only reports wins is not a result. These are measured and
reproducible.

**The uncertainty interval does not identify errors.** Error-detection AUC —
using sigma to rank which predictions are wrong — is **0.539**, against 0.5 for
chance. Mean sigma is 0.1024 on correct predictions and 0.1033 on incorrect
ones. The interval is an honest report of the model's output spread, and that
spread does respond to the input (sigma ranges 0.028–0.262 across the test
set, coefficient of variation 0.33), but it is close to useless as a
confidence signal. Treat it as "how much does dropout perturb this
prediction", not "how likely is this wrong".

**The probabilities are badly calibrated.** Expected Calibration Error is
**0.463**. Training uses `pos_weight=28.4` to counter the class imbalance,
which deliberately inflates predicted probabilities far above the 3.3% base
rate. The scores rank well and mean nothing in absolute terms — threshold them,
do not read them as probabilities.

**MC Dropout's effect on accuracy is inconsistent.** It exists here for the
uncertainty interval, not for accuracy, and the ~2.7x latency cost buys
nothing reliable in ranking quality:

| split | mode | AUC | AP |
| --- | --- | ---: | ---: |
| val | deterministic | 0.557 | 0.335 |
| val | MC Dropout, n=30 | 0.555 | 0.160 |
| test | deterministic | 0.755 | 0.246 |
| test | MC Dropout, n=30 | 0.776 | 0.275 |

Averaging helps test AP and halves validation AP. Note also that the
checkpoint is selected on *deterministic* validation AP but served with MC
averaging, so selection and deployment are not measuring the same quantity.

**Seed variance is large.** Best validation AP across five seeds: 0.212, 0.245,
0.267, 0.335, 0.338. Any single-run difference smaller than about 0.1 AP on
this dataset is noise, including differences in this README.

**Evaluation is stochastic** and therefore seeded (`--seed`, default 42). On an
identical checkpoint, unseeded runs moved error-detection AUC between 0.43 and
0.56, because the F1 threshold is chosen from sampled validation scores.

---

## How the detector was fixed

The checkpoint this repository previously shipped scored **ROC-AUC 0.481** —
worse than chance. It predicted "anomaly" for every window, and its MC Dropout
sigma was a near-constant 0.021 regardless of input, so the uncertainty
interval was decorative. Five causes, all now fixed and guarded by tests:

1. **The split was not a split.** `get_dataloaders` concatenated every series
   and *then* split the concatenation temporally. Because
   `machine_temperature_system_failure` is ~5.6x longer than
   `ec2_cpu_utilization_24ae8d`, both boundaries landed inside the first
   series: train and validation were 100% machine temperature, test was 100%
   EC2 CPU. It looked like 70/15/15 and was an out-of-distribution transfer
   test. Now the split is **grouped by series** — disjoint series per split,
   which measures generalisation to an unseen asset and keeps positive rates
   comparable (3.40% / 4.41% / 3.30%).

2. **The labels were ~43x too narrow.** `create_anomaly_mask` marked ±30
   wall-clock minutes around each annotation, which at NAB's 5-minute sampling
   is ±6 samples. NAB's own convention is a window totalling a fraction of the
   series length, divided across its anomalies — 567 points for machine
   temperature. The old labelling left roughly 11 positive training windows.
   `nab_anomaly_mask` now follows the NAB convention, with the fraction
   exposed as `--window-frac` (default 0.02; NAB scores with 0.10).

3. **Per-window z-scoring erased the signal.** Normalising each window to zero
   mean and unit variance discards level and scale, which is precisely what a
   machine-temperature failure is. It cannot simply be dropped — the service
   receives a bare window with no series identity, so preprocessing must be a
   pure function of the window, and a global scaler is worse still (the series
   span means of 0.1 to 87, and a global scaler measured AUC 0.125). The fix
   keeps per-window z-scoring for the sequence and **adds a feature branch**:
   ten statistical features, scaled by a `FeatureScaler` fitted on the training
   split and persisted in the checkpoint. Level information survives in the
   feature vector, and both transforms reproduce exactly at serve time.

4. **`nn.BCELoss` on a sigmoid output** with a manually applied ~30x weight.
   Now `BCEWithLogitsLoss(pos_weight=...)` on logits.

5. **Model selection on validation loss.** At a 3% positive rate, loss is
   dominated by the negative class and a model collapsing to the prior posts a
   respectable one. Selection is now on validation average precision.

Two smaller findings, both measured: the `ReduceLROnPlateau` scheduler *hurt*
(best validation AP 0.335 → 0.227 on seed 42, because plateau detection halves
the rate during dips the run recovers from) and is now opt-in; and training
runs on CPU by default, because MPS numerics shifted results enough that a
sweep run and a training run disagreed on identical hyperparameters.

`python scripts/train.py` with no flags reproduces the shipped checkpoint
byte-for-byte (sha256 `711eac756d17…`).

---

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /score` | Score one window; returns score, uncertainty interval, model version, server-side inference time |
| `GET /healthz` | Liveness. Does not touch the model |
| `GET /readyz` | Readiness. Reports model version, expected window size, default MC samples. 503 until loaded |
| `GET /drift` | Human-readable population drift against the training reference |
| `GET /metrics` | Prometheus text exposition |

Liveness and readiness are separate because loading the checkpoint takes long
enough that a combined probe reports a crash loop during a slow cold start —
exactly when it happens on a free-tier host.

`POST /score` takes `values` (exactly the model's window size; query `/readyz`)
and optional `mc_samples` (2–200). Both bound inference cost, so both are
validated: a wrong-length window is rejected with 422 rather than padded,
because silently reshaping the input would return a confident score for
something the caller did not ask about.

---

## Observability

`/metrics` answers the four questions an operator asks of a deployed model:

| Question | Metric |
| --- | --- |
| How much traffic? | `chrono_requests_total{endpoint,status}` |
| How fast? | `chrono_request_latency_seconds`, `chrono_inference_latency_seconds` |
| What is it saying? | `chrono_anomaly_score`, `chrono_uncertainty_std` |
| Is it still valid? | `chrono_drift_psi{feature}`, `chrono_drift_max_psi` |

Latency buckets are tuned for a CPU forward pass in the single-digit-to-tens
of milliseconds; the `prometheus_client` defaults start at 5 ms and would put
almost every observation in one bucket. No metric is labelled by anything
client-controlled — an unmatched route reports `endpoint="unmatched"` rather
than letting a caller mint unbounded time series.

### Drift

`scripts/build_reference.py` records how the model's inputs and outputs were
distributed **on the training split only**, into `outputs/reference.json`. The
service keeps a rolling buffer of recent requests and reports the Population
Stability Index against that reference, per input feature and for the score
distribution:

```
PSI < 0.10   stable
0.10 - 0.25  moderate
PSI > 0.25   significant
```

The reference is never rebuilt from live traffic. Doing so would let it drift
along with the data and report stability while the model quietly went stale,
which is the exact failure this is meant to catch. PSI is withheld until the
buffer holds at least 50 observations, because on fewer it is binning noise.

It works in both directions, which is the part worth testing: `tests/test_reference.py`
asserts PSI stays below 0.10 on an unshifted sample and exceeds 0.25 on a
three-sigma shift, on a variance change alone, and on an output-only collapse
with inputs held fixed.

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
        (scaled by the checkpoint's                 |
         FeatureScaler)                    concatenate
                                                    |
                                          classifier -> logit
                                                    |
                                       30x stochastic passes
                                                    |
                                        mean = score, sd = uncertainty
```

72,129 parameters. Both preprocessing paths are pure functions of the incoming
window plus fixed constants from the checkpoint, so training and serving apply
identical transforms.

The service enables dropout **once at load** and never toggles it. Toggling per
request needs a lock — one request's exit would disable dropout inside
another's forward pass and silently collapse its uncertainty to zero — and that
lock would serialise every forward pass. With dropout left on, scoring mutates
no shared state, `/score` is a sync endpoint dispatched to FastAPI's
threadpool, and requests genuinely overlap. That is what the concurrency table
above is measuring.

---

## Deployment

Actions builds and pushes to GitHub Container Registry on every `v*` tag; the
Hugging Face Space deploys that exact image.

```
git tag v0.2.1  ->  .github/workflows/release.yml  ->  ghcr.io/<owner>/chrono-sentinel:v0.2.1
                                                              |
                                                     spaces/Dockerfile
                                                     FROM ghcr.io/...:v0.2.1
                                                              |
                                                     Hugging Face Space
```

Three things about this chain are easy to get wrong:

- **Spaces cannot pull an arbitrary registry tag.** Its Docker SDK builds a
  Dockerfile in the Space repo. `spaces/Dockerfile` is that link — two lines,
  `FROM` a pinned GHCR tag — so what runs is the image CI tested, not a
  rebuild that might differ.
- **The GHCR package must be public**, or the Space builder cannot pull it.
  A package is created only by a successful push, so the release workflow
  has to go green before the visibility setting exists to change.
- **Pin a concrete tag, never `:latest`**, or the deployed artefact is not the
  one this README's numbers describe. The release workflow deliberately
  publishes no `latest` tag.

The image installs torch from the CPU wheel index. On Linux the default PyPI
wheel declares seven CUDA requirements (`cuda-toolkit`, `nvidia-cudnn`,
`cusparselt`, `nccl`, `nvshmem` and others) for a service that never sees a
GPU; the `+cpu` wheel drops all of them, and the running container reports
`torch 2.13.0+cpu`. ONNX is not an alternative: MC Dropout needs dropout live
at inference, and export folds it away.

**Image size, measured on linux/arm64:**

| image contents | size |
| --- | ---: |
| full research requirements installed | 1.87 GB |
| **runtime requirements only** | **1.35 GB** |

The service needs numpy and torch, not pandas, matplotlib, scikit-learn,
scipy or seaborn. Those were arriving through the package `__init__`, so
`threatsim/__init__.py` resolves its exports lazily (PEP 562) and the image
installs `requirements-runtime.txt` rather than `requirements.txt`. That is
worth 520 MB. torch is 635 MB of the remaining 873 MB of site-packages, which
is close to the floor without changing frameworks.
`tests/test_serving.py::TestImportFootprint` asserts the serving import chain
stays light, because a single module-level import would undo it silently.

Cold start from `docker run` to `/readyz` returning ready: **1.14 s** locally.
On free-tier Spaces it will be considerably slower — shared CPU, and the
image has to be pulled first. That figure is still TBD.

CI builds the image, runs a container, waits for `/readyz`, scores a real
window, asserts the uncertainty is non-zero, and prints both the image size
and the runner's free disk.

---

## Layout

```
threatsim/
  data.py           NAB loading, NAB-convention labelling, grouped splits, FeatureScaler
  features.py       10 statistical window features
  models.py         Transformer + batched MC Dropout
  reference.py      Drift reference profile and PSI
  utils.py          Seeding, checkpoint IO, plots
  serving/
    app.py          FastAPI application
    inference.py    Checkpoint loading and the preprocessing contract
    metrics.py      Prometheus collectors and the rolling drift buffer
    schemas.py      Request/response models and validation bounds
scripts/
  fetch_data.py       Download only the NAB series used
  train.py            Training
  evaluate.py         Metrics, calibration, uncertainty quality, baselines
  build_reference.py  Drift reference from the training split
  loadtest.py         Latency and throughput
  bench_batching.py   MC Dropout batching speedup and equivalence
tests/              107 tests
.github/workflows/  CI (test + build + container smoke test), Release (GHCR on tag)
spaces/             Hugging Face Space Dockerfile and README front-matter
```

## Reproducing every number here

```bash
python scripts/fetch_data.py
python scripts/train.py                       # checkpoint, byte-identical
python scripts/evaluate.py                    # model quality table, "what does not work"
python scripts/build_reference.py
python scripts/bench_batching.py              # batching table

uvicorn threatsim.serving.app:app --port 8077 &
python scripts/loadtest.py --requests 1000 --concurrency 8 --warmup 100 --sweep-mc 2,10,30,100
python scripts/loadtest.py --requests 800 --concurrency 8 --warmup 80 --mc-samples 30

pytest tests/ -q                              # 107 tests
```

Latency figures are machine-specific; the conditions are stated above so a
rerun elsewhere is comparable rather than merely different.

---

## Limitations

- The detector generalises across NAB series, which is a harder and more
  honest setting than a within-series split, but NAB is 12 series of one
  sample every five minutes. Nothing here has been tested on higher-frequency
  or multivariate data.
- Univariate only. `POST /score` takes one channel.
- The service holds one model. There is no A/B path, no shadow scoring, and
  no automatic retraining when drift fires — `/drift` reports, a human decides.
- Drift is measured over a rolling in-memory buffer, so it resets when the
  container restarts, and on a free-tier Space that happens whenever it sleeps.

## Licence

MIT. Educational and portfolio use.
