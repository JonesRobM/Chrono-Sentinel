# Chrono-Sentinel scoring service.
#
# There is no deep-learning framework in this image. The model is 76k
# parameters and the service only ever needs a forward pass, so the forward
# pass is implemented in numpy (threatsim/serving/forward.py) and the weights
# ship as a 282 KB .npz written by scripts/export_weights.py. torch stays in
# the training environment, where autograd and DataLoaders actually matter.
#
# That is worth roughly 1.2 GB of image. The risk it introduces -- two
# implementations of one function, which could silently diverge -- is held in
# check by tests/test_forward_parity.py, which pins the numpy path to torch's
# numbers on both the deterministic and the Monte Carlo paths.
#
# Dependencies are installed before application code is copied, so editing a
# source file does not reinstall them.
#
# Listens on $PORT, defaulting to 7860.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements-runtime.lock ./

RUN pip install --no-cache-dir -r requirements-runtime.lock


FROM python:3.12-slim

# Non-root, uid 1000 to match common container hosts, so files this user owns
# stay writable without a chown at startup.
RUN useradd --create-home --uid 1000 chrono

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    CHRONO_MODEL_PATH=/app/outputs/model.npz \
    CHRONO_REFERENCE_PATH=/app/outputs/reference.json \
    CHRONO_MC_SAMPLES=30 \
    CHRONO_WORKERS=1 \
    HOME=/home/chrono

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY threatsim/ ./threatsim/

# The served artefacts. Both are required: without the weights the service
# starts but reports itself unready, and without the reference it serves
# scores but reports no drift.
COPY outputs/model.npz outputs/reference.json ./outputs/

USER chrono

EXPOSE 7860

# Bind 0.0.0.0, not 127.0.0.1: a loopback bind is unreachable from outside the
# container and presents as a hung deploy.
#
# One worker by default, which suits a small host. Concurrency within a worker
# comes from FastAPI's threadpool (/score is sync and the numpy backend holds
# no lock), but numpy's many small operations hold the GIL more than torch's
# kernels do, so a single worker plateaus around 360 req/s. Raising
# CHRONO_WORKERS sidesteps that: 4 workers measured 475 req/s at ~58 MB RSS
# each, which is affordable now the image no longer carries a framework. On a
# 2-vCPU host leave it at 1 or 2.
CMD ["sh", "-c", "exec uvicorn threatsim.serving.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${CHRONO_WORKERS:-1}"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/healthz').read()" || exit 1
