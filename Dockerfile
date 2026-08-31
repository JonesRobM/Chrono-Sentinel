# Chrono-Sentinel scoring service.
#
# Two things about this image are load-bearing:
#
#   1. torch comes from the CPU wheel index. On Linux the default PyPI wheel
#      declares seven CUDA requirements (cuda-toolkit, cudnn, cusparselt,
#      nccl, nvshmem and friends) for a service that never sees a GPU. The
#      +cpu wheel drops all of them.
#   2. Dependencies are installed before the application code is copied, so
#      editing a source file does not reinstall torch.
#   3. Only runtime dependencies are installed, not the research set. See
#      requirements-runtime.txt.
#
# Listens on $PORT, defaulting to 7860 for Hugging Face Spaces.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# CPU-only torch first and on its own, so the large download is cached in its
# own layer and is not invalidated by a change to any other pin.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0

COPY requirements-runtime.txt requirements-serve.txt ./

# Runtime dependencies only. requirements.txt is the *research* set: pandas,
# matplotlib, scikit-learn, scipy and seaborn are used by training and
# evaluation and by nothing the service executes. Installing them here added
# ~300 MB to the image, so the package __init__ is lazy (PEP 562) and the
# serving path imports only numpy and torch.
RUN pip install --no-cache-dir -r requirements-runtime.txt -r requirements-serve.txt


FROM python:3.12-slim

# Non-root. Spaces runs containers as uid 1000, so match it: files this user
# owns stay writable there without a chown at startup.
RUN useradd --create-home --uid 1000 chrono

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    CHRONO_MODEL_PATH=/app/outputs/best_model.pt \
    CHRONO_REFERENCE_PATH=/app/outputs/reference.json \
    CHRONO_MC_SAMPLES=30 \
    CHRONO_TORCH_THREADS=1 \
    HOME=/home/chrono

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY threatsim/ ./threatsim/

# The served artefacts. Both are required at runtime: without the checkpoint
# the service starts but reports itself unready, and without the reference it
# serves scores but reports no drift.
COPY outputs/best_model.pt outputs/reference.json ./outputs/

USER chrono

EXPOSE 7860

# Bind 0.0.0.0, not 127.0.0.1: a loopback bind is unreachable from outside the
# container and presents as a hung deploy.
#
# One worker deliberately. The model is held in process memory and scoring is
# CPU-bound, so on a 2-vCPU host extra workers multiply memory and contend for
# the same cores. Concurrency comes from FastAPI's threadpool instead: /score
# is a sync endpoint and AnomalyScorer holds no lock.
CMD ["sh", "-c", "exec uvicorn threatsim.serving.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/healthz').read()" || exit 1
