# syntax=docker/dockerfile:1.7
# =============================================================================
# AI Service System — production image
#
# Base: NVIDIA CUDA 12.4 + cuDNN 9 runtime (Ubuntu 22.04).
# - Includes the CUDA/cuDNN libraries `onnxruntime-gpu>=1.19` links against.
# - Container starts on a CPU-only host too: ONNX Runtime drops the CUDA
#   provider at session creation and falls back to CPUExecutionProvider.
#
# Two-stage build keeps the final image free of compilers and pip caches.
# =============================================================================

ARG PYTHON_VERSION=3.11

# -----------------------------------------------------------------------------
# Stage 1 — builder: install Python + wheels into a clean prefix.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        build-essential \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py | python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel setuptools \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2 — runtime: copy the venv, drop the toolchain.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    APP_HOME=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
        libglib2.0-0 libgl1 \
        tini \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY --from=builder /opt/venv /opt/venv

# Non-root user for runtime
RUN groupadd --system app && useradd --system --gid app --home ${APP_HOME} app
WORKDIR ${APP_HOME}

COPY --chown=app:app app ./app
COPY --chown=app:app docker/entrypoint.sh /usr/local/bin/entrypoint.sh
# Strip Windows CRLF — otherwise tini reports "No such file or directory" (exit 127).
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8001/api/v1/health', timeout=3).status==200 else sys.exit(1)" \
    || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
