# Multi-stage: the build toolchain never reaches the runtime image.
#
# CPU wheels by default. The GPU torch build pulls in roughly 2.5 GB of CUDA
# libraries, which is pure waste in an image that has no GPU runtime -- see
# docker-compose.gpu.yml for the GPU path.

FROM python:3.12-slim AS builder

# requirements-cpu.txt by default; pass requirements.txt for the CUDA build:
#   docker compose build --build-arg REQUIREMENTS=requirements.txt
ARG REQUIREMENTS=requirements-cpu.txt

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential compiles the wheels that ship without binaries; it stays in
# this stage and never reaches runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements before source, so editing a .py does not invalidate the
# dependency layer -- that is the expensive one to rebuild.
COPY requirements-base.txt requirements-cpu.txt requirements.txt ./
RUN pip install -r "$REQUIREMENTS"


FROM python:3.12-slim AS runtime

# ffmpeg and libsndfile1 back soundfile/librosa decoding of mp3, m4a and ogg.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=configs/cpu.yml \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Only what the service needs at runtime: no tests, no scripts, no docs.
COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml README.md ./

# Non-root, and owner of the model cache it writes to on first start.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser /app
USER appuser

EXPOSE 8000

# No curl in the slim image, so probe with the interpreter already present.
# start-period is generous because the first boot downloads model weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
