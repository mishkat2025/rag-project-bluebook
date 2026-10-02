# syntax=docker/dockerfile:1

# EWU Bulletin RAG chatbot: retrieval, reranking and the terminal REPL.
#
# NOT in this image:
#   * the LLM. It is reached over HTTP at LLM_BASE_URL (LM Studio on the host
#     by default); its GPU use is that server's business, not this image's.
#   * the BGE-M3 and reranker weights (~7GB). They download on first run
#     into HF_HOME, which compose.yaml keeps in a named volume.
#
# Matches the venv this project was built and measured with. The BM25 index
# is a pickle, so the Python version is part of the index format.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models

WORKDIR /app

# Dependencies before source: this layer is rebuilt only when
# requirements.txt changes, not on every code edit.
#
# No nvidia/cuda base image is needed -- the cu130 torch wheel carries its own
# CUDA libraries, and the host's driver is passed in at run time (--gpus all).
# The same wheel runs on the CPU when DEVICE=cpu.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      --index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/
COPY eval/ eval/

# The bulletin and the index built from it (data/processed, data/indexes),
# so a user never runs the three build scripts and gets the exact index the
# numbers in PROGRESS.md were measured on. Build the index BEFORE the image.
COPY data/ data/

# Inside a container "localhost" is the container itself. Docker Desktop
# resolves this name to the host; on Linux compose.yaml maps it.
ENV LLM_BASE_URL=http://host.docker.internal:1234/v1

# Ties the published image to its repository on GitHub's registry.
LABEL org.opencontainers.image.source=https://github.com/mishkat2025/rag-project-bluebook

CMD ["python", "app/chat.py"]
