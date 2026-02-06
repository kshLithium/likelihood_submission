# FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/likelihood_submission

COPY env/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed -r /tmp/requirements.txt

COPY . /workspace/likelihood_submission

ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

# CMD ["python", "inference.py"]