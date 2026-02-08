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

WORKDIR /likelihood_submission

COPY env/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2* /usr/lib/python3/dist-packages/cv2* || true && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed -r /tmp/requirements.txt

COPY . /likelihood_submission

ENV OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

# CMD ["python", "inference.py"]