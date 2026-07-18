# ── Base image ─────────────────────────────────────────────────────────────────
# PyTorch 2.3.1, CUDA 12.1, Python 3.11, Ubuntu 22.04
# Competition server: Ubuntu 20.04, CUDA 12.9 — PyTorch 2.3 wheels are
# forward-compatible with CUDA 12.x patch versions.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        libsndfile1 \
        ffmpeg \
        build-essential \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python deps ────────────────────────────────────────────────────────────────

RUN git clone --depth 1 https://github.com/s3prl/s3prl.git /workspace/s3prl \
    && pip install --no-cache-dir -e /workspace/s3prl

RUN pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



RUN pip install s3prl

# ── Clone s3prl ────────────────────────────────────────────────────────────────
# Must live at /workspace/s3prl so that:
#   torch.hub.load('s3prl', 'wavlm_large', source='local')
# resolves correctly — source='local' looks for the repo name in cwd.
#
# We also install s3prl as a package so its Python imports work.


# ── Copy source code ───────────────────────────────────────────────────────────
# unispeech/ must be in your repo root — contains ECAPA_TDNN_SMALL
COPY src/              src/
COPY unispeech/        unispeech/
COPY infer.py          .
COPY download_models.py .


ARG FEAT_TYPE=wavlm_large
ENV FEAT_TYPE=${FEAT_TYPE}

# torch.hub cache — stays in image
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# Run from /workspace so that source='local' finds ./s3prl

RUN pip install --no-cache-dir gdown
RUN python download_models.py
ENV CHECKPOINT_ID=1NYnJkkhMHsC5DjHFzJ1IrKqO7DItA7Yt
RUN mkdir -p /workspace/checkpoints && \
    gdown ${CHECKPOINT_ID} -O /workspace/checkpoints/model.pt


# ── Runtime env ────────────────────────────────────────────────────────────────
# Override these with -e at docker run, or pass as --flags to infer.py.

ENV TEST_CSV=/data/test_public.csv
ENV DATA_DIR=/data
ENV MODEL_PATH=/checkpoints/model.pt

# ── Entry point ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "infer.py"]
CMD ["--help"]
