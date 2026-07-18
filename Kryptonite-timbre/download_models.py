#!/usr/bin/env python3
"""
download_models.py — Pre-download all model weights at Docker build time.

Called once from the Dockerfile RUN step. Bakes weights into the image
so inference containers start instantly with no network access.

Downloads:
  1. s3prl feature extractor for wavlm_large (~1.8 GB)
     via torch.hub.load('s3prl', 'wavlm_large', source='local')
  2. silero-vad (~3 MB)

Important: s3prl must already be cloned to /workspace/s3prl
before this script runs (the Dockerfile does this).
torch.hub.load with source='local' looks for the repo in the
current working directory, so this script must be run from /workspace.
"""

import os
import sys

print("=" * 60)
print("Pre-downloading model weights for offline inference")
print("=" * 60)

# ── 1. wavlm_large via s3prl ──────────────────────────────────────────────────
# torch.hub.load with source='local' requires the repo to be in cwd.
# It downloads the actual wavlm_large weights to ~/.cache/torch/hub/checkpoints/
FEAT_TYPE = os.environ.get("FEAT_TYPE", "wavlm_large")
print(f"\n[1/2] Pre-loading s3prl feature extractor: {FEAT_TYPE} ...")
print(       "      (downloads ~1.8 GB wavlm_large weights on first call)")

import torch
from unispeech.models.escapa_tdnn_p import ECAPA_TDNN_SMALL
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
# ── 2. silero-vad ─────────────────────────────────────────────────────────────

# This triggers the weight download and caches it.
del model
import gc; gc.collect()
print(f"      [✓] {FEAT_TYPE} weights cached.")
print("\n[✓] All downloads complete.\n")
