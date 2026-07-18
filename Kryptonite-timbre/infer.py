#!/usr/bin/env python3
"""
infer.py — Single-command inference entry point.

Uses ECAPA_TDNN_SMALL with wavlm_large backbone (via s3prl).

Full pipeline:
  1. Load ECAPA_TDNN_SMALL checkpoint
  2. Extract speaker embeddings with multi-chunk averaging + VAD
  3. Apply Query Expansion (--qe, on by default)
  4. Write submission.csv

Usage (single required parameter):
    python infer.py --k 10

All other args have sensible defaults or fall back to environment variables
set in the Dockerfile (TEST_CSV, DATA_DIR, MODEL_PATH).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import random
from typing import List, Optional, Tuple,Union,Any

import faiss
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import SpeakerDataset
from src.metrics import _l2_normalize_rows, precision_at_k


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / norm


def compute_k_recall(topk: int, n: int) -> int:
    if topk <= 10:
        k_recall = 100
    elif topk <= 100:
        k_recall = min(topk * 10,500)
    elif topk <= 500:
        k_recall = min(topk * 5,1500)
    else:
        k_recall = topk * 3

    return min(k_recall, max(n - 1, 1))


def save_topk_indices_csv_fused(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    filepaths: List[str],
    out_csv: str,
    topk: int = 10,
    w1: float = 0.5,
    w2: float = 0.5,
) -> None:
    emb1 = _l2_normalize(np.asarray(embeddings1, dtype=np.float32))
    emb2 = _l2_normalize(np.asarray(embeddings2, dtype=np.float32))

    N = emb1.shape[0]

    # адаптивный recall pool
    k_recall = compute_k_recall(topk, N)

    index1 = faiss.IndexFlatIP(emb1.shape[1])
    index2 = faiss.IndexFlatIP(emb2.shape[1])

    index1.add(emb1)
    index2.add(emb2)

    # +1 потому что self-match почти всегда попадёт в top-1
    search_k = min(k_recall + 1, N)
    _, I1 = index1.search(emb1, search_k)
    _, I2 = index2.search(emb2, search_k)

    I1 = I1.astype(np.int64)
    I2 = I2.astype(np.int64)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "neighbours"])

        for i in range(N):
            cand = set(I1[i].tolist()) | set(I2[i].tolist())
            cand.discard(i)

            if not cand:
                writer.writerow([filepaths[i], ""])
                continue

            cand = np.array(list(cand), dtype=np.int64)

            scores1 = emb1[cand] @ emb1[i]
            scores2 = emb2[cand] @ emb2[i]
            scores = w1 * scores1 + w2 * scores2

            kk = min(topk, len(cand))
            top_idx = np.argsort(-scores)[:kk]
            top_neighbors = cand[top_idx]

            writer.writerow([filepaths[i], ",".join(map(str, top_neighbors.tolist()))])

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, feat_type: str = "wavlm_large",
               device: str = "cuda") -> torch.nn.Module:
    """
    Load ECAPA_TDNN_SMALL with s3prl wavlm_large backbone.

    torch.hub.load(..., source='local') requires /workspace/s3prl to exist —
    the Dockerfile clones it there at build time.
    """
    from unispeech.models.escapa_tdnn_p import ECAPA_TDNN_SMALL

    print(f"[Model] Initialising ECAPA_TDNN_SMALL  feat_type={feat_type} ...")
    model = ECAPA_TDNN_SMALL(
        feat_dim=1024,
        feat_type=feat_type,
        config_path=None,
    ).to(device)

    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and wrapped checkpoints
    state = cp.get("model", cp)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[Model] Loaded — parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def load_audio(filepath: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)


def get_chunk(
    waveform: Union[torch.Tensor, np.ndarray],
    chunk_len: Union[Tuple[int, int], int],
    random_chunk: bool = True,
):
    """Get random chunk

    Args:
        waveform: torch.Tensor (1, samples) or (samples, )
        chunk_len: either a specific chunk length or a range within which the chunk length falls
        random_chunk: whether to take a random chunk or not
    Returns:
        torch.Tensor (1, exactly chunk_len) or (exactly chunk_len, )
    """
    ndim = len(waveform.shape)
    is_torch = False

    # first, convert to 1-dim numpy.array (to not confuse tile, repeat and so on)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().numpy()
        is_torch = True

    if ndim == 2:
        # squeeze it for now
        waveform = waveform[0]

    data_len = len(waveform)

    if isinstance(chunk_len, Tuple):
        chunk_len = int(np.random.uniform(*chunk_len))

    if data_len >= chunk_len:
        chunk_start = 0
        if random_chunk:
            chunk_start = random.randint(0, data_len - chunk_len)
        waveform = waveform[chunk_start : chunk_start + chunk_len]

    elif data_len > 0:
        repeat_factor = chunk_len // data_len + 1
        waveform = np.tile(waveform, repeat_factor)
        waveform = waveform[:chunk_len]
    else:
        print("Trying to pad an audio of zero length.")
        waveform = np.zeros(chunk_len, dtype=np.float32)

    if ndim == 2:
        waveform = waveform[None, :]

    if is_torch:
        waveform = torch.tensor(waveform)

    return waveform





# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    csv_path: str,
    sample_rate: int,
    batch_size: int,
    num_workers: int,
    device: str,
    base_dir: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        emb_np:    (N, D) float32 L2-normalised embeddings, one per utterance
        labels_np: (N,)   int64 speaker labels (-1 if no speaker_id column)
        filepaths: list of filepath strings in CSV order
    """

    ds = SpeakerDataset(csv_path, sample_rate, base_dir=base_dir,chunk_seconds=6.,is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device == "cuda"))

    model.eval()
    all_embl =[]
    all_embe = []
    all_lab = []

    with torch.no_grad():
        for i,(wave, lab) in enumerate(tqdm(loader, total=len(loader))):
            wave = wave.cuda()
            embl,embe = model(wave)
            all_embl.append(embl.detach().cpu().numpy())
            all_embe.append(embe.detach().cpu().numpy())
            all_lab.append(lab.numpy())

    embl_np    = np.concat(all_embl)
    embe_np    = np.concat(all_embe)
    labels_np = np.concat(all_lab)
    filepaths = ds.df[ds.file_col].astype(str).tolist()

    # print(f"[Embed] {emb_np.shape[0]} utterances × {emb_np.shape[1]}-dim")
    return embl_np,embe_np, labels_np, filepaths




# ─────────────────────────────────────────────────────────────────────────────
# Submission writer
# ─────────────────────────────────────────────────────────────────────────────

def write_submission(embeddings: np.ndarray, filepaths: List[str],
                     out_csv: str, topk: int) -> None:
    emb   = _l2_normalize_rows(embeddings.astype(np.float32))
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    _, I  = index.search(emb, topk + 1)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "neighbours"])
        for i, (fp, row) in enumerate(zip(filepaths, I)):
            nbrs = [int(x) for x in row if x >= 0 and x != i][:topk]
            w.writerow([fp, ",".join(str(x) for x in nbrs)])

    print(f"[✓] Submission → {out_csv}   ({len(filepaths)} rows, top-{topk})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Speaker ID inference — outputs submission.csv"
    )

    # ── The one required parameter ────────────────────────────────────────────
    parser.add_argument(
        "--k", type=int, required=True,
        help="Top-K neighbours per utterance written to submission.csv",
    )

    # ── Paths — all fall back to env vars set in Dockerfile ──────────────────
    parser.add_argument("--test_csv",    type=str,
                        default=os.environ.get("TEST_CSV",    "data/test_public.csv"))
    parser.add_argument("--data_dir",    type=str,
                        default=os.environ.get("DATA_DIR",    "data"))
    parser.add_argument("--model_path",  type=str,
                        default=os.environ.get("MODEL_PATH",  "checkpoints/model.pt"))

    parser.add_argument("--out",         type=str, default="submission.csv")

    # ── Model settings ────────────────────────────────────────────────────────
    parser.add_argument("--feat_type",   type=str, default="wavlm_large",
                        help="s3prl feature extractor name (default: wavlm_large)")

    # ── Inference knobs ───────────────────────────────────────────────────────
    parser.add_argument("--sample_rate",   type=int,   default=16000)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--device",        type=str,   default="cuda")

    args = parser.parse_args()

    # ── Resolve device ────────────────────────────────────────────────────────
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else "cpu" if args.device == "auto" else args.device
    )

    print(f"\n{'='*58}")
    print(f"  Speaker ID Inference — top-{args.k} neighbours")
    print(f"{'='*58}\n")

    t0 = time.perf_counter()

    # ── Validate required files ───────────────────────────────────────────────
    for label, path in [("--test_csv",   args.test_csv),
                        ("--model_path", args.model_path)]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    if not os.path.isdir("s3prl"):
        sys.exit(
            "[ERROR] ./s3prl directory not found.\n"
            "  Inside Docker this is pre-cloned at build time.\n"
            "  Locally: git clone https://github.com/s3prl/s3prl.git"
        )

    # ── Load model ────────────────────────────────────────────────────────────
    print("[1/4] Loading model ...")
    # import nemo.collections.asr as nemo_asr
    model = load_model(args.model_path, feat_type=args.feat_type, device=device)

    # ── Extract embeddings ────────────────────────────────────────────────────
    base_dir = args.data_dir or os.path.dirname(os.path.abspath(args.test_csv))
    embl,embe, labels, filepaths = extract_embeddings(
        model, args.test_csv, args.sample_rate,
        args.batch_size, args.num_workers, device, base_dir,
    )

    # ── Write submission ──────────────────────────────────────────────────────
    print(f"\n[4/4] Writing submission (top-{args.k}) ...")
    save_topk_indices_csv_fused(embl,embe, filepaths, args.out, topk=args.k,w1=0.7,w2=0.3)

    # ── Optional val metric ───────────────────────────────────────────────────
    if (labels != -1).any():
        metrics = precision_at_k(embl, labels, ks=(args.k,))
        print(f"\n[VAL] P@{args.k} = {metrics[f'precision@{args.k}']:.4f}")
        metrics = precision_at_k(embe, labels, ks=(args.k,))
        print(f"\n[VAL] P@{args.k} = {metrics[f'precision@{args.k}']:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"\n[✓] Done in {elapsed:.1f}s  →  {args.out}\n")


if __name__ == "__main__":
    main()
