import os
import random
from typing import Tuple, Union
import numpy as np
import pandas as pd
import torch
import torchaudio

from torch.utils.data import Dataset

from src.metrics import _l2_normalize_rows
import faiss

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


class SpeakerDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int,
        chunk_seconds: float,
        is_train: bool,
        base_dir: str | None = None,
        filepath_col: str = "filepath",
        speaker_id_col: str = "speaker_id",
    ):
        self.df = pd.read_csv(csv_path)
        self.file_col = filepath_col
        self.spk_col = speaker_id_col
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * float(chunk_seconds))
        self.is_train = is_train
        self.base_dir = base_dir

        assert (
            self.file_col in self.df.columns
        ), f"Column '{self.file_col}' not found in CSV"
        self.has_sid = self.spk_col in self.df.columns
        if self.has_sid:
            self.df[self.spk_col] = self.df[self.spk_col].astype(str)

        if self.has_sid:
            speakers = sorted(self.df[self.spk_col].unique())
            self.speaker_to_label = {spk: i for i, spk in enumerate(speakers)}
        else:
            self.speaker_to_label = {}
        self.num_speakers = len(self.speaker_to_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row[self.file_col]
        if not os.path.isabs(path) and self.base_dir:
            path = os.path.normpath(os.path.join(self.base_dir, path))

        try:
            waveform = load_audio(path, self.sample_rate)
        except Exception as e:
            print("BAD FILE:", path, e)
            return self.__getitem__(random.randint(0,len(self.df)-1))
        waveform = get_chunk(waveform, self.num_samples,self.is_train)
        if self.has_sid:
            spk = row[self.spk_col]
            label = self.speaker_to_label[spk]
        else:
            label = -1
        return waveform, label
    




class SpeakerDatasetMerge(Dataset):
    def __init__(
        self,
        csv_path: str,
        train_path: str,
        val_path: str,
        sample_rate: int,
        chunk_seconds: float,
        is_train: bool,
        base_dir: str | None = None,
        filepath_col: str = "filepath",
        speaker_id_col: str = "speaker_id",
    ):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        self.df = pd.read_csv(csv_path)
        self.file_col = filepath_col
        self.spk_col = speaker_id_col
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * float(chunk_seconds))
        self.is_train = is_train
        self.base_dir = base_dir

        assert (
            self.file_col in self.df.columns
        ), f"Column '{self.file_col}' not found in CSV"
        self.has_sid = self.spk_col in self.df.columns

        speakers_train = sorted(train_df[self.spk_col].unique())
        speaker_val = sorted(val_df[self.spk_col].unique())
        speaker_to_label_train = {spk: i for i, spk in enumerate(speakers_train)}
        speaker_to_label_val = {spk: i for i, spk in enumerate(speaker_val,start = len(speakers_train))}

        
        self.speaker_to_label = {**speaker_to_label_train, **speaker_to_label_val}
        
        self.num_speakers = len(self.speaker_to_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row[self.file_col]
        if not os.path.isabs(path) and self.base_dir:
            path = os.path.normpath(os.path.join(self.base_dir, path))

        waveform = load_audio(path, self.sample_rate)
        waveform = get_chunk(waveform, self.num_samples,self.is_train)
        if self.has_sid:
            spk = row[self.spk_col]
            label = self.speaker_to_label[spk]
        else:
            label = -1
        return waveform, label




# class MultiChunkDataset(Dataset):
#     """
#     Loads each utterance in full and splits it into overlapping chunks.

#     For each file, returns a (N_chunks, T) tensor where N_chunks is the number
#     of non-overlapping windows extracted. Embeddings are averaged at inference.

#     This is the single biggest improvement for retrieval P@K:
#     - Single 6s chunk:         one noisy estimate of the speaker embedding
#     - Average of N 3s chunks:  variance reduced by ~sqrt(N), cleaner centroid
#     """

#     def __init__(
#         self,
#         csv_path: str,
#         sample_rate: int,
#         chunk_seconds: float,
#         base_dir: str = "",
#         filepath_col: str = "filepath",
#         speaker_id_col: str = "speaker_id",
#         min_chunks: int = 1,
#     ):
#         self.df          = pd.read_csv(csv_path)
#         self.file_col    = filepath_col
#         self.spk_col     = speaker_id_col
#         self.sample_rate = sample_rate
#         self.chunk_len   = int(sample_rate * chunk_seconds)
#         self.base_dir    = base_dir
#         self.min_chunks  = min_chunks

#         self.has_sid = self.spk_col in self.df.columns
#         if self.has_sid:
#             speakers = sorted(self.df[self.spk_col].unique())
#             self.speaker_to_label = {s: i for i, s in enumerate(speakers)}
#         else:
#             self.speaker_to_label = {}

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row  = self.df.iloc[idx]
#         path = row[self.file_col]
#         if not os.path.isabs(path) and self.base_dir:
#             path = os.path.normpath(os.path.join(self.base_dir, path))

#         waveform = load_audio(path, self.sample_rate)  # (T,)
#         T = len(waveform)

#         # Split into non-overlapping chunks of chunk_len
#         # If the file is shorter than one chunk, pad and return 1 chunk
#         if T < self.chunk_len:
#             pad = torch.zeros(self.chunk_len)
#             pad[:T] = waveform
#             chunks = pad.unsqueeze(0)  # (1, chunk_len)
#         else:
#             n = T // self.chunk_len
#             chunks = waveform[:n * self.chunk_len].view(n, self.chunk_len)

#         label = -1
#         if self.has_sid:
#             label = self.speaker_to_label.get(row[self.spk_col], -1)

#         return chunks, label  # (N, T), int
    
# def collate_multichunk(batch):
#     """Each item has variable N chunks — keep them separate, track boundaries."""
#     all_chunks, all_labels, boundaries = [], [], [0]
#     for chunks, label in batch:
#         all_chunks.append(chunks)
#         all_labels.append(label)
#         boundaries.append(boundaries[-1] + chunks.shape[0])
#     return torch.cat(all_chunks, dim=0), torch.tensor(all_labels), boundaries

# ─────────────────────────────────────────────────────────────────────────────
# Multi-chunk dataset with VAD
# ─────────────────────────────────────────────────────────────────────────────
 

 # ─────────────────────────────────────────────────────────────────────────────
# Optional VAD  (silero-vad)
# ─────────────────────────────────────────────────────────────────────────────
 
def load_vad_model():
    """Load silero-VAD. Returns None if not installed."""
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        get_speech_ts = utils[0]
        return model, get_speech_ts
    except Exception as e:
        print(f"[VAD] silero-vad not available ({e}). Running without VAD.")
        return None, None
 
 
def strip_silence_vad(waveform: torch.Tensor, sr: int,
                      vad_model, get_speech_ts,
                      min_speech_s: float = 0.5) -> torch.Tensor:
    """
    Remove non-speech frames using silero-VAD.
 
    EDA shows test files have 27% silence vs 12% in train.
    Silence frames carry no speaker information and dilute the embedding
    by pushing it toward a "silence centroid" rather than the speaker centroid.
 
    Falls back to full waveform if:
      - VAD not available
      - Less than min_speech_s of speech detected (likely a bad file)
    """
    if vad_model is None:
        return waveform
 
    # silero-vad expects 1D tensor at 16kHz
    w = waveform.squeeze()
    try:
        speech_ts = get_speech_ts(w, vad_model, sampling_rate=sr,
                                  threshold=0.5, min_speech_duration_ms=200)
        if not speech_ts:
            return waveform  # no speech detected — keep original
 
        speech_frames = [w[s["start"]: s["end"]] for s in speech_ts]
        speech_only   = torch.cat(speech_frames)
 
        if len(speech_only) / sr < min_speech_s:
            return waveform  # too short after VAD — keep original
 
        return speech_only
    except Exception:
        return waveform
 
 
 
class MultiChunkDataset(Dataset):
    """
    Loads each utterance, optionally strips silence via VAD,
    then splits into non-overlapping chunks for averaging.
 
    EDA findings applied:
      - VAD: test has 27% silence — strip it before chunking
      - Chunk size 3s: test median=6.9s → ~2 chunks per file on average
    """
 
    def __init__(
        self,
        csv_path: str,
        sample_rate: int,
        chunk_seconds: float,
        base_dir: str = "",
        filepath_col: str = "filepath",
        speaker_id_col: str = "speaker_id",
        vad_model=None,
        get_speech_ts=None,
    ):
        self.df          = pd.read_csv(csv_path)
        self.file_col    = filepath_col
        self.spk_col     = speaker_id_col
        self.sample_rate = sample_rate
        self.chunk_len   = int(sample_rate * chunk_seconds)
        self.base_dir    = base_dir
        self.vad_model   = vad_model
        self.get_speech_ts = get_speech_ts
 
        self.has_sid = self.spk_col in self.df.columns
        if self.has_sid:
            speakers = sorted(self.df[self.spk_col].unique())
            self.speaker_to_label = {s: i for i, s in enumerate(speakers)}
        else:
            self.speaker_to_label = {}
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row[self.file_col]
        if not os.path.isabs(path) and self.base_dir:
            path = os.path.normpath(os.path.join(self.base_dir, path))
 
        waveform = load_audio(path, self.sample_rate)  # (T,)
 
        # Strip silence — test has 27% silence vs 12% train
        if self.vad_model is not None:
            waveform = strip_silence_vad(
                waveform, self.sample_rate,
                self.vad_model, self.get_speech_ts,
            )
 
        T = len(waveform)
        if T < self.chunk_len:
            pad        = torch.zeros(self.chunk_len)
            pad[:T]    = waveform
            chunks     = pad.unsqueeze(0)
        else:
            n      = T // self.chunk_len
            chunks = waveform[:n * self.chunk_len].view(n, self.chunk_len)
 
        label = -1
        if self.has_sid:
            label = self.speaker_to_label.get(row[self.spk_col], -1)
 
        return chunks, label
 
 
def collate_multichunk(batch):
    all_chunks, all_labels, boundaries = [], [], [0]
    for chunks, label in batch:
        all_chunks.append(chunks)
        all_labels.append(label)
        boundaries.append(boundaries[-1] + chunks.shape[0])
    return torch.cat(all_chunks, dim=0), torch.tensor(all_labels), boundaries




class EmbeddingDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        embeddings_path,
        sample_rate: int,
        chunk_seconds: float,
        is_train: bool,
        base_dir: str | None = None,
        filepath_col: str = "filepath",
        speaker_id_col: str = "speaker_id",
    ):
        self.df = pd.read_csv(csv_path)
        self.file_col = filepath_col
        self.spk_col = speaker_id_col
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * float(chunk_seconds))
        self.is_train = is_train
        self.base_dir = base_dir
        self.embeddings = np.load(embeddings_path)
        assert (
            self.file_col in self.df.columns
        ), f"Column '{self.file_col}' not found in CSV"
        self.has_sid = self.spk_col in self.df.columns

        if self.has_sid:
            speakers = sorted(self.df[self.spk_col].unique())
            self.speaker_to_label = {spk: i for i, spk in enumerate(speakers)}
        else:
            self.speaker_to_label = {}
        self.num_speakers = len(self.speaker_to_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        emb = self.embeddings[idx]
        row = self.df.iloc[idx]
        # path = row[self.file_col]
        # if not os.path.isabs(path) and self.base_dir:
        #     path = os.path.normpath(os.path.join(self.base_dir, path))

        # waveform = load_audio(path, self.sample_rate)
        # waveform = get_chunk(waveform, self.num_samples,self.is_train)
        if self.has_sid:
            spk = row[self.spk_col]
            label = self.speaker_to_label[spk]
        else:
            label = -1
        return emb, label


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """
    Dataset of (query, candidate) pairs for reranker training.

    Pairs are constructed as:
      Positives: same-speaker pairs sampled from the training set.
      Hard negatives: different-speaker pairs that rank in the top-K of
                      either model A or model B (false positives of the
                      base models — the hardest cases for the reranker to learn).

    Args:
        emb_a:          (N, D_a) embeddings from model A, L2-normalised.
        emb_b:          (N, D_b) embeddings from model B, L2-normalised.
        labels:         (N,) integer speaker IDs.
        topk_hard_neg:  Mine negatives from top-K of base model rankings.
        neg_ratio:      Number of negatives per positive pair.
        max_pairs:      Cap total pairs to avoid memory issues on large datasets.
    """

    def __init__(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        labels: np.ndarray,
        topk_hard_neg: int = 50,
        neg_ratio: int = 4,
        max_pairs: int = 1000000,
    ):
        self.emb_a  = torch.from_numpy(emb_a.astype(np.float32))
        self.emb_b  = torch.from_numpy(emb_b.astype(np.float32))
        self.labels = labels

        print("[Reranker] Building pair index ...")
        # pos_pairs, neg_pairs = self._build_pairs(
        #     emb_a, emb_b, labels, topk_hard_neg, neg_ratio, max_pairs
        # )
        pos_pairs = np.load("pos_pairs.npy")
        neg_pairs = np.load("neg_pairs.npy")
        self.pairs  = np.concatenate([pos_pairs, neg_pairs], axis=0)
        self.targets = np.concatenate([
            np.ones(len(pos_pairs),  dtype=np.float32),
            np.zeros(len(neg_pairs), dtype=np.float32),
        ])
        # Shuffle
        perm = np.random.permutation(len(self.pairs))
        self.pairs   = self.pairs[perm]
        self.targets = self.targets[perm]
        print(f"[Reranker] {len(pos_pairs)} positives + {len(neg_pairs)} hard negatives "
              f"= {len(self.pairs)} total pairs")

    @staticmethod
    def _build_pairs(
        emb_a: np.ndarray,
        emb_b: np.ndarray,
        labels: np.ndarray,
        topk: int,
        neg_ratio: int,
        max_pairs: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(labels)

        # Build speaker → indices map
        from collections import defaultdict
        spk_to_idx: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            spk_to_idx[int(lbl)].append(i)

        # Positive pairs: all same-speaker pairs (capped)
        pos_pairs = []
        for idxs in spk_to_idx.values():
            if len(idxs) < 2:
                continue
            for ii in range(len(idxs)):
                for jj in range(ii + 1, len(idxs)):
                    pos_pairs.append((idxs[ii], idxs[jj]))
        random.shuffle(pos_pairs)
        if max_pairs is not None:
            max_pos = max_pairs // (1 + neg_ratio)
            pos_pairs = pos_pairs[:max_pos]
        pos_pairs = np.array(pos_pairs, dtype=np.int64)

        # Hard negatives: top-K neighbours of each utterance from model A
        # that are different speakers (false positives of model A)
        print("[Reranker] Mining hard negatives from model A top-K ...")
        emb_a_norm = _l2_normalize_rows(emb_a.astype(np.float32))
        index_a    = faiss.IndexFlatIP(emb_a_norm.shape[1])
        index_a.add(emb_a_norm)
        _, I_a = index_a.search(emb_a_norm, topk + 1)

        # Also from model B
        print("[Reranker] Mining hard negatives from model B top-K ...")
        emb_b_norm = _l2_normalize_rows(emb_b.astype(np.float32))
        index_b    = faiss.IndexFlatIP(emb_b_norm.shape[1])
        index_b.add(emb_b_norm)
        _, I_b = index_b.search(emb_b_norm, topk + 1)

        neg_set: set[tuple[int, int]] = set()
        pos_set: set[tuple[int, int]] = set(map(tuple, pos_pairs.tolist()))

        for i in range(N):
            for nbr in I_a[i].tolist() + I_b[i].tolist():
                if nbr == i or nbr < 0:
                    continue
                if labels[nbr] != labels[i]:
                    key = (min(i, nbr), max(i, nbr))
                    neg_set.add(key)

        neg_pairs_list = list(neg_set)
        random.shuffle(neg_pairs_list)
        max_neg = len(pos_pairs) * neg_ratio
        neg_pairs = np.array(neg_pairs_list[:max_neg], dtype=np.int64)
        np.save('pos_pairs.npy',pos_pairs)
        np.save('neg_pairs.npy',neg_pairs)
        return pos_pairs, neg_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j     = self.pairs[idx]
        target   = self.targets[idx]
        return (
            self.emb_a[i], self.emb_a[j],
            self.emb_b[i], self.emb_b[j],
            torch.tensor(target),
        )

