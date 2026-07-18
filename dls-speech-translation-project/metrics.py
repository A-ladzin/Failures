# metrics.py

import torch
import torchaudio
import numpy as np
from speechbrain.inference.classifiers import EncoderClassifier

class SpeakerSimilarity:
    def __init__(self, device: str = None):
        """
        - device: "cuda" or "cpu". If None, it auto-detects.
        """
        # 1) Figure out device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2) Load a pretrained speaker-recognition model (ECAPA-TDNN in this example)
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )

        # 3) Get the target sample rate for that model (usually 16 kHz for ECAPA-TDNN)
        self.expected_sample_rate = 16000  # ECAPA-TDNN defaults to 16 kHz

    def embed(self, wav_path: str) -> np.ndarray:
        """
        Loads `wav_path`, resamples if needed, then returns a 1-D embedding vector.
        """
        # 1) Load waveform
        signal, sr = torchaudio.load(wav_path)  # signal: [channels, time]; sr: sample rate

        # 2) If multi-channel, average to mono
        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        # 3) Resample if the file's sr != what the model expects
        if sr != self.expected_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.expected_sample_rate)
            signal = resampler(signal)
            sr = self.expected_sample_rate

        # 4) Move signal to correct device
        signal = signal.to(self.device)

        # 5) In SpeechBrain’s API, `encode_batch(...)` expects a [batch, samples]-shaped tensor
        #    But right now `signal` is [1, num_samples]. That’s fine: it treats the first dimension as “channels.”
        #    We want [batch=1, time], so we should remove the channel dimension if it's exactly 1:
        signal = signal.squeeze(0).unsqueeze(0)  # → shape: [1, num_samples]

        # 6) Run the model’s encoder. The output is a dict; the actual embeddings live in "embeddings"
        embeddings = self.classifier.encode_batch(signal)  # → torch.Tensor of shape [1, embedding_dim]

        # 7) Move back to CPU, detach, convert to numpy, and flatten
        emb_np = embeddings.squeeze(0).cpu().detach().numpy()  # shape: [embedding_dim]
        return emb_np

    def evaluate(self, orig_wav: str, dubbed_wav: str) -> float:
        """
        Compute cosine similarity between the “original” speaker WAV and the “dubbed” WAV.

        Returns a float in [-1, 1].
        """
        emb_orig   = self.embed(orig_wav)    # shape: [dim]
        emb_dub    = self.embed(dubbed_wav)  # shape: [dim]

        # 8) Cosine similarity
        emb_orig = emb_orig.squeeze()
        emb_dub = emb_dub.squeeze()

        cos_sim = np.dot(emb_orig, emb_dub) / (
            np.linalg.norm(emb_orig) * np.linalg.norm(emb_dub)
        )

        return float(cos_sim)
