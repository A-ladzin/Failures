"""
augment.py — Optimized waveform and spectrogram augmentations.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _signal_power(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """RMS power per sample in a (B, T) batch."""
    return x.square().mean(dim=-1, keepdim=True).clamp_min_(eps)


def _mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add `noise` to `signal` at the requested SNR (dB)."""
    sig_p  = _signal_power(signal)
    noi_p  = _signal_power(noise)
    target = sig_p / (10 ** (snr_db / 10.0))
    scale  = (target / noi_p).sqrt()
    return signal + noise * scale



# # ─────────────────────────────────────────────────────────────────────────────
# # 6. Hardware & Network Artifacts
# # ─────────────────────────────────────────────────────────────────────────────


class TransmissionArtifactAugment(nn.Module):
    def __init__(
        self,
        clip_prob: float = 0.3,
        packet_loss_prob: float = 0.2,
        clipping_threshold_range: tuple = (0.3, 0.8),
        max_mask_length: int = 480, # ~30ms at 16kHz
    ):
        super().__init__()
        self.clip_prob = clip_prob
        self.packet_loss_prob = packet_loss_prob
        self.clipping_threshold_range = clipping_threshold_range
        self.max_mask_length = max_mask_length

    def _apply_clipping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purely vectorized clipping. No loops, no silencing bugs.
        x: (B_subset, T)
        """
        B_sub = x.size(0)
        low, high = self.clipping_threshold_range
        
        # Generate independent thresholds for each sample in the subset
        # Shape: (B_sub, 1) for broadcasting
        thresholds = low + (high - low) * torch.rand(B_sub, 1, device=x.device)
        
        # Determine the peak of each sample to ensure we clip relative to volume
        peaks = x.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-9)
        
        # target_clip is a value between (low * peak) and (high * peak)
        target_clip = peaks * thresholds
        
        # Perform vectorized clamp across the whole subset
        return torch.clamp(x, min=-target_clip, max=target_clip)

    def _apply_packet_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized burst packet loss using a binary mask.
        x: (B_subset, T)
        """
        B_sub, T = x.shape
        
        # 1. Randomize lengths and starts for every sample in the subset at once
        lengths = torch.randint(80, self.max_mask_length, (B_sub,), device=x.device)
        starts = torch.randint(0, max(1, T - self.max_mask_length), (B_sub,), device=x.device)
        ends = starts + lengths
        
        # 2. Create a coordinate grid (B_sub, T)
        # grid[b, t] = t
        grid = torch.arange(T, device=x.device).unsqueeze(0).expand(B_sub, -1)
        
        # 3. Create the mask: True where the time index is within the burst loss window
        # We use unsqueeze(1) to broadcast (B_sub,) against (B_sub, T)
        mask = (grid >= starts.unsqueeze(1)) & (grid < ends.unsqueeze(1))
        
        # 4. Zero out the signal where mask is True (inverted logic for faster multiplication)
        return x.masked_fill(mask, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        B = x.size(0)
        
        # --- Clipping (Hardware Distortion) ---
        clip_mask = torch.rand(B, device=x.device) < self.clip_prob
        clip_idx = clip_mask.nonzero(as_tuple=True)[0]
        if clip_idx.numel() > 0:
            x[clip_idx] = self._apply_clipping(x[clip_idx])

        # --- Packet Loss (Network Issues) ---
        loss_mask = torch.rand(B, device=x.device) < self.packet_loss_prob
        loss_idx = loss_mask.nonzero(as_tuple=True)[0]
        if loss_idx.numel() > 0:
            x[loss_idx] = self._apply_packet_loss(x[loss_idx])

        return x

# ─────────────────────────────────────────────────────────────────────────────
# 1. Colored noise
# ─────────────────────────────────────────────────────────────────────────────

class ColoredNoiseAugment(nn.Module):
    def __init__(
        self,
        prob: float = 0.7,
        snr_range: tuple = (5.0, 35.0),
        exp_range: tuple = (0.0, 2.0),
    ):
        super().__init__()
        self.prob      = prob
        self.snr_range = snr_range
        self.exp_range = exp_range

    def _colored_noise(self, shape: tuple, exponent: float, device: torch.device) -> torch.Tensor:
        B, T = shape
        white = torch.randn(B, T, device=device)

        if exponent == 0.0:
            return white

        freqs = torch.fft.rfftfreq(T, device=device).clamp_min_(1e-6)
        filter_ = freqs.pow(-exponent / 2.0)
        filter_[0] = 0.0  

        spectrum = torch.fft.rfft(white) * filter_
        return torch.fft.irfft(spectrum, n=T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(x.shape[0], device=x.device) < self.prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return x
        
        waveform = x[indices].clone()

        snr = random.uniform(*self.snr_range)
        exp = random.uniform(*self.exp_range)
        noise = self._colored_noise(waveform.shape, exp, waveform.device)
        waveform = _mix_at_snr(waveform, noise, snr)
        x[indices] = waveform
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic RIR (Fast FFT Convolve)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticRIRAugment(nn.Module):
    def __init__(
        self,
        prob: float = 0.6,
        rt60_range: tuple = (0.1, 1.0),
        sample_rate: int = 16000,
        rir_length_s: float = 1.0,
        drr_range: tuple = (0.0, 10.0),
    ):
        super().__init__()
        self.prob        = prob
        self.rt60_range  = rt60_range
        self.sample_rate = sample_rate
        self.rir_len     = int(rir_length_s * sample_rate)
        self.drr_range   = drr_range

    # def _make_rir(self, rt60: float, device: torch.device) -> torch.Tensor:
    #     t = torch.arange(self.rir_len, device=device, dtype=torch.float32)
    #     decay = 3.0 * math.log(10.0) / (rt60 * self.sample_rate)
    #     envelope = torch.exp(-decay * t)
    #     noise = torch.randn(self.rir_len, device=device)
    #     rir = noise * envelope
    #     return rir / rir.abs().max().clamp_min_(1e-9)
    
    def _make_rir(self, rt60, device):
        # Sampling independent decay for each batch element
        t = torch.arange(self.rir_len, device=device).float()
        # rt60 is now a tensor of shape (B, 1)
        decay = 3.0 * math.log(10.0) / (rt60 * self.sample_rate)
        envelope = torch.exp(-decay * t) # (B, rir_len)
        noise = torch.randn_like(envelope)
        rir = noise * envelope
        return rir / rir.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        
        apply_mask = torch.rand(x.shape[0], device=x.device) < self.prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return x
        
        waveform = x[indices].clone()
        
        B, T   = waveform.shape
        # rt60   = random.uniform(*self.rt60_range)
        # drr_db = random.uniform(*self.drr_range)

        rt60 = torch.empty(B, 1, device=x.device).uniform_(*self.rt60_range)
        drr_db = torch.empty(B, 1, device=x.device).uniform_(*self.drr_range)

        # Generate RIR and use O(N log N) FFT convolution instead of slow spatial conv1d
        # rir = self._make_rir(rt60, waveform.device).unsqueeze(0)  # (1, L)
        rir = self._make_rir(rt60, waveform.device) # (B, L)
        reverbed = AF.fftconvolve(waveform, rir, mode="full")[..., :T]

        dry_p  = _signal_power(waveform)
        wet_p  = _signal_power(reverbed)
        target = dry_p / (10 ** (drr_db / 10.0))
        scale  = (target / wet_p).sqrt()
        out    = waveform + reverbed * scale

        peak = out.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-9)

        waveform = out / peak * waveform.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-9)

        x[indices] = waveform
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Telephone / codec simulation (Cached Filters)
# ─────────────────────────────────────────────────────────────────────────────

class TelephoneAugment(nn.Module):
    def __init__(
        self,
        bandpass_prob: float = 0.4,
        resample_prob: float = 0.4,
        sample_rate: int = 16000,
        low_cut_range: tuple = (100.0, 300.),
        high_cut_range: tuple = (900, 3800.0),
    ):
        super().__init__()
        self.bandpass_prob  = bandpass_prob
        self.resample_prob  = resample_prob
        self.sample_rate    = sample_rate
        self.low_cut_range  = low_cut_range
        self.high_cut_range = high_cut_range
        
        # Precompute anti-aliasing kernels by using stateful transforms
        self.downsample = AT.Resample(sample_rate, 8000)
        self.upsample = AT.Resample(8000, sample_rate)

    def _bandpass(self, waveform: torch.Tensor) -> torch.Tensor:
        low_hz   = torch.tensor([random.uniform(*self.low_cut_range) for i in range(waveform.shape[0])])
        high_hz  = torch.tensor([random.uniform(*self.high_cut_range) for i in range(waveform.shape[0])])
        T = waveform.shape[-1]

        spectrum = torch.fft.rfft(waveform)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate, device=waveform.device)
        mask     = torch.stack([((freqs <= high_hz[i]) & (freqs[i] >= low_hz[i])).float() for i in range(waveform.shape[0])])
        return torch.fft.irfft(spectrum * mask, n=T)

    def _resample_codec(self, waveform: torch.Tensor) -> torch.Tensor:
        T = waveform.shape[-1]
        up = self.upsample(self.downsample(waveform))
        
        if up.shape[-1] > T:
            return up[..., :T]
        elif up.shape[-1] < T:
            return F.pad(up, (0, T - up.shape[-1]))
        return up

    def forward(self, x: torch.Tensor,is_lowpass) -> torch.Tensor:
        # if is_lowpass:
        #     return waveform
        #     return waveform



        apply_mask = (torch.rand(x.shape[0], device=x.device) < self.bandpass_prob)* (~is_lowpass)
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() > 0:
            waveform = x[indices].clone()
            waveform = self._bandpass(waveform)
            x[indices] = waveform


        apply_mask = torch.rand(x.shape[0], device=x.device) < self.resample_prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return x
        
        waveform = x[indices].clone()
        waveform = self._resample_codec(waveform)

        x[indices] = waveform
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. Volume perturbation
# ─────────────────────────────────────────────────────────────────────────────

class VolumeAugment(nn.Module):
    def __init__(self, prob: float = 0.5, gain_range: tuple = (0.6, 1.4)):
        super().__init__()
        self.prob       = prob
        self.gain_range = gain_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        apply_mask = torch.rand(x.shape[0], device=x.device) < self.prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return x
        
        waveform = x[indices].clone()
        waveform = waveform * random.uniform(*self.gain_range)
        x[indices] = waveform
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Low-pass filter augmentation
# ─────────────────────────────────────────────────────────────────────────────

class LowPassAugment(nn.Module):
    def __init__(
        self,
        prob: float = 0.6,
        cutoff_range_hz: tuple = (300.0, 1800.0),
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.prob            = prob
        self.cutoff_range_hz = cutoff_range_hz
        self.sample_rate     = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if random.random() > self.prob:
        #     return waveform,False

        apply_mask = torch.rand(x.shape[0], device=x.device) < self.prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return x,apply_mask
        
        waveform = x[indices].clone()

        T = waveform.shape[-1]
        cutoff_hz = torch.tensor([random.uniform(*self.cutoff_range_hz) for i in range(waveform.shape[0])])

        spectrum = torch.fft.rfft(waveform)
        freqs    = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate, device=waveform.device)

        mask     = torch.stack([(freqs <= cutoff_hz[i]).float() for i in range(waveform.shape[0])])
        filtered = torch.fft.irfft(spectrum * mask, n=T)

        orig_rms = _signal_power(waveform).sqrt()
        filt_rms = _signal_power(filtered).sqrt()
        scale    = orig_rms / filt_rms.clamp_min_(1e-9)

        waveform = filtered * scale
        x[indices] = waveform


        return x,apply_mask


# ─────────────────────────────────────────────────────────────────────────────
# 5. SpecAugment (Fully Vectorized)
# ─────────────────────────────────────────────────────────────────────────────

class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_masks: int = 2,
        time_masks: int = 2,
        freq_width: int = 27,
        time_width: int = 40,
        p: float = 1.0,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x of shape (B, T, F). Torchaudio mask_along_axis_iid applies independent
        masks to every item in the batch using optimized C++.
        """

            
        out = x.clone()
        
        # Axis 2 is frequency (F), Axis 1 is time (T)
        for _ in range(self.freq_masks):
            out = AF.mask_along_axis_iid(out, self.freq_width, 0.0, 2,self.p)
            
        for _ in range(self.time_masks):
            out = AF.mask_along_axis_iid(out, self.time_width, 0.0, 1,self.p)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Full waveform augmentation chain
# ─────────────────────────────────────────────────────────────────────────────

class WaveformAugment(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_prob: float = 0.2,
        noise_snr_db: tuple = (25.0, 50.0),
        reverb_prob: float = 0.6,
        rt60_range: tuple = (0.3, 1.2),
        lp_prob: float = 0.6,
        lp_cutoff_range_hz: tuple = (300.0, 4000.0),
        bandpass_prob: float = 0.15,
        resample_prob: float = 0.2,
        volume_prob: float = 0.4,
        volume_range: tuple = (0.7, 1.3),
        clip_prob = 0.25,
        packet_loss_prob = 0.15,
        bp_lowcut_range = (100,300),
        bp_highcut_range = (900,3800)
    ):
        super().__init__()
        self.noise     = ColoredNoiseAugment(prob=noise_prob, snr_range=noise_snr_db)
        self.reverb    = SyntheticRIRAugment(prob=reverb_prob, rt60_range=rt60_range,
                                             sample_rate=sample_rate)
        self.lowpass   = LowPassAugment(prob=lp_prob,
                                        cutoff_range_hz=lp_cutoff_range_hz,
                                        sample_rate=sample_rate)
        self.telephone = TelephoneAugment(bandpass_prob=bandpass_prob,
                                          resample_prob=resample_prob,
                                          sample_rate=sample_rate,
                                          low_cut_range=bp_lowcut_range,
                                          high_cut_range=bp_highcut_range)
        self.volume    = VolumeAugment(prob=volume_prob, gain_range=volume_range)
        self.artifacts = TransmissionArtifactAugment(clip_prob=clip_prob, packet_loss_prob=packet_loss_prob)
        self.energy_drop_threshold = 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)).clamp_min(1e-9)
        waveform = x.clone()
        waveform = self.noise(waveform)
        waveform = self.reverb(waveform)
        waveform,is_lowpass = self.lowpass(waveform)
        waveform = self.artifacts(waveform)
        waveform = self.telephone(waveform,is_lowpass)
        waveform = self.volume(waveform)
        out_rms = torch.sqrt(waveform.pow(2).mean(dim=-1, keepdim=True))
        energy_ratio = out_rms / in_rms
        is_dead = (energy_ratio < self.energy_drop_threshold)
        waveform = torch.where(is_dead, x, waveform)
        return waveform