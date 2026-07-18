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




import torch
import torch.nn as nn

class HardCompressorAugmentor(nn.Module):
    def __init__(
        self,
        sr = 16000,
        threshold_range=(-35.0, -10.0),
        ratio_range=(8.0, 40.0),
        makeup_gain_range=(1.0, 6.0),
        saturation_range=(1.0, 3.0),
        prob = 0.5,
        device="cuda"
    ):
        super().__init__()
        self.sr = sr
        self.device = device

        self.threshold_range = threshold_range
        self.ratio_range = ratio_range
        self.makeup_gain_range = makeup_gain_range
        self.saturation_range = saturation_range
        self.prob = prob

    def _rand(self, shape, low, high):
        return torch.rand(shape, device=self.device) * (high - low) + low

    def forward(self, waveform):
        """
        x: (B, T) waveform in [-1, 1]
        """


        apply_mask = torch.rand(waveform.shape[0], device=waveform.device) < self.prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return waveform,apply_mask
        
        x = waveform[indices].clone()

        x = x.to(self.device)

        B, T = x.shape
        eps = 1e-7

        # === per-sample params ===
        threshold = self._rand((B, 1), *self.threshold_range)
        ratio = self._rand((B, 1), *self.ratio_range)
        makeup = self._rand((B, 1), *self.makeup_gain_range)
        drive = self._rand((B, 1), *self.saturation_range)

        # === convert to log domain ===
        x_abs = torch.clamp(x.abs(), eps)
        x_db = 20.0 * torch.log10(x_abs)

        # === compression curve (hard knee) ===
        over = x_db - threshold

        gain_db = torch.where(
            over > 0,
            -over * (1.0 - 1.0 / ratio),
            torch.zeros_like(over)
        )

        # === back to linear gain ===
        gain = torch.pow(10.0, gain_db / 20.0)

        x = x * gain

        # === makeup gain ===
        x = x * makeup

        # === soft saturation (cheap limiter effect) ===
        x = torch.tanh(x * drive)
        waveform[indices] = x

        return waveform,apply_mask
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AnalogNoiseAugmentorTorch(nn.Module):
    def __init__(
        self,
        sr=16000,
        noise_level_range=(0.001, 0.01),
        crackle_prob_range=(0.0005, 0.003),
        wow_freq_range=(0.1, 0.5),
        wow_depth_range=(0.0005, 0.003),
        flutter_freq_range=(4.0, 10.0),
        flutter_depth_range=(0.0002, 0.0007),
        saturation_drive_range=(1.0, 1.5),
        lowpass_cutoff_range=(2000, 12000),
        dropout_prob=0.3,
        mode="tape",
        device="cuda",
        prob = 0.3
    ):
        super().__init__()
        self.sr = sr
        self.mode = mode
        self.device = device

        # ranges
        self.noise_level_range = noise_level_range
        self.crackle_prob_range = crackle_prob_range
        self.wow_freq_range = wow_freq_range
        self.wow_depth_range = wow_depth_range
        self.flutter_freq_range = flutter_freq_range
        self.flutter_depth_range = flutter_depth_range
        self.saturation_drive_range = saturation_drive_range
        self.lowpass_cutoff_range = lowpass_cutoff_range
        self.dropout_prob = dropout_prob
        self.prob = prob

    def _rand_uniform(self, shape, low, high):
        return torch.rand(shape, device=self.device) * (high - low) + low

    def forward(self, waveform):
        """
        x: (B, T) on CUDA
        """

        apply_mask = torch.rand(waveform.shape[0], device=waveform.device) < self.prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return waveform
        
        x = waveform[indices].clone()

        x = x.to(self.device)
        B, T = x.shape



        # === Sample per-example parameters ===
        noise_level = self._rand_uniform((B, 1), *self.noise_level_range)
        crackle_prob = self._rand_uniform((B, 1), *self.crackle_prob_range)
        wow_freq = self._rand_uniform((B, 1), *self.wow_freq_range)
        wow_depth = self._rand_uniform((B, 1), *self.wow_depth_range)
        flutter_freq = self._rand_uniform((B, 1), *self.flutter_freq_range)
        flutter_depth = self._rand_uniform((B, 1), *self.flutter_depth_range)
        saturation_drive = self._rand_uniform((B, 1), *self.saturation_drive_range)
        lowpass_cutoff = self._rand_uniform((B,), *self.lowpass_cutoff_range)

        # === 1. Hiss ===
        noise = torch.randn_like(x) * noise_level
        x = x + noise

        # === 2. Crackle (vinyl) ===
        if self.mode == "vinyl":
            rand = torch.rand(B, T, device=x.device)
            mask = (rand < crackle_prob).float()
            clicks = torch.randn_like(x) * 0.2 * mask
            x = x + clicks

        # === 3. Wow & Flutter ===
        t = torch.arange(T, device=x.device).float() / self.sr  # (T,)

        # Expand to (B, T)
        t = t.unsqueeze(0).expand(B, T)

        wow = wow_depth * torch.sin(2 * math.pi * wow_freq * t)
        flutter = flutter_depth * torch.sin(2 * math.pi * flutter_freq * t)
        mod = (wow + flutter) * self.sr  # convert to index shift

        base_idx = torch.arange(T, device=x.device).float()
        base_idx = base_idx.unsqueeze(0).expand(B, T)

        warped_idx = (base_idx + mod).clamp(0, T - 1)

        # Normalize for grid_sample
        grid = (warped_idx / (T - 1)) * 2 - 1
        grid = grid.unsqueeze(1).unsqueeze(-1)  # (B,1,T,1)

        x_ = x.unsqueeze(1).unsqueeze(-1)

        # Normalize warped indices to [-1, 1]
        x_coords = (warped_idx / (T - 1)) * 2 - 1  # (B, T)

        # y coords = 0 (since height = 1)
        y_coords = torch.zeros_like(x_coords)

        # Build grid (B, T, 2)
        grid = torch.stack((x_coords, y_coords), dim=-1)

        # Add height dim → (B, 1, T, 2)
        grid = grid.unsqueeze(1)

        # Input must be (B, C, H, W) = (B,1,1,T)
        x_ = x.unsqueeze(1).unsqueeze(2)

        # Apply sampling
        x = F.grid_sample(
            x_,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )

        # Back to (B, T)
        x = x.squeeze(1).squeeze(1)

        # === 4. Saturation ===
        x = torch.tanh(x * saturation_drive)

        # === 5. Low-pass (per sample cutoff) ===
        freqs = torch.fft.rfftfreq(T, d=1/self.sr).to(x.device)  # (F,)
        fft = torch.fft.rfft(x, dim=1)

        # Build per-sample masks
        mask = (freqs.unsqueeze(0) <= lowpass_cutoff.unsqueeze(1)).float()
        fft = fft * mask

        x = torch.fft.irfft(fft, n=T, dim=1)

        # === 6. Dropouts (tape) ===
        if self.mode == "tape":
            apply_dropout = torch.rand(B, device=x.device) < self.dropout_prob

            for b in range(B):
                if apply_dropout[b]:
                    start = torch.randint(0, T - 100, (1,), device=x.device).item()
                    length = torch.randint(20, 100, (1,), device=x.device).item()
                    scale = torch.rand(1, device=x.device).item() * 0.3
                    x[b, start:start+length] *= scale
        
        waveform[indices] = x

        return waveform


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
        exp_range: tuple = (-1.0, 2.0),
    ):
        super().__init__()
        self.prob      = prob
        self.snr_range = snr_range
        self.exp_range = exp_range

    def bit_crush(self,x, bits=4):
        levels = 2 ** bits
        return torch.round(x * levels) / levels
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
        noise = self.bit_crush(self._colored_noise(waveform.shape, exp, waveform.device),bits=2)
        waveform = _mix_at_snr(waveform, noise, snr)
        bits = torch.randint(6,10,[waveform.shape[0],1],device = x.device)
        waveform = self.bit_crush(waveform,bits=bits)
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
        prob: float = 0.4,
        sample_rate: int = 16000,
        low_cut_range: tuple = (100.0, 300.),
        high_cut_range: tuple = (900, 3800.0),
        peak_freq_range=(800, 1400),
        peak_gain_range=(4, 12.0),  # dB-ish feel
        peak_width_range=(200, 800),
        distortion_range=(0.8, 1.5),
        sr = 16000,
        device = 'cuda'
        
    ):
        super().__init__()
        self.prob  = prob
        self.sample_rate    = sample_rate
        self.low_cut_range  = low_cut_range
        self.high_cut_range = high_cut_range
        self.peak_freq_range = peak_freq_range
        self.peak_gain_range = peak_gain_range
        self.peak_width_range = peak_width_range
        self.distortion_range = distortion_range
        self.device = device
        self.sr = sr
    def _rand(self, shape, low, high):
        return torch.rand(shape, device=self.device) * (high - low) + low
        
    def forward(self, waveform,is_distorted):

        apply_mask = (torch.rand(waveform.shape[0], device=waveform.device) < self.prob) & ~(is_distorted)
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return waveform
        
        x = waveform[indices].clone()

        x = x.to(self.device)
        B, T = x.shape

        # === Random params per sample ===
        low_cut = self._rand((B,), *self.low_cut_range)
        high_cut = self._rand((B,), *self.high_cut_range)
        peak_freq = self._rand((B,), *self.peak_freq_range)
        peak_gain = self._rand((B,), *self.peak_gain_range)
        peak_width = self._rand((B,), *self.peak_width_range)
        distortion = self._rand((B, 1), *self.distortion_range)

        # === FFT ===
        freqs = torch.fft.rfftfreq(T, d=1/self.sr).to(x.device)  # (F,)
        X = torch.fft.rfft(x, dim=1)  # (B, F)

        # === Bandpass mask ===
        band_mask = (
            (freqs.unsqueeze(0) >= low_cut.unsqueeze(1)) &
            (freqs.unsqueeze(0) <= high_cut.unsqueeze(1))
        ).float()

        # === 1k-ish peak (Gaussian bump) ===
        # shape: (B, F)
        peak = torch.exp(
            -0.5 * ((freqs.unsqueeze(0) - peak_freq.unsqueeze(1)) / peak_width.unsqueeze(1))**2
        )

        peak = 1 + peak * peak_gain.unsqueeze(1)

        # === Apply filter ===
        X = X * band_mask * peak

        x = torch.fft.irfft(X, n=T, dim=1)

        # === Distortion ===
        x = torch.tanh(x * distortion)
        waveform[indices] = x

        return waveform


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

    def soft_lowpass(self,freqs, cutoff, slope=10.0):
        return torch.sigmoid((cutoff - freqs) * slope)
    
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

        order = 4  # controls steepness

        mask = torch.stack([
            self.soft_lowpass(freqs, cutoff_hz[i])
            for i in range(waveform.shape[0])
        ])
        filtered = torch.fft.irfft(spectrum * mask, n=T)

        orig_rms = _signal_power(waveform).sqrt()
        filt_rms = _signal_power(filtered).sqrt()
        scale    = orig_rms / filt_rms.clamp_min_(1e-9)

        waveform = filtered * scale
        x[indices] = waveform


        return x,apply_mask


class WaveformSpectralTilt(nn.Module):
    def __init__(self, tilt_range=(-0.9, 0.9), sample_rate=16000,tilt_prob=0.5):
        """
        Args:
            tilt_range: Range for alpha. 
                        Positive values = High-frequency boost.
                        Negative values = High-frequency attenuation.
            sample_rate: Included for pipeline compatibility.
        """
        super().__init__()
        self.tilt_range = tilt_range
        self.sample_rate = sample_rate
        self.prob = tilt_prob

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Samples) or (Batch, 1, Samples)
        Returns:
            Augmented tensor of the same shape.
        """

            
        apply_mask = torch.rand(x.shape[0], device=x.device) < self.prob
        indices = apply_mask.nonzero(as_tuple=True)[0]

        if indices.numel() == 0:
            return x

        waveform = x[indices].clone()

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, T)
        
        batch_size, _, num_samples = waveform.shape
        device = waveform.device

        # 1. Randomly sample alpha for each item in the batch
        # Shape: (Batch, 1, 1)
        alphas = torch.empty(batch_size, 1, 1, device=device).uniform_(*self.tilt_range)

        # 2. Construct the kernels for grouped convolution
        # Each kernel is [1, -alpha]. Shape: (Batch, 1, 2)
        # We want to treat each batch item as its own group.
        ones = torch.ones(batch_size, 1, 1, device=device)
        kernels = torch.cat([ones, -alphas], dim=2) 

        # 3. Apply padding to the time dimension (left-side) to keep output length same
        x_padded = F.pad(waveform, (1, 0))

        # 4. Grouped Convolution: Apply unique kernel to each batch element
        # groups=batch_size ensures kernel[0] only hits x[0], kernel[1] hits x[1], etc.
        # weight needs to be (OutChannels, InChannels/Groups, K) -> (B, 1, 2)
        augmented = F.conv1d(
            x_padded.view(1, batch_size, -1), 
            kernels, 
            groups=batch_size
        ).view(batch_size, 1, num_samples)

        # 5. Energy Normalization (Peak normalization to prevent clipping)
        # Keeps the output volume within range of the input
        max_orig = waveform.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        max_aug = augmented.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        augmented = augmented * (max_orig / max_aug)
        waveform = augmented.squeeze(1)

        x[indices] = waveform

        return x


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
        # if random.random() > self.p:
        #     return x

            
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
        noise_prob: float = 0.25,
        noise_snr_db: tuple = (25.0, 50.0),
        pre_reverb_prob: float = 0.45,
        pre_rt60_range: tuple = (0.1, 0.9),
        reverb_prob: float = 0.35,
        rt60_range: tuple = (0.1, 1.5),
        post_reverb_prob = 0.15,
        post_rt60_range: tuple = (0.1, 1.),
        lp_prob: float = 0.4,
        lp_cutoff_range_hz: tuple = (400.0, 2700.0),
        
        resample_prob: float = 0.5,
        volume_prob: float = 0.4,
        volume_range: tuple = (0.7, 1.3),
        clip_prob = 0.25,
        packet_loss_prob = 0.15,
        bp_lowcut_range = (200,500),
        bp_highcut_range = (1200,3000),
        tilt_pre_prob = 0.5,
        tilt_pre_range = (-0.9,0.5),
        tilt_post_prob = 0.8,
        tilt_post_range = (-0.9,0.9),
        vinyl_prob = 0.15,
        tape_prob = 0.25,
        telephone_prob: float = 0.3,
        comp_prob = 0.25,
        device = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.noise     = ColoredNoiseAugment(prob=noise_prob, snr_range=noise_snr_db)
        self.pre_reverb    = SyntheticRIRAugment(prob=pre_reverb_prob, rt60_range=pre_rt60_range,
                                             sample_rate=sample_rate)
        self.reverb    = SyntheticRIRAugment(prob=reverb_prob, rt60_range=rt60_range,
                                             sample_rate=sample_rate)
        self.lowpass   = LowPassAugment(prob=lp_prob,
                                        cutoff_range_hz=lp_cutoff_range_hz,
                                        sample_rate=sample_rate)
        self.telephone = TelephoneAugment(prob=telephone_prob,
                                          sample_rate=sample_rate,
                                          low_cut_range=bp_lowcut_range,
                                          high_cut_range=bp_highcut_range,device = self.device)
        self.volume    = VolumeAugment(prob=volume_prob, gain_range=volume_range)
        self.artifacts = TransmissionArtifactAugment(clip_prob=clip_prob, packet_loss_prob=packet_loss_prob)
        self.tilt_pre = WaveformSpectralTilt(tilt_range = tilt_pre_range,sample_rate = sample_rate,tilt_prob = tilt_pre_prob)
        self.tilt_post = WaveformSpectralTilt(tilt_range = tilt_post_range,sample_rate = sample_rate,tilt_prob = tilt_post_prob)

        self.analogue_vinyl = AnalogNoiseAugmentorTorch(mode='vinyl',prob = vinyl_prob,device = self.device)
        self.analogue_tape = AnalogNoiseAugmentorTorch(mode = 'tape',prob = tape_prob,device=self.device)
        self.post_reverb    = SyntheticRIRAugment(prob=post_reverb_prob, rt60_range=post_rt60_range,
                                        sample_rate=sample_rate)
        self.hard_compressor = HardCompressorAugmentor(prob = comp_prob,device=self.device)
        self.energy_drop_threshold = 1e-6
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)).clamp_min(1e-9)
        waveform = x.clone()
        waveform = self.pre_reverb(waveform)
        waveform = self.tilt_pre(waveform)
        waveform = self.noise(waveform)
        waveform,is_lowpassed = self.lowpass(waveform)
        waveform,is_compressed = self.hard_compressor(waveform)
        is_distorted = is_lowpassed | is_compressed
        waveform = self.noise(waveform)
        waveform = self.tilt_post(waveform)
        waveform = self.artifacts(waveform)
        waveform = self.reverb(waveform)
        waveform = self.telephone(waveform,is_distorted)
        waveform = self.volume(waveform)
        out_rms = torch.sqrt(waveform.pow(2).mean(dim=-1, keepdim=True))
        energy_ratio = out_rms / in_rms
        is_dead = (energy_ratio < self.energy_drop_threshold)
        waveform = torch.where(is_dead, x, waveform)
        # print(is_dead.sum())
        waveform = self.analogue_vinyl(waveform)
        waveform = self.analogue_tape(waveform)
        waveform = self.post_reverb(waveform)
        return waveform