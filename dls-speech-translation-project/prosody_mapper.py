# # prosody_mapper.py

# import numpy as np
# import librosa
# import parselmouth
# import pyrubberband as pyrb
# from scipy.interpolate import interp1d
# from typing import Union, List

# class RhythmProsodyMapper:
#     """
#     For each segment:
#       1) Load original and TTS-aligned audio (arrays or paths).
#       2) Extract timing points (onsets) from both.
#       3) Build a nonuniform warp map so TTS onsets → original onsets.
#       4) Time-warp TTS audio via pitch-preserving method.
#       5) Extract mean F0 from both, compute a semitone shift, apply it.
#       6) Extract mean energy from both, compute gain, apply it.
#       7) Return a float32 array exactly the same length as the input TTS audio.
#     """

#     def __init__(self, sample_rate: int = 24000):
#         self.sample_rate = sample_rate

#     def _load_if_path(self, audio: Union[str, np.ndarray]) -> np.ndarray:
#         """
#         If `audio` is a file-path (string or Path-like), load via librosa at self.sample_rate.
#         Otherwise, assume it's already a NumPy array.
#         """
#         if isinstance(audio, (str, bytes)) or hasattr(audio, "stem"):
#             wav, _ = librosa.load(str(audio), sr=self.sample_rate)
#             return wav.astype(np.float32)
#         elif isinstance(audio, np.ndarray):
#             return audio.astype(np.float32)
#         else:
#             raise ValueError(f"Cannot interpret {type(audio)} as audio data")

#     def _extract_onsets(self, wav: np.ndarray) -> np.ndarray:
#         """
#         Returns an array of onset times (in seconds) detected in `wav` via librosa.onset_detect.
#         """
#         # We do a broad bandpass to improve onset detection in speech
#         onsets = librosa.onset.onset_detect(
#             y=wav,
#             sr=self.sample_rate,
#             backtrack=True,
#             units="time",
#             hop_length=256
#         )
#         return onsets  # e.g. array([0.10, 0.35, 0.72, ...])

#     def _create_warp_map(self,
#                          tts_onsets: np.ndarray,
#                          orig_onsets: np.ndarray,
#                          tts_duration: float,
#                          orig_duration: float) -> (np.ndarray, np.ndarray):
#         """
#         Build a time-warping map that sends each TTS-onset → corresponding original-onset.
#         If the number of onsets differs, we linearly interpolate.

#         Returns:
#           - tts_time:    np.linspace(0, tts_duration, num=tts_n_samples)
#           - warped_time: f(tts_time) so that f(tts_onsets[i])≈orig_onsets[i].
#         """

#         # If either array is empty, fallback to uniform alignment:
#         if len(tts_onsets) == 0 or len(orig_onsets) == 0:
#             # identity warp
#             tts_time = np.linspace(0.0, tts_duration, int(tts_duration * self.sample_rate))
#             warped_time = tts_time.copy() * (orig_duration / tts_duration)
#             return tts_time, warped_time

#         # 1) If counts differ, interpolate the shorter to match the longer
#         n_t = len(tts_onsets)
#         n_o = len(orig_onsets)

#         if n_t != n_o:
#             # Re-sample both onto a uniform 0–1 domain
#             tts_norm = np.linspace(0.0, 1.0, n_t)
#             orig_norm = np.linspace(0.0, 1.0, n_o)
#             interp_orig = interp1d(orig_norm, orig_onsets, kind="linear", fill_value="extrapolate")
#             orig_onsets_matched = interp_orig(tts_norm)
#             # Now we have exactly n_t points in orig_onsets_matched,
#             # aligned to tts_onsets[i] → orig_onsets_matched[i].
#             matched_tts_onsets = tts_onsets
#             matched_orig_onsets = orig_onsets_matched
#         else:
#             # same number → pair them directly
#             matched_tts_onsets = tts_onsets
#             matched_orig_onsets = orig_onsets

#         # 2) Build full sample-level time arrays
#         tts_n_samples = int(tts_duration * self.sample_rate)
#         tts_time = np.linspace(0.0, tts_duration, num=tts_n_samples)

#         # 3) Construct key points, including start/end
#         src_times = np.concatenate(([0.0], matched_tts_onsets, [tts_duration]))
#         tgt_times = np.concatenate(([0.0], matched_orig_onsets, [orig_duration]))

#         # Clip to valid range
#         tgt_times = np.clip(tgt_times, 0.0, orig_duration)

#         # 4) Interpolate linearly from src_times→tgt_times
#         warp_func = interp1d(src_times, tgt_times, kind="linear",
#                              bounds_error=False, fill_value="extrapolate")

#         warped_time = warp_func(tts_time)
#         return tts_time, warped_time

#     def _time_warp(self, audio: np.ndarray, warped_time: np.ndarray) -> np.ndarray:
#         """
#         Apply pitch-preserving time warping so that the sample originally at
#         tts_time[k] moves to warped_time[k]. We approximate this by computing
#         an overall stretch factor = (max(warped_time)/max(tts_time)) and letting
#         pyrubberband handle pitch-preserving warp.

#         If anything fails, fall back to uniform librosa time-stretch.
#         """

#         # Original and target durations:
#         orig_len = len(audio)
#         tts_duration = orig_len / self.sample_rate
#         target_duration = float(np.max(warped_time))

#         if target_duration <= 0 or tts_duration <= 0:
#             return audio

#         stretch_factor = target_duration / tts_duration

#         try:
#             warped = pyrb.time_stretch(audio, self.sample_rate, stretch_factor)
#         except Exception:
#             warped = librosa.effects.time_stretch(audio, rate=1.0 / stretch_factor)

#         # warp may introduce tiny differences in length. Truncate/pad to match:
#         desired_len = int(tts_duration * self.sample_rate)
#         if len(warped) > desired_len:
#             warped = warped[:desired_len]
#         elif len(warped) < desired_len:
#             warped = np.pad(warped, (0, desired_len - len(warped)), mode="constant")

#         return warped.astype(np.float32)

#     def _compute_mean_f0(self, audio: np.ndarray) -> float:
#         """
#         Extract F₀ via Praat; on failure, fall back to librosa.piptrack.
#         Return the mean of all voiced frames.
#         """
#         try:
#             snd = parselmouth.Sound(audio, sampling_frequency=self.sample_rate)
#             pitch = snd.to_pitch(time_step=0.01)
#             f0_values = pitch.selected_array["frequency"]
#             voiced = f0_values[f0_values > 0.0]
#             if len(voiced) > 0:
#                 return float(np.mean(voiced))
#         except Exception:
#             pass

#         # fallback: librosa piptrack
#         f0, _ = librosa.piptrack(y=audio, sr=self.sample_rate, threshold=0.1)
#         f0 = np.max(f0, axis=0)
#         voiced = f0[f0 > 0.0]
#         if len(voiced) > 0:
#             return float(np.mean(voiced))
#         return 0.0

#     def _compute_mean_energy(self, audio: np.ndarray) -> float:
#         """
#         Return the mean RMS energy of `audio`, or 0.0 if empty.
#         """
#         econt = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
#         if econt.shape[0] == 0:
#             return 0.0
#         return float(np.mean(econt))



#     def map_prosody(self, orig_audio: Union[str, np.ndarray], tts_audio: Union[str, np.ndarray]) -> np.ndarray:
#         """
#         1) Load orig and TTS audio (if they’re file‐paths).
#         2) Compute mean F₀ for each. If both > 0, compute semitone shift = 12·log2(mean_f0_orig / mean_f0_tts).
#            Otherwise, semitone_shift = 0.0 (no pitch‐shift).
#         3) Pitch‐shift the entire TTS waveform by that many semitones (librosa).
#         4) Compute mean RMS energy of orig vs. shifted TTS, compute gain = mean_energy_orig / mean_energy_tts.
#         5) Multiply TTS waveform by gain. Clip to [−1,1], then pad/truncate to match original TTS length.
#         6) Return a float32 array of the same length as the *original TTS* waveform.
#         """

#         # (1) Load waveforms if needed
#         wav_orig = self._load_if_path(orig_audio)
#         wav_tts = self._load_if_path(tts_audio)

#         tts_onsets = self._extract_onsets(wav_tts)
#         orig_onsets = self._extract_onsets(wav_orig)

#         tts_duration = len(wav_tts) / self.sample_rate
#         orig_duration = len(wav_orig) / self.sample_rate

#         # (3) Build warp map (source_time, warped_time)
#         tts_time, warped_time = self._create_warp_map(
#             tts_onsets=tts_onsets,
#             orig_onsets=orig_onsets,
#             tts_duration=tts_duration,
#             orig_duration=orig_duration
#         )

#         # (4) Time-warp the TTS audio (preserve pitch)
#         tts_timewarped = self._time_warp(wav_tts, warped_time)

#         # (2) Mean F₀ for each
#         mean_f0_orig = self._compute_mean_f0(wav_orig)
#         mean_f0_tts = self._compute_mean_f0(wav_tts)

#         if mean_f0_orig > 0.0 and mean_f0_tts > 0.0:
#             semitone_shift = 12.0 * np.log2(mean_f0_orig / mean_f0_tts)
#         else:
#             semitone_shift = 0.0

#         # (3) Pitch‐shift the entire TTS waveform
#         if abs(semitone_shift) > 0.01:
#             wav_tts_shifted = librosa.effects.pitch_shift(
#                 wav_tts,
#                 sr=self.sample_rate,
#                 n_steps=semitone_shift
#             )
#         else:
#             wav_tts_shifted = wav_tts.copy()

#         # (4) Mean RMS energy
#         mean_energy_orig = self._compute_mean_energy(wav_orig)
#         mean_energy_tts_shifted = self._compute_mean_energy(wav_tts_shifted)

#         if mean_energy_tts_shifted > 1e-8:
#             gain = mean_energy_orig / mean_energy_tts_shifted
#         else:
#             gain = 1.0

#         # (5) Apply gain (clip to [−1,1])
#         mapped = wav_tts_shifted * gain
#         mapped = np.clip(mapped, -1.0, 1.0)

#         # (6) Ensure same length as original TTS waveform
#         desired_len = len(wav_tts)
#         if mapped.shape[0] > desired_len:
#             mapped = mapped[:desired_len]
#         elif mapped.shape[0] < desired_len:
#             pad_width = desired_len - mapped.shape[0]
#             mapped = np.concatenate([mapped, np.zeros(pad_width, dtype=np.float32)])

#         return mapped.astype(np.float32)


# prosody_mapper.py

import numpy as np
import librosa
import parselmouth
from typing import Union, List
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pyrubberband as pyrb  # for pitch-preserving stretch
import warnings


class ProsodyExtractor:
    """
    Extract F₀, energy, and timing (onset + spectral-change) points from audio.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate

    def extract_f0(self, audio: np.ndarray, method: str = "praat") -> np.ndarray:
        """
        Returns a 1D array of F₀ values (Hz) per frame. Unvoiced = 0.0.
        - Tries Praat (parselmouth). If that fails, falls back to librosa.piptrack.
        """
        # Praat path
        try:
            snd = parselmouth.Sound(audio, sampling_frequency=self.sr)
            pitch = snd.to_pitch(time_step=0.01)  # 10ms hop
            f0 = pitch.selected_array['frequency']  # zeros where unvoiced
            f0 = np.nan_to_num(f0)  # ensure no NaNs
            return f0.astype(np.float32)
        except Exception:
            # Fallback to librosa.piptrack (harvest‐style)
            f0_matrix, _ = librosa.piptrack(y=audio, sr=self.sr, threshold=0.1)
            # Take max per frame
            f0_harvest = np.max(f0_matrix, axis=0)
            return f0_harvest.astype(np.float32)

    def extract_energy(self, audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Returns a 1D array of RMS energy (float) per frame.
        """
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        return energy.astype(np.float32)

    def extract_timing_points(self, audio: np.ndarray, 
                              hop_length: int = 512) -> List[float]:
        """
        Returns a sorted list of key time‐points (in seconds) based on:
          1) Onset detection
          2) Spectral‐centroid changes (syllable candidates)
        Combined and deduplicated.
        """
        # 1) Onset times (in seconds)
        onset_times = librosa.onset.onset_detect(
            y=audio,
            sr=self.sr,
            units="time",
            backtrack=True,
            hop_length=hop_length
        )

        # 2) Spectral‐centroid changes
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr, hop_length=hop_length)[0]
        spectral_diff = np.diff(spectral_centroid)
        if len(spectral_diff) > 0:
            threshold = np.std(spectral_diff) * 1.5
            significant_idxs = np.where(np.abs(spectral_diff) > threshold)[0]
            # Convert frame indices → time
            syllable_times = librosa.frames_to_time(significant_idxs, sr=self.sr, hop_length=hop_length)
        else:
            syllable_times = np.array([])

        # Combine and sort unique
        all_times = np.concatenate([onset_times, syllable_times])
        if all_times.size == 0:
            return []
        unique_times = np.unique(all_times)
        return unique_times.tolist()


class TimingAligner:
    """
    Align one audio’s timing to another’s timing‐points while preserving pitch if desired.
    This uses a piecewise linear warp map between source_times → target_times.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate

    def align_audio_to_timing(self,
                              audio: np.ndarray,
                              source_timing_points: List[float],
                              target_timing_points: List[float],
                              preserve_pitch: bool = True) -> np.ndarray:
        """
        Warps `audio` so that each time in source_timing_points
        gets mapped to the corresponding time in target_timing_points.
        If the two lists have different lengths, we interpolate them onto each other.
        - Returns a new audio array (float32) of roughly the same total duration
          but with local stretching/compression between points.
        """

        # If no timing points, nothing to do:
        if len(source_timing_points) == 0 or len(target_timing_points) == 0:
            return audio

        # Ensure arrays are sorted
        st = np.array(sorted(source_timing_points))
        tt = np.array(sorted(target_timing_points))

        # If lengths differ, linearly interpolate the shorter to match the longer
        if len(st) != len(tt):
            # Normalize each to [0,1]
            src_norm = np.linspace(0, 1, len(st))
            tgt_norm = np.linspace(0, 1, len(tt))
            interp_func = interp1d(tgt_norm, tt, kind="linear", fill_value="extrapolate")
            tt = interp_func(src_norm)

        # Now we have equal‐length st & tt
        orig_duration = len(audio) / self.sr
        # Ensure we include 0.0 and full duration as endpoints
        src_times = np.concatenate([[0.0], st, [orig_duration]])
        tgt_times = np.concatenate([[0.0], tt, [orig_duration]])
        # Clip to valid range
        tgt_times = np.clip(tgt_times, 0.0, orig_duration)

        # Create a warp function mapping every time in [0,orig_duration] → new time
        num_samples = len(audio)
        original_time_axis = np.linspace(0.0, orig_duration, num=num_samples)
        warp_func = interp1d(src_times, tgt_times, kind="linear", fill_value="extrapolate", bounds_error=False)
        warped_time_axis = warp_func(original_time_axis)

        # Now we have, for each sample index i (time original_time_axis[i]),
        # a new time warped_time_axis[i]. We need to “resample” audio at those new times:
        # Convert times → sample indices
        original_sample_indices = original_time_axis * self.sr
        warped_sample_indices = warped_time_axis * self.sr

        if preserve_pitch:
            # For pitch‐preserving, we do a **uniform** time‐stretch by overall factor,
            # then trust that local warping is close enough. True local pitch‐preserving
            # would require advanced PSOLA/etc.; here we approximate by a single stretch:
            try:
                # Overall stretch factor = (final total time) / (original total time).
                final_total = warped_time_axis.max()
                overall_factor = final_total / orig_duration if orig_duration > 0 else 1
                stretched = pyrb.time_stretch(audio, self.sr, overall_factor)
                # If lengths differ slightly, trim/pad:
                desired_len = int(original_sample_indices[-1] * overall_factor)
                if len(stretched) > desired_len:
                    stretched = stretched[:desired_len]
                elif len(stretched) < desired_len:
                    pad_w = desired_len - len(stretched)
                    stretched = np.concatenate([stretched, np.zeros(pad_w, dtype=stretched.dtype)])
                return stretched.astype(np.float32)
            except Exception as e:
                warnings.warn(f"Rubberband warp failed; falling back to linear resample.\n{e}")

                # Fall through to standard resampling below.

        # Standard (pitch‐changing) resample via np.interp:
        audio_float = audio.astype(np.float32)
        mapped = np.interp(warped_sample_indices, original_sample_indices, audio_float)
        return mapped.astype(np.float32)


class ProsodyMapper:
    """
    1) Takes:
       - orig_audio (NumPy array or file path) for the original speaker segment.
       - tts_audio  (NumPy array or file path) for the duration‐aligned TTS output.
    2) Extracts timing/rhythm from orig & TTS, then warps TTS locally to match orig rhythm.
    3) Computes a global semitone shift so that mean F₀ matches orig’s mean F₀.
    4) Applies a gain so that mean RMS energy matches orig’s mean.
    5) Returns a new float32 array (same length as the original TTS waveform).
    """

    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        self.extractor = ProsodyExtractor(sample_rate)
        self.aligner = TimingAligner(sample_rate)

    def _load_if_path(self, audio: Union[str, np.ndarray]) -> np.ndarray:
        """
        If `audio` is str or Path-like, load via librosa at self.sr. Otherwise, return copy of the array.
        """
        if isinstance(audio, (str, bytes)) or hasattr(audio, "stem"):
            wav, _ = librosa.load(str(audio), sr=self.sr)
            return wav.astype(np.float32)
        elif isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
        else:
            raise ValueError(f"ProsodyMapper._load_if_path: Cannot interpret {type(audio)} as audio")

    def _compute_mean_f0(self, wav: np.ndarray) -> float:
        """
        Extract mean F₀ (Praat → fallback to piptrack). Return 0.0 if no voiced frames.
        """
        f0 = self.extractor.extract_f0(wav, method="praat")
        voiced = f0[f0 > 0.0]
        if len(voiced) == 0:
            return 0.0
        return float(np.mean(voiced))

    def _compute_mean_energy(self, wav: np.ndarray) -> float:
        """
        Extract mean RMS energy. Return 0.0 if contour empty.
        """
        energy = self.extractor.extract_energy(wav)
        if energy.shape[0] == 0:
            return 0.0
        return float(np.mean(energy))

    def map_prosody(self,
                    orig_audio: Union[str, np.ndarray],
                    tts_audio: Union[str, np.ndarray]) -> np.ndarray:
        """
        1) Load both waveforms if needed.
        2) Extract timing_points from each.
        3) Warp TTS to orig timing via TimingAligner.align_audio_to_timing(..., preserve_pitch=True).
        4) Compute mean F₀_orig vs mean F₀_warped, apply a global pitch shift.
        5) Compute mean energy_orig vs mean energy_shifted, apply gain.
        6) Trim/pad so the final output matches the length of the original TTS audio.
        """

        # (1) Load if paths
        wav_orig = self._load_if_path(orig_audio)
        wav_tts = self._load_if_path(tts_audio)

        # (2) Extract timing points
        timing_orig = self.extractor.extract_timing_points(wav_orig)
        timing_tts = self.extractor.extract_timing_points(wav_tts)

        # (3) Warp TTS → match orig rhythm
        warped = self.aligner.align_audio_to_timing(
            wav_tts,
            source_timing_points=timing_tts,
            target_timing_points=timing_orig,
            preserve_pitch=True
        )

        # (4) Pitch matching (global semitone shift)
        mean_f0_orig = self._compute_mean_f0(wav_orig)
        mean_f0_warped = self._compute_mean_f0(warped)
        if mean_f0_orig > 0.0 and mean_f0_warped > 0.0:
            semitone_shift = 12.0 * np.log2(mean_f0_orig / mean_f0_warped)
        else:
            semitone_shift = 0.0

        if abs(semitone_shift) > 0.01:
            warped_shifted = librosa.effects.pitch_shift(
                warped,
                sr=self.sr,
                n_steps=semitone_shift
            )
        else:
            warped_shifted = warped

        # (5) Energy matching
        mean_en_orig = self._compute_mean_energy(wav_orig)
        mean_en_shifted = self._compute_mean_energy(warped_shifted)
        if mean_en_shifted > 1e-8:
            gain = mean_en_orig / mean_en_shifted
        else:
            gain = 1.0

        mapped = warped_shifted * gain
        mapped = np.clip(mapped, -1.0, 1.0)

        # # (6) Ensure final length = length of original TTS audio
        # desired_len = len(wav_tts)
        # if len(mapped) > desired_len:
        #     mapped = mapped[:desired_len]
        # elif len(mapped) < desired_len:
        #     pad_w = desired_len - len(mapped)
        #     mapped = np.concatenate([mapped, np.zeros(pad_w, dtype=np.float32)])

        return mapped.astype(np.float32)
