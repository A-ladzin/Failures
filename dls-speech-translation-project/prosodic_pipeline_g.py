#!/usr/bin/env python3
"""
Advanced Video Translation & Voice Cloning Pipeline
Preserves intonation, tone, and prosodic features across languages
"""

import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import soundfile as sf
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import parselmouth
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import moviepy.editor as mp

from helpers import *
from translation_helpers import *

from scipy.io.wavfile import write
import numpy as np







# run_pipeline.py

import os
import argparse
import torch
import subprocess
import shutil
import os
import subprocess
import tempfile
import time
import whisper
import openai
import srt
from datetime import timedelta
from typing import List, Dict

# Your existing imports:
from data_pipeline import AudioPreprocessor
from tts_pipeline import TTSWrapper
from metrics import SpeakerSimilarity

from data_pipeline import AudioPreprocessor
from asr_wrapper import ASRWrapper
# from translation import TranslationWrapper
from tts_pipeline import TTSWrapper,synthesize_timealigned_segments
from metrics import SpeakerSimilarity
from pydub import AudioSegment
from TTS.api import TTS
import csv
from functools import reduce

from pydub import AudioSegment
import os


@dataclass
class SegmentData:
    """Data structure for audio segment information"""
    start_time: float
    end_time: float
    original_text: str
    translated_text: str
    original_audio: np.ndarray
    f0_curve: np.ndarray
    energy_curve: np.ndarray
    duration: float
    speaker_embedding: Optional[torch.Tensor] = None
    prosodic_features: Optional[Dict] = None

#!/usr/bin/env python3
"""
Advanced Video Translation & Voice Cloning Pipeline
Aligns audio timing while preserving original pitch characteristics
"""

import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import soundfile as sf
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import parselmouth
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import moviepy.editor as mp
import pyrubberband as pyrb  # For pitch-preserving time stretching

@dataclass
class SegmentData:
    """Data structure for audio segment information"""
    start_time: float
    end_time: float
    original_text: str
    translated_text: str
    original_audio: np.ndarray
    f0_curve: np.ndarray
    energy_curve: np.ndarray
    duration: float
    speaker_embedding: Optional[torch.Tensor] = None
    prosodic_features: Optional[Dict] = None
    timing_points: Optional[List[float]] = None  # Key timing points for alignment

class ProsodyExtractor:
    """Extract and manipulate prosodic features from audio"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
    
    def extract_f0(self, audio: np.ndarray, method: str = "harvest") -> np.ndarray:
        """Extract fundamental frequency using multiple methods"""
        if method == "harvest":
            f0, _ = librosa.piptrack(y=audio, sr=self.sr, threshold=0.1)
            f0 = np.max(f0, axis=0)
        elif method == "praat":
            # Using Praat via parselmouth for more accurate F0
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            pitch = sound.to_pitch(time_step=0.01)
            f0 = pitch.selected_array['frequency']
        
        # Smooth and interpolate F0
        f0 = self._smooth_f0(f0)
        return f0
    
    def extract_energy(self, audio: np.ndarray, frame_length: int = 2048) -> np.ndarray:
        """Extract energy contour"""
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=512)[0]
        return energy
    
    def extract_timing_points(self, audio: np.ndarray) -> List[float]:
        """Extract key timing points for alignment (onsets, syllable boundaries)"""
        # Onset detection for rhythm alignment
        onset_frames = librosa.onset.onset_detect(
            y=audio, 
            sr=self.sr, 
            units='time',
            backtrack=True,
            hop_length=512
        )
        
        # Add syllable-level timing points using spectral changes
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spectral_diff = np.diff(spectral_centroids)
        
        # Find significant spectral changes (potential syllable boundaries)
        threshold = np.std(spectral_diff) * 1.5
        syllable_points = np.where(np.abs(spectral_diff) > threshold)[0]
        syllable_times = librosa.frames_to_time(syllable_points, sr=self.sr, hop_length=512)
        
        # Combine and sort timing points
        all_timing_points = np.concatenate([onset_frames, syllable_times])
        all_timing_points = np.unique(all_timing_points)
        
        return all_timing_points.tolist()
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Dict:
        """Extract rhythm and timing features"""
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr, units='time')
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        
        # Spectral features for rhythm
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        
        return {
            'onsets': onset_frames,
            'tempo': tempo,
            'beats': beats,
            'spectral_centroid': spectral_centroid
        }
    
    def _smooth_f0(self, f0: np.ndarray, window_length: int = 15) -> np.ndarray:
        """Smooth F0 curve and handle unvoiced regions"""
        # Remove zeros and interpolate
        voiced_indices = f0 > 0
        if np.sum(voiced_indices) > 10:  # Need enough voiced frames
            f0_voiced = f0[voiced_indices]
            time_voiced = np.where(voiced_indices)[0]
            
            # Interpolate across unvoiced regions
            if len(time_voiced) > 1:
                interp_func = interp1d(time_voiced, f0_voiced, 
                                     kind='linear', bounds_error=False, 
                                     fill_value='extrapolate')
                f0_interpolated = interp_func(np.arange(len(f0)))
                
                # Smooth the curve
                if len(f0_interpolated) > window_length:
                    f0_smoothed = savgol_filter(f0_interpolated, window_length, 3)
                else:
                    f0_smoothed = f0_interpolated
                
                return f0_smoothed
        
        return f0

class TimingAligner:
    """Align audio timing to a given duration, with optional pitch preservation."""
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate

    def align_to_duration(self, audio: np.ndarray, target_duration: float, preserve_pitch: bool = True) -> np.ndarray:
        """
        Uniformly stretch (or compress) the entire audio so its length matches target_duration (in seconds).
        If preserve_pitch=True, use rubberband; otherwise, use librosa’s time_stretch (which changes pitch).
        """

        # 1) Compute current_duration (in seconds) of the array:
        current_duration = len(audio) / self.sr
        if current_duration <= 0 or target_duration <= 0:
            # We can’t do anything meaningful if durations are zero or negative.
            return audio

        # 2) Stretch factor: how many times longer (or shorter) we need to be.
        stretch_factor =  target_duration/current_duration
        #   e.g. if current = 2.0 s and target=3.0 s, stretch_factor = 1.5 → we need to make it 1.5× longer.

        if preserve_pitch:
            # Try rubberband first (high-quality, pitch-preserving).
            # Ruberband’s API: pyrb.time_stretch(audio, sr, stretch_rate)
            # where stretch_rate>1 makes it longer, <1 makes it shorter.
            try:
                aligned_audio = pyrb.time_stretch(audio, self.sr, 1./stretch_factor)
            except Exception:
                # Fallback to librosa’s phase-vocoder time_stretch:
                # librosa.effects.time_stretch(y, rate) returns length = len(y) / rate
                # so to get "len(y) * stretch_factor", we must do rate = 1/stretch_factor.
                aligned_audio = librosa.effects.time_stretch(audio, rate=1.0 / stretch_factor)
        else:
            # If you don’t care about pitch, just use librosa directly:
            aligned_audio = librosa.effects.time_stretch(audio, rate=1.0 / stretch_factor)

        return aligned_audio

    # (Optional) Remove or de-prioritize the old `align_audio_to_timing` if you don’t use it anymore.
    def align_audio_to_timing(self, audio: np.ndarray,
                              source_timing_points: List[float],
                              target_timing_points: List[float],
                              preserve_pitch: bool = True) -> np.ndarray:
        """
        (Legacy) Align audio to match target timing points while preserving pitch.
        Not used in the new uniform-stretch approach.
        """
        # … you can leave this in place if you want to keep “detailed warp” for future,
        # but it is not strictly necessary for simple duration alignment.
        if len(source_timing_points) == 0 or len(target_timing_points) == 0:
            return audio

        # If number of timing points differ, interpolate one set to the other
        if len(source_timing_points) != len(target_timing_points):
            target_timing_points = self._interpolate_timing_points(
                source_timing_points, target_timing_points
            )

        # Now create warp map where “total_duration” is STILL the original audio’s duration.
        total_duration = len(audio) / self.sr
        original_times, warped_times = self._create_warp_map(
            source_timing_points, target_timing_points, total_duration, desired_target_duration=total_duration
        )

        if preserve_pitch:
            return self._pitch_preserving_warp(audio, original_times, warped_times)
        else:
            return self._standard_time_warp(audio, original_times, warped_times)

    def _create_warp_map(self, source_points: List[float],
                         target_points: List[float],
                         original_duration: float,
                         desired_target_duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a time-warping map.
        - source_points: list of times (in seconds) detected in the translated audio
        - target_points: list of times (in seconds) from the *original* segment
        - original_duration: duration of the translated audio
        - desired_target_duration: the duration you actually want at the end (e.g. the original segment’s duration)
        """
        # 1) Insert start and end
        source_times = np.array([0.0] + list(source_points) + [original_duration])
        # IMPORTANT: use the **original segment’s** duration as the final time
        # rather than using translated audio’s duration again.
        target_times = np.array([0.0] + list(target_points) + [desired_target_duration])

        # Clip to avoid any target_time > desired_target_duration
        target_times = np.clip(target_times, 0.0, desired_target_duration)

        # For the entire audio (each sample index), we’ll interpolate linearly
        # between (source_times → target_times).
        num_samples = int(original_duration * self.sr)
        original_time_samples = np.linspace(0, original_duration, num_samples)

        warp_func = interp1d(source_times, target_times, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
        warped_time_samples = warp_func(original_time_samples)

        return original_time_samples, warped_time_samples

    def _pitch_preserving_warp(self, audio: np.ndarray,
                               original_times: np.ndarray,
                               warped_times: np.ndarray) -> np.ndarray:
        """
        Apply pitch-preserving warp. As a simplified fallback, we just uniformly
        stretch the entire audio by (desired_target_duration / original_duration).
        """
        # original_duration = original_times.max()
        # desired_target_duration = warped_times.max()
        if len(original_times) == 0 or len(warped_times) == 0:
            return audio

        original_duration = original_times.max()
        desired_target_duration = warped_times.max()
        if original_duration <= 0:
            return audio

        stretch_factor = desired_target_duration / original_duration
        try:
            aligned_audio = pyrb.time_stretch(audio, self.sr, stretch_factor)
        except Exception:
            aligned_audio = librosa.effects.time_stretch(audio, rate=1.0/stretch_factor)

        return aligned_audio

    def _standard_time_warp(self, audio: np.ndarray,
                            original_times: np.ndarray,
                            warped_times: np.ndarray) -> np.ndarray:
        """
        Non–pitch-preserving warp via sample interpolation.
        (Not used for simple duration alignment.)
        """
        try:
            original_indices = np.arange(len(audio), dtype=float)
            interp_func = interp1d(original_times * self.sr, warped_times * self.sr,
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            new_sample_positions = interp_func(original_indices)
            warped_audio = np.interp(new_sample_positions, original_indices, audio)
            return warped_audio
        except Exception:
            return audio


class ProsodyTransfer:
    """Transfer prosodic features from source to target speech without pitch changes"""
    
    def __init__(self):
        self.timing_aligner = TimingAligner()
    
    def align_prosody_preserve_pitch(self, source_f0: np.ndarray, 
                                   source_timing: List[float],
                                   target_timing: List[float],
                                   source_duration: float,
                                   target_duration: float) -> Tuple[np.ndarray, List[float]]:
        """Align prosodic features while preserving pitch characteristics"""
        
        # Align timing points without changing F0 values
        aligned_timing = self.timing_aligner._interpolate_timing_points(
            source_timing, target_timing
        )
        
        # Time-stretch F0 curve to match new timing without changing pitch values
        stretch_factor = target_duration / source_duration
        
        # Resample F0 curve to match new duration
        original_time = np.linspace(0, 1, len(source_f0))
        target_time = np.linspace(0, 1, int(len(source_f0) * stretch_factor))
        
        interp_func = interp1d(original_time, source_f0, kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
        aligned_f0 = interp_func(target_time)
        
        return aligned_f0, aligned_timing

class VoiceCloningPipeline:
    """Main pipeline for voice cloning with timing alignment (pitch-preserving)"""
    
    def __init__(self, model_path: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.tts_model = TTS(model_path, gpu=True)
        self.prosody_extractor = ProsodyExtractor()
        self.prosody_transfer = ProsodyTransfer()
        self.timing_aligner = TimingAligner()
        self.speaker_embeddings = {}
        self.speaker_f0_stats = {}
    
    def extract_speaker_profile(self, reference_audio_paths: List[str], speaker_id: str) -> Dict:
        """Extract speaker characteristics from reference audio"""
        all_f0 = []
        all_energy = []
        all_timing_points = []
        
        for audio_path in reference_audio_paths:
            audio, sr = librosa.load(audio_path, sr=24000)
            
            # Extract prosodic features
            f0 = self.prosody_extractor.extract_f0(audio)
            energy = self.prosody_extractor.extract_energy(audio)
            timing_points = self.prosody_extractor.extract_timing_points(audio)
            
            all_f0.extend(f0[f0 > 0])  # Only voiced segments
            all_energy.extend(energy)
            all_timing_points.extend(timing_points)
        
        # Calculate speaker statistics
        f0_stats = {
            'mean': np.mean(all_f0),
            'std': np.std(all_f0),
            'min': np.percentile(all_f0, 5),
            'max': np.percentile(all_f0, 95)
        }
        
        energy_stats = {
            'mean': np.mean(all_energy),
            'std': np.std(all_energy)
        }
        
        # Store speaker profile
        self.speaker_f0_stats[speaker_id] = f0_stats
        
        return {
            'f0_stats': f0_stats,
            'energy_stats': energy_stats,
            'reference_paths': reference_audio_paths,
            'typical_timing_density': len(all_timing_points) / sum([librosa.get_duration(filename=path) for path in reference_audio_paths])
        }
    
    def segment_audio_with_prosody(self, audio_path: str, segments_info: List[Dict]) -> List[SegmentData]:
        """Segment audio and extract prosodic features for each segment"""
        audio, sr = librosa.load(audio_path, sr=24000)
        segments = []
        
        for seg_info in segments_info:
            start_sample = int(seg_info['start_time'] * sr)
            end_sample = int(seg_info['end_time'] * sr)
            
            segment_audio = audio[start_sample:end_sample]
            
            # Extract prosodic features
            f0_curve = self.prosody_extractor.extract_f0(segment_audio)
            energy_curve = self.prosody_extractor.extract_energy(segment_audio)
            rhythm_features = self.prosody_extractor.extract_rhythm_features(segment_audio)
            timing_points = self.prosody_extractor.extract_timing_points(segment_audio)
            
            segment = SegmentData(
                start_time=seg_info['start_time'],
                end_time=seg_info['end_time'],
                original_text=seg_info['original_text'],
                translated_text=seg_info['translated_text'],
                original_audio=segment_audio,
                f0_curve=f0_curve,
                energy_curve=energy_curve,
                duration=seg_info['end_time'] - seg_info['start_time'],
                prosodic_features=rhythm_features,
                timing_points=timing_points
            )
            
            segments.append(segment)
        
        return segments

    def synthesize_with_timing_alignment(self,
                                         segment: SegmentData,
                                         speaker_reference: List[str],
                                         speaker_id: str,
                                         language: str = "en",
                                         preserve_pitch: bool = True) -> np.ndarray:
        """
        1) Run XTTS to get a raw waveform for segment.translated_text
        2) Don’t manually slice off zeros—use the entire returned array
        3) Call align_to_duration(...) so that the final array is exactly segment.duration seconds long.
        """

        # (1) Generate raw TTS output (might contain trailing zeros if the model pads).
        basic_output = self.tts_model.tts(
            text=segment.translated_text,
            speaker_wav=speaker_reference,
            language=language
        )
        out_b = [int(i != 0) for i in basic_output]
        durations_ph = np.where(np.array(out_b) != 0)[0].max()
        # durations_ph = len(output)
        basic_output = basic_output[:durations_ph]

        # (2) Load into a NumPy array. Coqui XTTS sometimes returns a numpy array directly,
        #     or a filename. Handle both cases:
        if isinstance(basic_output, str):
            translated_audio, _ = librosa.load(basic_output, sr=24000)
        else:
            translated_audio = np.array(basic_output)

        # (3) Now uniformly stretch/compress “translated_audio” to exactly segment.duration:
        aligned_audio = self.timing_aligner.align_to_duration(
            translated_audio,
            target_duration=segment.duration,
            preserve_pitch=preserve_pitch
        )

        # Return a float32 waveform of length target_duration * 24000
        return aligned_audio
    
    def process_video_translation(self, video_path: str, segments_info: List[Dict],
                                 speaker_references: List[str], output_path: str,
                                 target_language: str = "en") -> str:
        """Complete pipeline for video translation with prosody preservation"""
        
        # Extract audio from video
        video = mp.VideoFileClip(video_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        
        # Extract speaker profile
        speaker_id = "main_speaker"
        self.extract_speaker_profile(speaker_references, speaker_id)
        
        # Segment audio with prosody
        segments = self.segment_audio_with_prosody(audio_path, segments_info)
        
        # Process each segment
        translated_segments = []
        for idx,segment in tqdm(enumerate(segments)):
            print(f"Processing segment: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
            
            translated_audio = self.synthesize_with_timing_alignment(
                segment, speaker_references[idx], speaker_id, target_language
            )
            
            translated_segments.append({
                'audio': translated_audio,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration
            })
            print(len(translated_audio)/24000,segment.start_time,segment.end_time,segment.duration)
        
        # Reconstruct full audio track
        full_audio = self._reconstruct_audio_track(translated_segments, video.duration)
        # Convert float32 array to int16
        scaled = np.int16(full_audio * 32767)
        output_audio_path = os.path.join(args.output_dir,"output_ru.wav")
        write(output_audio_path, 24000, scaled)
        # Create new video with translated audio
        # final_video = video.set_audio(mp.AudioArrayClip(full_audio, fps=24000))
        combine_audio_video(args.video_path,output_audio_path,output_path)
        # final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Cleanup
        video.close()
        Path(audio_path).unlink(missing_ok=True)
        
        return output_path
    
    def _reconstruct_audio_track(self, segments: List[Dict], total_duration: float) -> np.ndarray:
        """Reconstruct full audio track from translated segments"""
        total_samples = int(total_duration * 24000)
        full_audio = np.zeros(total_samples)
        
        
        for segment in segments:
            start_sample = int(segment['start_time'] * 24000)
            audio_data = segment['audio']
            
            # Ensure we don't exceed array bounds
            end_sample = min(start_sample + len(audio_data), total_samples)
            audio_length = end_sample - start_sample
            
            if audio_length > 0:
                full_audio[start_sample:end_sample] = audio_data[:audio_length]
        
        return full_audio

# Fine-tuning pipeline for XTTS with prosody awareness
class ProsodyAwareXTTSFinetuning:
    """Fine-tune XTTS model with prosody-aware loss functions"""
    
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.prosody_extractor = ProsodyExtractor()
    
    def prepare_training_data_with_prosody(self, 
                                         audio_files: List[str],
                                         transcripts: List[str],
                                         output_dir: str) -> str:
        """Prepare training data with prosodic feature annotations"""
        
        training_data = []
        
        for audio_file, transcript in zip(audio_files, transcripts):
            audio, sr = librosa.load(audio_file, sr=24000)
            
            # Extract prosodic features
            f0 = self.prosody_extractor.extract_f0(audio)
            energy = self.prosody_extractor.extract_energy(audio)
            rhythm = self.prosody_extractor.extract_rhythm_features(audio)
            
            # Store training sample with prosodic annotations
            sample = {
                'audio_path': audio_file,
                'transcript': transcript,
                'prosody': {
                    'f0': f0.tolist(),
                    'energy': energy.tolist(),
                    'rhythm': rhythm
                }
            }
            
            training_data.append(sample)
        
        # Save training data with prosodic annotations
        output_file = Path(output_dir) / "prosody_training_data.json"
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        return str(output_file)
    
    def create_prosody_aware_config(self, base_config_path: str, 
                                  prosody_weight: float = 0.3) -> str:
        """Create configuration for prosody-aware training"""
        
        # This would require modifications to XTTS training code
        # to include prosodic loss functions
        config_template = {
            "prosody_loss_weight": prosody_weight,
            "f0_loss_weight": 0.15,
            "energy_loss_weight": 0.1,
            "rhythm_loss_weight": 0.05,
            "enable_prosody_conditioning": True,
            "prosody_feature_dim": 128
        }
        
        # Save modified config
        output_config = "prosody_aware_config.json"
        with open(output_config, 'w') as f:
            json.dump(config_template, f, indent=2)
        
        return output_config

# Example usage
def main():
    """Example usage of the voice cloning pipeline"""
    
    # Initialize pipeline
    pipeline = VoiceCloningPipeline()


    ap = AudioPreprocessor()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Extract audio from video
    extracted_wav = os.path.join(args.output_dir, "audio_en.wav")
    ap.extract_audio(args.video_path, extracted_wav)

    data_folder = os.path.join(args.output_dir, "dataset")
    os.makedirs(data_folder,exist_ok=True)

    segments = transcribe_with_whisper(extracted_wav, model_size="small.en")

    transcript_path = os.path.join(args.output_dir, "transcript_en.txt")
    # Optionally save the English transcript
    with open(transcript_path, "w", encoding="utf8") as f:
        for seg in segments:
            f.write(f"{seg['start']:.2f}–{seg['end']:.2f}: {seg['text']}\n")

    sentences = [x['text'] for x in segments]
    exemplars = extract_style_exemplars(sentences = sentences, num_exemplars=12)

    segments_info = batch_translate_segments(segments,exemplars)



    
    # Example segment information (you'd get this from your translation model)
    # segments_info = [
    #     {
    #         'start_time': 0.0,
    #         'end_time': 3.5,
    #         'original_text': "Hello, how are you today?",
    #         'translated_text': "Hola, ¿cómo estás hoy?"
    #     },
    #     {
    #         'start_time': 3.5,
    #         'end_time': 7.2,
    #         'original_text': "I'm doing well, thank you.",
    #         'translated_text': "Estoy bien, gracias."
    #     }
    # ]
    
    # Speaker reference files for voice cloning
    # speaker_references = [
    #     "reference_speaker_1.wav",
    #     "reference_speaker_2.wav"
    # ]
    
    speaker_references = slice_audio_with_margin(extracted_wav,segments,os.path.join(args.output_dir,"speaker_wavs"),margin=5)
    # Process video
    output_video = pipeline.process_video_translation(
        video_path=args.video_path,
        segments_info=segments_info,
        speaker_references=speaker_references,
        output_path=os.path.join(args.output_dir,args.output_filepath),
        target_language="ru"
    )
    
    print(f"Translated video saved to: {output_video}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run full speech translation pipeline (EN→RU)")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input English video (.mp4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs")
    parser.add_argument("--output_filepath", type=str, default="output.mp4",
                        help="Filepath to save final result")
    parser.add_argument("--asr_model", type=str,
                        default="openai/whisper-medium",
                        help="ASR model name (HuggingFace)")
    parser.add_argument("--mt_model", type=str,
                        default="Helsinki-NLP/opus-mt-en-ru",
                        help="MT model name (HuggingFace)")
    parser.add_argument("--tts_model", type=str,
                        default="tts_models/multilingual/multi-dataset/xtts_v2",
                        help="Coqui XTTSv2 model identifier")
    parser.add_argument("--speaker_wavs", nargs="*", default=[],
                        help="Optional list of WAV files (6+sec) for voice cloning")
    parser.add_argument("--split_length", type=int, default=30,
                        help="Segment length (sec) to split audio for ASR")
    parser.add_argument("--tts_on_cuda", action="store_true",
                        help="Whether to run TTS on GPU (requires enough VRAM)")
    parser.add_argument("--finetune", action="store_true",
                        help="Whether to run TTS on GPU (requires enough VRAM)")
    args = parser.parse_args()

    main()