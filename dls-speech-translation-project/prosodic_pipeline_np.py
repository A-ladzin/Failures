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

class ProsodyTransfer:
    """Transfer prosodic features from source to target speech"""
    
    def __init__(self):
        self.f0_shift_semitones = 0  # Will be calculated based on speaker characteristics
    
    def align_prosody(self, source_f0: np.ndarray, target_duration: float, 
                     source_duration: float) -> np.ndarray:
        """Align prosodic features to match target duration"""
        # Time-stretch F0 curve to match target duration
        stretch_factor = target_duration / source_duration
        
        # Resample F0 curve
        original_time = np.linspace(0, 1, len(source_f0))
        target_time = np.linspace(0, 1, int(len(source_f0) * stretch_factor))
        
        interp_func = interp1d(original_time, source_f0, kind='cubic', 
                              bounds_error=False, fill_value='extrapolate')
        aligned_f0 = interp_func(target_time)
        
        return aligned_f0
    
    def adapt_f0_range(self, source_f0: np.ndarray, target_speaker_f0_stats: Dict) -> np.ndarray:
        """Adapt F0 range to target speaker's characteristics"""
        # Calculate source statistics
        source_mean = np.mean(source_f0[source_f0 > 0])
        source_std = np.std(source_f0[source_f0 > 0])
        
        # Target statistics
        target_mean = target_speaker_f0_stats['mean']
        target_std = target_speaker_f0_stats['std']
        
        # Normalize and scale
        normalized_f0 = (source_f0 - source_mean) / source_std
        adapted_f0 = normalized_f0 * target_std + target_mean
        
        # Ensure positive values for voiced regions
        adapted_f0[source_f0 <= 0] = 0
        
        return adapted_f0

class VoiceCloningPipeline:
    """Main pipeline for voice cloning with prosody preservation"""
    
    def __init__(self, model_path: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.tts_model = TTS(model_path, gpu=True)
        self.prosody_extractor = ProsodyExtractor()
        self.prosody_transfer = ProsodyTransfer()
        self.speaker_embeddings = {}
        self.speaker_f0_stats = {}
    
    def extract_speaker_profile(self, reference_audio_paths: List[str], speaker_id: str) -> Dict:
        """Extract speaker characteristics from reference audio"""
        all_f0 = []
        all_energy = []
        
        for audio_path in reference_audio_paths:
            audio, sr = librosa.load(audio_path, sr=24000)
            
            # Extract prosodic features
            f0 = self.prosody_extractor.extract_f0(audio)
            energy = self.prosody_extractor.extract_energy(audio)
            
            all_f0.extend(f0[f0 > 0])  # Only voiced segments
            all_energy.extend(energy)
        
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
            'reference_paths': reference_audio_paths
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
            
            segment = SegmentData(
                start_time=seg_info['start_time'],
                end_time=seg_info['end_time'],
                original_text=seg_info['original_text'],
                translated_text=seg_info['translated_text'],
                original_audio=segment_audio,
                f0_curve=f0_curve,
                energy_curve=energy_curve,
                duration=seg_info['end_time'] - seg_info['start_time'],
                prosodic_features=rhythm_features
            )
            
            segments.append(segment)
        
        return segments
    
    def synthesize_with_prosody_transfer(self, segment: SegmentData, 
                                       speaker_reference: List[str],
                                       speaker_id: str,
                                       language: str = "en") -> np.ndarray:
        """Synthesize speech with transferred prosody"""
        
        # First pass: Generate basic translated speech
        basic_output = self.tts_model.tts_to_file(
            text=segment.translated_text,
            speaker_wav=speaker_reference,
            language=language,
            file_path=None  # Return audio data directly
        )
        
        # Load the generated audio for prosody manipulation
        if isinstance(basic_output, str):  # If file path returned
            translated_audio, _ = librosa.load(basic_output, sr=24000)
        else:  # If audio data returned directly
            translated_audio = np.array(basic_output)
        
        # Extract prosody from translated speech
        translated_f0 = self.prosody_extractor.extract_f0(translated_audio)
        
        # Transfer prosody from original segment
        if speaker_id in self.speaker_f0_stats:
            # Adapt original F0 to target speaker
            adapted_f0 = self.prosody_transfer.adapt_f0_range(
                segment.f0_curve, 
                self.speaker_f0_stats[speaker_id]
            )
            
            # Align to translated speech duration
            aligned_f0 = self.prosody_transfer.align_prosody(
                adapted_f0,
                len(translated_audio) / 24000,
                segment.duration
            )
            
            # Apply prosody transfer (this would require additional TTS model modifications)
            # For now, we'll use the basic output with some post-processing
            final_audio = self._apply_prosody_modifications(
                translated_audio, aligned_f0, segment.energy_curve
            )
        else:
            final_audio = translated_audio
        
        return final_audio
    
    def _apply_prosody_modifications(self, audio: np.ndarray, 
                                   target_f0: np.ndarray, 
                                   target_energy: np.ndarray) -> np.ndarray:
        """Apply prosodic modifications to synthesized audio"""
        # This is a simplified version - in practice, you'd use more sophisticated
        # pitch shifting and energy modification techniques
        
        # Basic pitch shifting using librosa
        try:
            # Calculate average pitch shift needed
            current_f0 = self.prosody_extractor.extract_f0(audio)
            
            if len(current_f0) > 0 and len(target_f0) > 0:
                current_mean = np.mean(current_f0[current_f0 > 0])
                target_mean = np.mean(target_f0[target_f0 > 0])
                
                if current_mean > 0 and target_mean > 0:
                    semitone_shift = 12 * np.log2(target_mean / current_mean)
                    
                    # Apply pitch shift
                    if abs(semitone_shift) > 0.5:  # Only shift if significant difference
                        audio = librosa.effects.pitch_shift(
                            audio, sr=24000, n_steps=semitone_shift
                        )
        except Exception as e:
            print(f"Prosody modification failed: {e}")
        
        return audio
    
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
        for segment in segments:
            print(f"Processing segment: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
            
            translated_audio = self.synthesize_with_prosody_transfer(
                segment, speaker_references, speaker_id, target_language
            )
            
            translated_segments.append({
                'audio': translated_audio,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration
            })
        
        # Reconstruct full audio track
        full_audio = self._reconstruct_audio_track(translated_segments, video.duration)
        
        # Create new video with translated audio
        final_video = video.set_audio(mp.AudioArrayClip(full_audio, fps=24000))
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
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
    


    
    # Example segment information (you'd get this from your translation model)
    segments_info = [
        {
            'start_time': 0.0,
            'end_time': 3.5,
            'original_text': "Hello, how are you today?",
            'translated_text': "Hola, ¿cómo estás hoy?"
        },
        {
            'start_time': 3.5,
            'end_time': 7.2,
            'original_text': "I'm doing well, thank you.",
            'translated_text': "Estoy bien, gracias."
        }
    ]
    
    # Speaker reference files for voice cloning
    speaker_references = [
        "reference_speaker_1.wav",
        "reference_speaker_2.wav"
    ]
    
    # Process video
    output_video = pipeline.process_video_translation(
        video_path="input_video.mp4",
        segments_info=segments_info,
        speaker_references=speaker_references,
        output_path="translated_video.mp4",
        target_language="es"
    )
    
    print(f"Translated video saved to: {output_video}")

if __name__ == "__main__":
    main()