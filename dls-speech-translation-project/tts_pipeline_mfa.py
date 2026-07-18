# tts_pipeline.py

import os
import torch
from TTS.api import TTS  # Coqui TTS
import json
import subprocess
import tempfile
import sys
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
import librosa


import tempfile
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig
# from TTS.tts.configs.shared_configs import GPTTrainerConfig, GPTArgs
from TTS.utils.manage import ModelManager
from typing import List,Dict
from prosody_mapper import ProsodyMapper
from phenome_aligner import align
class TTSWrapper:
    """
    Wrapper for Coqui XTTSv2 TTS model for voice cloning.
    Supports zero-shot (speaker_wav) and few-shot (fine-tune) synthesis.
    """
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", use_cuda=True):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        # Initialize TTS (will download model if needed)
        self.tts = TTS(model_name).to(self.device)

    def synthesize(self, text, speaker_wav = None, language="ru", output_path="tts_output.wav"):
        """
        Synthesize speech for `text` in target `language`.
        `speaker_wav` is list of paths (1 or more) of reference audio samples for the target voice.
        Returns path to output WAV.
        """
        
        result_wav = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            split_sentences=True
        )
        return result_wav


prosody_mapper = ProsodyMapper()
# ───────── Updated: Time-Aligned + Duration-Fitted TTS ─────────────────────────
def synthesize_timealigned_segments(
    translated_segments: List[Dict],
    tts_wrapper: TTSWrapper,
    speaker_wavs: List,
    temp_dir: str,
    speed_factor: float = 1.
) -> List[str]:
    """
    For each translated_segment {"start","end","ru_text"}, do:
      1) Dry-run TTS to get phoneme durations → predicted_seconds
      2) Compute length_scale = predicted_seconds / (end - start)
      3) Call tts_to_file with length_scale so output fits exactly segment length
      4) If start > prev_end, prepend silence of duration (start - prev_end)
      5) Return a list of file paths in chronological order
    """
    pieces: List[str] = []
    silence_indices = []

    # Grab audio config parameters from the model so we can convert frames → seconds
    # We can pull them from tts_wrapper after initialization:
    # hop_length = 256  # e.g. 256
    sample_rate = 24000  # e.g. 24000
    # frame_duration = hop_length / sample_rate  # seconds per mel frame
    prev_end = 0.
    target_durations = []
    # temp_dir_txt = temp_dir+"_txt"
    align_path = temp_dir+"_aligned"
    temp_dir_out = temp_dir+"_out"
    
    for idx, seg in tqdm(enumerate(translated_segments)):
        # pieces_for_seg: List[str] = []
        start = seg["start"]
        end   = seg["end"]
        ru_txt = seg["ru_text"]

        # gap = start - prev_end
        # if gap > .05:
        #     silent_path = os.path.join(temp_dir, f"silent_{idx}.wav")
        #     subprocess.run([
        #         "ffmpeg", "-y",
        #         "-f", "lavfi",
        #         "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
        #         "-t", f"{gap:.3f}",
        #         "-acodec", "pcm_s16le",
        #         silent_path
        #     ], check=True)
        #     pieces_for_seg.append(silent_path)
        # prev_end+=gap

        # seg_duration = end - start  # target duration in seconds
        
        # 1) Dry-run synthesize to get durations (no file written)
        # Coqui’s API: synthesizer.tts(text, speaker_wav, length_scale=1.0, ...)
        # returns a dict containing "durations" tensor (phoneme durations in frames).
        # We only need durations, so we directly call `tts_wrapper.tts.tts(...)`.
        # output = tts_wrapper.tts.tts(
        #     ru_txt,
        #     speaker_wav=speaker_wavs,
        #     language="ru",
        #     split_sentences=False
        # )


        
        # 3) Generate final TTS WAV with that length_scale
        tts_path = os.path.join(temp_dir, f"tts_seg_{idx}.wav")
        txt_path = os.path.join(temp_dir, f"tts_seg_{idx}.txt")
        tts_kwargs = {
            "text": ru_txt,
            "speaker_wav": speaker_wavs[idx],
            "language": "ru",
            "file_path":tts_path,
            # "speed":speed_factor,
            "split_sentences": False      # we give one chunk per call
        }

        output = tts_wrapper.tts.tts_to_file(
            **tts_kwargs
        )
        # target_durations.append(seg_duration)
        with open(txt_path, "w") as t:
            t.write(ru_txt)


    subprocess.run([
    "mfa", "align",
    temp_dir,
    "russian_mfa",
    "russian_mfa",
    align_path
    ])
    prev_end = 0.
    for idx, seg in tqdm(enumerate(translated_segments)):
        tts_path = os.path.join(temp_dir, f"tts_seg_{idx}.wav")
        tg_path = os.path.join(align_path,f"tts_seg_{idx}.TextGrid")
        out_path = os.path.join(temp_dir_out,f"tts_seg_{idx}.wav")
        os.makedirs(temp_dir_out, exist_ok=True)

        # pieces_for_seg: List[str] = []
        start = seg["start"]
        end   = seg["end"]
        # ru_txt = seg["ru_text"]
        target_durations = end - start
        gap = start - prev_end
        if gap > 0.:
            silent_path = os.path.join(temp_dir, f"silent_{idx}.wav")
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
                "-t", f"{gap:.3f}",
                "-acodec", "pcm_s16le",
                silent_path
            ], check=True)
            pieces.append(silent_path)
            # prev_end+=gap
        
        pieces.append(align(tts_path,tg_path,target_durations,out_path,translated_segments[idx]['silence']))
        prev_end=end
        # orig_audio,_ = librosa.load(speaker_wavs[idx],sr=sample_rate)
        
        # orig_audio = orig_audio[int(min(30.,float(start))*sample_rate):]
        # output = prosody_mapper.map_prosody(
        #         orig_audio=orig_audio,
        #         tts_audio=output
        # #     )
        # write(tts_path, sample_rate, (output * 32767).astype(np.int16))
        # durations_ph = len(output)/sample_rate

        # # 5) Append the TTS output itself
        # pieces_for_seg.append(tts_path)
        # print(durations_ph,seg_duration,durations_ph/seg_duration)
        # prev_end+=durations_ph

        # # 6) Extend the master pieces list
        # pieces.extend(pieces_for_seg)

    
    

    return pieces

