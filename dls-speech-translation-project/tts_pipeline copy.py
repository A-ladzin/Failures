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
        
        # if speaker_wav is None:
        #         # tts_pipeline.py — adjust synthesize to allow no speaker_wav
        #     kwargs = {"text": text, "file_path": output_path, "language": language}
        #     if speaker_wav:
        #         kwargs["speaker_wav"] = speaker_wav
        #     # If no speaker_wav, TTS will use its built-in default voice
        #     self.tts.tts_to_file(**kwargs)
        #     return output_path
        # The TTS API splits text into sentences internally; disable splitting if needed
        # self.tts.tts_to_file(
        #     text=text,
        #     file_path=output_path,
        #     speaker_wav=speaker_wav,
        #     language=language,
        #     split_sentences=True
        # )
        result_wav = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            split_sentences=True
        )
        return result_wav

    # def finetune(self,
    #             speaker_wav_dir: str,
    #             transcripts_file: str,
    #             output_dir: str,
    #             use_cuda: bool = True):
    #     """
    #     Fine-tune the XTTSv2 model using its original config with minimal overrides.
    #     """

    #     # 1) Locate or download original config for the base model
    #     from TTS.utils.manage import ModelManager
    #     manager = ModelManager()
    #     model_path, config_path, _ = manager.download_model(self.model_name)

    #     with open(config_path, "r") as f:
    #         config = json.load(f)
    #     print(config)

    #     # 2) Modify config for fine-tuning
    #     config["run_name"] = "xtts_finetune"
    #     config["output_path"] = output_dir

    #     config['eval_split_size'] = 0.1
    #     config["datasets"][0]["formatter"] = "coqui"
    #     config["datasets"][0]["meta_file_train"] = transcripts_file
    #     config["datasets"][0]["path"] = speaker_wav_dir



    #     # config["trainer"]["gpus"] = 1 if use_cuda else 0

    #     # 3) Save modified config to temp file
    #     with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp_config:
    #         json.dump(config, tmp_config, indent=2)
    #         tmp_config.flush()
    #         config_path = tmp_config.name

    #     # 4) Launch training using Coqui's training module
    #     cmd = [
    #         sys.executable, "-m", "TTS.bin.train_tts",
    #         "--config_path", config_path,
    #         "--restore_path", model_path
    #     ]

    #     try:
    #         print("Running fine-tuning: ", " ".join(str(x) for x in cmd))
    #         subprocess.run(cmd, check=True)
    #     finally:
    #         os.remove(config_path)

    # def finetune(self,
    #             speaker_wav_dir: str,
    #             transcripts_file: str,
    #             output_dir: str,
    #             use_cuda: bool = True):
    #     """
    #     Fine-tune only the GPT encoder of the XTTSv2 model using train_gpt_xtts.py.
    #     """

    #     import tempfile
    #     from pathlib import Path
    #     from TTS.utils.manage import ModelManager

    #     # Download model and get checkpoint path
    #     model_path, config_path, _ = ModelManager().download_model(self.model_name)

    #     # Build minimal config for GPT fine-tuning
    #     config_dict = {
    #         "restore_path": str(model_path),
    #         "output_path": str(output_dir),
    #         "run_name": "xtts_gpt_finetune",
    #         "datasets": [
    #             {
    #                 "formatter": "ljspeech",  # Assumes LJSpeech-style metadata
    #                 "meta_file_train": str(transcripts_file),
    #                 "path": str(speaker_wav_dir)
    #             }
    #         ],
    #         "batch_size": 16,
    #         "eval_batch_size": 16,
    #         "num_loader_workers": 2,
    #         "num_steps": 1000,
    #         "save_step": 100,
    #         "eval_step": 100,
    #         "lr": 1e-4,
    #         "mixed_precision": True,
    #         "output_logger": True,
    #         "eval_split_size": 0.1
    #     }

    #     # Write temporary config JSON
    #     with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp_config:
    #         json.dump(config_dict, tmp_config, indent=2)
    #         tmp_config.flush()
    #         config_path = tmp_config.name

    #     # Train GPT encoder using correct script
    #     train_script = Path("coqui-ai-TTS/recipes/ljspeech/xtts_v2/train_gpt_xtts.py")
    #     cmd = [
    #         sys.executable, str(train_script),
    #         "--config", config_path
    #     ]

    #     try:
    #         print("Running GPT encoder fine-tuning:", " ".join(str(x) for x in cmd))
    #         subprocess.run(cmd, check=True)
    #     finally:
    #         os.remove(config_path)

# def synthesize_timealigned_segments(
#     translated_segments: List[Dict],
#     tts_wrapper: TTSWrapper,
#     tts_speaker_wavs: List[str],
#     temp_dir: str
# ) -> List[str]:
#     """
#     For each entry in translated_segments [{"start","end","ru_text"}...], generate a WAV that
#     is pasted onto a silent cushion so that it begins at exactly 'start' seconds.
#     Returns a list of file‐paths (in chronological order), each is either:
#       - a silent WAV (if the gap > 0), or
#       - a TTS‐generated WAV for that ru_text.
#     We'll later concatenate them all end‐to‐end so that the final Russian WAV spans
#     exactly [0 ... last_segment_end].
#     """
#     pieces: List[str] = []
#     prev_end = 0.0

#     for idx, seg in enumerate(translated_segments):
#         start = seg["start"]
#         ru_text = seg["ru_text"]
#         end   = seg["end"]

#         # 3.1. Create a silent WAV if there's a gap between prev_end and this start
#         gap = start - prev_end
#         # if gap > 0.005:  # anything >5 ms => insert silence
#         #     silent_path = os.path.join(temp_dir, f"silent_{idx}.wav")
#         #     # Generate a silent WAV of duration = `gap` seconds at 24 kHz mono (TTS default)
#         #     subprocess.run([
#         #         "ffmpeg", "-y",
#         #         "-f", "lavfi",
#         #         "-i", f"anullsrc=channel_layout=mono:sample_rate=48000",
#         #         "-t", f"{gap:.3f}",
#         #         "-acodec", "pcm_s16le",
#         #         silent_path
#         #     ], check=True)
#         #     pieces.append(silent_path)

#         # 3.2. Generate the TTS of ru_text for this segment
#         tts_path = os.path.join(temp_dir, f"tts_seg_{idx}.wav")
#         tts_wrapper.synthesize(
#             text=ru_text,
#             speaker_wav=tts_speaker_wavs,
#             language="ru",
#             output_path=tts_path
#         )
#         pieces.append(tts_path)

#         prev_end = end

#     return pieces

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

    # Grab audio config parameters from the model so we can convert frames → seconds
    # We can pull them from tts_wrapper after initialization:
    hop_length = 256  # e.g. 256
    sample_rate = 24000  # e.g. 24000
    frame_duration = hop_length / sample_rate  # seconds per mel frame
    prev_end = 0.
    for idx, seg in tqdm(enumerate(translated_segments)):
        pieces_for_seg: List[str] = []
        start = seg["start"]
        end   = seg["end"]
        ru_txt = seg["ru_text"]

        gap = start - prev_end
        if gap > .05:
            silent_path = os.path.join(temp_dir, f"silent_{idx}.wav")
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
                "-t", f"{gap*0.9:.3f}",
                "-acodec", "pcm_s16le",
                silent_path
            ], check=True)
            pieces_for_seg.append(silent_path)
            prev_end+=gap*0.9

        seg_duration = end - prev_end  # target duration in seconds

        # 1) Dry-run synthesize to get durations (no file written)
        # Coqui’s API: synthesizer.tts(text, speaker_wav, length_scale=1.0, ...)
        # returns a dict containing "durations" tensor (phoneme durations in frames).
        # We only need durations, so we directly call `tts_wrapper.tts.tts(...)`.
        output = tts_wrapper.tts.tts(
            ru_txt,
            speaker_wav=speaker_wavs,
            language="ru",
            split_sentences=False
        )

        # durations_ph = output[0]["durations"][0].cpu().tolist()
        # print(output)
        output = [int(i != 0) for i in output]
        durations_ph = np.where(np.array(output) != 0)[0].max()
        

        
        # print(durations_ph)
        # Sum phoneme durations (in mel frames) → total_frames
        # total_frames = sum(durations_ph)
        # predicted_seconds = total_frames * frame_duration
        predicted_seconds = durations_ph/sample_rate

        # 2) Compute length_scale
        # If predicted_seconds = 3.0s and seg_duration = 4.0s, we want length_scale < 1 → slow down.
        # length_scale = predicted_seconds / seg_duration
        length_scale = predicted_seconds / seg_duration if seg_duration > 0 else 1.0

        # 3) Generate final TTS WAV with that length_scale
        tts_path = os.path.join(temp_dir, f"tts_seg_{idx}.wav")
        tts_kwargs = {
            "text": ru_txt,
            "speaker_wav": speaker_wavs[idx],
            "language": "ru",
            # "file_path": tts_path,
            "speed":np.clip(length_scale,speed_factor*0.5,speed_factor*1.5),
            # "length_scale": length_scale,  # override default 1.0
            "split_sentences": False      # we give one chunk per call
        }

        output = tts_wrapper.tts.tts(
            **tts_kwargs
        )
        output = np.array(output)

        out_b = [int(i != 0) for i in output]
        durations_ph = np.where(np.array(out_b) != 0)[0].max()
        # durations_ph = len(output)
        
        output = np.array(output[:durations_ph],dtype=np.float32)
        # output_path = "output.wav"
        sample_rate = 24000  # or the sample rate used by your TTS model
        write(tts_path, sample_rate, (output * 32767).astype(np.int16))
        # tts_wrapper.tts.tts_to_file(**tts_kwargs)
        
        
        orig_audio,_ = librosa.load(speaker_wavs[idx],sr=24000)
        
        orig_audio = orig_audio[int(min(30.,float(start))*24000):]
        output = prosody_mapper.map_prosody(
                orig_audio=orig_audio,
                tts_audio=output
            )
        write(tts_path, sample_rate, (output * 32767).astype(np.int16))
        durations_ph = len(output)/24000
        # 4) If there's a gap between prev_end and this segment's start, prepend silence
        # gap = start - real_end
        # if gap > .2:
        #     silent_path = os.path.join(temp_dir, f"silent_{idx}.wav")
        #     subprocess.run([
        #         "ffmpeg", "-y",
        #         "-f", "lavfi",
        #         "-i", "anullsrc=channel_layout=mono:sample_rate=24000",
        #         "-t", f"{gap*0.9:.3f}",
        #         "-acodec", "pcm_s16le",
        #         silent_path
        #     ], check=True)
        #     pieces_for_seg.append(silent_path)
        #     real_end+=gap*0.9

        # 5) Append the TTS output itself
        pieces_for_seg.append(tts_path)
        print(durations_ph,seg_duration,durations_ph/seg_duration)
        prev_end+=durations_ph

        # 6) Extend the master pieces list
        pieces.extend(pieces_for_seg)

    return pieces

