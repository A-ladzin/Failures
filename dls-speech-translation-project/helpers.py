from typing import List,Dict
import whisper
from pydub import AudioSegment
import subprocess
import tempfile

def transcribe_with_whisper(
    audio_path: str,
    model_size: str = "small.en"
) -> List[Dict]:
    """
    Runs Whisper on `audio_path` with per-word timestamps, then groups words
    into contiguous “speech” segments whenever the gap between consecutive words
    exceeds 0.05 s (50 ms). Returns a list of {"start": float, "end": float, "text": str}.
    """
    model = whisper.load_model(model_size)
    # Note: you must have a Whisper version ≥ v2024.4.30 that supports `word_timestamps=True`.
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose =True
    )
    # result["segments"] is a list of dicts, each has a "words" array:
    #    [ {"word": "Hello", "start": 0.02, "end": 0.30}, ... ]
    segments = []
    for seg in result["segments"]:
        words = seg.get("words", [])
        if not words:
            continue

        # Build “speech‐only” sub‐segments by looking at inter‐word gaps:
        current_start = words[0]["start"]
        current_text_parts = [words[0]["word"]]
        last_end = words[0]["end"]

        for w in words[1:]:
            gap = w["start"] - last_end
            if gap <= 1.:
                # Still “same” speech segment
                current_text_parts.append(w["word"])
                last_end = w["end"]
            else:
                # Gap > 50 ms → finalize the previous sub‐segment
                segments.append({
                    "start": round(current_start, 3),
                    "end":   round(last_end, 3),
                    "text":  " ".join(current_text_parts).strip()
                })
                # Start a new sub‐segment
                current_start = w["start"]
                last_end = w["end"]
                current_text_parts = [w["word"]]

        # Finalize the last sub‐segment of this Whisper segment
        segments.append({
            "start": round(current_start, 3),
            "end":   round(last_end, 3),
            "text":  " ".join(current_text_parts).strip()
        })

    # Sort by start time
    segments.sort(key=lambda x: x["start"])
    return segments

import os
import time
from huggingface_hub import InferenceClient






def slice_audio_with_margin(
    input_wav: str,
    segments: list[dict],
    output_dir: str,
    margin: float = 0.5
):
    """
    Slices `input_wav` into multiple clips according to `segments`, adding `margin` seconds
    of padding before and after each segment. Saves each clip as a separate WAV in `output_dir`.

    Args:
        input_wav: Path to the full audio file (must be WAV or a format pydub can read).
        segments:  A list of dicts, each with:
                       {
                         "start": float,  # start time in seconds
                         "end":   float   # end time in seconds
                       }
        output_dir: Where to write the sliced WAVs.
        margin:     Time in seconds to pad before `start` and after `end`.
                    (If `start - margin` < 0, it clips at 0; if `end + margin` > audio_length, it clips at end.)
    """
    # 1) Load the full audio as a pydub AudioSegment
    audio = AudioSegment.from_file(input_wav)
    total_ms = len(audio)  # total length in milliseconds

    # 2) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    # 3) For each segment, compute slice boundaries (in ms), apply margin, then export
    for idx, seg in enumerate(segments, start=1):
        start_sec = seg["start"]
        end_sec   = seg["end"]

        # Convert to milliseconds and apply margin
        start_ms = max(0, int((start_sec - margin) * 1000))
        end_ms   = min(total_ms, int((end_sec + margin) * 1000))

        # Slice out that portion
        clip = audio[start_ms:end_ms]

        # Build an output filename, e.g. segment_1_0.50_3.75.wav
        out_name = f"segment_{idx}_{start_sec - margin:.2f}_{end_sec + margin:.2f}.wav"
        out_path = os.path.join(output_dir, out_name)

        # Export as WAV (16-bit PCM, same sample rate as original)
        clip.export(out_path, format="wav")
        paths.append(out_path)
        # print(f"  → Saved clip {idx}: {out_path} [ {start_ms}–{end_ms} ms ]")
    return paths




def combine_audio_video(video_path: str, new_audio_path: str, output_path: str):
    """
    Replaces the audio track of `video_path` with a mix of the original audio
    and `new_audio_path`, then writes the result to `output_path`.

    Steps:
    1. Extract original audio from the input video to a temporary WAV.
    2. Resample both original and new audio to 48 kHz stereo (so they match).
    3. Mix them together (new audio at full volume, original at 20%).
    4. Remux the mixed audio back into the video frames.

    Requires ffmpeg on PATH.
    """
    # Create temporary files
    tmp_dir = tempfile.mkdtemp(prefix="av_mixer_")
    orig_audio = os.path.join(tmp_dir, "orig_audio.wav")
    orig_48k = os.path.join(tmp_dir, "orig_48k_stereo.wav")
    new_48k = os.path.join(tmp_dir, "new_48k_stereo.wav")
    mixed_audio = os.path.join(tmp_dir, "mixed_audio.wav")

    try:
        # 1) Extract original audio from the video (as WAV, keep channels & sample rate info)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            orig_audio
        ], check=True)

        # 2) Resample original to 48kHz stereo
        subprocess.run([
            "ffmpeg", "-y",
            "-i", orig_audio,
            "-ar", "48000",
            "-ac", "2",
            orig_48k
        ], check=True)

        # 3) Resample new dubbed audio to 48kHz stereo
        subprocess.run([
            "ffmpeg", "-y",
            "-i", new_audio_path,
            "-ar", "48000",
            "-ac", "2",
            new_48k
        ], check=True)

        # 4) Mix original (20% volume) + new audio (100% volume)
        #    - duration=first: stop when the shorter audio ends
        #    - dropout_transition=2: smooth fade if one stream ends early
        mix_filter = (
            "[0:a]volume=0.2[a0];"
            "[1:a]volume=1.0[a1];"
            "[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[m]"
        )
        subprocess.run([
            "ffmpeg", "-y",
            "-i", orig_48k,
            "-i", new_48k,
            "-filter_complex", mix_filter,
            "-map", "[m]",
            "-c:a", "pcm_s16le",
            mixed_audio
        ], check=True)

        # 5) Mux mixed audio back into the video (copying video stream)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", mixed_audio,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ], check=True)

    finally:
        # Clean up temporary files and directory
        for f in (orig_audio, orig_48k, new_48k, mixed_audio):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

