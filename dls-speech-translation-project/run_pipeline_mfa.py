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
from tts_pipeline_mfa import TTSWrapper,synthesize_timealigned_segments
from metrics import SpeakerSimilarity

from data_pipeline import AudioPreprocessor
from asr_wrapper import ASRWrapper
# from translation import TranslationWrapper
# from tts_pipeline import 
from metrics import SpeakerSimilarity
from pydub import AudioSegment
from TTS.api import TTS
import csv
from translation import translate_with_deepseek_r1,extract_style_exemplars,batch_translate_segments
from functools import reduce

from pydub import AudioSegment
import os

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
        end_ms   = min(total_ms, int((end_sec+margin) * 1000))

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



def extract_mono16k_wav(video_path: str, wav_path: str):
    """
    Use ffmpeg to dump audio as 16 kHz / mono / 16-bit PCM WAV.
    Whisper requires 16 kHz. 
    """
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                   # no video
        "-ac", "1",              # 1 channel (mono)
        "-ar", "16000",          # 16 kHz
        "-sample_fmt", "s16",    # 16-bit PCM
        wav_path
    ], check=True)

# ── 2)  Run Whisper with word_timestamps=True ─────────────────────────────────

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
    last_end = 0.
    for seg in result["segments"]:
        words = seg.get("words", [])
        if not words:
            continue

        # Build “speech‐only” sub‐segments by looking at inter‐word gaps:
        current_start = words[0]["start"]
        current_text_parts = [words[0]["word"]]
        total_gap = current_start-last_end
        last_end = words[0]["end"]
        

        for w in words[1:]:
            gap = w["start"] - last_end
            total_gap+=gap
            if gap <= .2:
                # Still “same” speech segment
                current_text_parts.append(w["word"])
                last_end = w["end"]
            else:
                # Gap > 50 ms → finalize the previous sub‐segment
                segments.append({
                    "start": round(current_start, 3),
                    "end":   round(last_end, 3),
                    "text":  " ".join(current_text_parts).strip(),
                    "silence": total_gap
                })
                # Start a new sub‐segment
                current_start = w["start"]
                last_end = w["end"]
                current_text_parts = [w["word"]]

        # Finalize the last sub‐segment of this Whisper segment
        segments.append({
            "start": round(current_start, 3),
            "end":   round(last_end, 3),
            "text":  " ".join(current_text_parts).strip(),
            "silence": total_gap
        })

    # Sort by start time
    segments.sort(key=lambda x: x["start"])
    return segments

def concatenate_wavs(input_wavs: List[str], output_path: str):
    """
    Given a list of WAV file paths, create a text file for ffmpeg's concat demuxer,
    then run ffmpeg -f concat -i list.txt -c copy output_path.
    """
    list_path = os.path.join(os.path.dirname(output_path), "concat_list.txt")
    with open(list_path, "w", encoding="utf8") as f:
        for wav in input_wavs:
            # ffmpeg requires each line: file 'path/to/that.wav'
            f.write(f"file '{os.path.abspath(wav)}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path
    ], check=True)

    # Optionally remove the list file
    os.remove(list_path)


# def transcribe_with_whisper(audio_path: str, model_size: str = "small.en") -> List[Dict]:
#     """
#     Run Whisper on `audio_path` to get time‐aligned English segments.
#     Returns a list of {"start": float, "end": float, "text": str}.
#     """
#     model = whisper.load_model(model_size)
#     result = model.transcribe(audio_path, verbose=False)
#     segments = []
#     for seg in result["segments"]:
#         segments.append({
#             "start": seg["start"],
#             "end":   seg["end"],
#             "text":  seg["text"].strip()
#         })
#     return segments


def extract_ref_sample(full_audio_path, sample_path="speaker_ref.wav", duration_s=15):
    sound = AudioSegment.from_file(full_audio_path)
    clip = sound[: duration_s * 1000 ]
    clip.export(sample_path, format="wav")
    return sample_path


def chunked_translate(mt, transcript, tokenizer_max_tokens=400):
    """
    Split `transcript` into chunks ≤ tokenizer_max_tokens tokens (approx.),
    translate each chunk with `mt.translate()`, and concatenate results.

    Args:
      mt: an instance of your TranslationWrapper (e.g. Helsinki-NLP/opus-mt-en-ru).
      transcript: the full English text (string).
      tokenizer_max_tokens: rough token budget per chunk (default: 400).
        MarianMT can handle ~512 tokens, but we leave some headroom.

    Returns:
      A single string containing the full Russian translation.
    """
    import re

    # 1) First, split by sentence‐ending punctuation (period, question mark, exclamation).
    #    We keep the punctuation at the end of each chunk so the model retains sentence boundaries.
    sentence_end_re = re.compile(r'(?<=[\.\?\!])\s+')
    sentences = sentence_end_re.split(transcript.strip())
    # sentences is now a list like ["Hello world.", "This is a test.", "…"]

    translations = []
    current_chunk = ""
    current_token_count = 0

    for sent in sentences:
        # Estimate token count by splitting on spaces (rough proxy).
        # If you want an exact count, you can use the HF tokenizer: len(tokenizer.encode(sent)).
        estimated_tokens = len(sent.split())
        if current_token_count + estimated_tokens < tokenizer_max_tokens:
            # Append to current chunk
            if current_chunk:
                current_chunk += " " + sent
            else:
                current_chunk = sent
            current_token_count += estimated_tokens
        else:
            # Translate the existing chunk
            if current_chunk:
                translated_chunk = mt.translate(current_chunk)
                translations.append(translated_chunk)
            # Start a new chunk with the current sentence
            current_chunk = sent
            current_token_count = estimated_tokens

    # Don’t forget the last chunk
    if current_chunk:
        translated_chunk = mt.translate(current_chunk)
        translations.append(translated_chunk)

    # Join all translated pieces with a space (or " . " if you want to reinsert periods)
    return " ".join(translations)


def chunk_text_for_tts(text: str, max_chars=150):
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            idx = text.rfind(".", start, end)
            if idx == -1:
                idx = text.rfind(" ", start, end)
            if idx == -1 or idx <= start:
                idx = end
            end = idx + 1
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]

def run_full_pipeline(args):
    ap = AudioPreprocessor()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Extract audio from video
    extracted_wav = os.path.join(args.output_dir, "audio_en.wav")
    ap.extract_audio(args.video_path, extracted_wav)

    data_folder = os.path.join(args.output_dir, "dataset")
    os.makedirs(data_folder,exist_ok=True)

    # 2. Split audio into 30s segments for ASR
    # split_len = args.split_length if args.split_length > 0 else 30
    # segments = ap.split_audio(extracted_wav, segment_length=split_len)
    
    # print(f"Split audio into {len(segments)} segments of ~{split_len}s each.")



    # # 3. ASR (on CPU to reduce GPU memory usage)
    # asr = ASRWrapper(model_name=args.asr_model, use_cuda=True,fp16=True)
    # transcript_pieces = []
    # for seg in segments:
    #     text = asr.transcribe(seg)
    #     transcript_pieces.append(text.lower())
    # transcript = " ".join(transcript_pieces).strip()
    segments = transcribe_with_whisper(extracted_wav, model_size="small.en")

    transcript_path = os.path.join(args.output_dir, "transcript_en.txt")
    # Optionally save the English transcript
    with open(transcript_path, "w", encoding="utf8") as f:
        for seg in segments:
            f.write(f"{seg['start']:.2f}–{seg['end']:.2f}: {seg['text']}\n")


    if args.finetune:
        transcript_path = os.path.join(args.output_dir, "transcript_en.txt")

        foldered_segments = []
        for file_path in segments:
            # if os.path.isfile(file_path):
            shutil.move(file_path, data_folder)
            foldered_segments.append(os.path.join(data_folder,file_path))
            # else:
            #     print(f"File not found: {file_path}")
        
        metadata_path = os.path.join(data_folder,"metadata.csv")

        with open(metadata_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["ID","Transcription","Normalized Transcription"])
            for wav, text in zip(foldered_segments, transcript_pieces):
                # Use only basename for XTTS compatibility
                writer.writerow([os.path.basename(wav), text,text])

        with open(transcript_path, "w") as f:
            f.write(transcript)
        print(f"Full English transcript saved to {transcript_path}")
        del asr
        torch.cuda.empty_cache()
        print("→ Starting TTS fine-tuning...")
        # Instantiate wrapper with base model
        tts_ft_wrapper = TTSWrapper(model_name=args.tts_model, use_cuda=args.tts_on_cuda)
        # Run fine-tune method
        tts_ft_wrapper.finetune(
            speaker_wav_dir=data_folder,
            transcripts_file="metadata.csv",
            output_dir=args.output_dir,
            use_cuda=args.tts_on_cuda
        )
        print("→ Fine-tuning complete. Reloading fine-tuned model for inference.")
        # After fine-tuning, point to the best checkpoint and its config
        best_checkpoint = os.path.join(args.output_dir, "models", "best_model.pth.tar")
        config_json = os.path.join(args.output_dir, "configs", "config.json")
        tts = TTSWrapper(model_name=best_checkpoint, use_cuda=args.tts_on_cuda)
        # Override internal TTS object to use fine-tuned weights
        tts.tts = TTS(model_name=best_checkpoint, config_path=config_json).to(tts.device)
    else:
        tts = TTSWrapper(model_name=args.tts_model, use_cuda=args.tts_on_cuda)

    # 4. MT: chunked translate to avoid token limit
    
    # mt = TranslationWrapper(model_name=args.mt_model)
    sentences = [x['text'] for x in segments]
    exemplars = extract_style_exemplars(sentences = sentences, num_exemplars=12)

    aligned = batch_translate_segments(segments,exemplars)
    # translation = translate_with_deepseek_r1(
    #     full_transcript=transcript,
    #     style_exemplars=exemplars,
    #     max_chars_per_chunk=2500,
    #     temperature=0.4
    # )
    # Optionally save the Russian segment list
    ru_path = os.path.join(args.output_dir, "aligned_ru.txt")
    with open(ru_path, "w", encoding="utf8") as f:
        for seg in aligned:
            f.write(f"{seg['start']:.2f}–{seg['end']:.2f}: {seg['ru_text']}\n")

    # translation = chunked_translate(mt, transcript,150)
    # trans_path = os.path.join(args.output_dir, "transcript_ru.txt")
    # with open(trans_path, "w") as f:
    #     f.write(translation)
    print(f"Full Russian translation saved to {ru_path}")
    # del mt
    torch.cuda.empty_cache()

    # 5. TTS: chunk the Russian translation into ~2k-char pieces

    # tts_chunks = chunk_text_for_tts(translation, max_chars=150)
    # print(f"Splitting Russian text into {len(tts_chunks)} TTS chunks.")
    # speaker_wavs = 
    # wav_parts = []
    # for i, chunk in enumerate(tts_chunks):
    #     part_wav = os.path.join(args.output_dir, f"tts_part_{i}.wav")
        
    #     tts.synthesize(text=chunk, speaker_wav= speaker_wavs,
    #                    language="ru", output_path=part_wav)
    #     part_wav_ = f"tts_part_{i}.wav"
    #     wav_parts.append(part_wav_)
    # del tts
    # torch.cuda.empty_cache()
# ####### Estimating Speed Factor
#     translation = " ".join([line['ru_text'] for line in aligned])

#     tts_chunks = chunk_text_for_tts(translation, max_chars=150)
#     print(f"Splitting Russian text into {len(tts_chunks)} TTS chunks.")
#     speaker_wavs = args.speaker_wavs[0] if len(args.speaker_wavs) else extract_ref_sample(extracted_wav,"speaker_ref.wav")
#     # wav_parts = []
#     raw_duration = 0
#     for i, chunk in enumerate(tts_chunks):
#         # part_wav = os.path.join(args.output_dir, f"tts_part_{i}.wav")
        
#         audio = tts.synthesize(text=chunk, 
#                                speaker_wav= extract_ref_sample(extracted_wav,duration_s=90),
#                                 language="ru")
#         # part_wav_ = f"tts_part_{i}.wav"
#         # wav_parts.append(part_wav_)
#         raw_duration+= len(audio)
        
#     total_length = reduce(lambda x,y: x+float(y['end'])-float(y['start']),aligned,0)
#     speed_factor = (raw_duration/24000)/(total_length)
#     print("Speed Factor: ",speed_factor)
# ########
    tmpdir = tempfile.mkdtemp(prefix="tts_align_")
    # tts = TTSWrapper(model_name="tts_models/multilingual/multi-dataset/xtts_v2", use_cuda=True, fp16=True)
    # speaker_wavs = []  # or your list of reference WAVs if you do zero‐shot cloning

    speaker_wavs = slice_audio_with_margin(extracted_wav,segments,os.path.join(args.output_dir,"speaker_wavs"),margin=20)
    pieces = synthesize_timealigned_segments(aligned, tts, speaker_wavs = speaker_wavs, temp_dir=tmpdir,speed_factor = 1.)
    final_ru_wav = os.path.join(args.output_dir, "audio_ru.wav")
    concatenate_wavs(pieces, final_ru_wav)
    del tts
    # 6. Concatenate all TTS parts into one audio_ru.wav
    # filelist_path = os.path.join(args.output_dir, "tts_filelist.txt")
    # with open(filelist_path, "w") as f:
    #     for wav in wav_parts:
    #         f.write(f"file '{wav}'\n")
    # # final_tts = os.path.join(args.output_dir, "audio_ru.wav")
    # final_tts = os.path.join(args.output_dir, "final_russian.wav")
    # concatenate_wavs(pieces, final_tts)
    # ffmpeg_cmd = (
    #     f"ffmpeg -y -f concat -safe 0 -i {filelist_path} "
    #     f"-c copy {final_tts}"
    # )
    # os.system(ffmpeg_cmd)
    print(f"Synthesized Russian audio saved to {final_ru_wav}")

    # 7. Re-mux audio+video into the final dubbed MP4
    final_video = os.path.join(args.output_dir, "dubbed_video.mp4")
    ap.combine_audio_video(args.video_path, final_ru_wav, final_video)
    print(f"Final dubbed video saved to {final_video}")

    # 8. (Optional) Speaker similarity metric
    # if args.speaker_wavs:
    sim = SpeakerSimilarity().evaluate(extracted_wav, final_ru_wav)
    print(f"Speaker cosine similarity: {sim:.4f}")
    # else:
    #     print("No speaker_wavs provided; skipping speaker similarity.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full speech translation pipeline (EN→RU)")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input English video (.mp4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs")
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
    run_full_pipeline(args)
