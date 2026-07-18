# run_pipeline.py

import os
import argparse
import torch
import subprocess
import shutil

from data_pipeline import AudioPreprocessor
from asr_wrapper import ASRWrapper
from translation import TranslationWrapper
from tts_pipeline import TTSWrapper
from metrics import SpeakerSimilarity
from pydub import AudioSegment
from TTS.api import TTS
import csv
from translation import translate_with_deepseek_r1,extract_style_exemplars



def extract_ref_sample(full_audio_path, sample_path="speaker_ref.wav", duration_s=20):
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
    split_len = args.split_length if args.split_length > 0 else 30
    segments = ap.split_audio(extracted_wav, segment_length=split_len)
    print(f"Split audio into {len(segments)} segments of ~{split_len}s each.")



    # 3. ASR (on CPU to reduce GPU memory usage)
    asr = ASRWrapper(model_name=args.asr_model, use_cuda=True,fp16=True)
    transcript_pieces = []
    for seg in segments:
        text = asr.transcribe(seg)
        transcript_pieces.append(text.lower())
    transcript = " ".join(transcript_pieces).strip()
    






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
    mt = TranslationWrapper(model_name=args.mt_model)
    exemplars = extract_style_exemplars(transcript, num_exemplars=3)
    translation = translate_with_deepseek_r1(
        full_transcript=transcript,
        style_exemplars=exemplars,
        max_chars_per_chunk=2500,
        temperature=0.4
    )
    # translation = chunked_translate(mt, transcript,150)
    trans_path = os.path.join(args.output_dir, "transcript_ru.txt")
    with open(trans_path, "w") as f:
        f.write(translation)
    print(f"Full Russian translation saved to {trans_path}")
    del mt
    torch.cuda.empty_cache()

    # 5. TTS: chunk the Russian translation into ~2k-char pieces
    
    tts_chunks = chunk_text_for_tts(translation, max_chars=150)
    print(f"Splitting Russian text into {len(tts_chunks)} TTS chunks.")
    speaker_wavs = args.speaker_wavs[0] if len(args.speaker_wavs) else extract_ref_sample(extracted_wav,"speaker_ref.wav")
    wav_parts = []
    for i, chunk in enumerate(tts_chunks):
        part_wav = os.path.join(args.output_dir, f"tts_part_{i}.wav")
        
        tts.synthesize(text=chunk, speaker_wav= speaker_wavs,
                       language="ru", output_path=part_wav)
        part_wav_ = f"tts_part_{i}.wav"
        wav_parts.append(part_wav_)
    del tts
    torch.cuda.empty_cache()

    # 6. Concatenate all TTS parts into one audio_ru.wav
    filelist_path = os.path.join(args.output_dir, "tts_filelist.txt")
    with open(filelist_path, "w") as f:
        for wav in wav_parts:
            f.write(f"file '{wav}'\n")
    final_tts = os.path.join(args.output_dir, "audio_ru.wav")
    ffmpeg_cmd = (
        f"ffmpeg -y -f concat -safe 0 -i {filelist_path} "
        f"-c copy {final_tts}"
    )
    os.system(ffmpeg_cmd)
    print(f"Synthesized Russian audio saved to {final_tts}")

    # 7. Re-mux audio+video into the final dubbed MP4
    final_video = os.path.join(args.output_dir, "dubbed_video.mp4")
    ap.combine_audio_video(args.video_path, final_tts, final_video)
    print(f"Final dubbed video saved to {final_video}")

    # 8. (Optional) Speaker similarity metric
    # if args.speaker_wavs:
    sim = SpeakerSimilarity().evaluate(extracted_wav, final_tts)
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
