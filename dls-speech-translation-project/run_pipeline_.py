# run_pipeline.py

import os
import argparse
from data_pipeline import AudioPreprocessor
from asr_wrapper import ASRWrapper
from translation import TranslationWrapper
from tts_pipeline import TTSWrapper
from metrics import SpeakerSimilarity



from pydub import AudioSegment

def extract_ref_sample(full_audio_path, sample_path="speaker_ref.wav", duration_s=6):
    sound = AudioSegment.from_file(full_audio_path)
    clip = sound[: duration_s * 1000 ]
    clip.export(sample_path, format="wav")
    return sample_path




def run_full_pipeline(args):
    # 1. Preprocess: extract audio from video
    ap = AudioPreprocessor()
    os.makedirs(args.output_dir, exist_ok=True)
    extracted_wav = os.path.join(args.output_dir, "audio_en.wav")
    ap.extract_audio(args.video_path, extracted_wav)
    print(f"Extracted audio to {extracted_wav}")


    # (Optional) Split audio if needed (depending on ASR model limits)
    segments = [extracted_wav]
    if args.split_length > 0:
        segments = ap.split_audio(extracted_wav, segment_length=args.split_length)
        print(f"Split audio into {len(segments)} segments.")

    # 2. ASR: Transcribe each segment
    asr = ASRWrapper(model_name=args.asr_model)
    transcript = ""
    for seg in segments:
        text = asr.transcribe(seg)
        transcript += text + " "
    transcript = transcript.strip()
    transcript_path = os.path.join(args.output_dir, "transcript_en.txt")
    with open(transcript_path, "w") as f:
        f.write(transcript)
    print(f"ASR transcription saved to {transcript_path}")

    # 3. MT: Translate to Russian
    mt = TranslationWrapper(model_name=args.mt_model)
    # (For simplicity, translate entire transcript at once)
    translation = mt.translate(transcript)
    trans_path = os.path.join(args.output_dir, "transcript_ru.txt")
    with open(trans_path, "w") as f:
        f.write(translation)
    print(f"Translated text saved to {trans_path}")

    # 4. TTS: Synthesize Russian speech
    tts = TTSWrapper(model_name=args.tts_model)
    output_tts = os.path.join(args.output_dir, "audio_ru.wav")
    tts.synthesize(
        text=translation,
        speaker_wav=args.speaker_wavs if len(args.speaker_wavs) else extract_ref_sample(extracted_wav),
        language="ru",
        output_path=output_tts
    )
    print(f"Synthesized speech saved to {output_tts}")

    # (Optional) Fine-tune TTS before synthesizing if requested
    if args.use_finetune:
        print("Starting TTS fine-tuning on speaker samples...")
        # Prepare transcripts_file mapping speaker_wavs to text (in this demo, re-use translation)
        # Real scenario: use known transcripts for each sample.
        transcripts_file = os.path.join(args.output_dir, "speaker_labels.txt")
        with open(transcripts_file, "w") as f:
            for wav in args.speaker_wavs:
                f.write(f"{wav}|{translation}\n")
        tts.finetune(args.speaker_wav_dir, transcripts_file, output_dir=os.path.join(args.output_dir, "tts_finetuned"))

    # 5. Align audio to video
    final_video = os.path.join(args.output_dir, "dubbed_video.mp4")
    ap.combine_audio_video(args.video_path, output_tts, final_video)
    print(f"Final dubbed video saved to {final_video}")

    # 6. Compute speaker similarity metric
    sim = SpeakerSimilarity().evaluate(args.speaker_wavs[0], output_tts)
    print(f"Speaker cosine similarity: {sim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full speech translation pipeline (EN→RU)")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input English video (e.g. .mp4)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--asr_model", type=str, default="openai/whisper-medium", help="ASR model name")
    parser.add_argument("--mt_model", type=str, default="Helsinki-NLP/opus-mt-en-ru", help="MT model name")
    parser.add_argument("--tts_model", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="XTTS model identifier")
    parser.add_argument("--speaker_wavs", nargs='*', default=[], help="Optional WAV files for voice cloning")
    parser.add_argument("--split_length", type=int, default=0, help="Segment length (seconds) for splitting long audio (0 = no split)")
    parser.add_argument("--use_finetune", action='store_true', help="Whether to fine-tune TTS on speaker samples")
    args = parser.parse_args()
    run_full_pipeline(args)
