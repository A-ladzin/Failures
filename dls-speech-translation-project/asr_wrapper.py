# asr_wrapper.py
import torch
from transformers import pipeline
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
import torch
import torchaudio




class ASRWrapper:
    """
    Wrapper for Automatic Speech Recognition. Uses a HuggingFace pipeline.
    """
    def __init__(self, model_name="openai/whisper-medium",use_cuda = True,fp16 = True):
        # Initialize ASR pipeline; cache_dir can be set in config if needed.
        device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        torch_dtype=torch.float16 if fp16 else torch.float32
        self.asr = pipeline("automatic-speech-recognition", 
                            torch_dtype=torch_dtype,
                            model=model_name,
                            device=device)

    def transcribe(self, audio_path):
        """
        Transcribe speech from `audio_path` to text (English).
        Returns a string of the transcription.
        """
        result = self.asr(audio_path)
        text = result["text"]
        # Postprocess (strip/cleanup)
        return text.strip()


    def get_whisper_token_timestamps(self, audio_path,model_name="openai/whisper-medium"):
        """
        Returns a list of dictionaries: [{"token": "Hello", "start": 0.02, "end": 0.54}, ...]
        for every word/subword that Whisper emits, covering the full audio.
        """
        # 1) Load audio (16 kHz)
        speech_array, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            speech_array = resampler(speech_array)
        speech = speech_array.squeeze().numpy()

        # 2) Load model + tokenizer
        tokenizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe", language="en")
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.config.forced_decoder_ids = tokenizer.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )

        # 3) Prepare inputs
        inputs = tokenizer(speech, return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features  # (1, seq_len)

        # 4) Generate with timestamps
        #    The `return_timestamps=True` flag instructs Whisper to return a list of (token_id, start, end).
        with torch.no_grad():
            # Outputs token IDs + start/end times
            generated_ids = model.generate(
                input_features,
                max_new_tokens=1000,
                return_timestamps=True,
                output_scores=False,
                return_dict_in_generate=True
            )
        # `generated_ids` has .sequences (token IDs) and .sequences_timestamps (start/end)
        token_ids = generated_ids.sequences[0].tolist()
        timestamps = generated_ids.sequences_timestamps[0].tolist()
        # `timestamps` is a flat list: [token_id, start_time, end_time, token_id2, start_time2, end_time2, ...]
        # We can zip them together:
        tokens, token_starts, token_ends = [], [], []
        for i in range(0, len(timestamps), 3):
            token_ids_i = timestamps[i]
            start_i = timestamps[i+1] / 1000.0  # Whisper returns ms
            end_i   = timestamps[i+2] / 1000.0
            token_str = tokenizer.decode([token_ids_i]).strip()
            # Skip special tokens (silence markers, <|notimestamps|>, etc.)
            if token_str and not token_str.startswith("<"):
                tokens.append(token_str)
                token_starts.append(start_i)
                token_ends.append(end_i)

        # Return as list of dicts
        return [
            {"token_en": tok, "start_s": st, "end_s": ed}
            for (tok, st, ed) in zip(tokens, token_starts, token_ends)
        ]

