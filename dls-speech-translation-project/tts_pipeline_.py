# tts_pipeline.py
import torch
from TTS.api import TTS  # Coqui TTS

class TTSWrapper:
    """
    Wrapper for Coqui XTTSv2 TTS model for voice cloning.
    Uses Coqui TTS v0.22 with GPT module disabled to avoid API issues.
    """
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", use_cuda=True, fp16=False):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        # Initialize TTS with optional mixed precision
        self.tts = TTS(
            model_name,
            # gpu=(self.device.type == "cuda"),
            # torch_dtype=(torch.float16 if self.device.type == "cuda" and fp16 else torch.float32)
        ).to(self.device)
        # Disable GPT prosody module if present
        try:
            self.tts.tts_config.use_gpt = False
        except AttributeError:
            pass

    def synthesize(self, text, speaker_wav=None, speaker_idx=None, language="ru", output_path="tts_output.wav"):  
        """
        Synthesize speech for `text` in target `language`.
        - If `speaker_wav` (list of reference WAV paths) is provided: zero-shot voice cloning.
        - Else if `speaker_idx` is provided: use that index for a multi-speaker model.
        - If neither is provided: default to speaker_idx=0.
        Returns path to output WAV.
        """
        kwargs = {
            "text": text,
            "file_path": output_path,
            "language": language,
            "split_sentences": True,
        }
        # Determine multi-speaker parameters
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif speaker_idx is not None:
            kwargs["speaker_idx"] = speaker_idx
        else:
            # Default to speaker index 0 if model is multi-speaker
            kwargs["speaker_idx"] = 0

        # Call TTS to file
        self.tts.tts_to_file(**kwargs)
        return output_path

    def finetune(self, speaker_wav_dir, transcripts_file, output_dir, config_template_path=None, use_cuda=True):
        """
        Fine-tune the XTTSv2 model on a new speaker using Coqui TTS v0.22 CLI.
        - `speaker_wav_dir`: Directory containing WAV files for the speaker.
        - `transcripts_file`: Path to a metadata file with lines "wav_filename|transcription".
        - `output_dir`: Output path to save fine-tuned model checkpoints.
        - `config_template_path`: (Optional) Path to a JSON template to base the config on.
        - `use_cuda`: Whether to use GPU.
        """
        import os, json, subprocess, tempfile

        # Create a temporary config file for CLI fine-tuning
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp_config:
            config_path = tmp_config.name

            # Base config structure for XTTSv2 fine-tuning
            config = {
                "model_name": "xtts_v2",
                "run_name": "xtts_finetune",
                "output_path": output_dir,
                "dataset": {
                    "formatter": "xtts",
                    "meta_file_train": transcripts_file,
                    "path": speaker_wav_dir
                },
                "trainer": {
                    "gpus": 1 if use_cuda else 0,
                    "max_steps": 1000,
                    "batch_size": 16,
                    "eval_batch_size": 16,
                    "eval_interval": 100
                },
                "pretrained_model_path": self.model_name
            }

            # If a config template is provided, load and update it
            if config_template_path and os.path.isfile(config_template_path):
                with open(config_template_path, 'r') as f:
                    base_config = json.load(f)
                # Merge nested dictionaries
                for key, val in base_config.items():
                    if key in config and isinstance(val, dict) and isinstance(config[key], dict):
                        config[key].update(val)
                    else:
                        config[key] = val

            json.dump(config, tmp_config, indent=2)
            tmp_config.flush()

        # Invoke Coqui TTS CLI for fine-tuning
        try:
            subprocess.run([
                "tts",
                "--config_path", config_path,
                "--continue_path", self.model_name,
                "--output_path", output_dir,
                "--use_cuda", str(use_cuda).lower()
            ], check=True)
        finally:
            os.remove(config_path)  # Clean up temp file
        
        return os.path.join(output_dir, "best_model.pth")  # path to fine-tuned checkpoint
