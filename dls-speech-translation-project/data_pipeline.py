# data_pipeline.py

import os
import subprocess
import ffmpeg
# from moviepy.editor import VideoFileClip
from pydub import AudioSegment

import tempfile
# data_pipeline.py


class AudioPreprocessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def extract_audio(self, video_path, output_wav):
        (
            ffmpeg
            .input(video_path)
            .output(output_wav,
                    ac=1,                # mono
                    ar=self.target_sr,   # sample rate
                    format="wav")
            .overwrite_output()
            .run(quiet=True)
        )


    def split_audio(self, audio_path, segment_length=30):
        """
        Splits `audio_path` into segments of `segment_length` seconds.
        Returns list of segment file paths.
        """
        sound = AudioSegment.from_file(audio_path)
        duration_ms = len(sound)
        segments = []
        for i in range(0, duration_ms, segment_length * 1000):
            segment = sound[i: i + segment_length * 1000]
            seg_path = f"{os.path.splitext(audio_path)[0]}_seg_{i//1000}.wav"
            segment.export(seg_path, format="wav")
            segments.append(seg_path)
        return segments

    # def combine_audio_video(self, video_path, audio_path, output_path):
    #     """
    #     Combines `audio_path` (new Russian speech) with original video frames from `video_path`.
    #     """
    #     cmd = [
    #         "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
    #         "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", output_path
    #     ]
    #     subprocess.run(cmd, check=True)



    def combine_audio_video(self, video_path: str, new_audio_path: str, output_path: str):
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

