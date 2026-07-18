import subprocess
import sys
import os

def extract_30s_clip(input_path: str, output_path: str, start_time: float = 0.0):
    """
    Extract a 30 second clip from `input_path`, beginning at `start_time` (in seconds),
    and save it to `output_path`, using ffmpeg stream‐copy (no re-encode).

    If ffmpeg is not in your PATH, provide the full path to the ffmpeg executable
    (e.g. '/usr/bin/ffmpeg' or 'C:\\ffmpeg\\bin\\ffmpeg.exe') instead of just 'ffmpeg'.

    Args:
        input_path:   Path to the source MP4.
        output_path:  Path where the 30 s subclip will be written.
        start_time:   Time (in seconds) where the 30 s clip should start. Default is 0.0.
    """

    # Verify input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    # Build the ffmpeg command.
    # Using "-ss START -i INPUT -t 30 -c copy" does a fast keyframe‐based cut.
    # If you need exact frame accuracy, you can move -ss after -i and/or drop "-c copy".
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                       # overwrite output if it already exists
        "-ss", str(start_time),     # seek to START (in seconds) before decoding
        "-i", input_path,           # input file
        "-t", str(duration),                 # duration of output = 30 seconds
        "-c", "copy",               # stream-copy (no re-encode)
        output_path
    ]

    try:
        # Run ffmpeg and wait for it to finish
        completed = subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # If ffmpeg failed, print stderr for debugging
        stderr_output = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed with error:\n{stderr_output}") from e

    print(f"Successfully wrote 30 s clip to: {output_path}")


if __name__ == "__main__":
    # Usage example from the command line:
    #   python extract_30s.py /path/to/input.mp4 /path/to/output30s.mp4 [start_time_in_seconds]
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python extract_30s.py <input.mp4> <output.mp4> [start_seconds]")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    duration   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    start_sec   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    

    extract_30s_clip(input_file, output_file, start_time=start_sec)
