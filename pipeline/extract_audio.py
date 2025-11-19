import os
import ffmpeg

def extract_audio(video_path: str, output_dir: str) -> str:
    filename = os.path.basename(video_path).rsplit(".", 1)[0]
    audio_path = os.path.join(output_dir, f"{filename}.wav")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, ac=1, ar=16000)  # моно, 16kHz – идеально для Whisper
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[AUDIO] Extracted → {audio_path}")
        return audio_path
    except Exception as e:
        print(f"[ERROR] ffmpeg audio extract: {e}")
        return ""



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.extract_audio <video_path>")
        exit(1)

    video_path = sys.argv[1]
    output_dir = "2_audio"
    os.makedirs(output_dir, exist_ok=True)

    extract_audio(video_path, output_dir)
