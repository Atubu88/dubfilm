import os
import subprocess
from config import INPUT_DIR, OUTPUT_DIR

FINAL_WAV = os.path.join(OUTPUT_DIR, "final_audio.wav")
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "final_video.mp4")


def find_input_video():
    """Возвращает путь к первому найденному видео в 1_input/"""
    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
            return os.path.join(INPUT_DIR, f)
    return None


def merge_video():
    video_path = find_input_video()

    if not video_path:
        print("❌ No video found in 1_input/")
        return

    if not os.path.exists(FINAL_WAV):
        print("❌ final_audio.wav not found — run merge_audio first!")
        return

    print(f"🎬 Input video: {video_path}")
    print(f"🎧 Dub audio:  {FINAL_WAV}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", FINAL_WAV,
        "-map", "0:v:0",   # берём видео из оригинала
        "-map", "1:a:0",   # берём аудио из final_audio.wav
        "-c:v", "copy",    # видео не перекодируем
        "-c:a", "aac",     # финальное аудио конвертируется в AAC
        "-b:a", "192k",    # битрейт для лучшего качества
        "-shortest",       # ограничиваем по самому короткому
        FINAL_VIDEO
    ]

    print("🚀 Running FFmpeg...")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # лог FFmpeg
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    if proc.returncode != 0:
        print("❌ FFmpeg failed!")
        return

    if os.path.exists(FINAL_VIDEO):
        print(f"🎉 FINAL VIDEO READY → {FINAL_VIDEO}")
    else:
        print("❌ FFmpeg finished but video file is missing!")


if __name__ == "__main__":
    merge_video()
