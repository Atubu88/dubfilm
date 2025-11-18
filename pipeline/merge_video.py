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

    # --- НОВЫЙ КОМПЛЕКСНЫЙ МИКС ---
    #   1) оригинал -> тише (через volume=0.25)
    #   2) дубляж -> нормальная громкость
    #   3) amix: смешиваем 2 дорожки с приоритетом дубляжа
    #   4) normalize=1 — выравниваем громкость

    ffmpeg_filter = (
        "[0:a]volume=0.25[orig];"      # оригинальный звук тише (сохраняем SFX)
        "[1:a]volume=1.0[dub];"        # дубляж нормальной громкости
        "[orig][dub]amix=inputs=2:weights=1 3:normalize=1[a]" # микс
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", FINAL_WAV,
        "-filter_complex", ffmpeg_filter,
        "-map", "0:v",       # видео оставляем оригинальное
        "-map", "[a]",       # готовая смешанная аудио-дорожка
        "-c:v", "copy",      # не перекодируем видео
        "-c:a", "aac",       # кодируем аудио в AAC
        "-b:a", "192k",      # качество звука
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
