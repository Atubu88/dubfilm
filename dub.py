import argparse
import os
import sys
import shutil
import subprocess

# ПОДКЛЮЧАЕМ ЕДИНУЮ СИСТЕМУ ПУТЕЙ
from pipeline.constants import (
    INPUT_DIR,
    AUDIO_DIR,
    WHISPER_DIR,
    TRANSLATION_DIR,
    CHUNKS_DIR,
    OUTPUT_DIR,
)

HOME = os.path.expanduser("~")
DOWNLOADS_DIR = os.path.join(HOME, "Загрузки")
DEFAULT_DOWNLOAD_FILE = os.path.join(DOWNLOADS_DIR, "input.mp4")


# -------------------------------------------------------
# FULL CLEAN — очищаем ВСЕ данные старого запуска
# -------------------------------------------------------
def clean_all():
    folders = [
        INPUT_DIR,
        AUDIO_DIR,
        WHISPER_DIR,
        TRANSLATION_DIR,
        CHUNKS_DIR,
        OUTPUT_DIR,
    ]

    print("🧹 Cleaning ALL pipeline folders...")
    for folder in folders:
        if not os.path.exists(folder):
            continue

        for f in os.listdir(folder):
            full_path = os.path.join(folder, f)
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
                else:
                    shutil.rmtree(full_path)
            except Exception as e:
                print(f"⚠️ Could not clean {full_path}: {e}")

    print("✔ All folders cleaned!")


# -------------------------------------------------------
# Копируем исходное видео → input.mp4
# -------------------------------------------------------
def copy_input_video(src_path: str):
    if not os.path.exists(src_path):
        print(f"❌ Video not found: {src_path}")
        sys.exit(1)

    dst = os.path.join(INPUT_DIR, "input.mp4")

    try:
        shutil.copyfile(src_path, dst)
        print(f"📥 Copied input video → {dst}")
    except Exception as e:
        print(f"❌ Failed to copy video: {e}")
        sys.exit(1)


# -------------------------------------------------------
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# -------------------------------------------------------
def run(cmd):
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("❌ ERROR — stopping pipeline")
        sys.exit(1)


# -------------------------------------------------------
# ОБЩИЕ ПРЕПРОЦЕССИНГ-ШАГИ
# -------------------------------------------------------
def run_common_steps(input_video: str, lang: str):
    print("\n🧱 Running shared preprocessing steps")

    clean_all()
    copy_input_video(input_video)

    run("python -m pipeline.extract_audio 1_input/input.mp4")
    run("python -m pipeline.whisper_transcribe")
    run(f"python -m pipeline.translate_chunks {lang}")


# -------------------------------------------------------
# DUBBING
# -------------------------------------------------------
def run_dubbing_pipeline(input_video: str, lang: str):
    print("\n🎬 Starting FULL DUBBING PIPELINE")
    run_common_steps(input_video, lang)

    run("python -m pipeline.split_chunks")
    run("python -m pipeline.generate_tts")
    run("python -m pipeline.stretch_audio")
    run("python -m pipeline.merge_audio")
    run("python -m pipeline.merge_video")

    print("\n🎉 ALL DONE!")
    print("🍿 Final video → 6_output/final_video.mp4")


# -------------------------------------------------------
# VOICE-OVER
# -------------------------------------------------------
def run_voice_over_pipeline(input_video: str, lang: str):
    print("\n🎬 Starting VOICE-OVER PIPELINE")
    run_common_steps(input_video, lang)

    run("python -m pipeline.voice_over_tts")
    run("python -m pipeline.voice_over_mix")
    run("python -m pipeline.merge_video --mode voice_over")

    print("\n🎉 VOICE-OVER DONE!")
    print("📼 Final voice-over video → 6_output/final_video.mp4")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", help="Path to video file")
    parser.add_argument("lang", help="Language code")
    parser.add_argument(
        "--mode", choices=["dubbing", "voice_over"], default="voice_over"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "dubbing":
        run_dubbing_pipeline(args.input_video, args.lang)
    else:
        run_voice_over_pipeline(args.input_video, args.lang)


if __name__ == "__main__":
    main()
