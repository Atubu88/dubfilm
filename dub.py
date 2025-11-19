import os
import sys
import subprocess
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "6_output")


def clean_output():
    """Удаляет ВСЕ файлы из 6_output, но не папку."""
    print("🧹 Cleaning 6_output/...")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        try:
            os.remove(path)
        except IsADirectoryError:
            shutil.rmtree(path)

    print("✔ 6_output cleaned!")


def run(cmd):
    print(f"\n🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("❌ ERROR — stopping pipeline")
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python dub.py input.mp4 ru")
        sys.exit(1)

    input_video = sys.argv[1]
    lang = sys.argv[2]

    print("\n🎬 Starting FULL DUBBING PIPELINE")
    print(f"🎥 Input video: {input_video}")
    print(f"🌐 Target language: {lang}")

    # 0. Clean output
    clean_output()

    # 1. Extract original audio
    run(f"python -m pipeline.extract_audio {input_video}")

    # 2. Whisper transcript
    run("python -m pipeline.whisper_transcribe")

    # 3. Translate transcript
    run(f"python -m pipeline.translate_chunks {lang}")

    # 4. Split translated text into chunks
    run("python -m pipeline.split_chunks")

    # 5. Generate TTS for each chunk
    run("python -m pipeline.generate_tts")

    # 6. Stretch TTS to match timing
    run("python -m pipeline.stretch_audio")

    # ❌ REMOVE_VOICE отключено — мы сохраняем звуки
    # run("python -m pipeline.remove_voice")

    # 7. Merge stretched TTS + pauses
    run("python -m pipeline.merge_audio")

    # 8. Mix original SFX + translated voice
    run(f"python -m pipeline.merge_video {input_video}")

    print("\n🎉 ALL DONE!")
    print("🍿 Final video → 6_output/final_video.mp4")


if __name__ == "__main__":
    main()
