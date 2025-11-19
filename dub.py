import argparse
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


def run_common_steps(input_video: str, lang: str):
    print("\n🧱 Running shared preprocessing steps")

    # 0. Clean output
    clean_output()

    # 1. Extract original audio
    run(f"python -m pipeline.extract_audio {input_video}")

    # 2. Whisper transcript
    run("python -m pipeline.whisper_transcribe")

    # 3. Translate transcript
    run(f"python -m pipeline.translate_chunks {lang}")


def run_dubbing_pipeline(input_video: str, lang: str):
    print("\n🎬 Starting FULL DUBBING PIPELINE")
    print(f"🎥 Input video: {input_video}")
    print(f"🌐 Target language: {lang}")
    print("🎚 Mode: dubbing")

    run_common_steps(input_video, lang)

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


def run_voice_over_pipeline(input_video: str, lang: str):
    print("\n🎬 Starting VOICE-OVER PIPELINE")
    print(f"🎥 Input video: {input_video}")
    print(f"🌐 Target language: {lang}")
    print("🎚 Mode: voice_over")

    run_common_steps(input_video, lang)

    print("\n🎙 Switching to voice-over specific steps")

    # Voice-over flow: re-use chunking / TTS and polish before final mux
    run("python -m pipeline.split_chunks")
    run("python -m pipeline.generate_tts")
    run("python -m pipeline.stretch_audio")
    run("python -m pipeline.merge_audio")
    run("python -m pipeline.mastering")
    run(f"python -m pipeline.merge_video {input_video}")

    print("\n🎉 VOICE-OVER DONE!")
    print("📼 Final voice-over video → 6_output/final_video.mp4")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dub or voice-over any video using the automated pipeline"
    )
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument(
        "lang",
        help="Target language code (e.g., ru, en, es)",
    )
    parser.add_argument(
        "--mode",
        choices=["dubbing", "voice_over"],
        default="dubbing",
        help="Processing mode: dubbing (default) or voice_over",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "dubbing":
        run_dubbing_pipeline(args.input_video, args.lang)
    elif args.mode == "voice_over":
        run_voice_over_pipeline(args.input_video, args.lang)
    else:
        # argparse choices prevent this, but keep a safeguard for clarity
        print(f"❌ Unsupported mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()