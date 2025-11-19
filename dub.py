import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))


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

    # ❌ REMOVE THIS — NO VOICE REMOVAL
    # run("python -m pipeline.remove_voice")

    # 7. Merge stretched TTS + silence pauses → final_audio.wav
    run("python -m pipeline.merge_audio")

    # 8. Combine: video + original sound + TTS (ducking)
    run(f"python -m pipeline.merge_video {input_video}")

    print("\n🎉 ALL DONE!")
    print("🍿 Final video → 6_output/final_video.mp4")


if __name__ == "__main__":
    main()
