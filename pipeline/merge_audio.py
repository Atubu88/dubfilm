import os
import wave
import json
import subprocess
from config import CHUNKS_DIR, OUTPUT_DIR, AUDIO_DIR

FINAL_AUDIO = os.path.join(OUTPUT_DIR, "final_audio.wav")


def get_wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def merge_audio():
    print("🎚 Starting MERGE_AUDIO (fixed WAV only)…")

    # ищем tts_fixed_XXX.wav
    fixed_wavs = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.startswith("tts_fixed_") and f.endswith(".wav")]
    )

    if not fixed_wavs:
        print("❌ ERROR: No stretched TTS WAV files found in 6_output/")
        print("   👉 Run: python -m pipeline.stretch_audio")
        return

    print(f"🔍 Found {len(fixed_wavs)} stretched WAV files")

    # загружаем чанк-тайминги
    chunks = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")])

    if not chunks:
        print("❌ ERROR: No chunk JSON files in 5_chunks/")
        return

    # список аудио-компонентов ffmpeg concat
    concat_list_path = os.path.join(OUTPUT_DIR, "concat_list.txt")
    with open(concat_list_path, "w", encoding="utf-8") as listfile:

        for chunk_json in chunks:
            idx = chunk_json.replace("chunk_", "").replace(".json", "")

            # нужный файл
            wav_path = os.path.join(OUTPUT_DIR, f"tts_fixed_{idx}.wav")
            if not os.path.exists(wav_path):
                print(f"❌ Missing stretched WAV: tts_fixed_{idx}.wav — SKIPPING")
                continue

            # добавляем в список
            listfile.write(f"file '{wav_path}'\n")

            print(f"   🔗 Added: tts_fixed_{idx}.wav "
                  f"({get_wav_duration(wav_path):.2f}s)")

    # финальный merge
    print("\n🚀 Running FFmpeg concat…")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        FINAL_AUDIO
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    if proc.returncode != 0:
        print("❌ FFmpeg concat ERROR!")
        return

    if os.path.exists(FINAL_AUDIO):
        print(f"🎉 FINAL AUDIO READY → {FINAL_AUDIO}")
        print(f"🎧 Duration: {get_wav_duration(FINAL_AUDIO):.2f}s")
    else:
        print("❌ ERROR: final_audio.wav was NOT created!")


if __name__ == "__main__":
    merge_audio()
