import os
import wave
import json
import subprocess
from pipeline.constants import CHUNKS_DIR, OUTPUT_DIR, AUDIO_DIR

FINAL_AUDIO = os.path.join(OUTPUT_DIR, "final_audio.wav")


def get_wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def create_silence(path, seconds):
    """Create silent WAV of given duration."""
    rate = 16000
    frames = int(rate * seconds)

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)


def merge_audio():
    print("🎚 Starting MERGE_AUDIO with real pauses…")

    fixed_wavs = sorted(
        [
            f
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith("tts_fixed_") and f.endswith(".wav")
        ]
    )

    if not fixed_wavs:
        print("❌ ERROR: No TTS WAV files found in 6_output/")
        return

    chunks = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")])
    if not chunks:
        print("❌ ERROR: No chunk JSON files!")
        return

    concat_list_path = os.path.join(OUTPUT_DIR, "concat_list.txt")

    with open(concat_list_path, "w", encoding="utf-8") as listfile:
        for i, chunk_json in enumerate(chunks):
            idx = chunk_json.replace("chunk_", "").replace(".json", "")
            wav_path = os.path.abspath(
                os.path.join(OUTPUT_DIR, f"tts_fixed_{idx}.wav")
            )

            listfile.write(f"file '{wav_path}'\n")

            # calculate pauses
            with open(os.path.join(CHUNKS_DIR, chunk_json), "r", encoding="utf-8") as f:
                data = json.load(f)

            if i < len(chunks) - 1:
                next_chunk_path = os.path.join(CHUNKS_DIR, chunks[i + 1])
                with open(next_chunk_path, "r", encoding="utf-8") as next_f:
                    next_data = json.load(next_f)

                pause = next_data["start"] - data["end"]
                if pause > 0.05:  # ignore micro gaps
                    silence_path = os.path.abspath(
                        os.path.join(OUTPUT_DIR, f"silence_{idx}.wav")
                    )
                    create_silence(silence_path, pause)
                    listfile.write(f"file '{silence_path}'\n")

    print("🚀 Running FFmpeg concat…")

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

    subprocess.run(cmd)
    print("🎉 FINAL_AUDIO ready!")


if __name__ == "__main__":
    merge_audio()