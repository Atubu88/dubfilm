import os
import wave
import json
import shutil
import subprocess
from config import CHUNKS_DIR, OUTPUT_DIR

FIXED_PREFIX = "tts_"
OUT_PREFIX = "tts_fixed_"


def get_wav_duration(path):
    """Возвращает длительность WAV."""
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)


def ffmpeg_apply(input_f, output_f, tempo):
    """Один шаг растяжения/сжатия аудио."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_f,
        "-af", f"atempo={tempo}",
        output_f
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0 or not os.path.exists(output_f):
        print("❌ FFmpeg ERROR during audio stretch!")
        print("Command:", " ".join(cmd))
        print("stderr:", result.stderr)
        raise RuntimeError("FFmpeg failed")


def ffmpeg_stretch(input_path, output_path, factor):
    """
    Растянуть/сжать WAV c ограничениями atempo (0.5–2.0).
    """
    current = input_path
    remaining = factor

    # Дробим на шаги, если factor >2 или <0.5
    while remaining > 2.0 or remaining < 0.5:
        step = 2.0 if remaining > 1 else 0.5
        tmp = output_path + ".tmp.wav"

        ffmpeg_apply(current, tmp, step)

        remaining /= step
        current = tmp

    # Финальный шаг
    ffmpeg_apply(current, output_path, remaining)

    # Чистим TMP
    tmp = output_path + ".tmp.wav"
    if os.path.exists(tmp):
        os.remove(tmp)


def stretch_audio():
    print("🎚 Starting audio stretching...")

    chunks = sorted(f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json"))

    if not chunks:
        print("❌ No chunks in 5_chunks/")
        return

    for name in chunks:
        with open(os.path.join(CHUNKS_DIR, name), "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = name.replace("chunk_", "").replace(".json", "")

        src = os.path.join(OUTPUT_DIR, f"{FIXED_PREFIX}{idx}.wav")
        dst = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}{idx}.wav")

        if not os.path.exists(src):
            print(f"❌ No TTS WAV for chunk {idx} → skipping")
            continue

        tgt = data["end"] - data["start"]
        cur = get_wav_duration(src)

        print(f"\n🔍 Chunk {idx}:")
        print(f"   Whisper target: {tgt:.2f}s")
        print(f"   TTS duration:    {cur:.2f}s")

        # Разница менее 30 мс — копируем
        if abs(cur - tgt) < 0.03:
            print("   ✅ Duration OK — copying")
            shutil.copy(src, dst)
            continue

        factor = tgt / cur
        print(f"   🎛 Stretch factor: {factor:.3f}")

        # Выполняем растяжение
        ffmpeg_stretch(src, dst, factor)

        new = get_wav_duration(dst)
        print(f"   🎧 New duration: {new:.2f}s")

        if not os.path.exists(dst) or new < 0.01:
            print(f"❌ ERROR: failed to create {dst}")
            raise RuntimeError("Stretching failed")

    print("\n🟢 ALL TTS FIXED → ready for merge_audio()")
