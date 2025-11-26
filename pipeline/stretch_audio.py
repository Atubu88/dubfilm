import os
import json
import json
import os
import subprocess
import wave

from pipeline.constants import OUTPUT_DIR, CHUNKS_DIR


def get_wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def stretch_audio():
    print("🎚 Starting NEW stretch_audio…")

    wav_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("tts_") and f.endswith(".wav")])
    chunks = sorted([f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")])

    for idx, wav_name in enumerate(wav_files):
        json_name = chunks[idx]

        wav_path = os.path.join(OUTPUT_DIR, wav_name)
        json_path = os.path.join(CHUNKS_DIR, json_name)

        # load whisper timing
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_dur = data["end"] - data["start"]
        tts_dur = get_wav_duration(wav_path)

        print(f"\n🔍 Chunk {idx+1}:")
        print(f"   SRT target: {original_dur:.2f}s")
        print(f"   TTS duration:    {tts_dur:.2f}s")

        # NEW RULE:
        # If the TTS is shorter than 70% — stretch MAX 30%, NOT fully.
        min_allowed = original_dur * 0.70

        if tts_dur < min_allowed:
            # max 30% stretch
            stretched_target = tts_dur * 1.3
            silence_needed = original_dur - stretched_target

            print("   ⚠ TTS too short — using LIMITED stretch + silence")
            print(f"   🎧 New stretched duration: {stretched_target:.2f}s")
            print(f"   🕒 Silence needed:          {silence_needed:.2f}s")

            stretched_path = wav_path.replace("tts_", "tts_fixed_")
            silence_path = stretched_path.replace(".wav", "_silence.wav")

            # Stretch only 30%
            tempo = tts_dur / stretched_target

            subprocess.run([
                "ffmpeg", "-y",
                "-i", wav_path,
                "-filter:a", f"atempo={tempo}",
                stretched_path
            ])

            # Create silence for remaining time
            if silence_needed > 0:
                rate = 16000
                frames = int(rate * silence_needed)
                with wave.open(silence_path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(b"\x00\x00" * frames)

            continue

        # NORMAL CASE (small corrections)
        stretch_factor = original_dur / tts_dur

        # ffmpeg tempo = speed, so we invert
        tempo = 1.0 / stretch_factor

        print(f"   🎛 Stretch factor: {stretch_factor:.3f}")
        print(f"   🎚 FFmpeg tempo:   {tempo:.3f}")

        stretched_path = wav_path.replace("tts_", "tts_fixed_")

        subprocess.run([
            "ffmpeg", "-y",
            "-i", wav_path,
            "-filter:a", f"atempo={tempo}",
            stretched_path
        ])

    print("\n🟢 ALL TTS FIXED → ready for merge_audio()")


if __name__ == "__main__":
    stretch_audio()
