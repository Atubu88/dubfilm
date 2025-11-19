import json
import os
import shutil
import subprocess
import wave
from typing import List

from openai import OpenAI

from config import OPENAI_API_KEY, OUTPUT_DIR, TRANSLATION_DIR

client = OpenAI(api_key=OPENAI_API_KEY)

TRANSLATED_JSON = os.path.join(TRANSLATION_DIR, "translated.json")
TRANSLATED_TXT = os.path.join(TRANSLATION_DIR, "translated.txt")
MP3_OUTPUT = os.path.join(OUTPUT_DIR, "voice_over_tts.mp3")
WAV_OUTPUT = os.path.join(OUTPUT_DIR, "voice_over_tts.wav")
FINAL_AUDIO = os.path.join(OUTPUT_DIR, "final_audio.wav")


def _load_lines_from_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError("❌ translated.json does not contain a list of segments")

    lines = []
    for segment in data:
        text = segment.get("dst", "").strip()
        if text:
            lines.append(text)
    return lines


def load_translated_script() -> str:
    if os.path.exists(TRANSLATED_JSON):
        print("📖 Loading translated.json …")
        lines = _load_lines_from_json(TRANSLATED_JSON)
        script = "\n".join(lines)
    elif os.path.exists(TRANSLATED_TXT):
        print("📄 Loading translated.txt …")
        with open(TRANSLATED_TXT, "r", encoding="utf-8") as f:
            script = f.read()
    else:
        raise FileNotFoundError("❌ No translated.json or translated.txt available")

    script = script.strip()
    if len(script) < 10:
        raise RuntimeError("❌ Translated script is too short for TTS")

    print(f"📝 Script length: {len(script)} characters")
    return script


def synthesize_script_to_mp3(text: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🎤 Sending full script to OpenAI TTS once…")
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="nova",
        input=text,
    )

    audio_bytes = response.read()
    with open(MP3_OUTPUT, "wb") as f:
        f.write(audio_bytes)

    print(f"💾 Saved MP3 → {MP3_OUTPUT}")
    return MP3_OUTPUT


def convert_mp3_to_wav(mp3_path: str) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        mp3_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        WAV_OUTPUT,
    ]

    print("🎚 Converting MP3 → 16kHz mono WAV…")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError("❌ Failed to convert MP3 to WAV via ffmpeg")
    return WAV_OUTPUT


def sanity_check_wav(path: str, min_duration: float = 3.0) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"❌ WAV not found → {path}")

    if os.path.getsize(path) == 0:
        raise RuntimeError("❌ WAV file is empty")

    with wave.open(path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)

    print(f"🔎 Voice-over WAV duration: {duration:.2f}s")
    if duration < min_duration:
        raise RuntimeError(
            f"❌ Voice-over WAV too short ({duration:.2f}s). Expected at least {min_duration}s"
        )


def copy_to_final_audio(wav_path: str) -> None:
    shutil.copyfile(wav_path, FINAL_AUDIO)
    print(f"📦 Copied voice-over track to FINAL_AUDIO → {FINAL_AUDIO}")


def generate_voice_over_track():
    text = load_translated_script()
    mp3_path = synthesize_script_to_mp3(text)
    wav_path = convert_mp3_to_wav(mp3_path)
    sanity_check_wav(wav_path)
    copy_to_final_audio(wav_path)
    print("🟢 Voice-over track ready!")


if __name__ == "__main__":
    generate_voice_over_track()
