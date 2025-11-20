import os
import json
import subprocess
from openai import OpenAI
from config import OPENAI_API_KEY
from pipeline.constants import CHUNKS_DIR, OUTPUT_DIR
from helpers.validators import assert_valid_tts_chunk

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_tts():
    files = sorted(f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json"))

    if not files:
        print("❌ NO CHUNKS FOUND")
        return

    print(f"🎧 Found {len(files)} chunks — starting TTS…")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for f in files:
        path = os.path.join(CHUNKS_DIR, f)

        with open(path, "r", encoding="utf-8") as jf:
            data = json.load(jf)

        text = data["text"].strip()
        idx = f.replace("chunk_", "").replace(".json", "")

        mp3_path = os.path.join(OUTPUT_DIR, f"tts_{idx}.mp3")
        wav_path = os.path.join(OUTPUT_DIR, f"tts_{idx}.wav")

        print(f"🎤 Generating TTS for chunk {idx}…")

        # 1) Получаем MP3/ACC от OpenAI
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="echo",  # ← выразительный мультяшный голос
            input=text
        )

        audio_bytes = response.read()

        # 2) Сохраняем MP3-файл
        with open(mp3_path, "wb") as f_audio:
            f_audio.write(audio_bytes)

        # 3) Конвертируем MP3 → WAV (PCM, 16kHz, mono)
        cmd = [
            "ffmpeg", "-y",
            "-i", mp3_path,
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            wav_path
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 4) Проверяем WAV-файл
        assert_valid_tts_chunk(wav_path, text)

        print(f"💾 Saved → {wav_path}")
        print(f"🟢 TTS OK for chunk {idx}")

    print("🎉 ALL TTS DONE")

if __name__ == "__main__":
    generate_tts()
