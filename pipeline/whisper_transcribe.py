import json
import os
import re
import time

import requests
from openai import OpenAI
from pydub import AudioSegment

from pipeline.semantic_segmenter import semantic_segment
from config import OPENAI_API_KEY, ASSEMBLYAI_API_KEY, TRANSCRIBE_PROVIDER
from helpers.gpt_cleaner import clean_segments_with_gpt
from helpers.validators import assert_valid_whisper
from pipeline.constants import AUDIO_DIR, WHISPER_DIR


client = OpenAI(api_key=OPENAI_API_KEY)
ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"


# ============================================================
# ⭐ 1. AssemblyAI (текст → semantic segmentation) ⭐
# ============================================================

def _transcribe_with_assemblyai(audio_path, expected_language=None):

    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("❌ Missing ASSEMBLYAI_API_KEY")

    headers = {"authorization": ASSEMBLYAI_API_KEY}

    def stream(path):
        with open(path, "rb") as f:
            while chunk := f.read(5_242_880):
                yield chunk

    print("⬆️ Uploading audio...")
    up = requests.post(
        f"{ASSEMBLYAI_API_URL}/upload",
        headers=headers,
        data=stream(audio_path)
    )
    up.raise_for_status()
    audio_url = up.json()["upload_url"]

    payload = {"audio_url": audio_url, "punctuate": True}
    if expected_language:
        payload["language_code"] = expected_language

    print("🛰 Requesting transcription...")
    r = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        json=payload,
        headers=headers
    )
    r.raise_for_status()
    transcript_id = r.json()["id"]

    poll_url = f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}"

    # Polling
    while True:
        poll = requests.get(poll_url, headers=headers).json()
        if poll["status"] == "completed":
            print("✅ AssemblyAI transcription completed")
            break
        if poll["status"] == "error":
            raise RuntimeError(poll.get("error"))
        time.sleep(2)

    # ------- RESULT -------
    full_text = poll.get("text", "").strip()

    # → главное: semantic segmentation (audio + text)
    segments = semantic_segment(audio_path, full_text)

    return {
        "text": full_text,
        "language": expected_language or poll.get("language_code"),
        "segments": segments,
        "duration": poll.get("audio_duration")
    }


# ============================================================
# ⭐ 2. Whisper API (текст → semantic segmentation) ⭐
# ============================================================

def _transcribe_with_whisper(audio_path, expected_language=None):

    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    data = resp.model_dump()

    full_text = data.get("text", "").strip()

    # → semantic segmentation
    segments = semantic_segment(audio_path, full_text)

    data["segments"] = segments
    return data


# ============================================================
# ⭐ 3. Основная функция ⭐
# ============================================================

def whisper_transcribe(audio_file="input.wav", expected_language=None):

    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio not found: {audio_path}")

    provider = TRANSCRIBE_PROVIDER.lower()
    print(f"🎧 Transcribing using: {provider}")

    if provider == "whisper":
        data = _transcribe_with_whisper(audio_path, expected_language)
    else:
        data = _transcribe_with_assemblyai(audio_path, expected_language)

    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    # save JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    # save TXT
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(data.get("text", ""))

    print("📄 JSON saved")
    print("📝 TXT saved")

    # validation
    assert_valid_whisper(json_path, expected_language)

    # GPT cleanup
    cleaned = clean_segments_with_gpt(data)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(cleaned, jf, ensure_ascii=False, indent=2)

    print("🟢 Whisper validation PASSED")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")
