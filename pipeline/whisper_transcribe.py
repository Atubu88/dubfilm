import json
import os
import time
import requests
from openai import OpenAI
from config import OPENAI_API_KEY, ASSEMBLYAI_API_KEY, TRANSCRIBE_PROVIDER
from pipeline.constants import WHISPER_DIR, AUDIO_DIR
from helpers.validators import assert_valid_whisper   # ← ДОБАВИЛИ
from helpers.gpt_cleaner import clean_segments_with_gpt

client = OpenAI(api_key=OPENAI_API_KEY)

ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"


def _format_assemblyai_segments(words):
    segments = []
    if not words:
        return segments

    buffer = []
    start_time = None

    for word in words:
        text = word.get("text", "").strip()
        if not text:
            continue

        if start_time is None:
            start_time = word.get("start", 0) / 1000

        buffer.append(text)
        end_time = word.get("end", 0) / 1000

        if text.endswith((".", "!", "?", "…")):
            segment_text = " ".join(buffer).strip()
            if segment_text:
                segments.append({
                    "id": len(segments),
                    "start": start_time,
                    "end": end_time,
                    "text": segment_text,
                })
            buffer = []
            start_time = None

    if buffer and start_time is not None:
        segment_text = " ".join(buffer).strip()
        if segment_text:
            segments.append({
                "id": len(segments),
                "start": start_time,
                "end": end_time,
                "text": segment_text,
            })

    return segments


def _transcribe_with_assemblyai(audio_path, expected_language=None):
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("❌ ASSEMBLYAI_API_KEY is missing")

    headers = {"authorization": ASSEMBLYAI_API_KEY}

    def _read_file(path):
        with open(path, "rb") as f:
            while True:
                data = f.read(5_242_880)
                if not data:
                    break
                yield data

    print("⬆️  Uploading audio to AssemblyAI...")
    upload_resp = requests.post(
        f"{ASSEMBLYAI_API_URL}/upload",
        headers=headers,
        data=_read_file(audio_path)
    )
    upload_resp.raise_for_status()
    audio_url = upload_resp.json()["upload_url"]

    payload = {
        "audio_url": audio_url,
        "speech_model": "universal",
        "punctuate": True,
    }
    if expected_language:
        payload["language_code"] = expected_language

    print("🛰  Requesting AssemblyAI transcription...")
    transcript_resp = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        json=payload,
        headers=headers
    )
    transcript_resp.raise_for_status()
    transcript_id = transcript_resp.json()["id"]

    polling_endpoint = f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}"

    while True:
        status_resp = requests.get(polling_endpoint, headers=headers)
        status_resp.raise_for_status()
        result = status_resp.json()

        status = result.get("status")
        if status == "completed":
            print("✅ AssemblyAI transcription completed")
            break
        if status == "error":
            raise RuntimeError(f"Transcription failed: {result.get('error')}")
        time.sleep(3)

    formatted = {
        "text": result.get("text", ""),
        "language": result.get("language_code") or expected_language,
        "segments": _format_assemblyai_segments(result.get("words", [])),
    }

    if not formatted["segments"] and result.get("text"):
        formatted["segments"] = [{
            "id": 0,
            "start": 0,
            "end": max(result.get("audio_duration", 1), 1),
            "text": result.get("text"),
        }]

    return formatted


def _transcribe_with_whisper(audio_path, expected_language=None):
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )

    return response.model_dump()


def whisper_transcribe(audio_file="input.wav", expected_language=None):
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ ERROR: Audio file not found → {audio_path}")

    provider = TRANSCRIBE_PROVIDER.lower()
    print(f"🎧 Transcribing: {audio_path} (provider={provider})")

    if provider == "whisper":
        whisper_json = _transcribe_with_whisper(audio_path, expected_language)
    else:
        whisper_json = _transcribe_with_assemblyai(audio_path, expected_language)

    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    # 📄 Сохраняем JSON — теперь правильно
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(whisper_json, jf, ensure_ascii=False, indent=2)

    # 📝 Сохраняем raw-текст
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(whisper_json.get("text", ""))

    print(f"📄 JSON saved → {json_path}")
    print(f"📝 TXT saved  → {txt_path}")

    # 🛡 Whisper validation
    assert_valid_whisper(json_path, expected_language)

    # 🧹 Clean segments with GPT
    whisper_json = clean_segments_with_gpt(whisper_json)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(whisper_json, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(whisper_json.get("text", ""))

    print("🟢 Whisper validation PASSED")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")
