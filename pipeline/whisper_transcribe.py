import json
import os
import re
import time

import numpy as np
import requests
from openai import OpenAI
from pydub import AudioSegment

from config import OPENAI_API_KEY, ASSEMBLYAI_API_KEY, TRANSCRIBE_PROVIDER
from helpers.gpt_cleaner import clean_segments_with_gpt
from helpers.validators import assert_valid_whisper
from pipeline.constants import AUDIO_DIR, WHISPER_DIR

client = OpenAI(api_key=OPENAI_API_KEY)
ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"


# ============================================================
# ⭐ 1. НАДЁЖНАЯ СЕГМЕНТАЦИЯ — ЭНЕРГЕТИЧЕСКИЙ VAD ⭐
# ============================================================

def _detect_speech_regions(audio_path: str, frame_ms: int = 15):
    """Возвращает интервалы речи [start_ms, end_ms] без сторонних VAD зависимостей.

    Используем энергию полосы 0..8к Гц, скользящее среднее и гистерезис, чтобы:
    - не залипать на шуме/музыке,
    - надёжно найти реальное начало речи (не 0.0),
    - уважать паузы между фразами.
    """

    audio = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    if samples.size == 0:
        return []

    frame_len = int(16000 * frame_ms / 1000)
    if frame_len <= 0:
        frame_len = 240

    # Нормализуем, чтобы пороги не зависели от громкости файла
    peak = np.max(np.abs(samples)) or 1.0
    samples = samples / peak

    energies = []
    for i in range(0, len(samples), frame_len):
        frame = samples[i:i + frame_len]
        if frame.size == 0:
            break
        energies.append(float(np.mean(np.abs(frame))))

    if not energies:
        return []

    energies = np.array(energies)

    # Сглаживание, чтобы одиночные выбросы не превращались в «речь»
    if len(energies) > 4:
        kernel = np.ones(5) / 5
        energies = np.convolve(energies, kernel, mode="same")

    noise_floor = np.percentile(energies, 20)
    speech_threshold = max(noise_floor * 3.5, np.percentile(energies, 70))
    release_threshold = speech_threshold * 0.55

    min_speech_frames = max(1, int(320 / frame_ms))  # >= 0.32s речи
    min_gap_frames = max(1, int(180 / frame_ms))      # >= 0.18s тишины, чтобы закрыть сегмент

    segments = []
    in_speech = False
    speech_start = 0
    below_count = 0

    for idx, energy in enumerate(energies):
        if not in_speech and energy >= speech_threshold:
            in_speech = True
            speech_start = idx
            below_count = 0
            continue

        if in_speech:
            if energy < release_threshold:
                below_count += 1
                if below_count >= min_gap_frames:
                    speech_end = idx - below_count + 1
                    if speech_end - speech_start >= min_speech_frames:
                        segments.append((speech_start, speech_end))
                    in_speech = False
                    below_count = 0
            else:
                below_count = 0

    if in_speech:
        speech_end = len(energies) - 1
        if speech_end - speech_start >= min_speech_frames:
            segments.append((speech_start, speech_end))

    if not segments:
        return []

    pad_ms = 120
    merged = []
    for start_f, end_f in segments:
        start_ms = max(0, start_f * frame_ms - pad_ms)
        end_ms = min(len(audio), (end_f + 1) * frame_ms + pad_ms)

        if merged and start_ms <= merged[-1][1] + 80:
            merged[-1][1] = max(merged[-1][1], end_ms)
        else:
            merged.append([start_ms, end_ms])

    return merged


def segment_by_silence(audio_path: str, full_text: str):
    """
    Делим аудио по паузам (энергетический VAD) → распределяем текст по сегментам.
    """

    padded_segments = _detect_speech_regions(audio_path)

    if not padded_segments:
        # fallback: один сегмент
        audio = AudioSegment.from_file(audio_path)
        return [{
            "id": 0,
            "start": 0.0,
            "end": len(audio) / 1000,
            "text": full_text.strip()
        }]

    # ⭐ Разбиваем текст на предложения с учетом арабского языка
    pattern = re.compile(r"[^.!?؟…]+(?:[.!?؟…]+|$)")
    sentences = [s.strip() for s in pattern.findall(full_text) if s.strip()]

    def count_words(s):
        return len(s.split())

    durations = [end - start for start, end in padded_segments]
    sentence_data = [(s, count_words(s)) for s in sentences]
    total_words = sum(cnt for _, cnt in sentence_data)

    whisper_segments = []

    for idx, (start, end) in enumerate(padded_segments):
        remaining_segments = len(padded_segments) - idx
        remaining_dur = sum(durations[idx:]) or 1

        if not sentence_data:
            break

        if remaining_segments == 1:
            picked = sentence_data
            sentence_data = []
        else:
            # Сколько слов реально поместится в сегмент (≈3.2 слова/сек)
            max_for_time = max(1, int(round((durations[idx] / 1000.0) * 3.2)))
            target_by_ratio = max(1, round(total_words * durations[idx] / remaining_dur))
            target = min(max_for_time, target_by_ratio)

            picked = []
            picked_words = 0

            while sentence_data:
                # Чтобы впереди остался хотя бы один sentence
                if len(sentence_data) <= (remaining_segments - 1) and picked:
                    break

                sent, count = sentence_data[0]

                # Не переполняем сегмент
                if picked and picked_words + count > target:
                    break

                picked.append(sentence_data.pop(0))
                picked_words += count

            # safety fallback
            if not picked:
                picked.append(sentence_data.pop(0))

        text = " ".join(s for s, _ in picked).strip()
        total_words -= sum(cnt for _, cnt in picked)

        whisper_segments.append({
            "id": len(whisper_segments),
            "start": start / 1000,
            "end": end / 1000,
            "text": text
        })

    return whisper_segments


# ============================================================
# ⭐ 2. AssemblyAI без их сегментов — мы делаем свои ⭐
# ============================================================

def _transcribe_with_assemblyai(audio_path, expected_language=None):
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("❌ ASSEMBLYAI_API_KEY missing")

    headers = {"authorization": ASSEMBLYAI_API_KEY}

    def _read_file(path):
        with open(path, "rb") as f:
            while chunk := f.read(5_242_880):
                yield chunk

    print("⬆️  Uploading audio...")
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
        "punctuate": True
    }
    if expected_language:
        payload["language_code"] = expected_language

    print("🛰  Requesting transcription...")
    resp = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        json=payload,
        headers=headers
    )
    resp.raise_for_status()
    transcript_id = resp.json()["id"]

    poll_url = f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}"

    while True:
        poll = requests.get(poll_url, headers=headers).json()
        if poll["status"] == "completed":
            print("✅ AssemblyAI transcription completed")
            break
        if poll["status"] == "error":
            raise RuntimeError(poll.get("error"))
        time.sleep(3)

    full_text = poll.get("text", "").strip()
    segments = segment_by_silence(audio_path, full_text)

    return {
        "text": full_text,
        "language": poll.get("language_code", expected_language),
        "segments": segments,
        "duration": poll.get("audio_duration")
    }


# ============================================================
# ⭐ 3. Whisper API → сегменты тоже по паузам ⭐
# ============================================================

def _transcribe_with_whisper(audio_path, expected_language=None):
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json"
        )
    result = resp.model_dump()

    full_text = result.get("text", "")
    segments = segment_by_silence(audio_path, full_text)

    result["segments"] = segments
    return result


# ============================================================
# ⭐ 4. Основная функция ⭐
# ============================================================

def whisper_transcribe(audio_file="input.wav", expected_language=None):
    audio_path = os.path.join(AUDIO_DIR, audio_file)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"❌ Audio file not found → {audio_path}")

    provider = TRANSCRIBE_PROVIDER.lower()
    print(f"🎧 Transcribing using {provider}: {audio_path}")

    if provider == "whisper":
        data = _transcribe_with_whisper(audio_path, expected_language)
    else:
        data = _transcribe_with_assemblyai(audio_path, expected_language)

    os.makedirs(WHISPER_DIR, exist_ok=True)

    json_path = os.path.join(WHISPER_DIR, "transcript.json")
    txt_path = os.path.join(WHISPER_DIR, "transcript.txt")

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(data.get("text", ""))

    print("📄 JSON saved")
    print("📝 TXT saved")

    # Валидация структуры
    assert_valid_whisper(json_path, expected_language)

    # GPT очистка сегментов
    cleaned = clean_segments_with_gpt(data)

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(cleaned, jf, ensure_ascii=False, indent=2)

    print("🟢 Whisper validation PASSED")
    return json_path


if __name__ == "__main__":
    whisper_transcribe(expected_language="ar")