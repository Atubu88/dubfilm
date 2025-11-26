# semantic_segmenter.py
# Улучшенный сегментатор под мультфильмы: паузы + семантика

import re
import numpy as np
from pydub import AudioSegment


# ---------------------------------------------------------
# 1. Детекция пауз по энергии (VAD без torchaudio/silero)
# ---------------------------------------------------------
def detect_pauses_energy(audio_path, frame_ms=20, pause_min_ms=300):
    audio = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    frame_len = int(16000 * frame_ms / 1000)
    energies = []

    for i in range(0, len(samples), frame_len):
        frame = samples[i:i + frame_len]
        if frame.size == 0:
            break
        energies.append(float(np.mean(np.abs(frame))))

    energies = np.array(energies)

    # сглаживание
    if len(energies) > 5:
        energies = np.convolve(energies, np.ones(5) / 5, mode="same")

    # определяем порог шума
    noise = np.percentile(energies, 20)
    speech_thr = noise * 2.2
    silence_thr = noise * 1.4

    in_speech = False
    quiet_frames = 0
    pauses = []

    for idx, e in enumerate(energies):
        if in_speech:
            if e < silence_thr:
                quiet_frames += 1
                if quiet_frames * frame_ms >= pause_min_ms:
                    pauses.append(idx * frame_ms)
                    in_speech = False
                    quiet_frames = 0
            else:
                quiet_frames = 0
        else:
            if e > speech_thr:
                in_speech = True

    return pauses


# ---------------------------------------------------------
# 2. Разбить текст на предложения
# ---------------------------------------------------------
def split_into_sentences(text: str):
    text = text.strip()
    pattern = r"[^.!?؟…]+(?:[.!?؟…]+|$)"
    sentences = [s.strip() for s in re.findall(pattern, text) if s.strip()]
    return sentences


# ---------------------------------------------------------
# 3. Склеить слишком короткие
# ---------------------------------------------------------
def merge_short(sentences, min_words=5):
    out = []
    buf = ""

    for s in sentences:
        if len(s.split()) < min_words:
            buf += " " + s
        else:
            if buf.strip():
                out.append(buf.strip())
                buf = ""
            out.append(s)

    if buf.strip():
        out.append(buf.strip())

    return out


# ---------------------------------------------------------
# 4. Разделить слишком длинные
# ---------------------------------------------------------
def split_long(sentences, max_words=22):
    out = []

    for s in sentences:
        words = s.split()
        if len(words) <= max_words:
            out.append(s)
        else:
            chunk = []
            for w in words:
                chunk.append(w)
                if len(chunk) >= max_words:
                    out.append(" ".join(chunk))
                    chunk = []
            if chunk:
                out.append(" ".join(chunk))

    return out


# ---------------------------------------------------------
# 5. Распределение текста по паузам
# ---------------------------------------------------------
def assign_text_to_pauses(sentences, pauses, total_ms):
    # формируем временные окна
    stops = [0] + pauses + [total_ms]

    # если пауз меньше, чем предложений — делим равномерно
    if len(stops) - 1 < len(sentences):
        # просто распределяем по очереди
        segs = []
        for idx, s in enumerate(sentences):
            start = stops[min(idx, len(stops) - 2)]
            end = stops[min(idx + 1, len(stops) - 1)]
            segs.append({
                "id": idx,
                "start": round(start / 1000, 3),
                "end": round(end / 1000, 3),
                "text": s
            })
        return segs

    # если пауз много — объединяем окна
    segments = []
    si = 0  # индекс предложения

    for i in range(len(stops) - 1):
        if si >= len(sentences):
            break

        start = stops[i]
        end = stops[i + 1]

        segments.append({
            "id": len(segments),
            "start": round(start / 1000, 3),
            "end": round(end / 1000, 3),
            "text": sentences[si]
        })
        si += 1

    return segments


# ---------------------------------------------------------
# 6. Главный интерфейс
# ---------------------------------------------------------
def semantic_segment(audio_path, full_text):
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    # 1. находи реальные паузы
    pauses = detect_pauses_energy(audio_path)

    # 2. предложения
    sents = split_into_sentences(full_text)
    sents = merge_short(sents, min_words=5)
    sents = split_long(sents, max_words=22)

    # 3. распределяем по паузам
    segments = assign_text_to_pauses(sents, pauses, total_ms)

    return segments
