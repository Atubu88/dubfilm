import wave
import os
import numpy as np

class MergeAudioError(Exception):
    pass


def _read_pcm_samples(path: str):
    """
    Читает WAV, возвращает:
    - samples: numpy array int16
    - rate: sample rate
    - channels: number of channels
    - frames: frame count
    """
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        return samples, w.getframerate(), w.getnchannels(), w.getnframes()


def assert_valid_final_audio(path: str, expected_rate=16000, expected_channels=1):
    """
    Проверка качества финального объединённого аудио.
    Важно: OpenAI TTS всегда возвращает WAV 16kHz 1ch.
    """

    if not os.path.exists(path):
        raise MergeAudioError(f"❌ FINAL AUDIO ERROR: File not found → {path}")

    try:
        samples, rate, channels, frames = _read_pcm_samples(path)
    except Exception as e:
        raise MergeAudioError(f"❌ FINAL AUDIO ERROR: Cannot read WAV → {e}")

    # --- Формат (исправлено!) ---
    if rate != expected_rate:
        raise MergeAudioError(
            f"❌ WRONG SAMPLE RATE → {rate}Hz (expected {expected_rate})"
        )

    if channels != expected_channels:
        raise MergeAudioError(
            f"❌ WRONG CHANNELS COUNT → {channels} (expected {expected_channels})"
        )

    # --- Продолжительность ---
    duration = frames / rate
    if duration < 0.5:
        raise MergeAudioError(f"❌ TOO SHORT FINAL AUDIO → {duration:.2f}s")

    # --- Пик громкости ---
    peak = int(np.max(np.abs(samples)))
    if peak > 30000:  # почти переполнение int16 (32767)
        raise MergeAudioError(f"❌ CLIPPING DETECTED → peak={peak}")

    # --- Проверка на длинные тишины ---
    near_zero = np.sum(np.abs(samples) < 50)
    silence_ratio = near_zero / len(samples)

    if silence_ratio > 0.25:
        print(f"⚠️ WARNING: Final audio contains too much silence ({silence_ratio:.0%})")

    print(f"✅ FINAL AUDIO OK → duration={duration:.2f}s, peak={peak}")
