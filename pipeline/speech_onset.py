import math

import numpy as np
from pydub import AudioSegment


def detect_speech_onset(
    wav_path: str,
    frame_ms: int = 20,
    smooth: int = 4,
    hold_ms: int = 160,
) -> float:
    """Детектирует первое уверенное появление речи в аудио.

    Улучшения по сравнению с предыдущей версией:
    - опора на скользящее окно с пересечением (шаг = frame_ms/2),
    - порог отталкивается от реального шума в начале записи,
    - требуется устойчивое превышение порога на протяжении hold_ms,
      чтобы одиночный всплеск/музыка не считались началом речи.
    """

    audio = AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    frame_len = int(16000 * frame_ms / 1000)
    step_len = max(frame_len // 2, 1)  # перекрытие 50% для более точного старта

    if frame_len <= 0 or len(samples) < frame_len:
        return 0.0

    # FFT частоты
    fft_freqs = np.fft.rfftfreq(frame_len, d=1 / 16000)

    # Частоты человеческой речи (форманты F1–F3)
    speech_band = (fft_freqs >= 250) & (fft_freqs <= 3400)

    energies = []

    # Окно анализа
    for start in range(0, len(samples) - frame_len, step_len):
        chunk = samples[start:start + frame_len]

        # FFT амплитуды
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))

        # Энергия только в речевой полосе
        speech_energy = spectrum[speech_band].mean()
        energies.append(speech_energy)

    energies = np.array(energies)

    # Сглаживание для устойчивости
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        energies = np.convolve(energies, kernel, mode="same")

    if len(energies) == 0:
        return 0.0

    # Оценка шума по первым ~0.5 секунды (или доступному минимуму)
    baseline_frames = max(int(0.5 * 1000 / (step_len * 1000 / 16000)), 5)
    noise_slice = energies[:baseline_frames]
    noise_level = np.percentile(noise_slice, 60)

    # Требуем явное превышение фонового шума
    threshold = max(noise_level * 4.0, noise_level + 1e-6)

    hold_frames = max(int(math.ceil(hold_ms / (step_len * 1000 / 16000))), 2)

    # Ищем первое устойчивое превышение порога
    for idx, energy in enumerate(energies):
        if energy <= threshold:
            continue

        window = energies[idx: idx + hold_frames]
        if len(window) < hold_frames:
            break

        if window.mean() > threshold:
            return idx * (step_len / 16000.0)

    return 0.0
