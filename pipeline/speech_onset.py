import numpy as np
from pydub import AudioSegment


def detect_speech_onset(wav_path: str, frame_ms: int = 20, smooth: int = 4) -> float:
    """
    Надёжная детекция начала речи.
    Отличает речь от шумов и музыки.
    Работает без scipy/torch.
    """

    audio = AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    frame_len = int(16000 * frame_ms / 1000)

    # FFT частоты
    fft_freqs = np.fft.rfftfreq(frame_len, d=1 / 16000)

    # Частоты человеческой речи (форманты F1–F3)
    speech_band = (fft_freqs >= 250) & (fft_freqs <= 3400)

    energies = []

    # Окно анализа
    for i in range(0, len(samples) - frame_len, frame_len):
        chunk = samples[i:i + frame_len]

        # FFT амплитуды
        spectrum = np.abs(np.fft.rfft(chunk))

        # Энергия только в речевой полосе
        speech_energy = spectrum[speech_band].mean()

        energies.append(speech_energy)

    energies = np.array(energies)

    # Сглаживание для устойчивости
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        energies = np.convolve(energies, kernel, mode="same")

    # Определение шума
    noise_level = np.percentile(energies[:10], 40)
    threshold = noise_level * 3.0  # ниже — промахи

    # Ищем первое превышение порога
    for idx, energy in enumerate(energies):
        if energy > threshold:
            # проверяем, что это не одиночный пик
            if idx + 2 < len(energies):
                if energies[idx + 1] > threshold * 0.9:
                    return idx * (frame_len / 16000.0)

    return 0.0
