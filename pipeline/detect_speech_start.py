import wave
import struct
import numpy as np


def detect_speech_start(path, frame_ms=20, threshold=0.015, min_speech_ms=120):
    """
    Лёгкий VAD: определяет начало реальной речи.

    threshold — чувствительность (0.01–0.03)
    min_speech_ms — речь считается настоящей, если длится ≥120 мс
    """

    wf = wave.open(path, "rb")
    rate = wf.getframerate()  # 16000
    channels = wf.getnchannels()  # 1
    width = wf.getsampwidth()  # 2 bytes
    n_frames = wf.getnframes()

    frame_size = int(rate * frame_ms / 1000)  # samples per frame
    min_speech_frames = int(min_speech_ms / frame_ms)

    # читаем все
    raw = wf.readframes(n_frames)
    wf.close()

    # int16 → numpy array
    data = np.array(struct.unpack("<" + "h" * (len(raw) // 2), raw), dtype=np.float32)
    data /= 32768.0  # нормализация

    energies = []
    for i in range(0, len(data), frame_size):
        frame = data[i: i + frame_size]
        if len(frame) == 0:
            break
        rms = np.sqrt(np.mean(frame ** 2))
        energies.append(rms)

    # Поиск первого участка речи
    i = 0
    while i < len(energies):
        if energies[i] > threshold:
            # проверяем, что речь продолжается > 120 мс
            ok = True
            for j in range(1, min_speech_frames):
                if i + j >= len(energies) or energies[i + j] < threshold:
                    ok = False
                    break
            if ok:
                # точка в секундах
                return i * frame_ms / 1000.0

        i += 1

    # если речь не найдена
    return 0.0
