import os
import wave

class AudioValidationError(Exception):
    pass

def assert_valid_audio(path: str, expected_sample_rate=16000, expected_channels=1):
    """
    Проверяет WAV-файл на пригодность для Whisper.
    """

    if not os.path.exists(path):
        raise AudioValidationError(f"❌ AUDIO ERROR: File does not exist → {path}")

    if os.path.getsize(path) < 1000:
        raise AudioValidationError(f"❌ AUDIO ERROR: File too small → {path}")

    try:
        with wave.open(path, "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            frames = wav.getnframes()
            duration = frames / float(rate)

            if channels != expected_channels:
                raise AudioValidationError(f"❌ AUDIO ERROR: Channels={channels}, expected={expected_channels}")

            if rate != expected_sample_rate:
                raise AudioValidationError(f"❌ AUDIO ERROR: SampleRate={rate}, expected={expected_sample_rate}")

            if duration < 0.5:
                raise AudioValidationError(f"❌ AUDIO ERROR: Too short duration ({duration}s)")

            print(f"✅ AUDIO OK → {duration:.2f}s, {channels}ch, {rate}Hz")

    except wave.Error as e:
        raise AudioValidationError(f"❌ AUDIO ERROR: Invalid WAV structure → {e}")
