import os
import subprocess

from pipeline.constants import AUDIO_DIR, OUTPUT_DIR

ORIGINAL_AUDIO = os.path.join(AUDIO_DIR, "input.wav")
TTS_AUDIO = os.path.join(OUTPUT_DIR, "voice_over_tts.wav")
VOICE_OVER_MIX = os.path.join(OUTPUT_DIR, "voice_over_audio.wav")


def mix_voice_over_tracks():
    if not os.path.exists(ORIGINAL_AUDIO):
        raise FileNotFoundError(f"❌ Original audio not found → {ORIGINAL_AUDIO}")

    if not os.path.exists(TTS_AUDIO):
        raise FileNotFoundError(f"❌ Voice-over TTS not found → {TTS_AUDIO}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🎚️ Mixing voice-over with original audio bed…")

    # Сильно приглушаем оригинальный голос, оставляя только фон, чтобы не было эффекта
    # «двойного» произношения, и ограничиваем полосу для ещё большей прозрачности.
    filter_complex = (
        "[0:a]highpass=f=120,lowpass=f=3000,volume=-35dB[a0];"
        "[a0][1:a]amix=inputs=2:duration=shortest:dropout_transition=0[aout]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        ORIGINAL_AUDIO,
        "-i",
        TTS_AUDIO,
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-c:a",
        "pcm_s16le",
        VOICE_OVER_MIX,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("❌ Failed to mix voice-over audio via ffmpeg")

    print(f"✅ Voice-over mix saved → {VOICE_OVER_MIX}")


if __name__ == "__main__":
    mix_voice_over_tracks()
