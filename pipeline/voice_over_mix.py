import os
import subprocess

from config import AUDIO_DIR, OUTPUT_DIR
from pipeline.remove_voice import remove_voice

ORIGINAL_AUDIO = os.path.join(AUDIO_DIR, "input.wav")
TTS_AUDIO = os.path.join(OUTPUT_DIR, "voice_over_tts.wav")
VOICE_OVER_MIX = os.path.join(OUTPUT_DIR, "voice_over_audio.wav")
BACKGROUND_SFX = os.path.join(OUTPUT_DIR, "background_sfx.wav")
ORIGINAL_VOICE = os.path.join(OUTPUT_DIR, "original_voice.wav")


def ensure_vocal_stems():
    """Run vocal separation (Demucs) once so music/SFX stay clean."""

    if os.path.exists(BACKGROUND_SFX) and os.path.exists(ORIGINAL_VOICE):
        return

    print("🔬 No separated stems detected — running Demucs via remove_voice.py…")
    remove_voice()

    if not os.path.exists(BACKGROUND_SFX) or not os.path.exists(ORIGINAL_VOICE):
        raise RuntimeError(
            "❌ Failed to generate background_sfx/original_voice stems. "
            "Check demucs installation or run pipeline.remove_voice manually."
        )


def mix_voice_over_tracks():
    if not os.path.exists(ORIGINAL_AUDIO):
        raise FileNotFoundError(f"❌ Original audio not found → {ORIGINAL_AUDIO}")

    if not os.path.exists(TTS_AUDIO):
        raise FileNotFoundError(f"❌ Voice-over TTS not found → {TTS_AUDIO}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ensure_vocal_stems()

    print("🎚️ Mixing clean SFX bed + quiet original voice + TTS…")

    filter_complex = (
        "[1:a]volume=-20dB[origquiet];"
        "[0:a][origquiet][2:a]amix=inputs=3:duration=longest:dropout_transition=0[aout]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        BACKGROUND_SFX,
        "-i",
        ORIGINAL_VOICE,
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
