import os
from pydub import AudioSegment
from pipeline.constants import OUTPUT_DIR
from helpers.validators import assert_valid_final_audio

FINAL_RAW = os.path.join(OUTPUT_DIR, "final_audio.wav")
FINAL_MASTERED = os.path.join(OUTPUT_DIR, "final_audio_mastered.wav")


# ------------------------------
# 🔥 НОРМАЛИЗАЦИЯ ГРОМКОСТИ
# ------------------------------
def normalize(audio: AudioSegment, target_dbfs: float = -1.0):
    change = target_dbfs - audio.max_dBFS
    return audio.apply_gain(change)


# ------------------------------
# 🎚 FADE-IN / FADE-OUT
# ------------------------------
def apply_fades(audio: AudioSegment,
                fade_in_ms=50,
                fade_out_ms=80):
    return audio.fade_in(fade_in_ms).fade_out(fade_out_ms)


# ------------------------------
# 🎛️ МАСТЕРИНГ
# ------------------------------
def master_audio():
    if not os.path.exists(FINAL_RAW):
        print("❌ final_audio.wav not found — run merge_audio first!")
        return

    print(f"🎧 Loading: {FINAL_RAW}")
    audio = AudioSegment.from_wav(FINAL_RAW)

    print("🎚 Normalizing volume…")
    audio = normalize(audio, target_dbfs=-1.0)

    print("✨ Applying fade-in/out…")
    audio = apply_fades(audio, fade_in_ms=50, fade_out_ms=80)

    print(f"💾 Saving mastered audio → {FINAL_MASTERED}")
    audio.export(FINAL_MASTERED, format="wav")

    # финальная проверка WAV
    assert_valid_final_audio(FINAL_MASTERED)

    print("🟢 Mastered audio is ready!")


if __name__ == "__main__":
    master_audio()
