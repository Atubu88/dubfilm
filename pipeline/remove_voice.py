import os
import subprocess
from pipeline.constants import AUDIO_DIR, OUTPUT_DIR

INPUT_AUDIO = os.path.join(AUDIO_DIR, "input.wav")
SFX_AUDIO = os.path.join(OUTPUT_DIR, "background_sfx.wav")
VOICE_AUDIO = os.path.join(OUTPUT_DIR, "original_voice.wav")

def remove_voice():
    print("🔍 Starting AI Voice Removal (Demucs)…")

    if not os.path.exists(INPUT_AUDIO):
        print("❌ ERROR: input.wav not found!")
        return

    # Run demucs separation
    cmd = [
        "demucs",
        "-n", "htdemucs",
        INPUT_AUDIO
    ]

    print("🚀 Running Demucs…")
    subprocess.run(cmd)

    # Demucs saves to ./separated/htdemucs/<filename>/
    demucs_dir = "separated/htdemucs"
    name = os.path.basename(INPUT_AUDIO).replace(".wav", "")

    input_dir = os.path.join(demucs_dir, name)

    # Demucs separation names
    sfx_src = os.path.join(input_dir, "no_vocals.wav")
    voice_src = os.path.join(input_dir, "vocals.wav")

    # Copy results into our pipeline
    os.rename(sfx_src, SFX_AUDIO)
    os.rename(voice_src, VOICE_AUDIO)

    print("🎉 Voice removed successfully!")
    print(f"🎵 SFX saved → {SFX_AUDIO}")
    print(f"🎙 Voice saved → {VOICE_AUDIO}")

if __name__ == "__main__":
    remove_voice()
