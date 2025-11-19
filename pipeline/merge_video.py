import os
import subprocess
from config import INPUT_DIR, OUTPUT_DIR

FINAL_WAV = os.path.join(OUTPUT_DIR, "final_audio.wav")
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "final_video.mp4")


def find_input_video():
    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
            return os.path.join(INPUT_DIR, f)
    return None


def merge_video():
    video = find_input_video()
    if not video:
        print("❌ No video found!")
        return

    if not os.path.exists(FINAL_WAV):
        print("❌ No final_audio.wav!")
        return

    print("🎬 Merging video with cinematic ducking…")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video,        # original video with sound
        "-i", FINAL_WAV,    # your dub voice
        "-filter_complex",

        # DUCKING:
        # when TTS speaks → original track becomes quieter
        # when TTS is silent → original at full volume
        "[0:a][1:a]sidechaincompress=threshold=0.1:ratio=8:attack=20:release=200[oac]; "
        "[oac][1:a]amix=inputs=2:weights=1 2:normalize=1[aout]",

        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        FINAL_VIDEO
    ]

    subprocess.run(cmd)
    print(f"🎉 FINAL VIDEO READY → {FINAL_VIDEO}")


if __name__ == "__main__":
    merge_video()
