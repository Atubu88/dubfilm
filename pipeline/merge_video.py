import argparse
import os
import subprocess
from config import INPUT_DIR, OUTPUT_DIR

FINAL_WAV = os.path.join(OUTPUT_DIR, "final_audio.wav")
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "final_video.mp4")
VOICE_OVER_WAV = os.path.join(OUTPUT_DIR, "voice_over_audio.wav")


def find_input_video():
    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
            return os.path.join(INPUT_DIR, f)
    return None


def merge_video(mode: str = "dubbing"):
    video = find_input_video()
    if not video:
        print("❌ No video found!")
        return

    if mode == "voice_over":
        audio_source = VOICE_OVER_WAV
    else:
        audio_source = FINAL_WAV

    if not os.path.exists(audio_source):
        print(f"❌ Audio track not found → {audio_source}")
        return

    if mode == "voice_over":
        print("🎬 Muxing original video with voice-over mix…")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video,
            "-i", audio_source,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            FINAL_VIDEO,
        ]
    else:
        print("🎬 Merging video with cinematic ducking…")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", video,        # original video with sound
            "-i", audio_source,  # your dub voice
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


def parse_args():
    parser = argparse.ArgumentParser(description="Merge processed audio with the input video")
    parser.add_argument(
        "--mode",
        choices=["dubbing", "voice_over"],
        default="dubbing",
        help="Selects which audio track to mux",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_video(mode=args.mode)
