import os
import subprocess

def download_youtube(url, output_dir, filename="input.mp4"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Fix shorts
    if "shorts/" in url:
        url = url.replace("shorts/", "watch?v=")

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", filepath,
        url
    ]

    print("⏬ Running:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("❌ yt-dlp error:", result.stderr.decode())
        return None

    print(f"🎬 Saved to {filepath}")
    return filepath
