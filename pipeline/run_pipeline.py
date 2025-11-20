import os
from helpers.downloader import download_youtube
from pipeline.extract_audio import extract_audio
from pipeline_guard.audio_checks import assert_valid_audio   # ⬅️ ДОБАВЛЕН
from pipeline.constants import INPUT_DIR, AUDIO_DIR


def run_pipeline(youtube_url: str):
    print(f"⏬ Downloading video from: {youtube_url}")

    video_path = download_youtube(youtube_url, INPUT_DIR, filename="input.mp4")
    if not video_path or not os.path.exists(video_path):
        return "❌ ERROR: Failed to download YouTube video"

    print(f"🎬 Video saved: {video_path}")
    print("🎧 Extracting audio...")

    audio_file = extract_audio(video_path, AUDIO_DIR)
    if not audio_file or not os.path.exists(audio_file):
        return "❌ ERROR: Audio extract failed"

    print(f"🔍 Checking audio integrity...")

    # ⛔ STOP PIPELINE HERE IF AUDIO IS INVALID
    assert_valid_audio(audio_file)

    print(f"✅ Audio is valid: {audio_file}")
    print("➡ You can now run Whisper")

    return "✅ STEP 1 DONE"


if __name__ == "__main__":
    test_url = input("🔗 Enter YouTube URL: ")
    result = run_pipeline(test_url)
    print(result)
