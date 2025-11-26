import os
from dotenv import load_dotenv

load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_TOKEN      = os.getenv("BOT_TOKEN")
TRANSCRIBE_PROVIDER = os.getenv("TRANSCRIBE_PROVIDER", "assemblyai").lower()

INPUT_DIR       = "1_input"
AUDIO_DIR       = "2_audio"
# Whisper stage replaced by SRT-based subtitles, keep legacy name for compatibility
WHISPER_DIR     = "3_srt"
TRANSLATION_DIR = "4_translated"
CHUNKS_DIR      = "5_chunks"
OUTPUT_DIR      = "6_output"

for folder in (
    INPUT_DIR,
    AUDIO_DIR,
    WHISPER_DIR,
    TRANSLATION_DIR,
    CHUNKS_DIR,
    OUTPUT_DIR,
):
    os.makedirs(folder, exist_ok=True)
