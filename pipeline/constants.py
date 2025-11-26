import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(ROOT, "1_input")
AUDIO_DIR = os.path.join(ROOT, "2_audio")
SRT_DIR = os.path.join(ROOT, "3_srt")
# Alias kept for backward-compatibility with older pipeline steps
WHISPER_DIR = SRT_DIR
TRANSLATION_DIR = os.path.join(ROOT, "4_translated")
CHUNKS_DIR = os.path.join(ROOT, "5_chunks")
OUTPUT_DIR = os.path.join(ROOT, "6_output")
