import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(ROOT, "1_input")
AUDIO_DIR = os.path.join(ROOT, "2_audio")
WHISPER_DIR = os.path.join(ROOT, "3_whisper")
TRANSLATION_DIR = os.path.join(ROOT, "4_translated")
CHUNKS_DIR = os.path.join(ROOT, "5_chunks")
OUTPUT_DIR = os.path.join(ROOT, "6_output")
