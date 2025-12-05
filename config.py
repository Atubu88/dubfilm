import os
from dotenv import load_dotenv

load_dotenv()

# Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")

# OpenAI (Whisper + GPT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
