import asyncio
import logging
import os
from pathlib import Path

from aiogram.exceptions import TelegramUnauthorizedError

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from config import TELEGRAM_VIDEO_UPLOAD_TIMEOUT
from ai.service import AIService
from ai.provider import AIProvider
from config import BOT_TOKEN, OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL
from handlers.media import router as media_router
from handlers.start import router as start_router
from handlers.subtitles import router as subtitles_router
from handlers.dub import router as dub_router
from handlers.transcribe_json import router as transcribe_json_router

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "dubfilm.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class UnauthorizedFailSafe(logging.Handler):
    """Hard-stop process when token becomes unauthorized (prevents endless retry loops)."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "TelegramUnauthorizedError" in msg or "Unauthorized" in msg and "fetch updates" in msg:
            try:
                logger.critical("AUTH_FAILSAFE: Unauthorized detected, stopping process.")
            finally:
                os._exit(12)


def setup_ai_service() -> AIService:
    provider = AIProvider(
        chat_model=OPENAI_CHAT_MODEL,
        whisper_model=OPENAI_WHISPER_MODEL
    )
    return AIService(provider=provider)


async def main():
    session = AiohttpSession(timeout=TELEGRAM_VIDEO_UPLOAD_TIMEOUT)
    bot = Bot(
        token=BOT_TOKEN,
        session=session,
    )

    # Fail-safe: if aiogram starts logging Unauthorized in polling loop, stop hard.
    logging.getLogger("aiogram.dispatcher").addHandler(UnauthorizedFailSafe())

    # Preflight auth check (fail-fast on bad token)
    try:
        me = await bot.get_me()
        logger.info("✅ Telegram auth OK: @%s (%s)", me.username, me.id)
    except TelegramUnauthorizedError:
        logger.critical("AUTH_FAILFAST: BOT_TOKEN unauthorized at startup")
        raise

    dp = Dispatcher()

    ai_service = setup_ai_service()
    # ✅ Правильный способ хранения сервиса в Aiogram 3
    bot.ai_service = ai_service

    dp.include_router(start_router)
    dp.include_router(media_router)
    dp.include_router(subtitles_router)
    dp.include_router(dub_router)
    dp.include_router(transcribe_json_router)

    logger.info("🤖 Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())