import asyncio
import logging

from aiogram import Bot, Dispatcher

from ai.service import AIService
from ai.provider import AIProvider
from config import BOT_TOKEN, OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL
from handlers.media import router as media_router
from handlers.search import router as search_router
from handlers.start import router as start_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_ai_service() -> AIService:
    provider = AIProvider(
        chat_model=OPENAI_CHAT_MODEL,
        whisper_model=OPENAI_WHISPER_MODEL
    )
    return AIService(provider=provider)


async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    ai_service = setup_ai_service()
    # ✅ Правильный способ хранения сервиса в Aiogram 3
    bot.ai_service = ai_service

    dp.include_router(start_router)
    dp.include_router(search_router)
    dp.include_router(media_router)

    logger.info("🤖 Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
