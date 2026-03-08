import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message

TOKEN = os.getenv("FILEID_BOT_TOKEN", "8554049314:AAGl3QAd0JjtvD9oDH855KwXD7Jub-JaYa4")
LOG_PATH = Path("/home/fanfan/projects/dubfilm/logs/fileid_bot.log")
SAVE_PATH = Path("/home/fanfan/projects/dubfilm/out/file_ids.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("fileid_bot")


dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Привет. Пришли видео (или video-документ), и я верну Telegram file_id."
    )


@dp.message(F.video)
async def on_video(message: Message) -> None:
    file_id = message.video.file_id
    name = message.video.file_name or "video"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAVE_PATH.open("a", encoding="utf-8") as f:
        f.write(f"video\t{name}\t{file_id}\n")
    await message.answer(f"file_id:\n{file_id}")


@dp.message(F.document)
async def on_document(message: Message) -> None:
    mime = message.document.mime_type or ""
    if not mime.startswith("video/"):
        await message.answer("Нужен именно видеофайл.")
        return
    file_id = message.document.file_id
    name = message.document.file_name or "document_video"
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAVE_PATH.open("a", encoding="utf-8") as f:
        f.write(f"document\t{name}\t{file_id}\n")
    await message.answer(f"file_id:\n{file_id}")


@dp.message()
async def on_other(message: Message) -> None:
    await message.answer("Пришли видео, я верну file_id.")


async def main() -> None:
    bot = Bot(token=TOKEN)
    me = await bot.get_me()
    logger.info("Bot started: @%s (%s)", me.username, me.id)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
