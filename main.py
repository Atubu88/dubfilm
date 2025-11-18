import asyncio
import os
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from config import BOT_TOKEN, INPUT_DIR
from pipeline import run_pipeline      # 👈 новый импорт

bot = Bot(BOT_TOKEN)
dp = Dispatcher()

@dp.message()
async def handle_video(msg: Message):
    if not msg.video:
        return await msg.answer("Отправь видео ✉️")

    file = await bot.get_file(msg.video.file_id)
    filename = f"{msg.video.file_unique_id}.mp4"
    path = os.path.join(INPUT_DIR, filename)

    await bot.download_file(file.file_path, path)
    await msg.answer("🎬 Видео получено! Начинаю дубляж...")

    output_file = run_pipeline(path)  # 🔥 ВОТ ЭТО — запуск всей обработки

    await msg.answer_video(open(output_file, "rb"), caption="🔥 Готово!")

async def main():
    print("🤖 Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
