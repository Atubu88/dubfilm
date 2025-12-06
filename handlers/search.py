import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from pipelines.search import run_search

router = Router()
logger = logging.getLogger(__name__)


@router.message(Command("search"))
async def handle_search(message: Message) -> None:
    if not message.text:
        await message.answer("Пришли запрос после команды /search.")
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("После команды /search добавь текст запроса.")
        return

    query = parts[1].strip()
    if not query:
        await message.answer("Запрос пустой, попробуй снова с текстом поиска.")
        return

    await message.answer("Ищу информацию, подожди пару секунд...")
    try:
        result = await run_search(query)
    except Exception:
        logger.exception("Search failed for query: %s", query)
        await message.answer("Не удалось выполнить поиск. Попробуй позже.")
        return

    await message.answer(result)
