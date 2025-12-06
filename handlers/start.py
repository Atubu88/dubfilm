import logging

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from pipelines.search import run_search

router = Router()
logger = logging.getLogger(__name__)


class StartMenuState(StatesGroup):
    waiting_for_search_query = State()


START_MENU_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(
                text="🎬 Перевести видео / аудио",
                callback_data="start:media",
            )
        ],
        [
            InlineKeyboardButton(
                text="🔍 Умный поиск",
                callback_data="start:search",
            )
        ],
    ]
)


@router.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "Выбери, что хочешь сделать:",
        reply_markup=START_MENU_KEYBOARD,
    )


@router.callback_query(F.data == "start:media")
async def handle_media_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.answer()
    if callback.message:
        await callback.message.answer("Отправь видео, аудио или ссылку на видео.")


@router.callback_query(F.data == "start:search")
async def handle_search_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(StartMenuState.waiting_for_search_query)
    await callback.answer()
    if callback.message:
        await callback.message.answer("Напиши, что тебе найти в интернете")


@router.message(StartMenuState.waiting_for_search_query)
async def process_search_query(message: Message, state: FSMContext) -> None:
    if not message.text:
        await message.answer("Пришли текстовый запрос для поиска.")
        return

    query = message.text.strip()
    if not query:
        await message.answer("Запрос пустой, попробуй снова с текстом поиска.")
        return

    await message.answer("Ищу информацию, подожди пару секунд...")
    try:
        result = await run_search(query)
    except Exception:
        logger.exception("Search failed for query: %s", query)
        await message.answer("Не удалось выполнить поиск. Попробуй позже.")
        await state.clear()
        return

    await message.answer(result)
    await state.clear()
