from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from handlers.subtitles import start_subtitles

router = Router()


def _build_start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🎧 Перевод аудио",
                    callback_data="pipeline:audio_translation",
                )
            ],
            [
                InlineKeyboardButton(
                    text="📹 Перевод видео",
                    callback_data="pipeline:video_translation",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🎞 Видео с субтитрами",
                    callback_data="pipeline:subtitles",
                )
            ],
        ]
    )


@router.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        (
            "Привет! Я могу перевести аудио, видео или добавить субтитры."
            " Выбери, что хочешь сделать:"
        ),
        reply_markup=_build_start_keyboard(),
    )


@router.callback_query(F.data == "pipeline:audio_translation")
async def handle_audio_translation_choice(
    callback: CallbackQuery, state: FSMContext
) -> None:
    await callback.answer()
    await state.clear()
    if callback.message:
        await callback.message.answer(
            (
                "Отправь аудио или голосовое сообщение, или ссылку на файл."
                " Я переведу и сделаю выжимку."
            )
        )


@router.callback_query(F.data == "pipeline:video_translation")
async def handle_video_translation_choice(
    callback: CallbackQuery, state: FSMContext
) -> None:
    await callback.answer()
    await state.clear()
    if callback.message:
        await callback.message.answer(
            (
                "Пришли видеофайл или ссылку на ролик."
                " Я извлеку аудио, переведу и кратко перескажу."
            )
        )


@router.callback_query(F.data == "pipeline:subtitles")
async def handle_subtitles_choice(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    await state.clear()
    if callback.message:
        await start_subtitles(callback.message, state)
