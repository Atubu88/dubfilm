from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from handlers.subtitles import start_subtitles
from handlers.dub import start_dub
from handlers.transcribe_json import start_transcribe_json
from handlers.subtitles import SubtitleState   # ✅ ВАЖНО: импортируем состояние

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
            [
                InlineKeyboardButton(
                    text="🎤 Видео перевод (озвучка)",
                    callback_data="pipeline:dub",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🧾 Чистая транскрибация (JSON)",
                    callback_data="pipeline:transcribe_json",
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


# ───────── AUDIO PIPELINE ─────────

@router.callback_query(F.data == "pipeline:audio_translation")
async def handle_audio_translation_choice(
    callback: CallbackQuery, state: FSMContext
) -> None:
    await callback.answer()

    # ✅ чистим состояние
    await state.clear()

    if callback.message:
        await callback.message.answer(
            (
                "Отправь аудио или голосовое сообщение, или ссылку на файл."
                " Я переведу и сделаю выжимку."
            )
        )


# ───────── VIDEO PIPELINE (TEXT SUMMARY) ─────────

@router.callback_query(F.data == "pipeline:video_translation")
async def handle_video_translation_choice(
    callback: CallbackQuery, state: FSMContext
) -> None:
    await callback.answer()

    # ✅ чистим состояние
    await state.clear()

    if callback.message:
        await callback.message.answer(
            (
                "Пришли видеофайл или ссылку на ролик."
                " Я извлеку аудио, переведу и кратко перескажу."
            )
        )


# ───────── SUBTITLES PIPELINE ─────────

@router.callback_query(F.data == "pipeline:subtitles")
async def handle_subtitles_choice(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    await callback.answer()

    # ❗ ВАЖНО:
    # ❌ НЕ вызываем state.clear() перед сабтайтлами,
    #    потому что start_subtitles сам управляет состоянием

    if callback.message:
        await start_subtitles(callback.message, state)


@router.callback_query(F.data == "pipeline:dub")
async def handle_dub_choice(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    await callback.answer()
    if callback.message:
        await start_dub(callback.message, state)


@router.callback_query(F.data == "pipeline:transcribe_json")
async def handle_transcribe_json_choice(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    await callback.answer()
    if callback.message:
        await start_transcribe_json(callback.message, state)
