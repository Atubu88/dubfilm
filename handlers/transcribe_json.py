import json
import logging
import re
from pathlib import Path
from uuid import uuid4

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import FSInputFile, Message

from ai.service import AIService
from config import TEMP_DIR
from pipelines.transcribe import run_transcription
from services.audio import MAX_FILE_SIZE_BYTES, get_media_size, prepare_audio_file
from services.downloader import download_audio_from_url, is_supported_media_url

router = Router()
logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"(https?://\S+)", re.IGNORECASE)


class TranscribeJsonState(StatesGroup):
    waiting_for_source = State()


async def _get_ai_service(message: Message) -> AIService:
    return message.bot.ai_service


def _extract_supported_url(text: str) -> str | None:
    for match in URL_PATTERN.finditer(text):
        url = match.group(1)
        if is_supported_media_url(url):
            return url
    return None


def _is_supported_document(message: Message) -> bool:
    if not message.document:
        return False
    mime = message.document.mime_type or ""
    return mime.startswith("audio/") or mime.startswith("video/")


def _build_clean_payload(source: str, transcription_data: dict) -> dict:
    raw_segments = transcription_data.get("segments") or []
    segments = []
    for idx, s in enumerate(raw_segments, start=1):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "id": idx,
                "start": float(s.get("start") or 0.0),
                "end": float(s.get("end") or 0.0),
                "text": text,
            }
        )

    return {
        "source": source,
        "language": transcription_data.get("language", "unknown"),
        "segment_count": len(segments),
        "segments": segments,
    }


async def _transcribe_and_send_json(message: Message, state: FSMContext, audio_path: Path, source_label: str) -> None:
    ai_service = await _get_ai_service(message)
    try:
        result = await run_transcription(audio_path=audio_path, ai_service=ai_service)
        payload = _build_clean_payload(source_label, result)

        out_path = TEMP_DIR / f"transcription_{uuid4().hex}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        await message.answer_document(
            FSInputFile(out_path),
            caption="Готово. Чистая транскрибация с таймингами в JSON.",
        )
    except Exception:
        logger.exception("Transcribe JSON failed for %s", source_label)
        await message.answer("Не удалось сделать транскрибацию. Попробуй другой файл/ссылку.")
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            out_path.unlink(missing_ok=True)  # type: ignore[name-defined]
        except Exception:
            pass
        await state.clear()


@router.message(Command("transcribe_json"))
async def start_transcribe_json(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(TranscribeJsonState.waiting_for_source)
    await message.answer("Пришли видео/аудио или ссылку — верну JSON с чистой транскрибацией и таймингами.")


@router.message(TranscribeJsonState.waiting_for_source, F.text)
async def handle_transcribe_json_link(message: Message, state: FSMContext) -> None:
    url = _extract_supported_url(message.text or "")
    if not url:
        await message.answer("Нужна валидная ссылка (YouTube/TikTok/Instagram/Facebook).")
        return

    await message.answer("Скачиваю и транскрибирую по ссылке...")
    try:
        audio_path = await download_audio_from_url(url)
    except Exception as exc:
        if "Video too long" in str(exc):
            await message.answer("Видео слишком длинное (лимит 5 минут для этого режима).")
        else:
            await message.answer("Не удалось скачать по ссылке.")
        return

    await _transcribe_and_send_json(message, state, audio_path, source_label=url)


@router.message(TranscribeJsonState.waiting_for_source, F.audio | F.voice | F.video | F.video_note | F.document)
async def handle_transcribe_json_media(message: Message, state: FSMContext) -> None:
    if message.document and not _is_supported_document(message):
        await message.answer("Поддерживаются только аудио/видео файлы.")
        return

    file_size = get_media_size(message)
    if file_size and file_size > MAX_FILE_SIZE_BYTES:
        await message.answer("Файл слишком большой. Максимум 20 МБ.")
        return

    await message.answer("Скачиваю файл и делаю транскрибацию...")
    try:
        audio_path = await prepare_audio_file(message.bot, message)
    except Exception as exc:
        if "Video too long" in str(exc):
            await message.answer("Видео слишком длинное (лимит 5 минут для этого режима).")
        else:
            await message.answer("Не удалось обработать файл.")
        return

    source_name = message.document.file_name if message.document else "telegram_media"
    await _transcribe_and_send_json(message, state, audio_path, source_label=source_name)


@router.message(TranscribeJsonState.waiting_for_source)
async def handle_transcribe_json_other(message: Message) -> None:
    await message.answer("Пришли видео/аудио файл или ссылку.")
