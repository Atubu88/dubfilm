from pathlib import Path
from typing import Any, Dict
import asyncio
import aiohttp
from openai import AsyncOpenAI

from ai.base import BaseAIProvider
from config import (
    OPENAI_API_KEY,
    OPENAI_WHISPER_MODEL,
    ASSEMBLYAI_API_KEY,
    TRANSCRIBE_PROVIDER,
)


class AIProvider(BaseAIProvider):
    def __init__(self, chat_model: str, whisper_model: str) -> None:
        self.chat_model = chat_model
        self.whisper_model = whisper_model
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # ============================================================
    # ✅ ЕДИНАЯ ТОЧКА ТРАНСКРИБАЦИИ (Whisper или AssemblyAI)
    # ============================================================
    async def transcribe(self, file_path: Path) -> Dict[str, Any]:
        if TRANSCRIBE_PROVIDER == "assemblyai":
            return await self._transcribe_with_assemblyai(file_path)

        # 🔥 ПО УМОЛЧАНИЮ — WHISPER
        return await self._transcribe_with_whisper(file_path)

    # ============================================================
    # ✅ WHISPER
    # ============================================================
    async def _transcribe_with_whisper(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "rb") as audio_file:
            response = await self.client.audio.transcriptions.create(
                model=self.whisper_model,          # whisper-1
                file=audio_file,
                response_format="verbose_json",
            )

        segments = []
        for segment in getattr(response, "segments", []) or []:
            segments.append(
                {
                    "start": getattr(segment, "start", 0.0),
                    "end": getattr(segment, "end", getattr(segment, "start", 0.0)),
                    "text": getattr(segment, "text", ""),
                }
            )

        return {
            "text": response.text,
            "language": response.language,
            "segments": segments,
        }

    # ============================================================
    # ✅ ASSEMBLYAI
    # ============================================================
    async def _transcribe_with_assemblyai(self, file_path: Path) -> Dict[str, Any]:
        headers = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            # 1️⃣ Загружаем файл
            async with session.post(
                "https://api.assemblyai.com/v2/upload",
                data=file_path.read_bytes(),
            ) as upload_resp:
                upload_data = await upload_resp.json()
                upload_url = upload_data["upload_url"]

            # 2️⃣ Запускаем транскрипцию
            transcript_payload = {
                "audio_url": upload_url,
                "language_detection": True,
            }

            async with session.post(
                "https://api.assemblyai.com/v2/transcript",
                json=transcript_payload,
            ) as transcript_resp:
                transcript_data = await transcript_resp.json()
                transcript_id = transcript_data["id"]

            # 3️⃣ Ожидаем результат
            while True:
                async with session.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
                ) as polling_resp:
                    result = await polling_resp.json()

                    status = result["status"]

                    if status == "completed":
                        audio_duration = float(result.get("audio_duration", 0))
                        segments = [
                            {
                                "start": 0.0,
                                "end": audio_duration,
                                "text": result.get("text", ""),
                            }
                        ]
                        return {
                            "text": result["text"],
                            "language": result.get("language_code", "unknown"),
                            "segments": segments,
                        }

                    if status == "error":
                        raise RuntimeError(f"AssemblyAI error: {result['error']}")

                await asyncio.sleep(2)

    # ============================================================
    # ✅ ПЕРЕВОД (ЖЁСТКИЙ)
    # ============================================================
    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        prompt = (
            "You are a professional translator. "
            "You MUST translate the text STRICTLY into {tgt} language. "
            "The final answer MUST contain ONLY {tgt} language. "
            "DO NOT leave any words or sentences in {src}. "
            "If the text is a dialogue, format it as a dialogue using dashes. "
            "Preserve the emotional tone and religious expressions. "
            "Avoid word-for-word translation. "
            "Return ONLY the translated text without comments."
        ).format(src=source_language, tgt=target_language)

        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )

        result = (response.choices[0].message.content or "").strip()

        # ✅ ПОВТОР ЕСЛИ ЯЗЫК НЕ СМЕНИЛСЯ
        if source_language.lower() in result.lower():
            retry_prompt = (
                "Translate the following text STRICTLY into {tgt} language only. "
                "DO NOT keep any {src} words. "
                "Return translation only.\n\n{text}"
            ).format(src=source_language, tgt=target_language, text=text)

            retry_response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": retry_prompt},
                    {"role": "user", "content": text},
                ],
            )

            result = (retry_response.choices[0].message.content or "").strip()

        return result

    # ============================================================
    # ✅ СМЫСЛОВАЯ ВЫЖИМКА
    # ============================================================
    async def summarize(self, original_text: str, translated_text: str, target_language: str) -> str:
        prompt = (
            "You are a skilled editor. "
            "Write a short, meaningful summary in {lang}. "
            "Explain the MAIN IDEA and MORAL of the text in 2–3 sentences. "
            "Do NOT retell the dialogue literally. "
            "Focus on the message and lesson."
        ).format(lang=target_language)

        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Original text:\n" + original_text +
                        "\n\nTranslation:\n" + translated_text
                    ),
                },
            ],
        )

        return (response.choices[0].message.content or "").strip()
