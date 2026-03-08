from pathlib import Path
from typing import Any, Dict
import asyncio
import random
import aiohttp
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from ai.base import BaseAIProvider
from config import (
    OPENAI_API_KEY,
    OPENAI_WHISPER_MODEL,
    OPENAI_TTS_MODEL,
    OPENAI_TTS_VOICE,
    OPENAI_TTS_FORMAT,
    DUB_TTS_STYLE,
    ASSEMBLYAI_API_KEY,
    ASSEMBLYAI_SPEECH_MODEL,
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
        if TRANSCRIBE_PROVIDER == "hybrid":
            return await self._transcribe_hybrid(file_path)

        # 🔥 ПО УМОЛЧАНИЮ — WHISPER
        return await self._transcribe_with_whisper(file_path)

    # ============================================================
    # ✅ WHISPER
    # ============================================================
    async def _transcribe_with_whisper(self, file_path: Path) -> Dict[str, Any]:
        # Network-safe retry loop for intermittent upstream read/connect timeouts.
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                with open(file_path, "rb") as audio_file:
                    response = await self.client.audio.transcriptions.create(
                        model=self.whisper_model,          # whisper-1 / gpt-4o-transcribe family
                        file=audio_file,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                    )
                break
            except (APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
                if attempt >= 5:
                    raise
                delay = min(20.0, (1.6 ** attempt) + random.uniform(0, 0.7))
                print(f"⚠️ Whisper transient error (attempt {attempt}/5): {exc}. Retry in {delay:.1f}s")
                await asyncio.sleep(delay)
            except Exception as exc:
                # Some connection-layer failures surface as httpx/httpcore errors.
                msg = str(exc)
                transient_signals = (
                    "ReadError",
                    "ConnectError",
                    "Timeout",
                    "timed out",
                    "Connection reset",
                    "Connection error",
                )
                if any(sig in msg for sig in transient_signals) and attempt < 5:
                    last_exc = exc
                    delay = min(20.0, (1.6 ** attempt) + random.uniform(0, 0.7))
                    print(f"⚠️ Whisper transport error (attempt {attempt}/5): {exc}. Retry in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                raise

        if response is None:
            raise RuntimeError(f"Whisper transcription failed after retries: {last_exc}")

        segments = []

        # Prefer word-level timestamps when available (more robust sync for social clips).
        words = list(getattr(response, "words", []) or [])
        if words:
            buf = []
            seg_start = None
            seg_end = None

            def flush_segment():
                nonlocal buf, seg_start, seg_end, segments
                if not buf or seg_start is None or seg_end is None:
                    return
                text = " ".join(buf).strip()
                if text:
                    segments.append({"start": float(seg_start), "end": float(seg_end), "text": text})
                buf = []
                seg_start = None
                seg_end = None

            prev_end = None
            for w in words:
                txt = (getattr(w, "word", None) or getattr(w, "text", None) or "").strip()
                w_start = getattr(w, "start", None)
                w_end = getattr(w, "end", None)
                if not txt or w_start is None or w_end is None:
                    continue

                # Split on natural pause between words (helps avoid merging neighboring phrases)
                if prev_end is not None and (float(w_start) - float(prev_end)) >= 0.65 and buf:
                    flush_segment()

                if seg_start is None:
                    seg_start = w_start
                seg_end = w_end
                buf.append(txt)
                prev_end = w_end

                # split by punctuation or conservative chunk size
                if txt.endswith((".", "!", "?", "…", ":", ";")) or len(buf) >= 9:
                    flush_segment()

            flush_segment()

        # Fallback: segment-level timestamps
        if not segments:
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

    async def _transcribe_hybrid(self, file_path: Path) -> Dict[str, Any]:
        """Whisper for robust text/timing + AssemblyAI for speaker labels."""
        whisper = await self._transcribe_with_whisper(file_path)
        try:
            asm = await self._transcribe_with_assemblyai(file_path)
        except Exception:
            # Keep robust fallback: if AssemblyAI fails, return Whisper as-is.
            return whisper

        asm_segments = asm.get("segments") or []
        if not asm_segments:
            return whisper

        def speaker_for_segment(start: float, end: float) -> str | None:
            mid = (start + end) / 2.0
            best = None
            best_overlap = 0.0
            for s in asm_segments:
                s_start = float(s.get("start", 0.0))
                s_end = float(s.get("end", s_start))
                spk = s.get("speaker")
                # midpoint containment gets priority
                if s_start <= mid <= s_end and spk is not None:
                    return str(spk)
                overlap = max(0.0, min(end, s_end) - max(start, s_start))
                if overlap > best_overlap and spk is not None:
                    best_overlap = overlap
                    best = str(spk)
            return best

        merged_segments = []
        for seg in whisper.get("segments", []):
            st = float(seg.get("start", 0.0))
            en = float(seg.get("end", st))
            merged = dict(seg)
            merged["speaker"] = speaker_for_segment(st, en)
            merged_segments.append(merged)

        return {
            "text": whisper.get("text", ""),
            "language": whisper.get("language", asm.get("language", "unknown")),
            "segments": merged_segments,
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
                upload_url = upload_data.get("upload_url")
                if not upload_url:
                    raise RuntimeError(f"AssemblyAI upload failed: status={upload_resp.status} body={upload_data}")

            # 2️⃣ Запускаем транскрипцию
            transcript_payload = {
                "audio_url": upload_url,
                "language_detection": True,
                "speaker_labels": True,
                "speech_models": [ASSEMBLYAI_SPEECH_MODEL],
            }

            async with session.post(
                "https://api.assemblyai.com/v2/transcript",
                json=transcript_payload,
            ) as transcript_resp:
                transcript_data = await transcript_resp.json()
                transcript_id = transcript_data.get("id")
                if not transcript_id:
                    err = transcript_data.get("error") or transcript_data
                    raise RuntimeError(
                        f"AssemblyAI transcript create failed: status={transcript_resp.status} body={err}"
                    )

            # 3️⃣ Ожидаем результат
            while True:
                async with session.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
                ) as polling_resp:
                    result = await polling_resp.json()

                    status = result.get("status")

                    if not status:
                        raise RuntimeError(f"AssemblyAI polling invalid response: {result}")

                    if status == "completed":
                        segments = []

                        # Best path for multi-voice: AssemblyAI utterances with speaker labels.
                        utterances = result.get("utterances") or []
                        if utterances:
                            for u in utterances:
                                txt = (u.get("text") or "").strip()
                                if not txt:
                                    continue
                                start = u.get("start")
                                end = u.get("end")
                                if start is None or end is None:
                                    continue
                                segments.append(
                                    {
                                        "start": float(start) / 1000.0,
                                        "end": float(end) / 1000.0,
                                        "text": txt,
                                        "speaker": str(u.get("speaker")) if u.get("speaker") is not None else None,
                                    }
                                )

                        # Fallback: timestamped segments from words.
                        if not segments:
                            words = result.get("words") or []
                            if words:
                                buf = []
                                seg_start = None
                                seg_end = None

                                def flush_segment():
                                    nonlocal buf, seg_start, seg_end, segments
                                    if not buf or seg_start is None or seg_end is None:
                                        return
                                    text = " ".join(buf).strip()
                                    if text:
                                        segments.append(
                                            {
                                                "start": float(seg_start) / 1000.0,
                                                "end": float(seg_end) / 1000.0,
                                                "text": text,
                                            }
                                        )
                                    buf = []
                                    seg_start = None
                                    seg_end = None

                                for w in words:
                                    txt = (w.get("text") or "").strip()
                                    if not txt:
                                        continue
                                    w_start = w.get("start")
                                    w_end = w.get("end")
                                    if w_start is None or w_end is None:
                                        continue

                                    if seg_start is None:
                                        seg_start = w_start
                                    seg_end = w_end
                                    buf.append(txt)

                                    # Split on sentence punctuation or very long chunks.
                                    if txt.endswith((".", "!", "?", "…")) or len(buf) >= 16:
                                        flush_segment()

                                flush_segment()

                        if not segments:
                            # Fallback: one full segment if provider returned no word timings.
                            audio_duration = float(result.get("audio_duration", 0))
                            segments = [
                                {
                                    "start": 0.0,
                                    "end": audio_duration,
                                    "text": result.get("text", ""),
                                }
                            ]

                        return {
                            "text": result.get("text", ""),
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

    async def tts(self, text: str, *, voice: str | None = None, audio_format: str | None = None) -> bytes:
        model = OPENAI_TTS_MODEL
        selected_voice = voice or OPENAI_TTS_VOICE
        fmt = (audio_format or OPENAI_TTS_FORMAT or "mp3").lower()

        payload = {
            "model": model,
            "voice": selected_voice,
            "input": text,
            "response_format": fmt,
        }
        if DUB_TTS_STYLE:
            payload["instructions"] = DUB_TTS_STYLE

        try:
            response = await self.client.audio.speech.create(**payload)
        except TypeError:
            # Backward compatibility if SDK/model doesn't support instructions.
            payload.pop("instructions", None)
            response = await self.client.audio.speech.create(**payload)

        # openai-python response exposes read()/iter_bytes() depending on transport version
        if hasattr(response, "read"):
            data = response.read()
            return data if isinstance(data, (bytes, bytearray)) else bytes(data)

        if hasattr(response, "content"):
            content = response.content
            return content if isinstance(content, (bytes, bytearray)) else bytes(content)

        if hasattr(response, "iter_bytes"):
            buf = bytearray()
            for chunk in response.iter_bytes():
                if chunk:
                    buf.extend(chunk)
            return bytes(buf)

        raise RuntimeError("Unexpected TTS response type from OpenAI client")
