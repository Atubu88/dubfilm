import asyncio
import json
from pathlib import Path

from ai.provider import AIProvider
from ai.service import AIService
from config import OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL
from services.dub import constrain_translated_segments, synthesize_segment_audios, compose_dubbed_video_from_segments
from services.subtitles import SubtitleSegment, translate_segments

OVERRIDE = Path('/home/fanfan/projects/dubfilm/out/613fea3b-2042-47ca-83e1-4bd5634ca2bc_segments_override.json')
OUT_DIR = Path('/home/fanfan/projects/dubfilm/out')


async def main() -> None:
    data = json.loads(OVERRIDE.read_text(encoding='utf-8'))
    video = Path(data['video'])
    source_language = data.get('source_language', 'arabic')
    segments = [
        SubtitleSegment(
            start=float(s['start']),
            end=float(s['end']),
            text=str(s['text']),
            speaker=str(s.get('speaker')) if s.get('speaker') is not None else None,
        )
        for s in data['segments']
    ]

    ai = AIService(provider=AIProvider(chat_model=OPENAI_CHAT_MODEL, whisper_model=OPENAI_WHISPER_MODEL))

    translated = await translate_segments(
        segments=segments,
        source_language=source_language,
        target_language='Russian',
        ai_service=ai,
    )
    translated = await constrain_translated_segments(
        translated,
        target_language='Russian',
        ai_service=ai,
    )
    tts_items = await synthesize_segment_audios(ai, translated, target_language='Russian')
    out = await compose_dubbed_video_from_segments(video, tts_items)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final = OUT_DIR / f'{video.stem}_ru_dub_manualroles.mp4'
    out.replace(final)
    print(final)


if __name__ == '__main__':
    asyncio.run(main())
