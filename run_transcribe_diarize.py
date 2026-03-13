import asyncio
import json
from pathlib import Path

from ai.provider import AIProvider
from ai.service import AIService
from config import OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL, DUB_TARGET_CHARS_PER_SEC
from pipelines.translate import run_translation
from services.subtitles import SubtitleSegment, extract_audio_from_video

IN_DIR = Path('/home/fanfan/projects/dubfilm/in')
OUT_DIR = Path('/home/fanfan/projects/dubfilm/out')


def _tc(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    m, rem = divmod(ms, 60000)
    s, msec = divmod(rem, 1000)
    return f"{m:02}:{s:02}.{msec:03}"


def _looks_sentence_end(text: str) -> bool:
    t = (text or '').strip()
    return t.endswith(('.', '!', '?', '؟', '…', ':', ';'))


def _fit_metrics(translation_ru: str, start: float, end: float) -> tuple[float, str, str]:
    dur = max(0.01, float(end) - float(start))
    cps = len((translation_ru or '').strip()) / dur

    # Best-practice thresholds for lecture profile.
    if cps <= DUB_TARGET_CHARS_PER_SEC:
        return cps, 'ok', 'в пределах целевой плотности текста'
    if cps <= DUB_TARGET_CHARS_PER_SEC * 1.10:
        return cps, 'borderline', 'погранично: возможен ускоренный темп в TTS'
    return cps, 'risk', 'риск: текст может не влезть без rewrite/ускорения'


def lecture_safe_merge_segments(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
    """Merge over-fragmented lecture chunks into more natural thought blocks.

    Conservative rules (lecture-only):
    - merge when pause is short (<= 0.85s)
    - keep merged chunk duration under 16s
    - prefer merge when previous chunk is not a sentence end
    """
    if not segments:
        return segments

    merged: list[SubtitleSegment] = [segments[0]]
    for cur in segments[1:]:
        prev = merged[-1]
        gap = max(0.0, float(cur.start) - float(prev.end))
        merged_dur = max(0.0, float(cur.end) - float(prev.start))

        should_merge = (
            gap <= 0.85
            and merged_dur <= 16.0
            and (not _looks_sentence_end(prev.text) or gap <= 0.45)
        )

        if should_merge:
            merged[-1] = SubtitleSegment(
                start=prev.start,
                end=cur.end,
                text=f"{prev.text.strip()} {cur.text.strip()}".strip(),
                speaker=prev.speaker if prev.speaker == cur.speaker else prev.speaker or cur.speaker,
            )
        else:
            merged.append(cur)

    return merged


def pick_latest_video() -> Path:
    candidates = sorted(
        [p for p in IN_DIR.iterdir() if p.suffix.lower() in {'.mp4', '.webm', '.mov', '.mkv'}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f'No video files found in {IN_DIR}')
    return candidates[0]


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = pick_latest_video()

    ai = AIService(provider=AIProvider(chat_model=OPENAI_CHAT_MODEL, whisper_model=OPENAI_WHISPER_MODEL))

    audio_path = await extract_audio_from_video(video)
    try:
        result = await ai.transcribe_audio(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)

    source_language = (result.get('language') or 'unknown')
    raw_segments = result.get('segments') or []

    base_segments: list[SubtitleSegment] = []
    for s in raw_segments:
        text = (s.get('text') or '').strip()
        if not text:
            continue
        base_segments.append(
            SubtitleSegment(
                start=float(s.get('start') or 0.0),
                end=float(s.get('end') or 0.0),
                text=text,
                speaker=s.get('speaker'),
            )
        )

    # Lecture-safe segmentation: reduce over-splitting for one-speaker lectures.
    base_segments = lecture_safe_merge_segments(base_segments)

    # Draft RU translation per segment (1:1) to avoid cross-segment drift.
    translated_texts: list[str] = []
    for seg in base_segments:
        try:
            tr = await run_translation(
                text=seg.text,
                source_language=source_language,
                target_language='Russian',
                ai_service=ai,
            )
        except Exception:
            tr = seg.text
        translated_texts.append((tr or '').strip())

    segments_payload = []
    for idx, seg in enumerate(base_segments, start=1):
        tr_text = translated_texts[idx - 1] if idx - 1 < len(translated_texts) else ''
        cps, fit_status, risk_note = _fit_metrics(tr_text, seg.start, seg.end)
        item = {
            'id': idx,
            'start': seg.start,
            'end': seg.end,
            'start_tc': _tc(seg.start),
            'end_tc': _tc(seg.end),
            'text': seg.text,
            'translation_ru': tr_text,
            'cps': round(cps, 2),
            'fit_status': fit_status,
            'risk_note': risk_note,
        }
        if seg.speaker is not None:
            item['speaker'] = seg.speaker
        segments_payload.append(item)

    payload = {
        'video': str(video),
        'language': source_language,
        'segment_count': len(segments_payload),
        'segments': segments_payload,
    }

    # Main output for edit -> render flow
    translated_json_path = OUT_DIR / 'lecture_segments_translated.json'
    translated_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    # Also keep per-video snapshot
    snapshot_path = OUT_DIR / f'{video.stem}_segments_translated_draft.json'
    snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(translated_json_path)
    print(snapshot_path)


if __name__ == '__main__':
    asyncio.run(main())
