import asyncio
import json
import os
import re
from pathlib import Path

from ai.provider import AIProvider
from ai.service import AIService
from config import OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL, DUB_TARGET_CHARS_PER_SEC
from pipelines.translate import run_translation
from services.subtitles import extract_audio_from_video

IN_DIR = Path('/home/fanfan/projects/dubfilm/in')
OUT2_DIR = Path('/home/fanfan/projects/dubfilm/out2')


def pick_latest_video() -> Path:
    candidates = sorted(
        [p for p in IN_DIR.iterdir() if p.suffix.lower() in {'.mp4', '.webm', '.mov', '.mkv'}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f'No video files found in {IN_DIR}')
    return candidates[0]


def _fit_metrics(translation_ru: str, start: float, end: float) -> tuple[float, str, str]:
    dur = max(0.01, float(end) - float(start))
    cps = len((translation_ru or '').strip()) / dur
    if cps <= DUB_TARGET_CHARS_PER_SEC:
        return cps, 'ok', 'в пределах целевой плотности текста'
    if cps <= DUB_TARGET_CHARS_PER_SEC * 1.10:
        return cps, 'borderline', 'погранично: возможен ускоренный темп в TTS'
    return cps, 'risk', 'риск: текст может не влезть без rewrite/ускорения'


def _split_long_segment_by_punct(start: float, end: float, text: str, speaker: str | None) -> list[dict]:
    """Conservative split for long segments, without speaker mixing."""
    t = (text or '').strip()
    if not t:
        return []
    dur = max(0.01, end - start)

    # Keep short segments as-is.
    if dur <= 7.2:
        return [{'start': start, 'end': end, 'text': t, 'speaker': speaker}]

    parts = [p.strip() for p in re.split(r'(?<=[؟!.,،])\s+', t) if p.strip()]
    if len(parts) <= 1:
        return [{'start': start, 'end': end, 'text': t, 'speaker': speaker}]

    # Avoid micro parts.
    parts = [p for p in parts if len(p.split()) >= 2]
    if len(parts) <= 1:
        return [{'start': start, 'end': end, 'text': t, 'speaker': speaker}]

    total = sum(max(1, len(p)) for p in parts)
    out = []
    cur = start
    for i, p in enumerate(parts):
        ratio = max(1, len(p)) / total
        pd = dur * ratio
        pe = end if i == len(parts) - 1 else min(end, cur + pd)
        out.append({'start': cur, 'end': pe, 'text': p, 'speaker': speaker})
        cur = pe
    return out


def _fill_missing_speakers(segments: list[dict]) -> list[dict]:
    out = [dict(s) for s in segments]
    for i, s in enumerate(out):
        if s.get('speaker') is not None:
            continue
        left = out[i - 1].get('speaker') if i > 0 else None
        right = out[i + 1].get('speaker') if i + 1 < len(out) else None
        if left is not None and left == right:
            s['speaker'] = left
        elif left is not None:
            s['speaker'] = left
        elif right is not None:
            s['speaker'] = right
    return out


def _build_segments(asm_segments: list[dict], whisper_segments: list[dict]) -> list[dict]:
    """Primary source: AssemblyAI speaker utterances (no proportional text split)."""
    if not asm_segments:
        base = [
            {
                'start': float(w.get('start', 0.0)),
                'end': float(w.get('end', float(w.get('start', 0.0)))),
                'text': (w.get('text') or '').strip(),
                'speaker': w.get('speaker'),
            }
            for w in whisper_segments
            if (w.get('text') or '').strip()
        ]
        return _fill_missing_speakers(base)

    out: list[dict] = []
    for s in asm_segments:
        st = float(s.get('start', 0.0))
        en = float(s.get('end', st))
        txt = (s.get('text') or '').strip()
        sp = s.get('speaker')
        if not txt or en <= st:
            continue
        out.extend(_split_long_segment_by_punct(st, en, txt, sp))

    out = [s for s in out if (s.get('text') or '').strip()]
    return _fill_missing_speakers(out)


async def main() -> None:
    OUT2_DIR.mkdir(parents=True, exist_ok=True)
    video = pick_latest_video()

    provider = AIProvider(chat_model=OPENAI_CHAT_MODEL, whisper_model=OPENAI_WHISPER_MODEL)
    ai = AIService(provider=provider)

    audio_path = await extract_audio_from_video(video)
    try:
        whisper = await provider._transcribe_with_whisper(audio_path)
        try:
            asm = await provider._transcribe_with_assemblyai(audio_path)
            asm_segments = asm.get('segments') or []
            language = whisper.get('language') or asm.get('language') or 'unknown'
        except Exception:
            asm_segments = []
            language = whisper.get('language') or 'unknown'
    finally:
        audio_path.unlink(missing_ok=True)

    split_segments = _build_segments(asm_segments=asm_segments, whisper_segments=whisper.get('segments') or [])

    normalized = []
    for idx, s in enumerate(split_segments, start=1):
        txt = (s.get('text') or '').strip()
        if not txt:
            continue
        st = float(s.get('start') or 0.0)
        en = float(s.get('end') or 0.0)
        sp = s.get('speaker')

        try:
            tr = await run_translation(
                text=txt,
                source_language=language,
                target_language='Russian',
                ai_service=ai,
            )
        except Exception:
            tr = txt

        cps, fit_status, risk_note = _fit_metrics(tr, st, en)
        review_reason = []
        if sp is None:
            review_reason.append('speaker_missing')
        if (en - st) < 0.85:
            review_reason.append('very_short_segment')
        if fit_status == 'risk':
            review_reason.append('fit_hard_risk')

        normalized.append(
            {
                'id': idx,
                'start': st,
                'end': en,
                'text': txt,
                'speaker': sp,
                'translation_ru': (tr or '').strip(),
                'voice': '',
                'fit_status': fit_status,
                'cps': round(cps, 2),
                'risk_note': risk_note,
                'needs_review': bool(review_reason),
                'review_reason': review_reason,
            }
        )

    payload = {
        'video': str(video),
        'language': language,
        'segment_count': len(normalized),
        'segments': normalized,
        'meta': {
            'transcribe_provider': 'whisper',
            'diarization_provider': os.getenv('CARTOON_DIARIZATION_PROVIDER', 'assemblyai').strip().lower(),
            'speaker_split_mode': 'v2_assemblyai_primary_no_proportional_split',
            'profile': 'cartoon_cast_strong',
            'output_dir': str(OUT2_DIR),
        },
    }

    main_json = OUT2_DIR / 'cartoon_segments_translated.json'
    snapshot_json = OUT2_DIR / f'{video.stem}_cartoon_segments_draft.json'

    main_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    snapshot_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(main_json)
    print(snapshot_json)


if __name__ == '__main__':
    asyncio.run(main())
