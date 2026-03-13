import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from ai.provider import AIProvider
from ai.service import AIService
from config import OPENAI_CHAT_MODEL, OPENAI_WHISPER_MODEL, DUB_TARGET_CHARS_PER_SEC
from pipelines.translate import run_translation
from services.subtitles import extract_audio_from_video

IN_DIR = Path('/home/fanfan/projects/dubfilm/in')
OUT2_DIR = Path('/home/fanfan/projects/dubfilm/out2')

AR_STOPWORDS = {
    'من', 'في', 'إلى', 'الى', 'على', 'عن', 'أن', 'ان', 'لن', 'لا', 'ثم', 'و', 'ف', 'ب', 'ل',
}


def pick_latest_video() -> Path:
    candidates = sorted(
        [p for p in IN_DIR.iterdir() if p.suffix.lower() in {'.mp4', '.webm', '.mov', '.mkv'}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f'No video files found in {IN_DIR}')
    return candidates[0]


def _tc(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    m, rem = divmod(ms, 60000)
    s, msec = divmod(rem, 1000)
    return f"{m:02}:{s:02}.{msec:03}"


def _stable_id(text: str, start: float, end: float) -> str:
    key = f"{(text or '').strip().lower()}|{round(float(start), 2):.2f}|{round(float(end), 2):.2f}"
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]


def _fit_metrics(translation_ru: str, start: float, end: float) -> tuple[float, str, str]:
    dur = max(0.01, float(end) - float(start))
    cps = len((translation_ru or '').strip()) / dur
    if cps <= DUB_TARGET_CHARS_PER_SEC:
        return cps, 'ok', 'в пределах целевой плотности текста'
    if cps <= DUB_TARGET_CHARS_PER_SEC * 1.10:
        return cps, 'borderline', 'погранично: возможен ускоренный темп в TTS'
    return cps, 'risk', 'риск: текст может не влезть без rewrite/ускорения'


def _split_text_proportionally(text: str, parts: int) -> list[str]:
    words = (text or '').strip().split()
    if parts <= 1 or len(words) <= 1:
        return [text.strip()]
    parts = max(1, min(parts, len(words)))
    base = len(words) // parts
    rem = len(words) % parts
    out: list[str] = []
    i = 0
    for p in range(parts):
        take = base + (1 if p < rem else 0)
        out.append(' '.join(words[i:i + take]).strip())
        i += take

    # avoid dangling stopword at segment end
    for k in range(len(out) - 1):
        cur = out[k].split()
        nxt = out[k + 1].split()
        if len(cur) >= 2 and cur[-1] in AR_STOPWORDS:
            moved = cur.pop()
            nxt.insert(0, moved)
            out[k] = ' '.join(cur).strip()
            out[k + 1] = ' '.join(nxt).strip()

    return [x for x in out if x]


def _speaker_for_window(start: float, end: float, asm_segments: list[dict]) -> str | None:
    best_sp = None
    best_overlap = 0.0
    mid = (start + end) / 2.0
    for a in asm_segments:
        a_st = float(a.get('start', 0.0))
        a_en = float(a.get('end', a_st))
        sp = a.get('speaker')
        if sp is None:
            continue
        if a_st <= mid <= a_en:
            return str(sp)
        overlap = max(0.0, min(end, a_en) - max(start, a_st))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sp = str(sp)
    return best_sp


def _fill_missing_speakers(segments: list[dict]) -> list[dict]:
    out = [dict(s) for s in segments]
    for i, s in enumerate(out):
        if s.get('speaker') is not None:
            continue
        left = out[i - 1].get('speaker') if i > 0 else None
        right = out[i + 1].get('speaker') if i + 1 < len(out) else None
        if left == right and left is not None:
            s['speaker'] = left
        elif left is not None:
            s['speaker'] = left
        elif right is not None:
            s['speaker'] = right
    return out


def _append_flag(seg: dict, flag: str) -> None:
    flags = seg.get('_auto_flags') or []
    if flag not in flags:
        flags.append(flag)
    seg['_auto_flags'] = flags


def _merge_into(dst: dict, src: dict) -> None:
    dst['end'] = float(src.get('end', dst.get('end', 0.0)))
    t1 = (dst.get('text') or '').strip()
    t2 = (src.get('text') or '').strip()
    dst['text'] = f"{t1} {t2}".strip() if t1 and t2 else (t1 or t2)


def _resolve_overlaps(segments: list[dict], tiny_overlap: float = 0.08, keep_gap: float = 0.02) -> list[dict]:
    if not segments:
        return []
    out: list[dict] = [dict(segments[0])]
    for raw in segments[1:]:
        cur = dict(raw)
        prev = out[-1]
        p_start, p_end = float(prev.get('start', 0.0)), float(prev.get('end', 0.0))
        c_start = float(cur.get('start', 0.0))
        c_end = float(cur.get('end', c_start))
        ov = p_end - c_start
        if ov <= 0:
            out.append(cur)
            continue

        # Small overlap: trim boundary only.
        if ov <= tiny_overlap:
            new_start = p_end + keep_gap
            if new_start < c_end:
                cur['start'] = new_start
            else:
                mid = (p_start + p_end) / 2.0
                prev['end'] = max(p_start + 0.05, mid - keep_gap)
                cur['start'] = min(c_end - 0.05, mid + keep_gap)
            _append_flag(cur, 'overlap_trimmed')
            out.append(cur)
            continue

        # Large overlap: split boundary by midpoint and mark for review.
        mid = (max(p_start, c_start) + min(p_end, c_end)) / 2.0
        prev['end'] = max(p_start + 0.05, mid - keep_gap)
        cur['start'] = min(c_end - 0.05, mid + keep_gap)
        _append_flag(prev, 'overlap_midpoint_split')
        _append_flag(cur, 'overlap_midpoint_split')
        _append_flag(prev, 'needs_manual_overlap_review')
        _append_flag(cur, 'needs_manual_overlap_review')
        out.append(cur)

    return out


def _post_smooth_split_segments(segments: list[dict]) -> list[dict]:
    # Best-practice v1 post-processing:
    # 1) sort timeline
    # 2) resolve overlaps
    # 3) merge micro-segments
    # 4) merge same-speaker tiny gaps
    # 5) fill missing speakers + flags for review

    if not segments:
        return []

    ordered = sorted(
        [dict(s) for s in segments if (s.get('text') or '').strip()],
        key=lambda s: (float(s.get('start', 0.0)), float(s.get('end', 0.0))),
    )

    # Normalize impossible windows early
    sane: list[dict] = []
    for s in ordered:
        st = float(s.get('start', 0.0))
        en = float(s.get('end', st))
        if en <= st:
            en = st + 0.05
            s['end'] = en
            _append_flag(s, 'fixed_bad_window')
        sane.append(s)

    sane = _resolve_overlaps(sane)

    MIN_DUR = 0.90
    MIN_WORDS = 3
    SAME_GAP = 0.25

    out: list[dict] = []
    for seg in sane:
        st, en = float(seg.get('start', 0.0)), float(seg.get('end', 0.0))
        dur = en - st
        txt = (seg.get('text') or '').strip()
        words = txt.split()

        if out:
            prev = out[-1]
            p_sp, c_sp = prev.get('speaker'), seg.get('speaker')
            gap = float(seg.get('start', 0.0)) - float(prev.get('end', 0.0))
            same_or_unknown = (p_sp == c_sp) or (p_sp is None) or (c_sp is None)

            # Merge tiny/micro segments into previous if speaker-compatible.
            is_micro = (dur < MIN_DUR) or (len(words) < MIN_WORDS)
            if same_or_unknown and gap <= SAME_GAP and is_micro:
                _merge_into(prev, seg)
                _append_flag(prev, 'merged_micro')
                continue

            # Merge same speaker across tiny gap.
            if (p_sp == c_sp) and gap <= SAME_GAP:
                _merge_into(prev, seg)
                _append_flag(prev, 'merged_same_speaker_gap')
                continue

        out.append(seg)

    out = _fill_missing_speakers(out)

    # Mark suspicious speaker flips in very short windows for manual review.
    for i in range(1, len(out) - 1):
        cur = out[i]
        prev = out[i - 1]
        nxt = out[i + 1]
        dur = float(cur.get('end', 0.0)) - float(cur.get('start', 0.0))
        if dur <= 1.0 and prev.get('speaker') == nxt.get('speaker') and cur.get('speaker') != prev.get('speaker'):
            _append_flag(cur, 'speaker_flip_short_review')

    return out


def _hard_split_by_speaker(whisper_segments: list[dict], asm_segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    min_chunk = 0.45

    for w in whisper_segments:
        w_st = float(w.get('start', 0.0))
        w_en = float(w.get('end', w_st))
        txt = (w.get('text') or '').strip()
        if not txt or w_en <= w_st:
            continue

        boundaries = {w_st, w_en}
        for a in asm_segments:
            a_st = float(a.get('start', 0.0))
            a_en = float(a.get('end', a_st))
            if w_st < a_st < w_en:
                boundaries.add(a_st)
            if w_st < a_en < w_en:
                boundaries.add(a_en)

        cuts = sorted(boundaries)
        windows: list[tuple[float, float]] = []
        cur_st = cuts[0]
        for nxt in cuts[1:]:
            if (nxt - cur_st) < min_chunk:
                continue
            windows.append((cur_st, nxt))
            cur_st = nxt
        if not windows:
            windows = [(w_st, w_en)]

        text_parts = _split_text_proportionally(txt, len(windows))
        if len(text_parts) < len(windows):
            text_parts += [''] * (len(windows) - len(text_parts))
        elif len(text_parts) > len(windows):
            text_parts = text_parts[:len(windows)]

        for (c_st, c_en), c_txt in zip(windows, text_parts):
            c_txt = (c_txt or '').strip()
            if not c_txt:
                continue
            out.append(
                {
                    'start': c_st,
                    'end': c_en,
                    'text': c_txt,
                    'speaker': _speaker_for_window(c_st, c_en, asm_segments),
                }
            )

    return _post_smooth_split_segments(out)


def _forced_align_with_aeneas(audio_path: Path, segments: list[dict]) -> tuple[list[dict], str]:
    """Optional forced alignment pass. Requires aeneas CLI installed.
    Returns (possibly updated segments, status).
    """
    if not segments:
        return segments, 'skipped:no_segments'

    if os.getenv('CARTOON_FORCED_ALIGN', '1') not in {'1', 'true', 'yes', 'on'}:
        return segments, 'skipped:disabled'

    with tempfile.TemporaryDirectory(prefix='aeneas_') as td:
        td_path = Path(td)
        txt_path = td_path / 'fragments.txt'
        out_json = td_path / 'sync.json'

        lines = []
        for s in segments:
            t = (s.get('text') or '').strip().replace('\n', ' ')
            lines.append(t if t else '_')
        txt_path.write_text('\n'.join(lines), encoding='utf-8')

        cmd = [
            'python3', '-m', 'aeneas.tools.execute_task',
            str(audio_path),
            str(txt_path),
            'task_language=ara|is_text_type=plain|os_task_file_format=json',
            str(out_json),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        except Exception as e:
            return segments, f'skipped:error:{e.__class__.__name__}'

        if proc.returncode != 0 or not out_json.exists():
            return segments, 'skipped:aeneas_unavailable_or_failed'

        try:
            data = json.loads(out_json.read_text(encoding='utf-8'))
            frags = data.get('fragments') or []
            if len(frags) != len(segments):
                return segments, f'skipped:length_mismatch:{len(frags)}!={len(segments)}'

            aligned: list[dict] = []
            for s, f in zip(segments, frags):
                try:
                    st = float(f.get('begin', s.get('start', 0.0)))
                    en = float(f.get('end', s.get('end', st)))
                except Exception:
                    st = float(s.get('start', 0.0))
                    en = float(s.get('end', st))
                if en <= st:
                    st = float(s.get('start', 0.0))
                    en = float(s.get('end', st))
                ns = dict(s)
                ns['start'] = st
                ns['end'] = en
                aligned.append(ns)
            return aligned, 'enabled:aeneas'
        except Exception:
            return segments, 'skipped:parse_error'


async def main() -> None:
    OUT2_DIR.mkdir(parents=True, exist_ok=True)
    video = pick_latest_video()

    provider = AIProvider(chat_model=OPENAI_CHAT_MODEL, whisper_model=OPENAI_WHISPER_MODEL)
    ai = AIService(provider=provider)

    audio_path = await extract_audio_from_video(video)
    align_status = 'skipped:not_run'
    diarization_provider = os.getenv('CARTOON_DIARIZATION_PROVIDER', 'assemblyai').strip().lower()
    diarization_effective = 'none'

    try:
        whisper = await provider._transcribe_with_whisper(audio_path)
        language = whisper.get('language') or 'unknown'
        diar_segments = []

        # Best-practice fallback chain:
        # 1) chosen provider (pyannoteai|pyannote|assemblyai)
        # 2) assemblyai fallback
        # 3) no diarization (speaker=None)
        if diarization_provider == 'pyannoteai':
            try:
                pyc = await provider._diarize_with_pyannoteai(audio_path)
                diar_segments = pyc.get('segments') or []
                diarization_effective = 'pyannoteai'
            except Exception:
                try:
                    asm = await provider._transcribe_with_assemblyai(audio_path)
                    diar_segments = asm.get('segments') or []
                    diarization_effective = 'assemblyai_fallback'
                    language = whisper.get('language') or asm.get('language') or language
                except Exception:
                    diar_segments = []
                    diarization_effective = 'none'
        elif diarization_provider == 'pyannote':
            try:
                py = await provider._diarize_with_pyannote(audio_path)
                diar_segments = py.get('segments') or []
                diarization_effective = 'pyannote'
            except Exception:
                try:
                    asm = await provider._transcribe_with_assemblyai(audio_path)
                    diar_segments = asm.get('segments') or []
                    diarization_effective = 'assemblyai_fallback'
                    language = whisper.get('language') or asm.get('language') or language
                except Exception:
                    diar_segments = []
                    diarization_effective = 'none'
        else:
            try:
                asm = await provider._transcribe_with_assemblyai(audio_path)
                diar_segments = asm.get('segments') or []
                diarization_effective = 'assemblyai'
                language = whisper.get('language') or asm.get('language') or language
            except Exception:
                diar_segments = []
                diarization_effective = 'none'

        split_segments = _hard_split_by_speaker(whisper.get('segments') or [], diar_segments)
        split_segments, align_status = _forced_align_with_aeneas(audio_path, split_segments)
    finally:
        audio_path.unlink(missing_ok=True)

    normalized = []
    for idx, s in enumerate(split_segments, start=1):
        txt = (s.get('text') or '').strip()
        if not txt:
            continue
        start = float(s.get('start') or 0.0)
        end = float(s.get('end') or 0.0)
        try:
            tr = await run_translation(
                text=txt,
                source_language=language,
                target_language='Russian',
                ai_service=ai,
            )
        except Exception:
            tr = txt

        cps, fit_status, risk_note = _fit_metrics(tr, start, end)
        review_reasons = list(s.get('_auto_flags') or [])
        dur = max(0.01, end - start)
        if s.get('speaker') is None:
            review_reasons.append('speaker_missing')
        if dur < 0.9:
            review_reasons.append('very_short_segment')
        if cps > (DUB_TARGET_CHARS_PER_SEC * 1.35):
            review_reasons.append('fit_hard_risk')

        normalized.append(
            {
                'id': idx,
                'stable_id': _stable_id(txt, start, end),
                'start': start,
                'end': end,
                'start_tc': _tc(start),
                'end_tc': _tc(end),
                'text': txt,
                'speaker': s.get('speaker'),
                'translation_ru': (tr or '').strip(),
                'voice': '',
                'fit_status': fit_status,
                'cps': round(cps, 2),
                'risk_note': risk_note,
                'needs_review': bool(review_reasons),
                'review_reason': sorted(set(review_reasons)),
            }
        )

    payload = {
        'video': str(video),
        'language': language,
        'segment_count': len(normalized),
        'segments': normalized,
        'meta': {
            'transcribe_provider': 'whisper',
            'diarization_provider': diarization_effective,
            'speaker_split_mode': 'hard_boundaries',
            'forced_alignment': align_status,
            'profile': 'cartoon_cast_strong',
            'qc': 'needs_review + review_reason flags enabled',
            'output_dir': str(OUT2_DIR),
        },
    }

    generated_json = OUT2_DIR / 'cartoon_segments_generated.json'
    manual_json = OUT2_DIR / 'cartoon_segments_manual.json'
    snapshot_json = OUT2_DIR / f'{video.stem}_cartoon_segments_draft.json'

    generated_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    snapshot_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    # Backward-compat mirror (can be removed later)
    compat_json = OUT2_DIR / 'cartoon_segments_translated.json'
    compat_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    # Never overwrite manual once it exists
    if not manual_json.exists():
        manual_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    print(generated_json)
    print(manual_json)
    print(snapshot_json)


if __name__ == '__main__':
    asyncio.run(main())
