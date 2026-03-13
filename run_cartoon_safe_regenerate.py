import json
from pathlib import Path

OUT2 = Path('/home/fanfan/projects/dubfilm/out2')
GENERATED = OUT2 / 'cartoon_segments_generated.json'
MANUAL = OUT2 / 'cartoon_segments_manual.json'
REPORT = OUT2 / 'cartoon_merge_report.json'

PRESERVE_FIELDS = {'translation_ru', 'voice', 'character', 'lock_text'}


def _overlap(a: dict, b: dict) -> float:
    a1, a2 = float(a.get('start', 0.0)), float(a.get('end', 0.0))
    b1, b2 = float(b.get('start', 0.0)), float(b.get('end', 0.0))
    return max(0.0, min(a2, b2) - max(a1, b1))


def _norm_text(s: str) -> str:
    return ' '.join((s or '').strip().lower().split())


def _find_match(g: dict, manual_segments: list[dict], used: set[int]) -> tuple[int | None, str]:
    sid = g.get('stable_id')
    if sid:
        for i, m in enumerate(manual_segments):
            if i in used:
                continue
            if m.get('stable_id') == sid:
                return i, 'stable_id'

    gt = _norm_text(g.get('text') or '')
    best_i, best_score = None, 0.0
    for i, m in enumerate(manual_segments):
        if i in used:
            continue
        ov = _overlap(g, m)
        if ov <= 0:
            continue
        mt = _norm_text(m.get('text') or '')
        score = ov
        if gt and mt and gt == mt:
            score += 10.0
        if score > best_score:
            best_score, best_i = score, i
    if best_i is not None:
        return best_i, 'overlap_text'
    return None, 'none'


def main() -> None:
    if not GENERATED.exists():
        raise FileNotFoundError(f'Missing generated JSON: {GENERATED}')

    gdoc = json.loads(GENERATED.read_text(encoding='utf-8'))

    if not MANUAL.exists():
        MANUAL.write_text(json.dumps(gdoc, ensure_ascii=False, indent=2), encoding='utf-8')
        REPORT.write_text(json.dumps({'status': 'manual_created_from_generated'}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(MANUAL)
        print(REPORT)
        return

    mdoc = json.loads(MANUAL.read_text(encoding='utf-8'))
    gsegs = gdoc.get('segments') or []
    msegs = mdoc.get('segments') or []

    merged = []
    used = set()
    matched = 0
    unmatched = []

    for g in gsegs:
        i, mode = _find_match(g, msegs, used)
        out = dict(g)
        if i is not None:
            used.add(i)
            matched += 1
            m = msegs[i]
            for f in PRESERVE_FIELDS:
                if f in m:
                    out[f] = m.get(f)
        else:
            unmatched.append(g.get('id'))
        merged.append(out)

    # carry over manual-only locked segments (rare, but safer)
    carried = 0
    for i, m in enumerate(msegs):
        if i in used:
            continue
        if m.get('lock_text'):
            merged.append(dict(m))
            carried += 1

    merged_doc = dict(gdoc)
    merged_doc['segments'] = merged
    merged_doc['segment_count'] = len(merged)
    MANUAL.write_text(json.dumps(merged_doc, ensure_ascii=False, indent=2), encoding='utf-8')

    report = {
        'generated': str(GENERATED),
        'manual': str(MANUAL),
        'generated_count': len(gsegs),
        'manual_count_before': len(msegs),
        'matched': matched,
        'unmatched_generated_ids': unmatched,
        'carried_manual_locked': carried,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(MANUAL)
    print(REPORT)


if __name__ == '__main__':
    main()
