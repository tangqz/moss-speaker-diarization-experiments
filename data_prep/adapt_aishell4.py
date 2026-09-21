#!/usr/bin/env python3
"""AISHELL-4 adapter -> MOSS conversation JSONL.

Official splits:
    train = train_L + train_M + train_S   (191 sessions)
    test  = test                          (20 sessions, official)

Source layout per part directory:
    wav/<session>.flac            (8ch, 16 kHz, includes FLAC)
    TextGrid/<session>.TextGrid   (one IntervalTier per speaker; intervals with
                                   empty text are non-speech and skipped)
    TextGrid/<session>.rttm       (official diarization reference - used for eval)

Inline transcript tags such as <sil> and <#> are removed (policy: keep text
only; drop a segment if its text becomes empty).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_data_common import (  # noqa: E402
    AUDIO16K, EXTRACTED, JSONL, MANIFESTS, Segment, append_manifest,
    clamp_segments, clean_text, convert_audio_ch1, format_target, log,
    make_record, overlap_stats, parse_textgrid, reset_manifest,
    speaker_mapping, write_jsonl,
)

DATASET = "aishell4"
SPLITS = {
    "train": ["train_L", "train_M", "train_S"],
    "test": ["test"],
}


def convert_split(split: str, parts, limit: int, force_audio: bool) -> None:
    man_path = MANIFESTS / f"{DATASET}_{split}.jsonl"
    reset_manifest(man_path)
    records = []
    stats = {"no_audio": 0, "no_segments": 0, "sessions": 0}
    t0 = time.time()
    total = 0
    for part in parts:
        tg_files = sorted((EXTRACTED / part / "TextGrid").glob("*.TextGrid"))
        total += len(tg_files)
    done = 0
    for part in parts:
        tg_dir = EXTRACTED / part / "TextGrid"
        wav_dir = EXTRACTED / part / "wav"
        for tg in sorted(tg_dir.glob("*.TextGrid")):
            if limit and done >= limit:
                break
            done += 1
            stem = tg.stem
            src = wav_dir / f"{stem}.flac"
            if not src.exists():
                stats["no_audio"] += 1
                log(f"  [{DATASET}/{split}] {stem}: NO AUDIO - skipped")
                continue
            raw_segments = []
            for tier_name, intervals in parse_textgrid(tg):
                for a, b, text in intervals:
                    text = clean_text(text)
                    if not text or b <= a:
                        continue
                    raw_segments.append(Segment(a, b, tier_name, text))
            if not raw_segments:
                stats["no_segments"] += 1
                log(f"  [{DATASET}/{split}] {stem}: NO SEGMENTS - skipped")
                continue
            raw_segments.sort(key=lambda s: (s.start, s.end))
            dst = AUDIO16K / DATASET / f"{stem}.wav"
            info = convert_audio_ch1(src, dst, force=force_audio)
            duration = info["duration"]
            segments = clamp_segments(raw_segments, duration, stats)
            mapping = speaker_mapping(segments)
            segments = [Segment(s.start, s.end, mapping[s.speaker], s.text)
                        for s in segments]
            target = format_target(segments)
            records.append(make_record(dst, target))
            speech, overlap = overlap_stats(segments)
            append_manifest(man_path, {
                "dataset": DATASET, "split": split, "session_id": stem,
                "part": part,
                "audio_path": str(dst), "audio_src": str(src),
                "ann_src": str(tg), "duration": round(duration, 3),
                "n_segments": len(segments), "n_speakers": len(mapping),
                "speech_time": round(speech, 3), "overlap_time": round(overlap, 3),
                "speaker_map": mapping,
            })
            stats["sessions"] += 1
            if done % 25 == 0 or done == total:
                log(f"  [{DATASET}/{split}] {done}/{total}  ({time.time() - t0:.0f}s)")
    n = write_jsonl(JSONL / f"{DATASET}_{split}.jsonl", records)
    log(f"== {DATASET}/{split}: {n} records; skipped no_audio={stats['no_audio']} "
        f"no_segments={stats['no_segments']} end_clamped={stats.get('end_clamped', 0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-audio", action="store_true")
    args = ap.parse_args()
    for split in args.splits.split(","):
        split = split.strip()
        if split:
            convert_split(split, SPLITS[split], args.limit, args.force_audio)


if __name__ == "__main__":
    main()
