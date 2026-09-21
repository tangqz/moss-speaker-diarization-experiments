#!/usr/bin/env python3
"""AliMeeting (M2MeT) far-field adapter -> MOSS conversation JSONL.

Official splits:
    train = Train_Ali_far        (212 sessions)
    dev   = Eval_Ali/Eval_Ali_far   (8 sessions, official "Eval")
    test  = Test_Ali/Test_Ali_far   (20 sessions, official "Test")

Source layout per split:
    audio_dir/<Rxxxx_Mxxxx>_<MSxxx>.wav   (8ch, 16 kHz)
    textgrid_dir/<Rxxxx_Mxxxx>.TextGrid   (one IntervalTier per speaker)

Target: far-field channel 1 -> 16 kHz mono wav; speaker tiers -> segments.
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

DATASET = "alimeeting"
SPLITS = {
    "train": EXTRACTED / "Train_Ali_far",
    "dev": EXTRACTED / "Eval_Ali" / "Eval_Ali_far",
    "test": EXTRACTED / "Test_Ali" / "Test_Ali_far",
}


def convert_split(split: str, limit: int, force_audio: bool) -> None:
    root = SPLITS[split]
    tg_dir = root / "textgrid_dir"
    wav_dir = root / "audio_dir"
    tg_files = sorted(tg_dir.glob("*.TextGrid"))
    if limit:
        tg_files = tg_files[:limit]
    man_path = MANIFESTS / f"{DATASET}_{split}.jsonl"
    reset_manifest(man_path)
    records = []
    stats = {"no_audio": 0, "no_segments": 0}
    t0 = time.time()
    for i, tg in enumerate(tg_files, 1):
        stem = tg.stem
        wavs = sorted(wav_dir.glob(stem + "_*.wav"))
        if not wavs:
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
        info = convert_audio_ch1(wavs[0], dst, force=force_audio)
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
            "audio_path": str(dst), "audio_src": str(wavs[0]),
            "ann_src": str(tg), "duration": round(duration, 3),
            "n_segments": len(segments), "n_speakers": len(mapping),
            "speech_time": round(speech, 3), "overlap_time": round(overlap, 3),
            "speaker_map": mapping,
        })
        if i % 25 == 0 or i == len(tg_files):
            log(f"  [{DATASET}/{split}] {i}/{len(tg_files)}  ({time.time() - t0:.0f}s)")
    n = write_jsonl(JSONL / f"{DATASET}_{split}.jsonl", records)
    log(f"== {DATASET}/{split}: {n} records written; skipped no_audio={stats['no_audio']} "
        f"no_segments={stats['no_segments']} end_clamped={stats.get('end_clamped', 0)} "
        f"dropped={stats.get('neg_start_dropped', 0) + stats.get('past_end_dropped', 0) + stats.get('zero_len_dropped', 0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--limit", type=int, default=0, help="max sessions per split (debug)")
    ap.add_argument("--force-audio", action="store_true")
    args = ap.parse_args()
    for split in args.splits.split(","):
        split = split.strip()
        if split:
            convert_split(split, args.limit, args.force_audio)


if __name__ == "__main__":
    main()
