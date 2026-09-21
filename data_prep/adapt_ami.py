#!/usr/bin/env python3
"""AMI Meeting Corpus adapter -> MOSS conversation JSONL.

Official split (Full-corpus-ASR partition, via pyannote/BUT AMI-diarization-setup):
    train = 136 meetings, dev = 18, test = 16   (170 total)

Source:
    audio/<meeting>.Mix-Headset.wav       (16 kHz mono, used as-is)
    words/<meeting>.<letter>.words.xml    (per-participant word tokens with
                                           starttime/endtime; punc="true" items
                                           are punctuation without timing)

Merge policy: consecutive words of the same speaker with a silence gap of
<= MERGE_GAP seconds are merged into one segment; larger pauses start a new
segment.  Punctuation tokens are appended to the current segment text.
Speaker letters are mapped to S01..Sxx by order of first appearance.
"""
from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_data_common import (  # noqa: E402
    AUDIO16K, EXTRACTED, EXTERNAL, JSONL, MANIFESTS, Segment, append_manifest,
    audio_info, clamp_segments, convert_audio_ch1, format_target, log,
    make_record, overlap_stats, reset_manifest, speaker_mapping, write_jsonl,
)

DATASET = "ami"
LISTS = EXTERNAL / "ami" / "ami_setup" / "lists"
WORDS_DIR = EXTRACTED / "ami_annotations" / "words"
AUDIO_DIR = EXTERNAL / "ami" / "audio"
MERGE_GAP = 0.5


def load_meeting_words(meeting: str):
    """Return {speaker_letter: [(start, end, token, is_punc), ...]}."""
    per_speaker = {}
    for fp in sorted(WORDS_DIR.glob(f"{meeting}.*.words.xml")):
        parts = fp.name.split(".")
        if len(parts) < 3 or parts[0] != meeting:
            continue
        letter = parts[1]
        words = []
        root = ET.parse(str(fp)).getroot()
        for el in root:
            tag = el.tag.rsplit("}", 1)[-1]
            if tag != "w":
                continue  # skip disfmarkers etc.
            st, en = el.get("starttime"), el.get("endtime")
            if st is None or en is None:
                continue
            token = (el.text or "").strip()
            is_punc = el.get("punc") == "true"
            words.append((float(st), float(en), token, is_punc))
        words.sort(key=lambda w: (w[0], w[1]))
        if words:
            per_speaker[letter] = words
    return per_speaker


def merge_words(words):
    """Merge word tokens into segments -> list[(start, end, text)]."""
    segments = []
    cur = None  # [start, end, text]
    for st, en, token, is_punc in words:
        if is_punc:
            if cur is not None and token:
                cur[2] += token
            continue
        if not token:
            continue
        if cur is None or st - cur[1] > MERGE_GAP:
            if cur is not None:
                segments.append(tuple(cur))
            cur = [st, en, token]
        else:
            cur[1] = max(cur[1], en)
            cur[2] += " " + token
    if cur is not None:
        segments.append(tuple(cur))
    return segments


def convert_split(split: str, limit: int, force_audio: bool) -> None:
    meetings = [m.strip() for m in (LISTS / f"{split}.meetings.txt")
                .read_text(encoding="utf-8").splitlines() if m.strip()]
    if limit:
        meetings = meetings[:limit]
    man_path = MANIFESTS / f"{DATASET}_{split}.jsonl"
    reset_manifest(man_path)
    records = []
    stats = {"missing_audio": 0, "no_words": 0, "converted_audio": 0}
    t0 = time.time()
    for i, meeting in enumerate(meetings, 1):
        src = AUDIO_DIR / f"{meeting}.Mix-Headset.wav"
        if not src.exists():
            stats["missing_audio"] += 1
            log(f"  [{DATASET}/{split}] {meeting}: AUDIO MISSING - skipped")
            continue
        per_speaker = load_meeting_words(meeting)
        raw = []
        for letter, words in per_speaker.items():
            for st, en, text in merge_words(words):
                raw.append(Segment(st, en, letter, text))
        raw.sort(key=lambda s: (s.start, s.end))
        if not raw:
            stats["no_words"] += 1
            log(f"  [{DATASET}/{split}] {meeting}: NO WORDS - skipped")
            continue
        info = audio_info(src)
        if info["sr"] == 16000 and info["channels"] == 1:
            audio_path = src
            duration = info["duration"]
        else:
            dst = AUDIO16K / DATASET / f"{meeting}.wav"
            conv = convert_audio_ch1(src, dst, force=force_audio)
            audio_path, duration = dst, conv["duration"]
            stats["converted_audio"] += 1
        # AMI word timestamps can extend slightly past the released audio
        # (e.g. TS3007c); clamp to the actual duration per validation policy.
        raw = clamp_segments(raw, duration, stats)
        if not raw:
            stats["no_words"] += 1
            log(f"  [{DATASET}/{split}] {meeting}: ALL SEGMENTS OUT OF BOUNDS - skipped")
            continue
        mapping = speaker_mapping(raw)
        segments = [Segment(s.start, s.end, mapping[s.speaker], s.text)
                    for s in raw]
        target = format_target(segments)
        records.append(make_record(audio_path, target))
        speech, overlap = overlap_stats(segments)
        append_manifest(man_path, {
            "dataset": DATASET, "split": split, "session_id": meeting,
            "audio_path": str(audio_path), "audio_src": str(src),
            "ann_src": str(WORDS_DIR / f"{meeting}.*.words.xml"),
            "duration": round(duration, 3),
            "n_segments": len(segments), "n_speakers": len(mapping),
            "speech_time": round(speech, 3), "overlap_time": round(overlap, 3),
            "speaker_map": mapping,
        })
        if i % 25 == 0 or i == len(meetings):
            log(f"  [{DATASET}/{split}] {i}/{len(meetings)}  ({time.time() - t0:.0f}s)")
    n = write_jsonl(JSONL / f"{DATASET}_{split}.jsonl", records)
    log(f"== {DATASET}/{split}: {n} records; skipped missing_audio={stats['missing_audio']} "
        f"no_words={stats['no_words']} converted_audio={stats['converted_audio']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-audio", action="store_true")
    args = ap.parse_args()
    for split in args.splits.split(","):
        split = split.strip()
        if split:
            convert_split(split, args.limit, args.force_audio)


if __name__ == "__main__":
    main()
