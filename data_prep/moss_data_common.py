#!/usr/bin/env python3
"""Shared utilities for converting AliMeeting / AISHELL-4 / AMI into MOSS JSONL.

Output JSONL record (verified against official finetune.py):
  {"conversation": [
      {"role": "user",  "message_type": "text",  "content": <prompt>},
      {"role": "user",  "message_type": "audio", "content": <audio path>},
      {"role": "assistant", "message_type": "text", "content": <target>}]}

Target text format: ``[start][Sxx]text[end]`` blocks concatenated, seconds with
two decimals, e.g. ``[0.28][S01]Hello[2.32][3.22][S01]world[7.71]``.

Policies (documented in reports/):
* mono 16 kHz derived audio (channel 1 for 8-channel corpora);
* speaker IDs are session-local, assigned by first appearance -> S01, S02, ...;
* segments sorted by (start, end); overlapping cross-speaker segments kept;
* segments with empty text (after cleaning) are dropped.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/work/qt28/moss")
DATA = ROOT / "data"
EXTERNAL = DATA / "external"
EXTRACTED = DATA / "extracted"
AUDIO16K = DATA / "audio16k"
MANIFESTS = DATA / "manifests"
JSONL = DATA / "moss_jsonl"
REPORTS = DATA / "reports"
REFS = DATA / "refs"

TARGET_SR = 16000

# Official default prompt of MOSS-Transcribe-Diarize (Chinese, from
# moss_transcribe_diarize.inference_utils.DEFAULT_PROMPT).  At runtime the
# exact string is imported from the installed package when available.
_FALLBACK_PROMPT = (
    "请将音频转写为文本，每一段需以开始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
try:  # pragma: no cover - depends on runtime env
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT  # type: ignore
except Exception:  # noqa: BLE001
    DEFAULT_PROMPT = _FALLBACK_PROMPT


@dataclass
class Segment:
    start: float
    end: float
    speaker: str  # raw speaker id (source); replaced by Sxx after mapping
    text: str


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def read_text_auto(path: Path) -> str:
    data = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


_TIER_BLOCK = re.compile(r"item\s*\[\d+\]:(.*?)(?=item\s*\[\d+\]:|\Z)", re.S)
_TIER_NAME = re.compile(r'name\s*=\s*"((?:[^"\\]|\\.)*)"')
_INTERVAL = re.compile(
    r"intervals\s*\[\d+\]:\s*"
    r"xmin\s*=\s*(-?[0-9.eE+]+)\s*"
    r"xmax\s*=\s*(-?[0-9.eE+]+)\s*"
    r'text\s*=\s*"((?:[^"\\]|\\.)*)"',
    re.S,
)
_ESC = re.compile(r'\\(["\\nt])')
_ESC_MAP = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}


def _unescape(s: str) -> str:
    return _ESC.sub(lambda m: _ESC_MAP[m.group(1)], s)


def parse_textgrid(path: Path):
    """Parse a Praat TextGrid file.

    Returns list of (tier_name, [(start, end, text), ...]) in file order.
    Tolerates both the tab-indented (AliMeeting) and blank-line separated
    (AISHELL-4) flavours.
    """
    text = read_text_auto(path).replace("\r", "")
    tiers = []
    for block in _TIER_BLOCK.findall(text):
        m = _TIER_NAME.search(block)
        name = _unescape(m.group(1)) if m else ""
        intervals = [
            (float(a), float(b), _unescape(t))
            for a, b, t in _INTERVAL.findall(block)
        ]
        tiers.append((name, intervals))
    return tiers


_TAG_RE = re.compile(r"<[^>]*>")


def clean_text(s: str) -> str:
    """Normalise whitespace and drop inline annotation tags like <sil> / <#>."""
    s = _TAG_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Segment / target helpers
# ---------------------------------------------------------------------------

def speaker_mapping(segments) -> dict:
    """Map raw speaker ids to S01.. by order of first appearance."""
    order = []
    for seg in sorted(segments, key=lambda x: (x.start, x.end)):
        if seg.speaker not in order:
            order.append(seg.speaker)
    return {raw: f"S{i + 1:02d}" for i, raw in enumerate(order)}


def format_target(segments) -> str:
    return "".join(
        f"[{seg.start:.2f}][{seg.speaker}]{seg.text}[{seg.end:.2f}]"
        for seg in segments
    )


def clamp_segments(segments, duration: float, stats: dict):
    """Drop/clamp segments outside [0, duration]; update stats counters."""
    out = []
    for seg in segments:
        if seg.start < 0:
            stats["neg_start_dropped"] = stats.get("neg_start_dropped", 0) + 1
            continue
        if seg.start >= duration + 0.05:
            stats["past_end_dropped"] = stats.get("past_end_dropped", 0) + 1
            continue
        if seg.end > duration + 0.05:
            stats["end_clamped"] = stats.get("end_clamped", 0) + 1
            seg = Segment(seg.start, min(seg.end, duration), seg.speaker, seg.text)
        if seg.end <= seg.start:
            stats["zero_len_dropped"] = stats.get("zero_len_dropped", 0) + 1
            continue
        out.append(seg)
    return out


def overlap_stats(segments):
    """Return (speech_seconds, overlap_seconds) using 10 ms resolution."""
    if not segments:
        return 0.0, 0.0
    step = 0.01
    max_t = max(s.end for s in segments)
    n = int(max_t / step) + 2
    active = np.zeros(n, dtype=np.int16)
    for seg in segments:
        a = max(0, int(seg.start / step))
        b = min(n, int(np.ceil(seg.end / step)))
        active[a:b] += 1
    speech = float((active > 0).sum()) * step
    overlap = float((active > 1).sum()) * step
    return speech, overlap


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def convert_audio_ch1(src: Path, dst: Path, force: bool = False) -> dict:
    """Read ``src`` (using channel 0), resample to 16 kHz, write PCM-16 mono wav.

    Returns {"duration": seconds, "src_sr": int, "src_channels": int}.
    """
    if dst.exists() and dst.stat().st_size > 0 and not force:
        info = sf.info(str(dst))
        return {"duration": info.frames / info.samplerate, "src_sr": None,
                "src_channels": None, "cached": True}
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    channels = data.shape[1]
    audio = data[:, 0]
    if sr != TARGET_SR:
        import soxr  # noqa: PLC0415
        audio = soxr.resample(audio, sr, TARGET_SR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), audio, TARGET_SR, subtype="PCM_16")
    return {"duration": len(audio) / TARGET_SR, "src_sr": sr,
            "src_channels": channels}


def audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {"duration": info.frames / max(info.samplerate, 1),
            "sr": info.samplerate, "channels": info.channels}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def make_record(audio_path, target: str, prompt: str = DEFAULT_PROMPT) -> dict:
    return {"conversation": [
        {"role": "user", "message_type": "text", "content": prompt},
        {"role": "user", "message_type": "audio", "content": str(audio_path)},
        {"role": "assistant", "message_type": "text", "content": target},
    ]}


def write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_manifest(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_manifest(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def log(msg: str):
    print(msg, flush=True)
