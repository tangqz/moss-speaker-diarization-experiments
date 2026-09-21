#!/usr/bin/env python3
"""Stage evaluation references for the three official test sets.

  AliMeeting : RTTM + UEM generated from the far-field TextGrid (raw speaker
               ids as tier names).
  AISHELL-4  : official per-session RTTM copied from the corpus; UEM generated
               from our derived audio.
  AMI        : official "only_words" RTTM + UEM from AMI-diarization-setup
               (pyannote/BUT fork), copied for all splits.

Outputs under data/refs/<dataset>/<split>/.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_data_common import (  # noqa: E402
    AUDIO16K, EXTRACTED, EXTERNAL, REFS, clean_text, log, parse_textgrid,
)

ALI_FAR = {
    "dev": EXTRACTED / "Eval_Ali" / "Eval_Ali_far",
    "test": EXTRACTED / "Test_Ali" / "Test_Ali_far",
    "train": EXTRACTED / "Train_Ali_far",
}
AMI_SETUP = EXTERNAL / "ami" / "ami_setup"


def gen_ali_refs(split: str) -> int:
    root = ALI_FAR[split]
    out_dir = REFS / "alimeeting" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for tg in sorted((root / "textgrid_dir").glob("*.TextGrid")):
        stem = tg.stem
        audio = AUDIO16K / "alimeeting" / f"{stem}.wav"
        if not audio.exists():
            continue
        import soundfile as sf  # noqa: PLC0415
        info = sf.info(str(audio))
        duration = info.frames / info.samplerate
        rows = []
        for tier_name, intervals in parse_textgrid(tg):
            speaker = tier_name[2:] if tier_name.startswith("N_") else tier_name
            for a, b, text in intervals:
                if not clean_text(text) or b <= a:
                    continue
                rows.append(f"SPEAKER {stem} 1 {a:.3f} {b - a:.3f} <NA> <NA> {speaker} <NA> <NA>")
        rows.sort(key=lambda r: float(r.split()[3]))
        (out_dir / f"{stem}.rttm").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (out_dir / f"{stem}.uem").write_text(f"{stem} 1 0.000 {duration:.3f}\n", encoding="utf-8")
        n += 1
    return n


def copy_aishell_refs() -> int:
    out_dir = REFS / "aishell4" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rttm in sorted((EXTRACTED / "test" / "TextGrid").glob("*.rttm")):
        shutil.copy2(rttm, out_dir / rttm.name)
        stem = rttm.stem
        audio = AUDIO16K / "aishell4" / f"{stem}.wav"
        if audio.exists():
            import soundfile as sf  # noqa: PLC0415
            info = sf.info(str(audio))
            (out_dir / f"{stem}.uem").write_text(
                f"{stem} 1 0.000 {info.frames / info.samplerate:.3f}\n", encoding="utf-8")
        n += 1
    return n


def copy_ami_refs() -> int:
    n = 0
    for split in ("train", "dev", "test"):
        rttm_dir = AMI_SETUP / "only_words" / "rttms" / split
        uem_dir = AMI_SETUP / "uems" / split
        out_dir = REFS / "ami" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for fp in sorted(rttm_dir.glob("*.rttm")):
            shutil.copy2(fp, out_dir / fp.name)
            n += 1
        if uem_dir.exists():
            for fp in sorted(uem_dir.glob("*.uem")):
                shutil.copy2(fp, out_dir / fp.name)
    return n


def main() -> None:
    for split in ("train", "dev", "test"):
        log(f"alimeeting/{split}: {gen_ali_refs(split)} rttm")
    log(f"aishell4/test: {copy_aishell_refs()} rttm")
    log(f"ami: {copy_ami_refs()} rttm")
    log("refs staged under " + str(REFS))


if __name__ == "__main__":
    main()
