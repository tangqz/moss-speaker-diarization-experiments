#!/usr/bin/env python
"""Phase 0 smoke test: load MOSS-Transcribe-Diarize and transcribe one short audio file.

Usage:
    python smoke_infer.py [audio_path]

Environment variables:
    MOSS_MODEL_ID        default OpenMOSS-Team/MOSS-Transcribe-Diarize
    MOSS_MAX_NEW_TOKENS  default 2048
"""
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)


def main() -> int:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/work/qt28/moss/data/samples/jfk.wav"
    model_id = os.environ.get("MOSS_MODEL_ID", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
    max_new_tokens = int(os.environ.get("MOSS_MAX_NEW_TOKENS", "2048"))

    device = resolve_device("auto")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[smoke] audio  = {audio_path}")
    print(f"[smoke] model  = {model_id}")
    print(f"[smoke] device = {device} dtype = {dtype}")

    try:
        import soundfile as sf
        info = sf.info(audio_path)
        print(f"[smoke] audio info: {info.samplerate} Hz, {info.duration:.2f} s, {info.channels} ch")
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] audio info unavailable: {type(exc).__name__}: {exc}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype="auto",
        attn_implementation="sdpa",
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_s = time.time() - t0
    print(f"[smoke] model loaded in {load_s:.1f}s")
    if device.type == "cuda":
        print(f"[smoke] gpu mem after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    messages = build_transcription_messages(audio_path)
    t1 = time.time()
    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    gen_s = time.time() - t1
    print(f"[smoke] generation took {gen_s:.1f}s")
    if device.type == "cuda":
        print(f"[smoke] gpu mem peak: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    text = result["text"]
    print("===== RAW OUTPUT =====")
    print(text)
    print("===== PARSED SEGMENTS =====")
    try:
        segments = parse_transcript(text)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] parse failed: {type(exc).__name__}: {exc}")
        return 2
    for seg in segments:
        print(f"[{seg.start:8.2f} -> {seg.end:8.2f}] {seg.speaker}: {seg.text}")
    print(f"[smoke] parsed segments: {len(segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
