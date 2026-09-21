"""Transformers fixed-prefix replay with next-token margin recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, LogitsProcessor


BASE = Path("/work/qt28/moss/models/MOSS-Transcribe-Diarize")
REPO = Path("/work/qt28/moss/MOSS-Transcribe-Diarize")
REPEAT_TOKEN_ID = 47815


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class MarginRecorder(LogitsProcessor):
    def __init__(self, repeat_token_id: int, top_k: int = 5):
        self.repeat_token_id = repeat_token_id
        self.top_k = top_k
        self.rows: list[dict] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        values, indices = torch.topk(scores[0], k=self.top_k)
        repeated = scores[0, self.repeat_token_id]
        non_repeat = next(
            float(value.item())
            for value, token in zip(values, indices)
            if int(token.item()) != self.repeat_token_id
        )
        self.rows.append(
            {
                "step": len(self.rows),
                "repeat_logit": float(repeated.item()),
                "best_non_repeat_logit": non_repeat,
                "repeat_margin": float(repeated.item()) - non_repeat,
                "top_ids": [int(value) for value in indices.tolist()],
                "top_logits": [float(value) for value in values.tolist()],
            }
        )
        return scores


def prefix_ids(processor, case: dict) -> list[int]:
    if "prefix_ids" in case:
        ids = [int(value) for value in case["prefix_ids"]]
    else:
        ids = processor.tokenizer.encode(case["prefix_text"], add_special_tokens=False)
    ids.extend([REPEAT_TOKEN_ID] * int(case.get("append_repeat_token_count", 0)))
    if ids and ids[-1] == 151645:
        raise ValueError("fixed prefix must be an unfinished assistant response without EOS")
    return ids


def leading_run(ids: list[int], token: int) -> int:
    count = 0
    for value in ids:
        if value != token:
            break
        count += 1
    return count


def run_case(model, processor, base_inputs: dict, case: dict, continuation_tokens: int, dtype_name: str) -> dict:
    ids = prefix_ids(processor, case)
    prompt_ids = base_inputs["input_ids"]
    full_ids = torch.cat(
        [prompt_ids, torch.tensor([ids], dtype=prompt_ids.dtype, device=prompt_ids.device)], dim=1
    )
    full_attention = torch.ones_like(full_ids)
    recorder = MarginRecorder(REPEAT_TOKEN_ID)
    kwargs = {
        "input_ids": full_ids,
        "attention_mask": full_attention,
        "input_features": base_inputs["input_features"],
        "audio_feature_lengths": base_inputs["audio_feature_lengths"],
        "audio_chunk_mapping": base_inputs["audio_chunk_mapping"],
        "max_new_tokens": continuation_tokens,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "eos_token_id": 151645,
        "pad_token_id": 151643,
        "use_cache": True,
        "logits_to_keep": 1,
        "logits_processor": [recorder],
    }
    start = time.perf_counter()
    context = torch.autocast("cuda", dtype=torch.bfloat16) if dtype_name == "bf16" else torch.no_grad()
    with torch.inference_mode(), context:
        output = model.generate(**kwargs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    continuation = [int(value) for value in output[0, full_ids.shape[1] :].tolist()]
    ended_eos = bool(continuation and continuation[-1] == 151645)
    decoded = processor.tokenizer.decode(continuation, skip_special_tokens=True)
    prefix_text = processor.tokenizer.decode(ids, skip_special_tokens=True)
    full_output_ids = ids + continuation
    full_text = processor.tokenizer.decode(full_output_ids, skip_special_tokens=True)

    sys.path.insert(0, str(REPO))
    from moss_transcribe_diarize import parse_transcript

    prefix_parsed = parse_transcript(prefix_text)
    parsed = parse_transcript(full_text)
    prefix_ends_in_repeat = bool(ids and ids[-1] == REPEAT_TOKEN_ID)
    leading_repeat_tokens = leading_run(continuation, REPEAT_TOKEN_ID)
    prefix_last_end = float(prefix_parsed[-1].end) if prefix_parsed else None
    replay_last_end = float(parsed[-1].end) if parsed else None
    return {
        "case": case["name"],
        "source": case["source"],
        "note": case["note"],
        "status": "ok",
        "dtype": dtype_name,
        "prefix_tokens": len(ids),
        "prefix_ids_sha256": sha_ids(ids),
        "prefix_ends_in_repeat": prefix_ends_in_repeat,
        "generated_ids": continuation,
        "generated_ids_sha256": sha_ids(continuation),
        "generated_tokens": len(continuation),
        "ended_eos": ended_eos,
        "leading_repeat_tokens": leading_repeat_tokens,
        "escaped_repeat": bool(
            prefix_ends_in_repeat and continuation and leading_repeat_tokens < len(continuation)
        ),
        "escaped_immediately": bool(
            prefix_ends_in_repeat and continuation and continuation[0] != REPEAT_TOKEN_ID
        ),
        "continuation_text": decoded,
        "parsed_segments_before_replay": len(prefix_parsed),
        "last_parsed_end_before_replay": prefix_last_end,
        "parsed_segments_after_replay": len(parsed),
        "last_parsed_end_after_replay": replay_last_end,
        "timestamp_advanced": bool(
            prefix_last_end is not None and replay_last_end is not None and replay_last_end > prefix_last_end
        ),
        "margin_trajectory": recorder.rows,
        "generation_seconds": seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model-label", required=True, choices=("base", "step60", "step90", "step150"))
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--scope", choices=("d0", "d1"), default="d1")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    spec = read(args.experiment / "cases.json")
    item = read(args.experiment / "input.json")
    selected_names = set(spec[f"{args.scope}_cases"])
    cases = [case for case in spec["cases"] if case["name"] in selected_names]
    model_path = Path(spec["models"][args.model_label])
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    external = processor.tokenizer.encode("外", add_special_tokens=False)
    if external != [REPEAT_TOKEN_ID]:
        raise RuntimeError(f"tokenizer invariant failed: 外 -> {external}")

    sys.path.insert(0, str(REPO))
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    ).cuda().eval()
    # Feature extraction is done once in FP32 outside CUDA autocast, then reused
    # for every fixed history of this model/backend run.
    prepared = prepare_inputs(
        processor,
        build_transcription_messages(item["audio"], item["prompt"]),
        device=None,
    )
    base_inputs = {key: value.to("cuda") for key, value in prepared.items()}
    if int(base_inputs["input_ids"].shape[1]) != int(item["prompt_len"]):
        raise RuntimeError("expanded audio prompt length changed")

    out = args.experiment / "hf" / args.dtype / args.model_label
    write(
        out / "run.json",
        {
            "backend": "transformers",
            "model_label": args.model_label,
            "model_path": str(model_path),
            "dtype": args.dtype,
            "scope": args.scope,
            "gpu": torch.cuda.get_device_name(args.gpu),
            "torch_version": torch.__version__,
            "prompt_tokens": int(base_inputs["input_ids"].shape[1]),
            "prompt_ids_sha256": tensor_sha(base_inputs["input_ids"]),
            "input_features_sha256": tensor_sha(base_inputs["input_features"]),
            "input_features_dtype": str(base_inputs["input_features"].dtype),
            "feature_protocol": "processor FP32 outside autocast; identical tensor reused across cases",
        },
    )
    failures = 0
    for case in cases:
        torch.cuda.reset_peak_memory_stats()
        try:
            result = run_case(model, processor, base_inputs, case, spec["continuation_tokens"], args.dtype)
        except Exception as exc:
            failures += 1
            result = {
                "case": case["name"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write(out / f'{case["name"]}.json', result)
        print(json.dumps({"model": args.model_label, "dtype": args.dtype, **result}, ensure_ascii=True), flush=True)
    write(out / "complete.json", {"complete": failures == 0, "failures": failures, "cases": len(cases)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
