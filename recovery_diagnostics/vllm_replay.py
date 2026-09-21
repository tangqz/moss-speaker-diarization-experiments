"""vLLM fixed-prefix replay for D0 token-sequence parity and determinism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


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


def prefix_ids(processor, case: dict) -> list[int]:
    if "prefix_ids" in case:
        ids = [int(value) for value in case["prefix_ids"]]
    else:
        ids = processor.tokenizer.encode(case["prefix_text"], add_special_tokens=False)
    ids.extend([REPEAT_TOKEN_ID] * int(case.get("append_repeat_token_count", 0)))
    if ids and ids[-1] == 151645:
        raise ValueError("fixed prefix must not end with EOS")
    return ids


def leading_run(ids: list[int], token: int) -> int:
    count = 0
    for value in ids:
        if value != token:
            break
        count += 1
    return count


def logprob_rows(output) -> list[dict]:
    rows = []
    for step, mapping in enumerate(output.logprobs or []):
        ordered = sorted(mapping.items(), key=lambda pair: pair[1].rank or 10**9)
        repeat = mapping.get(REPEAT_TOKEN_ID)
        best_other = next((entry for token, entry in ordered if int(token) != REPEAT_TOKEN_ID), None)
        repeat_value = float(repeat.logprob) if repeat is not None else None
        other_value = float(best_other.logprob) if best_other is not None else None
        rows.append(
            {
                "step": step,
                "repeat_logprob": repeat_value,
                "best_non_repeat_logprob": other_value,
                "repeat_margin": repeat_value - other_value if repeat_value is not None and other_value is not None else None,
                "top_ids": [int(token) for token, _ in ordered],
                "top_logprobs": [float(entry.logprob) for _, entry in ordered],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model-label", required=True, choices=("base", "step60", "step90", "step150"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[args.gpu]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch
    import vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(REPO))
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, process_audio_info

    spec = read(args.experiment / "cases.json")
    item = read(args.experiment / "input.json")
    cases = [case for case in spec["cases"] if case["name"] in set(spec["d0_cases"])]
    model_path = Path(spec["models"][args.model_label])
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    if processor.tokenizer.encode("外", add_special_tokens=False) != [REPEAT_TOKEN_ID]:
        raise RuntimeError("repeat-token invariant failed")
    messages = build_transcription_messages(item["audio"], item["prompt"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)
    prompt_ids = processor(text=text, audio=audios, return_tensors="pt")["input_ids"][0].tolist()
    if len(prompt_ids) != int(item["prompt_len"]):
        raise RuntimeError("expanded audio prompt length changed")

    config = {
        "model": str(model_path),
        "tokenizer": str(BASE),
        "trust_remote_code": True,
        "model_impl": "vllm",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 131072,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 49152,
        "gpu_memory_utilization": 0.72,
        "enable_prefix_caching": False,
        "generation_config": "vllm",
        "seed": 0,
        "limit_mm_per_prompt": {"audio": 1},
    }
    start = time.perf_counter()
    llm = LLM(**config)
    startup_seconds = time.perf_counter() - start
    out = args.experiment / "vllm" / "bf16" / args.model_label
    write(
        out / "run.json",
        {
            "backend": "vllm",
            "backend_version": vllm.__version__,
            "torch_version": torch.__version__,
            "model_label": args.model_label,
            "model_path": str(model_path),
            "gpu": torch.cuda.get_device_name(0),
            "startup_seconds": startup_seconds,
            "config": config,
            "prompt_tokens": len(prompt_ids),
            "prompt_ids_sha256": sha_ids(prompt_ids),
        },
    )

    params = SamplingParams(
        temperature=0,
        top_p=1,
        top_k=-1,
        repetition_penalty=1,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=int(spec["continuation_tokens"]),
        stop_token_ids=[151645],
        ignore_eos=False,
        skip_special_tokens=True,
        seed=0,
        logprobs=20,
    )
    failures = 0
    for case in cases:
        try:
            prefix = prefix_ids(processor, case)
            full_ids = prompt_ids + prefix
            request = {
                "prompt_token_ids": full_ids,
                "multi_modal_data": {"audio": (audios[0], processor.feature_extractor.sampling_rate)},
            }
            repeats = []
            for repetition in (1, 2):
                generation_start = time.perf_counter()
                result = llm.generate([request], params, use_tqdm=True)[0]
                generation_seconds = time.perf_counter() - generation_start
                if list(result.prompt_token_ids) != full_ids:
                    raise RuntimeError("vLLM changed supplied fixed-prefix prompt IDs")
                generated = [int(value) for value in result.outputs[0].token_ids]
                repeats.append(
                    {
                        "repetition": repetition,
                        "generated_ids": generated,
                        "generated_ids_sha256": sha_ids(generated),
                        "generated_tokens": len(generated),
                        "leading_repeat_tokens": leading_run(generated, REPEAT_TOKEN_ID),
                        "ended_eos": bool(generated and generated[-1] == 151645)
                        or result.outputs[0].stop_reason == 151645,
                        "finish_reason": result.outputs[0].finish_reason,
                        "stop_reason": result.outputs[0].stop_reason,
                        "continuation_text": processor.tokenizer.decode(generated, skip_special_tokens=True),
                        "margin_trajectory": logprob_rows(result.outputs[0]),
                        "generation_seconds": generation_seconds,
                    }
                )
            record = {
                "case": case["name"],
                "source": case["source"],
                "note": case["note"],
                "status": "ok",
                "backend": "vllm",
                "dtype": "bf16",
                "prefix_tokens": len(prefix),
                "prefix_ids_sha256": sha_ids(prefix),
                "prefix_ends_in_repeat": bool(prefix and prefix[-1] == REPEAT_TOKEN_ID),
                "deterministic_repeat": repeats[0]["generated_ids"] == repeats[1]["generated_ids"],
                "runs": repeats,
            }
        except Exception as exc:
            failures += 1
            record = {
                "case": case["name"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write(out / f'{case["name"]}.json', record)
        print(json.dumps({"model": args.model_label, **record}, ensure_ascii=True), flush=True)
    write(out / "complete.json", {"complete": failures == 0, "failures": failures, "cases": len(cases)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
