"""Run full-meeting decoding sweeps for the known problematic sample."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path("/work/qt28/moss")
BASE = ROOT / "models/MOSS-Transcribe-Diarize"
REPO = ROOT / "MOSS-Transcribe-Diarize"
TRAIN_RUN = ROOT / "results/ms-swift-63643"
TARGET_KEY = "alimeeting/test/R8005_M8009"

CONFIGS = {
    "penalty": [
        {"name": "A0_seed0", "temperature": 0.0, "repetition_penalty": 1.00, "seed": 0},
        {"name": "A1_seed0", "temperature": 0.0, "repetition_penalty": 1.02, "seed": 0},
        {"name": "A2_seed0", "temperature": 0.0, "repetition_penalty": 1.05, "seed": 0},
        {"name": "A3_seed0", "temperature": 0.0, "repetition_penalty": 1.10, "seed": 0},
    ],
    "sampling": [
        {"name": f"B1_seed{seed}", "temperature": 0.2, "repetition_penalty": 1.00, "seed": seed}
        for seed in (0, 1, 2)
    ]
    + [
        {"name": f"B2_seed{seed}", "temperature": 0.4, "repetition_penalty": 1.00, "seed": seed}
        for seed in (0, 1, 2)
    ],
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def leading_run(ids: list[int], token: int) -> int:
    count = 0
    for value in ids:
        if value != token:
            break
        count += 1
    return count


def make_params(config: dict, max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=config["temperature"],
        top_p=1,
        top_k=-1,
        min_p=0,
        repetition_penalty=config["repetition_penalty"],
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=max_tokens,
        stop_token_ids=[151645],
        ignore_eos=False,
        skip_special_tokens=True,
        seed=config["seed"],
    )


def generate(llm, request, params):
    start = time.perf_counter()
    result = llm.generate([request], params, use_tqdm=True)[0]
    seconds = time.perf_counter() - start
    output = result.outputs[0]
    ids = [int(value) for value in output.token_ids]
    eos = bool(ids and ids[-1] == 151645) or output.stop_reason == 151645
    return result, ids, eos, seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-label", choices=("base", "sft"), required=True)
    parser.add_argument("--family", choices=("penalty", "sampling"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[args.gpu]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch
    import vllm
    from transformers import AutoProcessor
    from vllm import LLM

    sys.path.insert(0, str(REPO))
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, process_audio_info

    sys.path.insert(0, str(TRAIN_RUN / "source/evaluation"))
    from diagnostics import output_diagnostics

    item = read(args.out / "runs/A0_seed0/inputs.json")[0]
    model_path = BASE if args.model_label == "base" else TRAIN_RUN / "vllm_exports/checkpoint-150"

    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    messages = build_transcription_messages(item["audio"], item["prompt"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)
    prompt_ids = processor(text=text, audio=audios, return_tensors="pt")["input_ids"][0].tolist()
    if len(prompt_ids) != int(item["prompt_len"]):
        raise RuntimeError("expanded prompt length changed")

    engine_config = {
        "model": str(model_path),
        "tokenizer": str(BASE),
        "trust_remote_code": True,
        "model_impl": "vllm",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 131072,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 49152,
        "gpu_memory_utilization": 0.45,
        "enable_prefix_caching": False,
        "generation_config": "vllm",
        "seed": 0,
        "limit_mm_per_prompt": {"audio": 1},
    }
    startup = time.perf_counter()
    llm = LLM(**engine_config)
    startup_seconds = time.perf_counter() - startup
    write(
        args.out / "workers" / f"{args.model_label}-{args.family}.json",
        {
            "model": args.model_label,
            "family": args.family,
            "model_path": str(model_path),
            "backend": "vllm",
            "backend_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "engine_config": engine_config,
            "startup_seconds": startup_seconds,
        },
    )

    failures = 0
    for config in CONFIGS[args.family]:
        run = args.out / "runs" / config["name"]
        try:
            full_request = {
                "prompt": text,
                "multi_modal_data": {"audio": (audios[0], processor.feature_extractor.sampling_rate)},
            }
            full_cap = min(40000, 131072 - len(prompt_ids))
            full_result, full_ids, full_eos, full_seconds = generate(
                llm, full_request, make_params(config, full_cap)
            )
            if list(full_result.prompt_token_ids) != prompt_ids:
                raise RuntimeError("vLLM full-meeting prompt IDs changed")
            raw = processor.tokenizer.decode(full_ids, skip_special_tokens=True).strip()
            segments = [dataclasses.asdict(segment) for segment in parse_transcript(raw)]
            full_record = {
                **item,
                "model": args.model_label,
                "model_path": str(model_path),
                "backend": "vllm",
                "backend_version": vllm.__version__,
                "config": config,
                "status": "ok",
                "raw_text": raw,
                "segments": segments,
                "generated_ids": full_ids,
                "generated_tokens": len(full_ids),
                "ended_eos": full_eos,
                "truncated": not full_eos,
                "max_new_tokens": full_cap,
                "parse_empty": not segments,
                "finish_reason": full_result.outputs[0].finish_reason,
                "stop_reason": full_result.outputs[0].stop_reason,
                "e2e_seconds": full_seconds,
                "generation_seconds": full_seconds,
                "rtf": full_seconds / float(item["duration"]),
                "peak_allocated_gib": None,
                "peak_reserved_gib": None,
            }
            full_record["diagnostics"] = output_diagnostics(full_record)
            write(run / "predictions" / args.model_label / f"{TARGET_KEY}.json", full_record)
        except Exception as exc:
            failures += 1
            error = {
                "model": args.model_label,
                "config": config,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write(run / "errors" / f"{args.model_label}.json", error)
            print(json.dumps(error, ensure_ascii=True), flush=True)
        else:
            print(
                json.dumps(
                    {
                        "model": args.model_label,
                        "config": config["name"],
                        "full_tokens": full_record["generated_tokens"],
                        "full_eos": full_record["ended_eos"],
                        "full_longest_run": full_record["diagnostics"]["longest_identical_run"],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    write(
        args.out / "workers" / f"{args.model_label}-{args.family}-complete.json",
        {"complete": failures == 0, "failures": failures, "configs": len(CONFIGS[args.family])},
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
