"""Run one rank of the gated Base/Step-150 full Test re-evaluation."""

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


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-label", choices=("base", "sft"), required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[args.rank]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import torch
    import vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(REPO))
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, process_audio_info
    sys.path.insert(0, str(TRAIN_RUN / "source/evaluation"))
    from diagnostics import output_diagnostics

    config = read(args.run / "selected_config.json")
    inputs = [item for item in read(args.run / "inputs.json") if int(item["rank"]) == args.rank]
    model_path = BASE if args.model_label == "base" else TRAIN_RUN / "vllm_exports/checkpoint-150"
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    engine_config = {
        "model": str(model_path),
        "tokenizer": str(BASE),
        "trust_remote_code": True,
        "model_impl": "vllm",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 131072,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max(8192, ((max(item["prompt_len"] for item in inputs) + 8191) // 8192) * 8192),
        "gpu_memory_utilization": 0.45,
        "enable_prefix_caching": False,
        "generation_config": "vllm",
        "seed": config["seed"],
        "limit_mm_per_prompt": {"audio": 1},
    }
    start = time.perf_counter()
    llm = LLM(**engine_config)
    startup_seconds = time.perf_counter() - start
    write(
        args.run / "workers" / f"{args.model_label}-rank-{args.rank}-engine.json",
        {
            "model": args.model_label,
            "rank": args.rank,
            "model_path": str(model_path),
            "backend": "vllm",
            "backend_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "startup_seconds": startup_seconds,
            "engine_config": engine_config,
        },
    )
    failures = 0
    for item in inputs:
        destination = args.run / "predictions" / args.model_label / f'{item["key"]}.json'
        record = {
            **item,
            "model": args.model_label,
            "model_path": str(model_path),
            "backend": "vllm",
            "backend_version": vllm.__version__,
            "decoding_config": config,
            "engine_config": engine_config,
            "engine_startup_seconds": startup_seconds,
        }
        try:
            e2e_start = time.perf_counter()
            messages = build_transcription_messages(item["audio"], item["prompt"])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)
            cap = min(65536, 131072 - int(item["prompt_len"]))
            params = SamplingParams(
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                min_p=config["min_p"],
                repetition_penalty=config["repetition_penalty"],
                presence_penalty=config["presence_penalty"],
                frequency_penalty=config["frequency_penalty"],
                max_tokens=cap,
                stop_token_ids=[151645],
                ignore_eos=False,
                skip_special_tokens=True,
                seed=config["seed"],
            )
            request_start = time.perf_counter()
            result = llm.generate(
                [{"prompt": text, "multi_modal_data": {"audio": (audios[0], processor.feature_extractor.sampling_rate)}}],
                params,
                use_tqdm=True,
            )[0]
            generation_seconds = time.perf_counter() - request_start
            if len(result.prompt_token_ids) != int(item["prompt_len"]):
                raise RuntimeError("expanded prompt length changed")
            output = result.outputs[0]
            ids = [int(value) for value in output.token_ids]
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            segments = [dataclasses.asdict(segment) for segment in parse_transcript(raw)]
            e2e_seconds = time.perf_counter() - e2e_start
            eos = bool(ids and ids[-1] == 151645) or output.stop_reason == 151645
            record.update(
                status="ok",
                raw_text=raw,
                segments=segments,
                generated_ids=ids,
                generated_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                generated_tokens=len(ids),
                ended_eos=eos,
                truncated=not eos,
                max_new_tokens=cap,
                parse_empty=not segments,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
                e2e_seconds=e2e_seconds,
                generation_seconds=generation_seconds,
                rtf=e2e_seconds / float(item["duration"]),
                peak_allocated_gib=None,
                peak_reserved_gib=None,
            )
            record["diagnostics"] = output_diagnostics(record)
        except Exception as exc:
            failures += 1
            record.update(
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        write(destination, record)
        print(
            json.dumps(
                {"key": item["key"], "model": args.model_label, "rank": args.rank, "status": record["status"]},
                ensure_ascii=True,
            ),
            flush=True,
        )
        if failures:
            break
    write(
        args.run / "workers" / f"{args.model_label}-rank-{args.rank}-complete.json",
        {"complete": failures == 0, "failures": failures, "items": len(inputs)},
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
