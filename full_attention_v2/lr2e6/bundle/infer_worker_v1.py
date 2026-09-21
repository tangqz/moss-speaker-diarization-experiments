"""One independent GPU worker. Full meetings, identical decoding for both weights."""
import argparse
import copy
import dataclasses
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path('/work/qt28/moss')
sys.path.insert(0, str(ROOT / 'MOSS-Transcribe-Diarize'))
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages, prepare_inputs, ProgressStreamer,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.rank)
    device = torch.device('cuda', args.rank)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    items = [r for r in json.loads((args.run / 'inputs.json').read_text()) if r['rank'] == args.rank]
    base = ROOT / 'models/MOSS-Transcribe-Diarize'
    models = [('base', base), ('sft', ROOT / 'checkpoints/full-attention-ddp4-62994')]
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, local_files_only=True)
    status = dict(rank=args.rank, gpu=torch.cuda.get_device_name(device), pid=os.getpid())
    def progress(**changes):
        status.update(changes)
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.run / f'worker-{args.rank}-status.json', status)
        print(json.dumps(status), flush=True)
    failures = 0
    for split in ['dev', 'test']:
        for label, path in models:
            todo = []
            for item in items:
                if item['split'] != split:
                    continue
                dest = args.run / 'predictions' / label / (item['key'] + '.json')
                if dest.is_file() and json.loads(dest.read_text()).get('status') == 'ok':
                    continue
                todo.append((item, dest))
            if not todo:
                continue
            progress(stage='loading_model', model=label, split=split)
            tload = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
            torch.cuda.synchronize(device)
            load_seconds = time.perf_counter() - tload
            # Warm-up with a separate short sample. Never part of benchmark metrics.
            warm = prepare_inputs(processor, build_transcription_messages(
                ROOT / 'data/samples/jfk.wav'), device=device).to(device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                model.generate(**warm, max_new_tokens=4, do_sample=False, num_beams=1,
                    logits_to_keep=1, use_cache=True)
            del warm
            torch.cuda.empty_cache()
            progress(stage='model_ready', model_load_seconds=load_seconds)
            for item, dest in todo:
                inputs = outputs = generated = None
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                free_before, total = torch.cuda.mem_get_info(device)
                start = time.perf_counter()
                progress(stage='preprocessing', model=label, split=split, key=item['key'],
                         generated_tokens=0, audio_seconds=item['duration'])
                record = dict(item, model=label, model_path=str(path), rank=args.rank,
                    gpu=status['gpu'], model_load_seconds=load_seconds,
                    free_gpu_bytes_before=free_before, total_gpu_bytes=total,
                    decoding=dict(do_sample=False, num_beams=1, max_new_tokens_cap=65536,
                                  max_context=131072, attention='sdpa', dtype='bfloat16',
                                  logits_to_keep=1, use_cache=True),
                    started_utc=datetime.now(timezone.utc).isoformat())
                try:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        inputs = prepare_inputs(processor, build_transcription_messages(
                            item['audio'], item['prompt']), max_length=131072, device=device).to(device)
                    prompt_len = int(inputs['attention_mask'].sum().item())
                    assert prompt_len == item['prompt_len'], (prompt_len, item['prompt_len'])
                    max_new = min(65536, 131072 - prompt_len)
                    config = copy.deepcopy(model.generation_config)
                    config.do_sample = False
                    config.num_beams = 1
                    config.max_new_tokens = max_new
                    config.use_cache = True
                    # Use the original EOS/pad IDs explicitly for both models.
                    config.eos_token_id = processor.tokenizer.eos_token_id
                    config.pad_token_id = processor.tokenizer.pad_token_id
                    torch.cuda.synchronize(device)
                    gen_start = time.perf_counter()
                    first_token = []
                    def token_event(n):
                        if not first_token:
                            first_token.append(time.perf_counter())
                        if n == 1 or n % 256 == 0:
                            progress(stage='generating', generated_tokens=n,
                                     elapsed_seconds=round(time.perf_counter() - start, 2))
                    streamer = ProgressStreamer(token_event)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model.generate(**inputs, generation_config=config,
                            logits_to_keep=1, streamer=streamer)
                    torch.cuda.synchronize(device)
                    generation_seconds = time.perf_counter() - gen_start
                    generated = outputs[0, prompt_len:]
                    ids = generated.tolist()
                    raw = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    segments = [dataclasses.asdict(s) for s in parse_transcript(raw)]
                    eos_ids = config.eos_token_id
                    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                    ended_eos = bool(ids) and ids[-1] in eos_ids
                    e2e = time.perf_counter() - start
                    record.update(status='ok', raw_text=raw, segments=segments,
                        generated_ids=ids, generated_tokens=len(ids), ended_eos=ended_eos,
                        truncated=not ended_eos, max_new_tokens=max_new,
                        parse_empty=not segments, e2e_seconds=e2e,
                        preprocessing_seconds=gen_start-start, generation_seconds=generation_seconds,
                        first_token_seconds=(first_token[0]-gen_start if first_token else None),
                        rtf=e2e/item['duration'], generated_tokens_per_second=len(ids)/generation_seconds,
                        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
                except Exception as exc:
                    failures += 1
                    record.update(status='error', error_type=type(exc).__name__, error=str(exc),
                                  traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start)
                    traceback.print_exc()
                record['finished_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(dest, record)
                progress(stage='record_saved', key=item['key'], outcome=record['status'])
                del inputs, outputs, generated
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    progress(stage='done', failures=failures)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
