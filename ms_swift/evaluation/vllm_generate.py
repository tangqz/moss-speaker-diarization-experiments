"""Official vLLM MOSS inference with unchanged prompts, parser and scoring schema."""
import argparse
import dataclasses
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

from common import BASE, REPO, read, write, emit, now
from diagnostics import output_diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--model-label', choices=['base', 'sft'], default='sft')
    ap.add_argument('--smoke-tokens', type=int, default=0)
    a = ap.parse_args()
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible[a.rank]
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    sys.path.insert(0, str(REPO))
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, process_audio_info

    torch.set_num_threads(4)
    all_items = read(a.run/'inputs.json')
    items = [r for r in all_items if r['rank'] == a.rank]
    if not items:
        return
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    engine_config = dict(model=str(a.checkpoint), tokenizer=str(BASE),
        trust_remote_code=True, model_impl='vllm', dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=131072, max_num_seqs=1,
        max_num_batched_tokens=max(8192, ((max(r['prompt_len'] for r in all_items)+8191)//8192)*8192),
        gpu_memory_utilization=.55, enable_prefix_caching=False,
        generation_config='vllm', seed=0,
        limit_mm_per_prompt={'audio': 1})
    worker_start = time.perf_counter()
    emit('vllm_loading', rank=a.rank, model=a.model_label, checkpoint=str(a.checkpoint))
    llm = LLM(**engine_config)
    startup = time.perf_counter()-worker_start
    write(a.run/f'{a.model_label}-worker-{a.rank}-engine.json', dict(
        config=engine_config, version=vllm.__version__, torch=torch.__version__,
        startup_seconds=startup, memory_semantics='per-request worker allocator peak unavailable; see native initialization logs', utc=now()))
    failures = 0
    for item in items:
        dest = a.run/'predictions'/a.model_label/f"{item['key']}.json"
        start = time.perf_counter()
        record = dict(item, model=a.model_label, model_path=str(a.checkpoint), rank=a.rank,
            gpu=torch.cuda.get_device_name(0), backend='vllm', backend_version=vllm.__version__,
            engine_config=engine_config, started_utc=now(), engine_startup_seconds=startup,
            model_manifest=read(a.run/'model_manifest.json'),
            decoding=dict(temperature=0, repetition_penalty=1, presence_penalty=0,
                frequency_penalty=0, max_new_tokens_cap=65536, max_context=131072,
                dtype='bfloat16', eos_token_id=151645), engineering_smoke=bool(a.smoke_tokens))
        write(a.run/f'{a.model_label}-worker-{a.rank}-status.json',dict(
            stage='generating', key=item['key'], utc=now()))
        try:
            messages = build_transcription_messages(item['audio'], item['prompt'])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)
            assert len(audios) == 1
            expected_ids = None
            verification_seconds = 0.
            if a.smoke_tokens:
                verification_start=time.perf_counter()
                expected_ids = processor(text=text, audio=audios, return_tensors='pt')['input_ids'][0].tolist()
                verification_seconds=time.perf_counter()-verification_start
                start+=verification_seconds
            cap = min(65536, 131072-item['prompt_len'])
            if a.smoke_tokens:
                cap = min(cap, a.smoke_tokens)
            params = SamplingParams(temperature=0, top_p=1, top_k=-1,
                repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
                max_tokens=cap, stop_token_ids=[151645], ignore_eos=False,
                skip_special_tokens=True, seed=0)
            gen_start = time.perf_counter()
            result = llm.generate([dict(prompt=text, multi_modal_data={
                'audio': (audios[0], processor.feature_extractor.sampling_rate)})],
                params, use_tqdm=True)[0]
            generation_seconds = time.perf_counter()-gen_start
            assert len(result.prompt_token_ids) == item['prompt_len'], (
                'expanded_prompt_length_changed', len(result.prompt_token_ids), item['prompt_len'])
            if expected_ids is not None:
                assert result.prompt_token_ids == expected_ids, 'vLLM prompt tokens differ from official processor'
            output = result.outputs[0]
            ids = list(output.token_ids)
            raw = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            segments = [dataclasses.asdict(s) for s in parse_transcript(raw)]
            eos = bool(ids and ids[-1] == 151645) or output.stop_reason == 151645
            e2e = time.perf_counter()-start
            record.update(status='ok', raw_text=raw, segments=segments, generated_ids=ids,
                generated_tokens=len(ids), generated_absolute_position_start=item['prompt_len'],
                prompt_ids_sha256=hashlib.sha256(str(result.prompt_token_ids).encode()).hexdigest(),
                official_prompt_ids_checked=expected_ids is not None,
                input_verification_seconds=verification_seconds,
                ended_eos=eos, truncated=not eos, max_new_tokens=cap, parse_empty=not segments,
                finish_reason=output.finish_reason, stop_reason=output.stop_reason,
                e2e_seconds=e2e, preprocessing_seconds=gen_start-start,
                generation_seconds=generation_seconds, first_token_seconds=None,
                rtf=e2e/item['duration'], generated_tokens_per_second=len(ids)/generation_seconds,
                peak_allocated_gib=None, peak_reserved_gib=None,
                memory_measurement='not available through native safe worker RPC')
            record['diagnostics'] = output_diagnostics(record)
        except Exception as exc:
            failures += 1
            record.update(status='error', error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
            traceback.print_exc()
        record['finished_utc'] = now()
        write(dest, record)
        emit('vllm_record_saved', key=item['key'], status=record['status'], rank=a.rank)
        if failures:
            break
    write(a.run/f'{a.model_label}-worker-{a.rank}-status.json',dict(stage='done',
        failures=failures, total_seconds=time.perf_counter()-worker_start, utc=now()))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
