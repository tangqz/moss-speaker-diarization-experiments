"""Matched vLLM Full/R-SWA quality evaluation with output-only penalty 1.02."""
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
from input_receipt import receipt as input_receipt
from protocol import ATTENTION_BACKEND


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--model-label', choices=['base', 'sft'], default='sft')
    ap.add_argument('--smoke-tokens', type=int, default=0)
    ap.add_argument('--window',required=True)
    ap.add_argument('--allow-inference-ablation',action='store_true')
    a = ap.parse_args()
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible[a.rank]
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    os.environ['MOSS_KV_AUDIT_PATH']=str(a.run/f'{a.model_label}-worker-{a.rank}-kv.jsonl')
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from moss_vllm_rswa import overrides
    sys.path.insert(0, str(REPO))
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, process_audio_info

    torch.set_num_threads(4)
    all_items = read(a.run/'inputs.json')
    items = [r for r in all_items if r['rank'] == a.rank]
    if not items:
        return
    processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    window=None if a.window=='full' else int(a.window)
    engine_config = dict(model=str(a.checkpoint), tokenizer=str(BASE),
        trust_remote_code=True, model_impl='vllm', dtype='bfloat16', tensor_parallel_size=1,
        max_model_len=131072, max_num_seqs=1,
        max_num_batched_tokens=max(8192, ((max(r['prompt_len'] for r in all_items)+8191)//8192)*8192),
        gpu_memory_utilization=.55, enable_prefix_caching=False,mm_processor_cache_gb=0,
        disable_log_stats=False,
        generation_config='vllm', seed=0,
        limit_mm_per_prompt={'audio': 1},
        attention_config={'backend':ATTENTION_BACKEND},
        worker_extension_cls='moss_vllm_rswa.WorkerExtension')
    worker_start = time.perf_counter()
    emit('vllm_loading', rank=a.rank, model=a.model_label, checkpoint=str(a.checkpoint))
    llm = LLM(**engine_config,hf_overrides=overrides(window,a.allow_inference_ablation))
    engine_config.update(rswa_window=window,penalty_scope='all_generated_tokens_only',model_runner='v1',flashinfer_sampler=False)
    receipt=llm.collective_rpc('rswa_runtime_receipt')
    write(a.run/f'{a.model_label}-worker-{a.rank}-attention.json',receipt)
    startup = time.perf_counter()-worker_start
    write(a.run/f'{a.model_label}-worker-{a.rank}-engine.json', dict(
        config=engine_config, version=vllm.__version__, torch=torch.__version__,
        startup_seconds=startup, memory_semantics='per-request worker allocator peak includes preallocated vLLM KV pool; active blocks recorded separately', utc=now()))
    failures = 0
    for item in items:
        dest = a.run/'predictions'/a.model_label/f"{item['key']}.json"
        start = time.perf_counter()
        record = dict(item, model=a.model_label, model_path=str(a.checkpoint), rank=a.rank,
            gpu=torch.cuda.get_device_name(0), backend='vllm', backend_version=vllm.__version__,
            engine_config=engine_config, started_utc=now(), engine_startup_seconds=startup,
            model_manifest=read(a.run/'model_manifest.json'),
            decoding=dict(temperature=0, repetition_penalty=1.02, penalty_scope='all_generated_tokens_only', presence_penalty=0,
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
            verification_start=time.perf_counter()
            official=processor(text=text,audio=audios,return_tensors='pt')
            expected_ids=official['input_ids'][0].tolist()
            write(a.run/'input_receipts'/f"{item['key']}.json",dict(
                input_receipt(processor,expected_ids,official['audio_feature_lengths'],official['audio_chunk_mapping']),
                key=item['key'],
                resampled_audio_samples=len(audios[0]),
                audio_sample_rate=processor.feature_extractor.sampling_rate,
                processor_config_sha256=hashlib.sha256((BASE/'processor_config.json').read_bytes()).hexdigest(),
                time_marker_every_seconds=5,tokens_per_second=12.5,use_time_markers=True))
            verification_seconds=time.perf_counter()-verification_start
            start+=verification_seconds
            del official
            cap = min(65536, 131072-item['prompt_len'])
            if a.smoke_tokens:
                cap = min(cap, a.smoke_tokens)
            params = SamplingParams(temperature=0, top_p=1, top_k=-1,
                repetition_penalty=1.02, presence_penalty=0, frequency_penalty=0,
                max_tokens=cap, stop_token_ids=[151645], ignore_eos=False,
                skip_special_tokens=True, seed=0)
            request=dict(prompt=text,multi_modal_data={'audio':(audios[0],processor.feature_extractor.sampling_rate)})
            warm_start=time.perf_counter()
            # Warm prefill and one decode step. This discarded engineering
            # request has no quality score; the real request above honors EOS.
            warm_params=SamplingParams(temperature=0,repetition_penalty=1.02,max_tokens=2,ignore_eos=True,seed=0)
            llm.generate([request],warm_params,use_tqdm=False)
            llm.llm_engine.reset_encoder_cache()
            assert llm.collective_rpc('rswa_encoder_cache_size')==[0]
            warm_seconds=time.perf_counter()-warm_start
            start+=warm_seconds
            llm.collective_rpc('rswa_reset_memory_stats')
            gen_start = time.perf_counter()
            result = llm.generate([request],params, use_tqdm=True)[0]
            generation_seconds = time.perf_counter()-gen_start
            memory=llm.collective_rpc('rswa_memory_stats')[0]
            assert len(result.prompt_token_ids) == item['prompt_len'], (
                'expanded_prompt_length_changed', len(result.prompt_token_ids), item['prompt_len'])
            if expected_ids is not None:
                assert result.prompt_token_ids == expected_ids, 'vLLM prompt tokens differ from official processor'
            output = result.outputs[0]
            timing=result.metrics
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
                generation_seconds=generation_seconds,
                first_token_seconds=timing.first_token_latency if timing else None,
                prefill_seconds=timing.first_token_ts-timing.scheduled_ts if timing else None,
                decode_seconds=timing.last_token_ts-timing.first_token_ts if timing else None,
                engine_request_timing=dataclasses.asdict(timing) if timing else None,
                warmup_seconds=warm_seconds,warmup_excluded=True,encoder_cache_cleared_after_warmup=True,
                rtf=e2e/item['duration'], generated_tokens_per_second=len(ids)/generation_seconds,
                peak_allocated_gib=memory['peak_allocated_bytes']/2**30,
                peak_reserved_gib=memory['peak_reserved_bytes']/2**30,
                memory_measurement=memory['interpretation'],worker_memory=memory)
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
