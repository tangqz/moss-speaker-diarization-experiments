"""Isolate the CUDA audio-preprocessing autocast context; inference only."""
import contextlib
import copy
import dataclasses
import gc
import hashlib
import os
from pathlib import Path
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path('/work/qt28/moss')
SOURCE = ROOT / 'results/full-attention-v2-63421/source'
sys.path.insert(0, str(SOURCE))
from common import BASE, REPO, now, read, sha, write
from diagnostics import output_diagnostics
sys.path.insert(0, str(REPO))
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    run = ROOT / 'results' / f"full-attention-v2-diagnostic-{os.environ['SLURM_JOB_ID']}"
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, run / 'preprocessing_probe.py')
    write(run / 'status.json', dict(stage='input_preprocessing_probe', updated_utc=now(), training_updates=0))
    try:
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        device = torch.device('cuda', 0)
        processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
        source_eval = ROOT / 'results/full-attention-v2-63421/evaluations/sentinel-50'
        item = next(x for x in read(source_eval / 'inputs.json') if x['key'] == 'ami/dev/TS3004c')
        inputs, metadata = {}, {}
        for mode in ['outside_autocast', 'production_bf16_autocast']:
            context = torch.autocast('cuda', dtype=torch.bfloat16) if mode.startswith('production') else contextlib.nullcontext()
            with torch.inference_mode(), context:
                batch = prepare_inputs(processor, build_transcription_messages(item['audio'], item['prompt']),
                                       max_length=131072, device=device).to(device)
            inputs[mode] = batch
            metadata[mode] = {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=tensor_sha(v)) for k, v in batch.items()}
        a, b = inputs.values()
        differences = {}
        for key in a:
            if a[key].is_floating_point():
                delta = a[key].float() - b[key].float()
                differences[key] = dict(equal=torch.equal(a[key], b[key]),
                    max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                    changed_fraction=float((delta != 0).float().mean()))
            else:
                differences[key] = dict(equal=torch.equal(a[key], b[key]))
                assert differences[key]['equal'], key
        write(run / 'input_comparison.json', dict(metadata=metadata, differences=differences,
            input=item, audio_sha256=sha(Path(item['audio'])),
            processor_source_sha256=sha(REPO / 'moss_transcribe_diarize/inference_utils.py')))
        historical = read(source_eval / 'predictions/base/ami/dev/TS3004c.json')['generated_ids']
        recent_root = ROOT / 'results/full-attention-v2-diagnostic-63439'
        results = []
        for step in [0, 10]:
            model_path = BASE if step == 0 else ROOT / f'checkpoints/full-attention-v2-63421/checkpoint-{step}'
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
            warm = prepare_inputs(processor, build_transcription_messages(ROOT / 'data/samples/jfk.wav'), device=device).to(device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                model.generate(**warm, max_new_tokens=4, do_sample=False, num_beams=1, logits_to_keep=1, use_cache=True)
            del warm
            recent = read(recent_root / f'checkpoint-{step}/predictions/sft/ami/dev/TS3004c.json')['generated_ids']
            for mode, batch in inputs.items():
                config = copy.deepcopy(model.generation_config)
                for key, value in dict(max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True,
                        eos_token_id=151645, pad_token_id=151643, repetition_penalty=1., no_repeat_ngram_size=0).items():
                    setattr(config, key, value)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model.generate(**batch, generation_config=config, logits_to_keep=1)
                length = int(batch['attention_mask'].sum())
                ids = output[0, length:].tolist()
                raw = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
                record = dict(step=step, mode=mode, generated_tokens=len(ids), generated_ids=ids,
                    raw_text=raw, segments=[dataclasses.asdict(s) for s in parse_transcript(raw)],
                    recent_prefix_equal=ids == recent[:len(ids)], recent_common_prefix=common_prefix(ids, recent),
                    historical_base_prefix_equal=(ids == historical[:len(ids)]) if step == 0 else None,
                    historical_base_common_prefix=common_prefix(ids, historical) if step == 0 else None,
                    prefix_only=True, training_updates=0)
                record['diagnostics'] = output_diagnostics(record)
                write(run / f'prefix-{step}-{mode}.json', record)
                results.append({k: v for k, v in record.items() if k not in ['generated_ids', 'raw_text', 'segments']})
                write(run / 'status.json', dict(stage='prefix_probe', step=step, mode=mode, updated_utc=now(), training_updates=0))
                del output
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write(run / 'outcome.json', dict(status='completed', completed_utc=now(), training_updates=0,
            input_differences=differences, results=results,
            scope='512-token paired prefixes only; not full generation or a quality acceptance test.'))
        write(run / 'status.json', dict(stage='completed', updated_utc=now(), training_updates=0))
    except Exception as exc:
        write(run / 'status.json', dict(stage='failed', updated_utc=now(), error=str(exc)))
        raise


if __name__ == '__main__':
    main()
