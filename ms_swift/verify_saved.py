"""Read-only checkpoint diagnostics and production-environment loading smoke."""
import argparse
import json
import math
from pathlib import Path
import sys
import torch
from safetensors import safe_open


def update_diagnostics(run):
    base = Path('/work/qt28/moss/models/MOSS-Transcribe-Diarize')
    index = {}
    for path in base.glob('*.safetensors'):
        with safe_open(path, framework='pt') as handle:
            index.update({key: path for key in handle.keys()})
    groups = {}
    single = run/'single/checkpoint-1/model.safetensors'
    parallel = run/'sp4/checkpoint-1/model.safetensors'
    with safe_open(single, framework='pt') as a, safe_open(parallel, framework='pt') as b:
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            with safe_open(index[key], framework='pt') as base_file:
                original = base_file.get_tensor(key).float()
            x = a.get_tensor(key).float()-original
            y = b.get_tensor(key).float()-original
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            group = next((name for name in ['whisper_encoder', 'vq_adaptor', 'language_model', 'lm_head']
                          if name in key), 'other')
            values = groups.setdefault(group, dict(single_sq=0., sp4_sq=0., diff_sq=0., dot=0., count=0))
            values['single_sq'] += float(x.square().sum())
            values['sp4_sq'] += float(y.square().sum())
            values['diff_sq'] += float((x-y).square().sum())
            values['dot'] += float((x*y).sum())
            values['count'] += x.numel()
    for values in groups.values():
        values['single_update_norm'] = math.sqrt(values['single_sq'])
        values['sp4_update_norm'] = math.sqrt(values['sp4_sq'])
        values['cosine'] = values['dot']/max(math.sqrt(values['single_sq']*values['sp4_sq']), 1e-30)
        values['relative_update_difference'] = math.sqrt(values['diff_sq']/max(values['single_sq'], 1e-30))
    (run/'update_comparison.json').write_text(json.dumps(groups, indent=2))
    print(json.dumps(groups), flush=True)


def production_load(run):
    from transformers import AutoModelForCausalLM, AutoProcessor
    sys.path.insert(0, '/work/qt28/moss/MOSS-Transcribe-Diarize')
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
    base = '/work/qt28/moss/models/MOSS-Transcribe-Diarize'
    checkpoint = run/'sp4/checkpoint-1'
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        batch = prepare_inputs(processor, build_transcription_messages(
            '/work/qt28/moss/data/samples/jfk.wav'), device=torch.device('cuda:0'))
        output = model.generate(**batch, max_new_tokens=32, do_sample=False,
                                num_beams=1, logits_to_keep=1, use_cache=True)
    ids = output[0, batch['input_ids'].shape[1]:].tolist()
    assert len(ids) > 0
    receipt = dict(checkpoint=str(checkpoint), production_python=sys.executable,
        generated_ids=ids, decoded=processor.tokenizer.decode(ids, skip_special_tokens=True),
        model_class=type(model).__name__, loading_and_generation_passed=True,
        qualification='Short format/loading smoke, not full-dev accuracy')
    (run/'production_loading_check.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--mode', choices=['updates', 'production'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    (update_diagnostics if args.mode == 'updates' else production_load)(args.run)
