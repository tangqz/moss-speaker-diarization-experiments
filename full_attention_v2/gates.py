"""Real four-GPU, longest-meeting, accumulation, serialization and resume gates."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common import ROOT, BASE, TRAIN, HERE, REPO, HORIZON, read, write, sha, emit


def execute(command, log, env=None):
    emit('gate_command', log=str(log), command=command)
    with log.open('w') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def freeze(run):
    protocol = read(HERE/'design_protocol.json')
    for name, expected in read(HERE/'source_manifest.json').items():
        assert sha(HERE/name) == expected, name
    expected = {
        BASE/'model-00000-of-00001.safetensors': protocol['initialization']['weights_sha256'],
        TRAIN: protocol['data']['sha256'],
        REPO/'moss_transcribe_diarize/modeling_moss_transcribe_diarize.py': 'e5fed522f246a315036a20d545e9f804ea01d81f5b0d246ba833ea635a1f4056',
    }
    hashes = {}
    for path, value in expected.items():
        actual = sha(path)
        assert actual == value, (str(path), actual)
        hashes[str(path)] = actual
    for name in ['finetune.py','moss_transcribe_diarize/processing_moss_transcribe_diarize.py',
                 'moss_transcribe_diarize/transcript_parser.py']:
        path = REPO/name
        hashes[str(path)] = sha(path)
        destination = run/'frozen_repository'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    versions = {p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','safetensors']}
    assert versions['transformers'] == '5.17.0', versions
    assert versions['torch'].startswith('2.14.0'), versions
    free = shutil.disk_usage(run).free/2**30
    assert free >= 110, free
    import torch
    assert torch.cuda.device_count() == 4
    devices = [dict(index=i, name=torch.cuda.get_device_name(i), free_gib=torch.cuda.mem_get_info(i)[0]/2**30,
                    total_gib=torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(4)]
    assert all(d['free_gib'] >= 32 for d in devices), devices
    record = dict(hashes=hashes, versions=versions, free_disk_gib=free, devices=devices,
                  source_manifest=read(HERE/'source_manifest.json'))
    write(run/'environment.json', record)
    emit('environment_pass', **record)


def records(folder, step):
    rows = []
    for file in folder.glob('train-rank-*.jsonl'):
        rows.extend(json.loads(s) for s in file.read_text().splitlines() if s.strip())
    return sorted([r for r in rows if r['step']==step], key=lambda x:x['rank'])


def loss_and_order(rows):
    micros = [m for r in rows for m in r['microbatches']]
    tokens = sum(m['local_supervised_tokens'] for m in micros)
    assert len(micros) == 4
    assert all(m['global_supervised_tokens']==tokens for m in micros)
    return sum(m['loss_numerator'] for m in micros)/tokens, sorted(i for m in micros for i in m['indices']), tokens


def tensors(path):
    import safetensors
    files = sorted(path.glob('model*.safetensors'))
    assert files
    for file in files:
        with safetensors.safe_open(file, framework='pt', device='cpu') as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def validate(run):
    import torch
    ddp = run/'gates/ddp'
    single = run/'gates/accumulation'
    resumed = run/'gates/resumed'
    a, b = [loss_and_order(records(p, 1)) for p in [ddp, single]]
    assert a[1:] == b[1:]
    assert abs(a[0]-b[0]) < 2e-4, (a,b)
    grads = [torch.load(p/'gradients-step-1.pt', map_location='cpu', weights_only=True) for p in [ddp,single]]
    assert grads[0].keys() == grads[1].keys()
    error = reference = 0.
    per_tensor = []
    for name in grads[0]:
        x,y = grads[0][name], grads[1][name]
        e, d = float((x-y).double().square().sum()), float(y.double().square().sum())
        error += e; reference += d
        relative = (e/max(d,1e-30))**.5
        per_tensor.append(dict(name=name, relative_l2=relative))
        assert relative < .02, (name,relative)
    relative = (error/reference)**.5
    assert relative < .005, relative
    del grads
    continuous_rows, resumed_rows = records(ddp, 2), records(resumed,2)
    assert len(continuous_rows) == len(resumed_rows) == 4
    c,d = loss_and_order(continuous_rows),loss_and_order(resumed_rows)
    assert c[1:] == d[1:]
    assert abs(c[0]-d[0]) < 1e-6, (c,d)
    assert max(m['sequence_tokens'] for r in continuous_rows for m in r['microbatches']) == 116527
    for x,y in zip(continuous_rows,resumed_rows):
        assert x['microbatches'] == y['microbatches'], (x['rank'],'RNG/data/loss continuation differs')
        assert x['lr'] == y['lr'] == 1e-5*(1-1/HORIZON)
    # Check every live parameter against its serialized value, not just samples.
    for folder in [ddp/'checkpoint-1', ddp/'checkpoint-2', resumed/'checkpoint-2']:
        live = read(folder/'live_parameter_hashes.json')
        found = set()
        for name,value in tensors(folder):
            assert value.dtype == torch.float32
            if name in live:
                assert hashlib.sha256(value.numpy().tobytes()).hexdigest() == live[name], name
                found.add(name)
        assert found == set(live), sorted(set(live)-found)
    left = read(ddp/'checkpoint-2/live_parameter_hashes.json')
    right = read(resumed/'checkpoint-2/live_parameter_hashes.json')
    assert left == right, 'Resumed weights are not bitwise identical to uninterrupted weights'
    # Optimizer files may serialize storage IDs differently; compare their tensors.
    optimizers = [torch.load(p/'checkpoint-2/optimizer.pt', map_location='cpu', weights_only=True)
                  for p in [ddp,resumed]]
    assert optimizers[0]['param_groups'] == optimizers[1]['param_groups']
    assert optimizers[0]['state'].keys() == optimizers[1]['state'].keys()
    for key in optimizers[0]['state']:
        for field,x in optimizers[0]['state'][key].items():
            y = optimizers[1]['state'][key][field]
            assert torch.equal(x,y) if torch.is_tensor(x) else x==y
            if field in ['exp_avg','exp_avg_sq']:
                assert x.dtype == torch.float32
    del optimizers
    gc.collect()
    evidence = dict(loss_ddp=a[0], loss_accumulation=b[0], global_tokens=a[2],
        gradient_relative_l2=relative, gradient_per_tensor=per_tensor,
        longest_context=116527, all_parameters_serialized_exactly=True,
        resume_parameters_bitwise_equal=True, resume_optimizer_bitwise_equal=True,
        resume_data_rng_and_losses_equal=True, scheduler_horizon=HORIZON,
        longest_batch_loss=c[0], ddp_peak_gib=[r['peak_allocated_gib'] for r in continuous_rows],
        ddp_process_rss_gib=[r['peak_process_rss_gib'] for r in continuous_rows])
    write(run/'gate_numerical_results.json', evidence)
    emit('numerical_gates_pass', **{k:v for k,v in evidence.items() if k!='gradient_per_tensor'})


def inference_reload(run, checkpoint=None):
    import torch
    sys.path.insert(0,str(REPO))
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    path = checkpoint if checkpoint is not None else run/'gates/ddp/checkpoint-2'
    processor = AutoProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    item = next(r for r in read(HERE/'dev_inputs.json') if r['key']=='ami/dev/IS1008a')
    inputs = prepare_inputs(processor,build_transcription_messages(item['audio'],item['prompt']),
                            max_length=131072,device=torch.device('cuda',0)).to('cuda:0')
    outputs = []
    for label,dtype in [('cast_fp32_saved_live_equivalent',torch.float32),('direct_bf16_reload',torch.bfloat16)]:
        model = AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,local_files_only=True,
            dtype=dtype,attn_implementation='sdpa').to(dtype=torch.bfloat16,device='cuda:0').eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out = model(**inputs,logits_to_keep=1,use_cache=False).logits.float().cpu()
        outputs.append(out)
        del model
        gc.collect();torch.cuda.empty_cache()
    assert torch.equal(*outputs), 'FP32 cast vs direct BF16 reload logits differ'
    evidence = dict(key=item['key'],prompt_len=inputs['input_ids'].shape[-1],
        logits_max_abs=float((outputs[0]-outputs[1]).abs().max()), top1_same=True,
        live_serialization_exact_verified_separately=True)
    write(run/'gate_inference_reload.json',evidence)
    emit('inference_reload_pass',**evidence)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--validate-only',action='store_true')
    a=ap.parse_args()
    a.run.mkdir(parents=True,exist_ok=True)
    log=a.run/'logs';log.mkdir(exist_ok=True)
    if not a.validate_only:
        freeze(a.run)
        data=read(HERE/'preflight.json')['samples']
        shortest=[r['index'] for r in sorted(data,key=lambda r:(r['tokens'],r['index']))[:7]]
        indices=shortest[:4]+[1]+shortest[4:7]
        assert len(set(indices))==8
        write(a.run/'gate_indices.json',indices)
        ddp=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(HERE/'train.py')]
        common=['--gate-indices',json.dumps(indices)]
        execute(ddp+['--output',str(a.run/'gates/ddp'),'--stop','2']+common,log/'gate-ddp.log')
        env=dict(os.environ)
        env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')[0]
        execute([sys.executable,str(HERE/'train.py'),'--output',str(a.run/'gates/accumulation'),
                 '--stop','1','--accumulation','4']+common,log/'gate-accumulation.log',env)
        execute(ddp+['--output',str(a.run/'gates/resumed'),'--stop','2',
                     '--resume',str(a.run/'gates/ddp/checkpoint-1')]+common,log/'gate-resume.log')
    validate(a.run)
    inference_reload(a.run)
    write(a.run/'entry_gates_passed.json',dict(passed=True, numerical='gate_numerical_results.json',
        inference='gate_inference_reload.json', source_manifest=read(HERE/'source_manifest.json')))


if __name__=='__main__':
    main()
