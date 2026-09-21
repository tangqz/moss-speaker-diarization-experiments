"""Bounded, isolated diagnostics for the next MOSS SFT; never edits old results."""
import argparse
import collections
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path('/work/qt28/moss')
sys.path[:0] = [str(ROOT/'MOSS-Transcribe-Diarize'), str(ROOT/'dkucc/memory_sft')]
from finetune import ConversationDataset, DataCollator
from memory_sft import projected_causal_loss, install_memory_forward, install_bounded_audio_encoder
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, LogitsProcessor
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor

BASE = ROOT/'models/MOSS-Transcribe-Diarize'
OLD = ROOT/'checkpoints/full-attention-ddp4-62994'
EVAL = ROOT/'results/eval-62994-63363'
REPLAY_KEYS = ['alimeeting/dev/R8001_M8004', 'alimeeting/dev/R8003_M8001']
VAL_KEYS = ['ami/dev/IS1008a', 'alimeeting/dev/R8009_M8018']


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), ensure_ascii=True), flush=True)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def load(path, dtype=torch.bfloat16):
    return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
        local_files_only=True, dtype=dtype, attn_implementation='sdpa').to(DEVICE).eval()


def move(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def amp():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def stats(logits, token, eos):
    z = logits.reshape(-1).float()
    p = z.softmax(-1)
    top = z.topk(5)
    other = z.clone()
    other[token] = -torch.inf
    return dict(token=int(token), probability=p[token].item(),
        rank=int((z > z[token]).sum())+1, margin=(z[token]-other.max()).item(),
        entropy_nats=(-(p*p.clamp_min(1e-30).log()).sum()).item(),
        eos_probability=p[eos].item(), top_ids=top.indices.tolist(),
        top_probabilities=p[top.indices].tolist())


def compare(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return dict(max_abs=(a-b).abs().max().item(), rms=(a-b).square().mean().sqrt().item(),
        top1_same=bool(a.argmax() == b.argmax()), top1_a=int(a.argmax()), top1_b=int(b.argmax()),
        kl_a_b=(a.softmax(-1)*(a.log_softmax(-1)-b.log_softmax(-1))).sum().item())


def longest_run(ids):
    best = (0, 0, None)
    i = 0
    while i < len(ids):
        j = i+1
        while j < len(ids) and ids[j] == ids[i]:
            j += 1
        if j-i > best[1]:
            best = (i, j-i, ids[i])
        i = j
    return best


def with_history(prompt, history):
    batch = dict(prompt)
    batch['input_ids'] = torch.cat([prompt['input_ids'], torch.tensor([history], device=DEVICE, dtype=torch.long)], -1)
    batch['attention_mask'] = torch.ones_like(batch['input_ids'])
    return batch


def contracts(run, processor):
    # Independent position-by-position reference, with no label-shift helper.
    labels = torch.tensor([[-100, -100, 2, 3, 4, 5]])
    hidden = torch.zeros(1, 6, 7)
    for j in range(2, 6):
        hidden[0, j-1, labels[0, j]] = 12
    hidden.requires_grad_()
    weight = torch.eye(7, requires_grad=True)
    actual = projected_causal_loss(hidden, weight, labels, chunk_size=2)
    manual = sum(-F.log_softmax(hidden[0,j-1] @ weight.T, -1)[labels[0,j]] for j in range(2,6))/4
    g1 = torch.autograd.grad(actual, (hidden, weight), retain_graph=True)
    g2 = torch.autograd.grad(manual, (hidden, weight))
    assert abs(actual.item()-manual.item()) < 1e-6
    assert all((x-y).abs().max() < 1e-6 for x,y in zip(g1,g2))
    same_position = torch.stack([-F.log_softmax(hidden[0,j] @ weight.T,-1)[labels[0,j]] for j in range(2,6)]).mean()
    assert same_position > actual+1
    result = dict(toy_alignment=dict(loss=actual.item(), independent_loss=manual.item(),
        incorrect_same_position_loss=same_position.item(), gradients_match=True))

    # Compare full prompt+target encoding with independently encoded boundary.
    train = ConversationDataset(str(ROOT/'data/moss_jsonl/train_mix.jsonl'))
    boundary_bad = []
    for i, row in enumerate(train):
        prompt = processor.apply_chat_template(build_transcription_messages(row['audio'], row['prompt']),
            tokenize=False, add_generation_prompt=True)
        target = row['target']+processor.tokenizer.eos_token
        a = processor.tokenizer.encode(prompt, add_special_tokens=False)+processor.tokenizer.encode(target, add_special_tokens=False)
        b = processor.tokenizer.encode(prompt+target, add_special_tokens=False)
        if a != b:
            boundary_bad.append(i)
    result['all_training_text_boundaries'] = dict(records=len(train), bad_indices=boundary_bad)
    assert not boundary_bad, boundary_bad

    # Visibility contracts that can be reused for mask SFT. This tests a reference,
    # not a production sparse backend or cache eviction implementation.
    torch.manual_seed(0)
    q,k,v = [torch.randn(1,2,19,8,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    pos = torch.arange(19)
    causal = pos[None,:] <= pos[:,None]
    def rswa(window):
        return causal & ((pos[None,:] < 7) | (pos[None,:] >= pos[:,None]-window+1))
    full = F.scaled_dot_product_attention(q,k,v,attn_mask=causal)
    wide = F.scaled_dot_product_attention(q,k,v,attn_mask=rswa(19))
    assert torch.equal(full,wide)
    narrow = rswa(3)
    assert not narrow.triu(1).any() and narrow[7:,:7].all()
    result['reference_mask_contract'] = dict(full_equals_wide_rswa=True, reference_visible=True, future_hidden=True)
    write(run/'contracts.json',dict(status='running',**result))

    model = load(BASE)
    sample = ConversationDataset(str(ROOT/'data/smoke_train.jsonl'))[0]
    batch = move(DataCollator(processor,8192)([sample]))
    ids, lab = batch['input_ids'], batch['labels']
    first = int(lab[0].ne(-100).nonzero()[0])
    result['real_supervision_boundary'] = dict(sequence_tokens=ids.shape[-1], first_target_index=first,
        predicting_query_index=first-1, previous_input_id=int(ids[0,first-1]), target_id=int(lab[0,first]),
        target_text_matches=processor.tokenizer.decode(lab[0,lab[0].ne(-100)].tolist(),skip_special_tokens=False)==sample['target']+processor.tokenizer.eos_token,
        eos_supervised=int(lab[0,-1])==processor.tokenizer.eos_token_id)
    assert result['real_supervision_boundary']['target_text_matches']
    with torch.inference_mode(), amp():
        out = model(**batch, use_cache=False)
        logits = out.logits
        j = lab[0].ne(-100).nonzero().flatten()
        paired = logits[0,j-1].float()
        independent = (torch.logsumexp(paired,-1)-paired.gather(1,lab[0,j,None]).squeeze(1)).mean()
        result['real_loss'] = dict(upstream=out.loss.item(), independent=independent.item(),
            error=abs(out.loss.item()-independent.item()))
        assert result['real_loss']['error'] < 1e-5
        # Mutate later input tokens only; retain length, audio and positions.
        changed = {k:v for k,v in batch.items() if k!='labels'}
        changed['input_ids'] = ids.clone()
        cut = max(first+1, ids.shape[-1]-16)
        changed['input_ids'][0,cut:] = processor.tokenizer.encode(' hello',add_special_tokens=False)[0]
        altered = model(**changed,use_cache=False).logits
        earlier_error = (logits[:,:cut].float()-altered[:,:cut].float()).abs().max().item()
        result['real_future_invariance'] = dict(changed_from=cut, checked_through=cut-1,max_abs=earlier_error)
        assert earlier_error < 1e-5
        del logits, altered, out, paired

        # Forced token histories, preserving the full audio reference.
        n = min(first+8, ids.shape[-1]-4)
        prefix = {k:v for k,v in batch.items() if k not in ['labels','input_ids','attention_mask']}
        prefix.update(input_ids=ids[:,:n],attention_mask=batch['attention_mask'][:,:n])
        cached = model(**prefix,use_cache=True,logits_to_keep=1)
        past = cached.past_key_values
        comparisons = []
        for offset in range(3):
            t = n+offset
            cached = model(input_ids=ids[:,t:t+1],attention_mask=torch.ones_like(ids[:,:t+1]),
                position_ids=torch.tensor([[t]],device=DEVICE),past_key_values=past,use_cache=True,logits_to_keep=1)
            past = cached.past_key_values
            all_input = {k:v for k,v in batch.items() if k not in ['labels','input_ids','attention_mask']}
            all_input.update(input_ids=ids[:,:t+1],attention_mask=torch.ones_like(ids[:,:t+1]))
            recomputed = model(**all_input,use_cache=False,logits_to_keep=1)
            comparisons.append(compare(cached.logits,recomputed.logits))
        result['short_audio_cache_vs_recompute'] = comparisons
        del past,cached,recomputed

        class ForceEOS(LogitsProcessor):
            def __call__(self, input_ids, scores):
                scores.fill_(-torch.inf)
                scores[:,processor.tokenizer.eos_token_id] = 0
                return scores
        prompt = prepare_inputs(processor,build_transcription_messages(sample['audio'],sample['prompt']),device=DEVICE).to(DEVICE)
        cfg = copy.deepcopy(model.generation_config)
        cfg.eos_token_id = processor.tokenizer.eos_token_id
        cfg.pad_token_id = processor.tokenizer.pad_token_id
        cfg.use_cache = True
        cfg.save_pretrained(run/'config_roundtrip')
        reload_cfg = GenerationConfig.from_pretrained(run/'config_roundtrip')
        forced = model.generate(**prompt,generation_config=reload_cfg,logits_processor=[ForceEOS()],
            max_new_tokens=4,do_sample=False,logits_to_keep=1)
        new = forced[0,prompt['input_ids'].shape[-1]:].tolist()
        assert new == [processor.tokenizer.eos_token_id]
        result['eos_roundtrip'] = dict(generated=new,pad_id=processor.tokenizer.pad_token_id,passes=True)
    result['base_generation_config'] = json.loads((BASE/'generation_config.json').read_text())
    result['sft_generation_config'] = json.loads((OLD/'generation_config.json').read_text())
    result['status'] = 'completed'
    write(run/'contracts.json',result)
    emit('contracts_complete',result=result)


def replay(run, processor):
    result = []
    for label,path in [('base',BASE),('sft402',OLD)]:
        model = load(path)
        for key in REPLAY_KEYS:
            prediction = json.loads((EVAL/'predictions/sft'/f'{key}.json').read_text())
            start,count,c = longest_run(prediction['generated_ids'])
            assert count > 1000 and prediction['split']=='dev'
            history = prediction['generated_ids'][:start]
            prompt = prepare_inputs(processor,build_transcription_messages(prediction['audio'],prediction['prompt']),
                max_length=131072,device=DEVICE).to(DEVICE)
            full = with_history(prompt,history)
            emit('replay_start',model=label,key=key,prompt_tokens=prompt['input_ids'].shape[-1],history_tokens=start)
            with torch.inference_mode(),amp():
                out = model(**full,use_cache=True,logits_to_keep=1)
                past = out.past_key_values
                points = [dict(k=0,**stats(out.logits,c,processor.tokenizer.eos_token_id))]
                comparison = []
                for k in range(1,33):
                    n = full['input_ids'].shape[-1]+k-1
                    out = model(input_ids=torch.tensor([[c]],device=DEVICE),
                        attention_mask=torch.ones(1,n+1,device=DEVICE,dtype=torch.long),
                        position_ids=torch.tensor([[n]],device=DEVICE),past_key_values=past,use_cache=True,logits_to_keep=1)
                    past = out.past_key_values
                    if k in [1,2,4,8,16,32]:
                        points.append(dict(k=k,**stats(out.logits,c,processor.tokenizer.eos_token_id)))
                    if k in [8,32]:
                        complete = with_history(prompt,history+[c]*k)
                        uncached = model(**complete,use_cache=False,logits_to_keep=1)
                        comparison.append(dict(k=k,**compare(out.logits,uncached.logits),
                            cached_repeat=stats(out.logits,c,processor.tokenizer.eos_token_id),
                            recomputed_repeat=stats(uncached.logits,c,processor.tokenizer.eos_token_id)))
                        del uncached,complete
                result.append(dict(key=key,model=label,loop_start=start,saved_loop_count=count,
                    repeated_token=c,repeated_text=processor.tokenizer.decode([c]),
                    prefix_sha256=hashlib.sha256(json.dumps(history,separators=(',',':')).encode()).hexdigest(),
                    points=points,cache_vs_recompute=comparison))
                del out,past
            write(run/'replay.json',dict(status='running',records=result))
            del full,prompt
            gc.collect(); torch.cuda.empty_cache()
        del model
        gc.collect(); torch.cuda.empty_cache()
    write(run/'replay.json',dict(status='completed',records=result,
        boundary='Two known dev failures; same saved SFT prefix, full original audio. Not a prevalence estimate or natural-recovery test.'))
    emit('replay_complete',records=len(result))


def get_val(processor):
    rows = []
    for file in ['ami_dev.jsonl','alimeeting_dev.jsonl']:
        rows.extend(ConversationDataset(str(ROOT/'data/moss_jsonl'/file)).samples)
    inputs = json.loads((EVAL/'inputs.json').read_text())
    chosen = [(key,next(row for row in rows if row['audio']==next(x['audio'] for x in inputs if x['key']==key))) for key in VAL_KEYS]
    return [(key,DataCollator(processor,131072)([row])) for key,row in chosen]


def eval_loss(model, batches):
    result = []
    model.eval()
    with torch.inference_mode(), amp():
        for key,cpu in batches:
            b = move(cpu)
            out = model(**b,use_cache=False)
            result.append(dict(key=key,loss=out.loss.item(),supervised_tokens=int(b['labels'].ne(-100).sum())))
            del b,out
    model.train()
    return result


def update_sample(model):
    selected = ['model.whisper_encoder.layers.0.fc1.weight', 'model.whisper_encoder.layers.23.fc2.weight',
        'model.vq_adaptor.layers.0.weight','model.language_model.layers.0.self_attn.q_proj.weight',
        'model.language_model.layers.14.mlp.down_proj.weight','model.language_model.layers.27.mlp.down_proj.weight',
        'model.language_model.embed_tokens.weight']
    result = {}
    for name,p in model.named_parameters():
        if name not in selected:
            continue
        idx = torch.linspace(0,p.numel()-1,min(65536,p.numel()),device=p.device).long().unique()
        before = p.detach().flatten()[idx].float().clone()
        grad = p.grad.detach().flatten()[idx].float().clone()
        result[name] = (p,idx,before,grad)
    return result


def audit_updates(samples, lr):
    rows = []
    for name,(p,idx,before,grad) in samples.items():
        after = p.detach().flatten()[idx].float()
        # Independent FP32 Adam first step, same *actual clipped gradient*.
        reference = before-lr*grad/(grad.abs()+1e-8)
        desired = reference-before
        observed = after-before
        valid = desired.ne(0)
        rows.append(dict(parameter=name,sampled=int(before.numel()),reference_nonzero=int(valid.sum()),
            reference_nonzero_but_actual_zero=int((valid & observed.eq(0)).sum()),
            lost_fraction=(valid & observed.eq(0)).sum().item()/max(valid.sum().item(),1),
            nonzero_gradient=int(grad.ne(0).sum()),actual_update_norm=observed.norm().item(),
            reference_update_norm=desired.norm().item(), update_error_norm=(observed-desired).norm().item()))
    return rows


def pilot(run, processor, precision):
    torch.manual_seed(0)
    dataset = ConversationDataset(str(ROOT/'data/moss_jsonl/train_mix.jsonl'))
    order = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(0))[:16].tolist()
    dtype = torch.float32 if precision=='fp32' else torch.bfloat16
    model = load(BASE,dtype)
    model.tie_weights()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    install_bounded_audio_encoder(model,8)
    install_memory_forward(model,512,offload_backbone=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0,fused=True)
    vals = get_val(processor)
    result = dict(precision=precision,seed=0,start='original_base',attention='full',sp=1,
        data_indices=order,data_audios=[dataset[i]['audio'] for i in order],effective_batch=4,
        steps_planned=4,schedule_horizon=402,warmup=0,learning_rate=1e-5,weight_decay=0.0,
        note='One GPU with accumulation 4, full meetings; isolated diagnostic, not DDP qualification or final training.',
        initial_validation=eval_loss(model,vals),steps=[])
    write(run/f'pilot-{precision}.json',result)
    for step in range(4):
        lr = 1e-5*(1-step/402)
        for group in optimizer.param_groups:
            group['lr'] = lr
        batch_ids = order[step*4:(step+1)*4]
        cpu_batches = [DataCollator(processor,131072)([dataset[i]]) for i in batch_ids]
        denom = sum(int(b['labels'][:,1:].ne(-100).sum()) for b in cpu_batches)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        for index,b in zip(batch_ids,cpu_batches):
            emit('pilot_microbatch',precision=precision,step=step+1,index=index,tokens=b['input_ids'].shape[-1])
            b = move(b)
            with amp():
                out = model(**b,use_cache=False,num_items_in_batch=denom)
            loss_value += out.loss.item()
            out.loss.backward()
            del b,out
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        samples = update_sample(model) if step==0 else None
        optimizer.step()
        if samples is not None:
            result['first_step_update_audit'] = audit_updates(samples,lr)
            result['optimizer_state_dtypes'] = dict(collections.Counter(str(v.dtype) for state in optimizer.state.values() for k,v in state.items() if k in ['exp_avg','exp_avg_sq']))
            del samples
        torch.cuda.synchronize(DEVICE)
        record = dict(step=step+1,loss=loss_value,lr=lr,grad_norm=norm.item(),
            seconds=time.perf_counter()-start,supervised_tokens=denom,
            peak_allocated_gib=torch.cuda.max_memory_allocated(DEVICE)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(DEVICE)/2**30)
        result['steps'].append(record)
        write(run/f'pilot-{precision}.json',result)
        emit('pilot_step',precision=precision,**record)
        del cpu_batches
    optimizer.zero_grad(set_to_none=True)
    result['final_validation'] = eval_loss(model,vals)
    # Keep the diagnostic checkpoint isolated; no old checkpoint is overwritten.
    model.save_pretrained(run/f'pilot-{precision}-checkpoint',safe_serialization=True)
    processor.save_pretrained(run/f'pilot-{precision}-checkpoint')
    torch.save(dict(optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(DEVICE),
        step=4,order=order,schedule_horizon=402),run/f'pilot-{precision}-checkpoint'/'pilot_state.pt')
    result['status'] = 'completed'
    write(run/f'pilot-{precision}.json',result)
    emit('pilot_complete',precision=precision,final_validation=result['final_validation'])


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['contracts','replay','pilot'])
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--device',type=int,required=True)
    ap.add_argument('--precision',choices=['bf16','fp32'])
    a=ap.parse_args()
    torch.cuda.set_device(a.device)
    DEVICE=torch.device('cuda',a.device)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    a.run.mkdir(parents=True,exist_ok=True)
    free,total=torch.cuda.mem_get_info(DEVICE)
    emit('start',mode=a.mode,precision=a.precision,device=a.device,gpu=torch.cuda.get_device_name(DEVICE),
        free_gib=free/2**30,total_gib=total/2**30,torch=torch.__version__)
    processor_cls=AutoProcessor if a.mode=='replay' else MossTranscribeDiarizeProcessor
    processor=processor_cls.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    if a.mode=='contracts': contracts(a.run,processor)
    elif a.mode=='replay': replay(a.run,processor)
    else: pilot(a.run,processor,a.precision)
