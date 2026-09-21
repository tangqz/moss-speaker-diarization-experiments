"""FP32 persistent parameters/Adam, BF16 autocast, exact global-token CE.

Trainer retains the original seeded data loader. A callback pauses execution at
the requested validation boundary while max_steps and the scheduler remain 402.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import resource
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from common import BASE, TRAIN, HERE, REPO, HORIZON, read, write, sha, emit, append, now
sys.path.insert(0, str(REPO))
from finetune import ConversationDataset, DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, set_seed
from memory_sft import install_memory_forward, install_bounded_audio_encoder


def rng_digest():
    states = [random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()]
    return [hashlib.sha256(pickle.dumps(s) if not torch.is_tensor(s) else s.cpu().numpy().tobytes()).hexdigest()
            for s in states]


def assert_precision(model, optimizer=None, gradients=False):
    params = list(model.parameters())
    assert params and all(p.requires_grad and p.dtype == torch.float32 for p in params)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    if gradients:
        grads = [p.grad for p in params]
        assert all(g is not None and g.dtype == torch.float32 for g in grads)
        norms = torch.stack([g.detach().norm() for g in grads])
        assert bool(torch.isfinite(norms).all()), 'Nonfinite gradient; refusing optimizer update'
    moments = collections.Counter()
    if optimizer is not None:
        for state in optimizer.state.values():
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    value = state[key]
                    assert value.dtype == torch.float32, (key, value.dtype)
                    moments[str(value.dtype)] += 1
    return dict(parameters=len(params), parameter_dtype='torch.float32', adam_moments=dict(moments), tied=True)


class IndexedDataset(ConversationDataset):
    def __init__(self, indices=None):
        super().__init__(str(TRAIN))
        self.indices = list(range(len(self.samples))) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        return dict(self.samples[original], __index=original)


class CheckedCollator(DataCollator):
    def __init__(self, processor):
        super().__init__(processor, 131072)
        self.expected = {r['index']:r for r in read(HERE/'preflight.json')['samples']}

    def __call__(self, samples):
        assert len(samples) == 1
        batch = super().__call__(samples)
        index = samples[0]['__index']
        exp = self.expected[index]
        assert samples[0]['audio'] == exp['audio']
        assert batch['input_ids'].shape[-1] == exp['tokens'] <= 131072
        assert int(batch['labels'][:, 1:].ne(-100).sum()) == exp['labels']
        assert int(batch['labels'][0, -1]) == self.processor.tokenizer.eos_token_id
        batch['sample_index'] = torch.tensor([index])
        return batch


def sampled_gradients(model, count=4096):
    result = {}
    for name, p in model.named_parameters():
        n = min(count, p.numel())
        idx = torch.arange(n, device=p.device, dtype=torch.long)*(p.numel()-1)//max(n-1, 1)
        result[name] = p.grad.detach().flatten()[idx].cpu().clone()
    return result


UPDATE_COMPONENTS = {
    'model.language_model.embed_tokens.weight',
    'model.language_model.layers.0.self_attn.q_proj.weight',
    'model.language_model.layers.14.mlp.down_proj.weight',
    'model.language_model.layers.27.mlp.down_proj.weight',
    'model.whisper_encoder.layers.0.fc1.weight',
    'model.whisper_encoder.layers.23.fc2.weight',
    'model.vq_adaptor.layers.0.weight',
}


class Audit(TrainerCallback):
    def __init__(self, output, stop, gate):
        self.output, self.stop, self.gate = output, stop, gate
        self.trainer = None
        self.samples = {}

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        result = assert_precision(model, optimizer)
        if state.global_step:
            assert result['adam_moments'].get('torch.float32') == 2*result['parameters']
            expected_lr = 1e-5*(1-state.global_step/HORIZON)
            assert abs(optimizer.param_groups[0]['lr']-expected_lr) < 1e-12
        emit('precision_guard', rank=args.process_index, step=state.global_step, **result)

    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.perf_counter()
        self.trainer.micro_records = []
        torch.cuda.reset_peak_memory_stats()

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        assert_precision(model, optimizer, gradients=True)
        self.lr = optimizer.param_groups[0]['lr']
        assert abs(self.lr-1e-5*(1-state.global_step/HORIZON)) < 1e-12
        if self.gate and args.process_index == 0:
            torch.save(sampled_gradients(model), self.output/f'gradients-step-{state.global_step+1}.pt')
        self.samples = {}
        if args.process_index == 0:
            for name, p in model.named_parameters():
                if name in UPDATE_COMPONENTS:
                    n = min(4096, p.numel())
                    idx = torch.arange(n, device=p.device, dtype=torch.long)*(p.numel()-1)//max(n-1, 1)
                    self.samples[name] = (p, idx, p.detach().flatten()[idx].clone(), p.grad.detach().flatten()[idx].clone())

    def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        precision = assert_precision(model, optimizer)
        assert precision['adam_moments'].get('torch.float32') == 2*precision['parameters']
        updates = []
        for name, (p, idx, before, gradient) in self.samples.items():
            delta = p.detach().flatten()[idx]-before
            row = dict(parameter=name, sampled=len(idx), nonzero=int(delta.ne(0).sum()), norm=float(delta.norm()))
            if state.global_step == 0:
                desired = before-self.lr*gradient/(gradient.abs()+1e-8)-before
                valid = desired.ne(0)
                row.update(reference_nonzero=int(valid.sum()), lost=int((valid & delta.eq(0)).sum()))
                assert row['lost'] == 0, row
            updates.append(row)
        self.samples.clear()
        self.last_updates = updates
        if args.process_index == 0:
            assert len(updates) == 7

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        row = dict(event='step', utc=now(), rank=args.process_index, step=state.global_step,
                   epoch=state.epoch, lr=self.lr, seconds=time.perf_counter()-self.start,
                   microbatches=self.trainer.micro_records,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                   updates=self.last_updates, precision='FP32_parameters_gradients_Adam_BF16_autocast')
        append(self.output/f'train-rank-{args.process_index}.jsonl', row)
        write(self.output/f'rank-{args.process_index}-status.json', row)
        emit('optimizer_step', rank=args.process_index, step=state.global_step,
             peak_gib=row['peak_allocated_gib'], seconds=row['seconds'])
        if self.gate or state.global_step % 25 == 0 or state.global_step in [134,268,402]:
            control.should_save = True
        if state.global_step >= self.stop:
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.process_index == 0 and logs:
            append(self.output/'trainer-log.jsonl', dict(utc=now(), step=state.global_step, **logs))


class AuditedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        indices = inputs.pop('sample_index').tolist()
        local_tokens = int(inputs['labels'][:,1:].ne(-100).sum())
        assert num_items_in_batch is not None
        global_tokens = int(num_items_in_batch)
        rng_before = rng_digest() if self.gate else None
        loss, out = super().compute_loss(model, inputs, True, num_items_in_batch)
        assert bool(torch.isfinite(loss.detach()))
        # Trainer compensates for DDP's mean by multiplying by world size.
        local_numerator = float(loss.detach())*global_tokens/self.args.world_size
        self.micro_records.append(dict(indices=indices, sequence_tokens=inputs['input_ids'].shape[-1],
            local_supervised_tokens=local_tokens, global_supervised_tokens=global_tokens,
            loss_numerator=local_numerator, local_mean_loss=local_numerator/local_tokens,
            rng_before=rng_before))
        return (loss, out) if return_outputs else loss

    def create_optimizer(self):
        result = super().create_optimizer()
        assert_precision(self.model, self.optimizer)
        return result

    def _load_rng_state(self, checkpoint):
        super()._load_rng_state(checkpoint)
        emit('rng_restored', rank=self.args.process_index, checkpoint=str(checkpoint), digest=rng_digest())

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        dist.barrier() if self.args.world_size > 1 else None
        if self.is_world_process_zero():
            folder = Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
            evidence = assert_precision(self.model, self.optimizer)
            write(folder/'v2_contract.json', dict(step=self.state.global_step, scheduler_horizon=HORIZON,
                precision=evidence, protocol_sha256=sha(HERE/'design_protocol.json'),
                data_sha256=sha(TRAIN), world_size=self.args.world_size,
                sampler='Trainer_seedable_random' if not self.gate else 'gate_sequential',
                source_manifest=read(HERE/'source_manifest.json'), use_cache_at_inference=True,
                attention='full_causal_sdpa', sp=1))
            required = ['optimizer.pt','scheduler.pt','trainer_state.json']
            required += ([f'rng_state_{r}.pth' for r in range(self.args.world_size)]
                         if self.args.world_size>1 else ['rng_state.pth'])
            assert all((folder/n).is_file() for n in required), required
            if self.gate:
                # Full live-weight hashes prove that serialization preserved every parameter.
                hashes = {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                          for n,p in self.model.named_parameters()}
                write(folder/'live_parameter_hashes.json', hashes)
            write(folder/'checkpoint_complete.json', dict(step=self.state.global_step, utc=now(), complete=True))
        dist.barrier() if self.args.world_size > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stop', type=int, required=True)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--gate-indices', type=str)
    ap.add_argument('--parameter-dtype', choices=['float32'], default='float32')
    ap.add_argument('--accumulation', type=int, default=1)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get('WORLD_SIZE','1'))
    assert world*a.accumulation == 4, 'All runs must retain global batch four'
    assert 0 < a.stop <= HORIZON
    indices = json.loads(a.gate_indices) if a.gate_indices else None
    if indices is None:
        assert world == 4 and a.accumulation == 1
        assert len(IndexedDataset()) == 536
    set_seed(0)
    model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa')
    model.tie_weights()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.generation_config.eos_token_id = 151645
    model.generation_config.pad_token_id = 151643
    install_bounded_audio_encoder(model, 8)
    install_memory_forward(model, 512, offload_backbone=True)
    processor = MossTranscribeDiarizeProcessor.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
    args = TrainingArguments(output_dir=str(a.output), max_steps=HORIZON, num_train_epochs=3,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.accumulation,
        learning_rate=1e-5, lr_scheduler_type='linear', warmup_steps=0,
        weight_decay=0, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
        optim='adamw_torch_fused', max_grad_norm=1., bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        average_tokens_across_devices=True, ddp_find_unused_parameters=False,
        logging_steps=1, logging_nan_inf_filter=False, report_to='none', disable_tqdm=True,
        save_strategy='no', save_total_limit=None, dataloader_num_workers=0,
        remove_unused_columns=False, label_names=['labels'], seed=0, data_seed=0,
        train_sampling_strategy='sequential' if indices is not None else 'random',
        ignore_data_skip=False, restore_callback_states_from_checkpoint=False)
    audit = Audit(a.output, a.stop, indices is not None)
    trainer = AuditedTrainer(model=model, args=args, train_dataset=IndexedDataset(indices),
        data_collator=CheckedCollator(processor), processing_class=processor, callbacks=[audit])
    audit.trainer = trainer
    trainer.gate = indices is not None
    trainer.micro_records = []
    assert trainer.model_accepts_loss_kwargs
    assert_precision(model)
    if a.resume:
        contract = read(a.resume/'v2_contract.json')
        assert contract['scheduler_horizon'] == HORIZON
        assert contract['world_size'] == world
        assert read(a.resume/'checkpoint_complete.json')['complete']
        assert contract['protocol_sha256'] == sha(HERE/'design_protocol.json')
        assert contract['source_manifest'] == read(HERE/'source_manifest.json')
    result = trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
    assert trainer.state.global_step == a.stop
    if trainer.is_world_process_zero():
        write(a.output/f'phase-{a.stop}-complete.json', dict(step=a.stop, utc=now(), metrics=result.metrics,
            checkpoint=str(a.output/f'checkpoint-{a.stop}'), schedule_horizon=HORIZON))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
