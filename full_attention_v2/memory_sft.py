"""MOSS full-attention SFT with checkpointed projection + exact FP32 CE.

Only the loaded MOSS instance's supervised forward is adapted. Its backbone,
attention, inference forward, parameter names, and installed libraries remain
the official implementation. Full vocabulary logits are transient per chunk.
This loss-only forward is for SFT; RL entropy needs a separate statistics path.
"""
import functools
import contextlib
import json
import os
from pathlib import Path
import sys
import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _project_ce(hidden, weight, target):
    return F.cross_entropy(F.linear(hidden, weight).float(), target, reduction="sum")


def install_bounded_audio_encoder(model, chunk_batch=8):
    """Bound Whisper FFN intermediates; leave attention's batching unchanged."""
    encoder = model.model.whisper_encoder
    for key in ("dropout", "attention_dropout", "activation_dropout", "layerdrop"):
        if getattr(encoder.config, key, 0) != 0:
            raise ValueError(f"Deterministic encoder rebatching requires {key}=0")
    def install_layer(layer):
        original = layer.forward
        @functools.wraps(type(layer).forward)
        def forward(self, hidden_states, attention_mask, **kwargs):
            if not self.training or hidden_states.shape[0] <= chunk_batch:
                return original(hidden_states, attention_mask, **kwargs)
            # Mirrors transformers 5.17.0 WhisperEncoderLayer; only the
            # token-independent fc1/GELU/fc2 computation is checkpointed in parts.
            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)
            hidden_states, _ = self.self_attn(hidden_states=hidden_states,
                attention_mask=attention_mask, **kwargs)
            hidden_states = residual + F.dropout(hidden_states,p=self.dropout,training=self.training)
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            def ffn(part):
                part = self.activation_fn(self.fc1(part))
                part = F.dropout(part,p=self.activation_dropout,training=self.training)
                return F.dropout(self.fc2(part),p=self.dropout,training=self.training)
            result = [checkpoint(ffn,hidden_states[i:i+chunk_batch],use_reentrant=False)
                      for i in range(0,hidden_states.shape[0],chunk_batch)]
            hidden_states = residual + torch.cat(result,dim=0)
            if hidden_states.dtype == torch.float16:
                bound = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states,min=-bound,max=bound)
            return hidden_states
        layer.forward = types.MethodType(forward,layer)
    for layer in encoder.layers:
        install_layer(layer)


def projected_causal_loss(hidden, weight, labels, num_items_in_batch=None, chunk_size=512):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    targets = F.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
    states = hidden.reshape(-1, hidden.shape[-1])
    valid = targets.ne(-100)
    # Ignored prompt/padding positions have exactly zero direct CE gradient.
    states, targets = states[valid], targets[valid]
    total = states.sum() * 0.0 + weight.reshape(-1)[0].float() * 0.0
    for start in range(0, targets.numel(), chunk_size):
        args = (states[start:start + chunk_size], weight, targets[start:start + chunk_size])
        if torch.is_grad_enabled():
            term = checkpoint(_project_ce, *args, use_reentrant=False)
        else:
            term = _project_ce(*args)
        total = total + term
    denom = targets.numel() if num_items_in_batch is None else num_items_in_batch
    denom = torch.as_tensor(denom, device=hidden.device, dtype=torch.float32).clamp_min(1)
    return total / denom


def install_memory_forward(model, chunk_size=512, offload_backbone=False):
    from transformers.modeling_outputs import CausalLMOutputWithPast
    if model.config.model_type != "moss_transcribe_diarize":
        raise ValueError("Only the verified MOSS architecture is supported")
    original = model.forward

    @functools.wraps(type(model).forward)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                input_features=None, audio_feature_lengths=None, audio_chunk_mapping=None,
                logits_to_keep=0, **kwargs):
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=position_ids, past_key_values=past_key_values,
                      inputs_embeds=inputs_embeds, use_cache=use_cache,
                      output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                      input_features=input_features, audio_feature_lengths=audio_feature_lengths,
                      audio_chunk_mapping=audio_chunk_mapping)
        if labels is None:
            return original(**inputs, return_dict=return_dict, logits_to_keep=logits_to_keep, **kwargs)
        if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
            raise ValueError("SFT expects unsliced labels and hidden states")
        denominator = kwargs.pop("num_items_in_batch", None)
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload_backbone else contextlib.nullcontext()
        with context:
            outputs = self.model(**inputs, return_dict=True, **kwargs)
        loss = projected_causal_loss(outputs.last_hidden_state, self.lm_head.weight,
                                     labels, denominator, chunk_size)
        if return_dict is False:
            return (loss,)
        # A loss-only result also prevents Accelerate upcasting unused full logits.
        return CausalLMOutputWithPast(loss=loss)

    model.forward = types.MethodType(forward, model)
    model.accepts_loss_kwargs = True
    return model


def main():
    sys.path.insert(0, os.environ.get("MOSS_ROOT", "/work/qt28/moss/MOSS-Transcribe-Diarize"))
    from finetune import ScriptArguments, ConversationDataset, DataCollator
    from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
    from transformers import (AutoModelForCausalLM, HfArgumentParser, Trainer,
                              TrainerCallback, TrainingArguments, set_seed)
    script, args = HfArgumentParser((ScriptArguments, TrainingArguments)).parse_args_into_dataclasses()
    args.remove_unused_columns = False
    args.label_names = ["labels"]
    if args.label_smoothing_factor != 0:
        raise ValueError("This exact CE implementation requires label_smoothing_factor=0")
    set_seed(args.seed)
    processor = MossTranscribeDiarizeProcessor.from_pretrained(script.model_name_or_path, trust_remote_code=True)
    dataset = ConversationDataset(script.train_jsonl)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(script.model_name_or_path, trust_remote_code=True,
                                               dtype=dtype, attn_implementation=script.attn_implementation)
    model.tie_weights()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    audio_chunk_batch = int(os.environ.get("MOSS_AUDIO_CHUNK_BATCH", "8"))
    install_bounded_audio_encoder(model, audio_chunk_batch)
    chunk_size = int(os.environ.get("MOSS_PROJECTION_CHUNK", "512"))
    offload_ranks = {int(r) for r in os.environ.get("MOSS_CPU_OFFLOAD_RANKS", "").split(",") if r}
    offload = args.process_index in offload_ranks
    install_memory_forward(model, chunk_size, offload_backbone=offload)
    print(json.dumps(dict(event="memory_policy",rank=args.process_index,offload_backbone=offload)),flush=True)

    class MemoryLog(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            print(json.dumps(dict(event="optimizer_step", rank=args.process_index,
                step=state.global_step, peak_gib=round(torch.cuda.max_memory_allocated()/2**30, 3))), flush=True)

    trainer = Trainer(model=model, args=args, train_dataset=dataset,
        data_collator=DataCollator(processor, script.max_length), processing_class=processor,
        callbacks=[MemoryLog()])
    if trainer.is_world_process_zero():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        metadata = dict(train_jsonl=script.train_jsonl, records=len(dataset), max_length=script.max_length,
                        attention=script.attn_implementation, sequence_parallel_size=1,
                        world_size=args.world_size, chunk_size=chunk_size, loss="checkpointed_projection_fp32_ce",
                        cpu_offload_ranks=sorted(offload_ranks),
                        audio_chunk_batch=audio_chunk_batch,
                        average_tokens_across_devices=args.average_tokens_across_devices,
                        model_accepts_loss_kwargs=trainer.model_accepts_loss_kwargs,
                        model_class=str(type(model)), torch=torch.__version__, job=os.environ.get("SLURM_JOB_ID"))
        (Path(args.output_dir)/"memory_sft_config.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
    if not trainer.model_accepts_loss_kwargs:
        raise RuntimeError("Trainer must pass the global valid-token denominator")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if os.environ.get("MOSS_SKIP_FINAL_SAVE") != "1":
        trainer.save_model()
        trainer.save_state()
    trainer.save_metrics("train", result.metrics)


if __name__ == "__main__":
    main()
