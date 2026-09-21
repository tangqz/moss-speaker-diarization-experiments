"""Compare actual Whisper outputs and gradients across independent chunk batches."""
import json
import os
import sys
import torch
sys.path.insert(0,os.environ.get("MOSS_ROOT","/work/qt28/moss/MOSS-Transcribe-Diarize"))
from finetune import ConversationDataset, DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from transformers import AutoModelForCausalLM
from memory_sft import install_bounded_audio_encoder

root="/work/qt28/moss"
processor=MossTranscribeDiarizeProcessor.from_pretrained(root+"/models/MOSS-Transcribe-Diarize",trust_remote_code=True)
sample=ConversationDataset(root+"/data/smoke_train.jsonl")[0]
features=DataCollator(processor,8192)([sample])["input_features"].cuda().to(torch.bfloat16).repeat(9,1,1)
model=AutoModelForCausalLM.from_pretrained(root+"/models/MOSS-Transcribe-Diarize",
    trust_remote_code=True,dtype=torch.bfloat16,attn_implementation="sdpa")
encoder=model.model.whisper_encoder.cuda().train()
encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
torch.manual_seed(918)
reference=encoder(features,return_dict=True).last_hidden_state
probe=torch.randn_like(reference)
(reference.float()*probe.float()).mean().backward()
expected=reference.detach().clone()
grads={n:p.grad.detach().clone() for n,p in encoder.named_parameters() if n in {"conv1.weight","layers.0.fc1.weight","layers.0.self_attn.q_proj.weight"}}
assert len(grads)==3
del reference
encoder.zero_grad(set_to_none=True)
install_bounded_audio_encoder(model,chunk_batch=2)
actual=encoder(features,return_dict=True).last_hidden_state
(actual.float()*probe.float()).mean().backward()
output_relative=float((actual.detach().float()-expected.float()).norm()/expected.float().norm())
relative={n:float((p.grad.float()-grads[n].float()).norm()/grads[n].float().norm()) for n,p in encoder.named_parameters() if n in grads}
print(json.dumps(dict(event="whisper_ffn_comparison",output_relative_error=output_relative,gradient_relative_errors=relative)),flush=True)
assert output_relative<0.02,output_relative
assert all(e<0.04 for e in relative.values()),relative
print(json.dumps(dict(event="whisper_ffn_equivalence_pass",chunks=9,chunk_batch=2,
    output_relative_error=output_relative,gradient_relative_errors=relative)),flush=True)
