"""Check the real MOSS supervised path and untouched inference against upstream."""
import json
import os
import sys
import torch
sys.path.insert(0, os.environ.get("MOSS_ROOT", "/work/qt28/moss/MOSS-Transcribe-Diarize"))
from finetune import ConversationDataset, DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from transformers import AutoModelForCausalLM
from memory_sft import install_memory_forward

root = "/work/qt28/moss"
torch.manual_seed(0)
processor = MossTranscribeDiarizeProcessor.from_pretrained(root+"/models/MOSS-Transcribe-Diarize",trust_remote_code=True)
dataset = ConversationDataset(root+"/data/smoke_train.jsonl")
batch = {k:v.cuda() for k,v in DataCollator(processor,8192)([dataset[0]]).items()}
model = AutoModelForCausalLM.from_pretrained(root+"/models/MOSS-Transcribe-Diarize",
    trust_remote_code=True,dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
model.tie_weights()
model.config.use_cache = False
model.config.text_config.use_cache = False
out = model(**batch)
loss_ref = out.loss.detach().clone()
out.loss.backward()
grad_ref = model.lm_head.weight.grad.detach().clone()
del out
model.zero_grad(set_to_none=True)
inference_inputs = {k:v for k,v in batch.items() if k!="labels"}
with torch.no_grad():
    # Compare identical projection shapes; full-sequence vs one-row GEMM has
    # ordinary BF16 rounding differences even with the unmodified model.
    last_logits = model(**inference_inputs,logits_to_keep=1).logits.detach().clone()
offload = "0" in os.environ.get("MOSS_CPU_OFFLOAD_RANKS", "").split(",")
install_memory_forward(model,512,offload_backbone=offload)
out = model(**batch)
assert out.logits is None
out.loss.backward()
relative_grad_error = (model.lm_head.weight.grad.float()-grad_ref.float()).norm()/grad_ref.float().norm()
torch.testing.assert_close(out.loss,loss_ref,atol=2e-4,rtol=2e-4)
assert relative_grad_error < 0.03, relative_grad_error.item()
with torch.no_grad():
    inference = model(**inference_inputs,logits_to_keep=1)
torch.testing.assert_close(inference.logits,last_logits,atol=0,rtol=0)
print(json.dumps(dict(event="real_moss_equivalence_pass",tokens=batch["input_ids"].shape[-1],
    reference_loss=float(loss_ref),memory_loss=float(out.loss.detach()),relative_head_grad_error=float(relative_grad_error),
    inference_exact=True,offload_backbone=offload)),flush=True)
