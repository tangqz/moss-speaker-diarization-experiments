"""Audit manifest membership/audio paths; compute exact full-context lengths."""
import hashlib
import json
import os
from pathlib import Path
import sys
import soundfile as sf
sys.path.insert(0,os.environ.get("MOSS_ROOT","/work/qt28/moss/MOSS-Transcribe-Diarize"))
from finetune import ConversationDataset, DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor, _compute_audio_token_length
from moss_transcribe_diarize.inference_utils import build_transcription_messages

root=Path("/work/qt28/moss")
folder=root/"data/moss_jsonl"
train=folder/"train_mix.jsonl"
rows=[json.loads(l) for l in train.read_text().splitlines() if l.strip()]
paths=[r["conversation"][1]["content"] for r in rows]
assert len(paths)==len(set(paths))==536
assert all(Path(p).is_file() for p in paths)
heldout=set()
for name in ["alimeeting_dev","alimeeting_test","aishell4_test","ami_dev","ami_test"]:
    heldout.update(json.loads(l)["conversation"][1]["content"] for l in (folder/(name+".jsonl")).read_text().splitlines() if l.strip())
assert not heldout.intersection(paths)
processor=MossTranscribeDiarizeProcessor.from_pretrained(root/"models/MOSS-Transcribe-Diarize",trust_remote_code=True)
dataset=ConversationDataset(str(train))
collator=DataCollator(processor,131072)
lengths=[]
for i in range(len(dataset)):
    sample=dataset[i]
    info=sf.info(sample["audio"])
    assert info.samplerate==processor.feature_extractor.sampling_rate and info.channels==1
    assert info.frames>0
    chunk=int(processor.feature_extractor.n_samples)
    audio_tokens=sum(_compute_audio_token_length(min(chunk,info.frames-start),
        processor.feature_extractor,processor.audio_merge_size) for start in range(0,info.frames,chunk))
    prompt=processor.apply_chat_template(build_transcription_messages(sample["audio"],sample["prompt"]),
        tokenize=False,add_generation_prompt=True)
    text=prompt+sample["target"]+processor.tokenizer.eos_token
    ids=processor.expand_audio_token(text,audio_tokens,131072)
    prompt_ids=processor.expand_audio_token(prompt,audio_tokens,131072)
    length=len(ids)
    label_count=length-len(prompt_ids)
    assert label_count>0 and ids[-1]==processor.tokenizer.eos_token_id
    lengths.append(dict(index=i,tokens=length,labels=label_count,audio=paths[i]))
    if (i+1)%50==0:
        print(json.dumps(dict(event="data_length_audit",done=i+1,total=len(dataset),max_tokens=max(x["tokens"] for x in lengths))),flush=True)
for i in {min(range(len(lengths)),key=lambda i:lengths[i]["tokens"]),max(range(len(lengths)),key=lambda i:lengths[i]["tokens"])}:
    batch=collator([dataset[i]])
    assert int(batch["input_ids"].shape[-1])==lengths[i]["tokens"]
    assert int(batch["labels"].ne(-100).sum())==lengths[i]["labels"]
    print(json.dumps(dict(event="exact_length_crosscheck_pass",index=i,tokens=lengths[i]["tokens"])),flush=True)
report=dict(records=len(rows),train_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),heldout_overlap=0,
            length_method="official_chunk_length_and_expand_audio_token; full_collator_checked_on_min_max",
            max_tokens=max(x["tokens"] for x in lengths),min_tokens=min(x["tokens"] for x in lengths),samples=lengths)
dest=root/"results"/("preflight-"+os.environ["SLURM_JOB_ID"]+".json")
dest.write_text(json.dumps(report,indent=2))
order=sorted(range(len(rows)),key=lambda i:lengths[i]["tokens"])
picked=[order[i] for i in [0,len(order)//4,len(order)//2,3*len(order)//4,-4,-3,-2,-1]]
smoke=folder/("codex_smoke_"+os.environ["SLURM_JOB_ID"]+".jsonl")
smoke.write_text("".join(json.dumps(rows[i],ensure_ascii=False)+"\n" for i in picked))
print(json.dumps(dict(event="data_preflight_pass",records=len(rows),max_tokens=report["max_tokens"],smoke=str(smoke))),flush=True)
