"""Read-only CPU audit of targets and optimizer metadata; no model/audio load."""
import collections
import importlib.util
import json
from pathlib import Path
import re
import sys

root=Path('/work/qt28/moss')
spec=importlib.util.spec_from_file_location('audit_parser',root/'MOSS-Transcribe-Diarize/moss_transcribe_diarize/transcript_parser.py')
pa=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pa;spec.loader.exec_module(pa)
blocks=re.compile(r'\[(\d+(?:\.\d+)?)\]\[(S\d+)\]([^\[\]]*)\[(\d+(?:\.\d+)?)\]')
data=[json.loads(line) for line in (root/'data/moss_jsonl/train_mix.jsonl').read_text().splitlines() if line.strip()]
stats=collections.defaultdict(lambda:dict(meetings=0,chars=0,segments=0,bad_target_grammar=0,parser_loss_meetings=0,largest_repeat=0,zero_length_segments=0,backward_end_segments=0))
issues=[];prompts=collections.Counter()
for i,r in enumerate(data):
    c=r['conversation'];audio=c[1]['content'];target=c[2]['content'];prompts[c[0]['content']]+=1
    ds='ami' if '/ami/' in audio else 'alimeeting' if '/alimeeting/' in audio else 'aishell4'
    s=stats[ds];s['meetings']+=1;s['chars']+=len(target)
    matches=list(blocks.finditer(target));parsed=pa.parse_transcript(target)
    s['segments']+=len(matches);s['zero_length_segments']+=sum(float(m[1])==float(m[4]) for m in matches)
    s['backward_end_segments']+=sum(float(m[1])>float(m[4]) for m in matches)
    uncovered=len(target)-sum(m.end()-m.start() for m in matches)
    if uncovered:
        s['bad_target_grammar']+=1;issues.append(dict(index=i,audio=audio,uncovered_chars=uncovered,head=target[:120]))
    if len(parsed)!=len(matches):
        s['parser_loss_meetings']+=1;issues.append(dict(index=i,audio=audio,parser_segments=len(parsed),regex_segments=len(matches)))
    longest=max((len(m[0]) for m in re.finditer(r'(.)\1+',target)),default=1)
    s['largest_repeat']=max(s['largest_repeat'],longest)

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
checkpoint=root/'checkpoints/full-attention-ddp4-62994/checkpoint-402'
with FakeTensorMode():
    optimizer=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True)
states=collections.Counter((key,str(value.dtype)) for item in optimizer['state'].values() for key,value in item.items() if torch.is_tensor(value))
args=torch.load(checkpoint/'training_args.bin',map_location='cpu',weights_only=False)
fields=['learning_rate','num_train_epochs','per_device_train_batch_size','gradient_accumulation_steps',
    'optim','adam_beta1','adam_beta2','adam_epsilon','weight_decay','max_grad_norm',
    'lr_scheduler_type','warmup_steps','warmup_ratio','eval_strategy','save_steps','save_total_limit',
    'bf16','average_tokens_across_devices','label_smoothing_factor']
print(json.dumps(dict(records=len(data),stats=dict(stats),issues=issues,prompts=dict(prompts),
    optimizer_state_dtypes={str(k):v for k,v in states.items()},
    optimizer_parameter_groups=[{k:v for k,v in group.items() if k!='params'}|{'parameters':len(group['params'])} for group in optimizer['param_groups']],
    training_arguments={k:str(getattr(args,k,None)) for k in fields}),ensure_ascii=False))
