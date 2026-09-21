"""Verify native Swift template tensors against the original official collator."""
import json,sys,os
from pathlib import Path
import torch
import moss_plugin
from swift.model import get_model_processor
from swift.template import get_template
from finetune import DataCollator,ConversationDataset

TASK=Path(__file__).resolve().parent
BASE='/work/qt28/moss/models/MOSS-Transcribe-Diarize'
_,processor=get_model_processor(BASE,model_type=moss_plugin.MODEL_TYPE,load_model=False,torch_dtype=torch.float32)
template=get_template(processor,template_type=moss_plugin.MODEL_TYPE,max_length=131072)
template.set_mode('train')
parallel_template=get_template(processor,template_type=moss_plugin.MODEL_TYPE,max_length=131072,
                               sequence_parallel_size=4)
parallel_template.set_mode('train')
dataset=ConversationDataset('/work/qt28/moss/data/moss_jsonl/train_mix.jsonl')
receipt=json.loads((TASK/'data_receipts.json').read_text())
results=[]
for index in [receipt['smoke_index'],receipt['longest_index']]:
    r=dataset[index]
    swift_row=dict(messages=[dict(role='user',content='<audio>\n'+r['prompt']),
                              dict(role='assistant',content=r['target'])],audios=[r['audio']])
    os.environ['MOSS_IMPLICIT_CAUSAL']='0'
    actual=template.data_collator([template.encode(swift_row)])
    expected=DataCollator(processor,131072)([r])
    for key in expected:
        assert key in actual and torch.equal(actual[key],expected[key]), (index,key)
    os.environ['MOSS_IMPLICIT_CAUSAL']='1'
    adapted=parallel_template.data_collator([parallel_template.encode(swift_row)])
    length=expected['input_ids'].shape[1]
    assert 'attention_mask' not in adapted
    assert adapted['input_ids'].shape[1]%4==0
    for key in ['input_ids','labels']:
        assert torch.equal(adapted[key][:,:length],expected[key]), (index,key,'real_prefix')
    assert bool(adapted['labels'][:,length:].eq(-100).all())
    assert torch.equal(adapted['position_ids'][0],torch.arange(adapted['input_ids'].shape[1]))
    for key in ['input_features','audio_feature_lengths','audio_chunk_mapping']:
        assert torch.equal(adapted[key],expected[key]), (index,key,'audio')
    results.append(dict(index=index,tokens=actual['input_ids'].shape[-1],
                        supervised_tokens=int(actual['labels'].ne(-100).sum()),all_tensors_equal=True,
                        implicit_causal_real_prefix_equal=True,right_padding_labels_ignored=True,
                        padded_tokens=adapted['input_ids'].shape[1]))
    print(json.dumps(results[-1]),flush=True)
(TASK/'adapter_check.json').write_text(json.dumps(dict(passed=True,records=results),indent=2))
