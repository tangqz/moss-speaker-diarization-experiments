"""Official tensor parity and explicit prefix metadata, before SP reshaping."""
import json,os
from pathlib import Path
import torch
import moss_rswa_plugin as plugin
from swift.model import get_model_processor
from swift.template import get_template
from finetune import DataCollator,ConversationDataset

TASK=Path('/work/qt28/moss/dkucc/rswa_20260920')
_,processor=get_model_processor('/work/qt28/moss/models/MOSS-Transcribe-Diarize',
    model_type=plugin.MODEL_TYPE,load_model=False,torch_dtype=torch.float32)
template=get_template(processor,template_type=plugin.MODEL_TYPE,max_length=131072,sequence_parallel_size=4)
template.set_mode('train')
dataset=ConversationDataset('/work/qt28/moss/data/moss_jsonl/train_mix.jsonl')
receipt=json.loads((TASK/'evidence/data_metadata.json').read_text())['original_receipts']
results=[]
for index in [receipt['smoke_index'],receipt['longest_index']]:
    row=dataset[index]
    swift_row=dict(messages=[dict(role='user',content='<audio>\n'+row['prompt']),
                             dict(role='assistant',content=row['target'])],audios=[row['audio']])
    actual=template.data_collator([template.encode(swift_row)])
    expected=DataCollator(processor,131072)([row])
    n=expected['input_ids'].shape[1]
    for key in ['input_ids','labels','attention_mask']:
        assert torch.equal(actual[key][:,:n],expected[key]),(index,key)
    for key in ['input_features','audio_feature_lengths','audio_chunk_mapping']:
        assert torch.equal(actual[key],expected[key]),(index,key)
    p=int(torch.where(expected['labels'][0]!=-100)[0][0])
    assert actual['rswa_prefix_length']==p and actual['rswa_valid_length']==n
    assert actual['input_ids'].shape[1]%4==0
    assert actual['labels'][:,n:].eq(-100).all()
    assert actual['attention_mask'][:,n:].eq(0).all()
    results.append(dict(index=index,length=n,prefix=p,padded_length=actual['input_ids'].shape[1],passed=True))
    print(json.dumps(results[-1]),flush=True)
Path(os.environ['RSWA_GATE_RUN'],'adapter_check.json').write_text(json.dumps(dict(passed=True,records=results),indent=2))
