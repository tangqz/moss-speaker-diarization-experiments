"""Swift registration: original CE and collator, explicit R-SWA metadata."""
import importlib.util,json,os,sys
import hashlib
from pathlib import Path
import torch
from swift.model import Model,ModelGroup,ModelMeta,register_model
from swift.template import TemplateMeta,register_template
from model_adapter import install

BASE_PLUGIN=Path(os.environ.get('MOSS_BASE_PLUGIN','/work/qt28/moss/dkucc/ms_swift_20260914/moss_plugin.py'))
spec=importlib.util.spec_from_file_location('moss_official_registration',BASE_PLUGIN)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
MODEL_TYPE='moss_rswa'


class RswaLoader(base.MossLoader):
    def get_model(self,model_dir,config,processor,model_kwargs):
        model=super().get_model(model_dir,config,processor,model_kwargs)
        value=os.environ.get('MOSS_RSWA_WINDOW','128')
        window=None if value=='full' else int(value)
        existing=getattr(config,'moss_rswa',None)
        if existing is not None and existing['window']!=window:
            raise ValueError('Checkpoint window differs from requested training mode')
        return install(model,window,backend=os.environ.get('MOSS_RSWA_BACKEND','flex'))


class RswaTemplate(base.MossTemplate):
    def _encode(self,inputs):
        row=super()._encode(inputs)
        labels=row.get('labels')
        if labels is None:
            prefix=len(row['input_ids'])
        else:
            prefix=next(i for i,x in enumerate(labels) if x!=-100)
            target=inputs.messages[1]['content'].strip()+self.tokenizer.eos_token
            expected=self.tokenizer(target,add_special_tokens=False)['input_ids']
            actual=[x for x in labels if x!=-100]
            if actual!=expected:
                raise ValueError('Full target changed/truncated; refusing this sample')
        row['rswa_prefix_length']=prefix
        row['rswa_valid_length']=len(row['input_ids'])
        row['rswa_sample_key']=str(inputs.audios[0])
        receipts=os.environ.get('MOSS_RSWA_INPUT_RECEIPTS')
        if receipts and os.environ.get('RANK','0')=='0':
            from input_receipt import receipt
            directory=Path(receipts);directory.mkdir(parents=True,exist_ok=True)
            dest=directory/(hashlib.sha256(row['rswa_sample_key'].encode()).hexdigest()+'.json')
            evidence=receipt(self.processor,row['input_ids'][:prefix],row['audio_feature_lengths'],row['audio_chunk_mapping'])
            evidence['audio']=row['rswa_sample_key']
            if dest.exists():
                if json.loads(dest.read_text())!=evidence:raise ValueError('Training input changed across stages')
            else:dest.write_text(json.dumps(evidence))
        return row

    def _data_collator(self,batch,*,padding_to=None):
        if len(batch)!=1:
            raise ValueError('Initial R-SWA contract requires microbatch=1')
        if os.environ.get('MOSS_IMPLICIT_CAUSAL')=='1':
            raise ValueError('The old implicit-full-causal shortcut is incompatible with R-SWA')
        encoded=super()._data_collator(batch,padding_to=padding_to)
        row=batch[0]
        length=int(row['rswa_valid_length'])
        if length>self.max_length:
            raise ValueError('Overlength meeting cannot be deleted or truncated')
        # Preserve scalar metadata before Swift shifts/splits the supervision.
        encoded['rswa_prefix_length']=int(row['rswa_prefix_length'])
        encoded['rswa_valid_length']=length
        encoded['rswa_sample_key']=row['rswa_sample_key']
        extra=(-length)%self.sequence_parallel_size
        if extra:
            encoded['input_ids']=torch.nn.functional.pad(encoded['input_ids'],(0,extra),value=self.tokenizer.pad_token_id)
            encoded['attention_mask']=torch.nn.functional.pad(encoded['attention_mask'],(0,extra),value=0)
            if 'labels' in encoded:
                encoded['labels']=torch.nn.functional.pad(encoded['labels'],(0,extra),value=-100)
        encoded['position_ids']=torch.arange(length+extra).unsqueeze(0)
        audit=os.environ.get('MOSS_RSWA_COLLATION_LOG')
        if audit and os.environ.get('RANK','0')=='0':
            with open(audit,'a') as f:
                f.write(json.dumps(dict(audio=row['rswa_sample_key'],prefix=row['rswa_prefix_length'],length=length))+'\n')
        return encoded


register_template(TemplateMeta(template_type=MODEL_TYPE,prefix=[],prompt=['{{QUERY}}'],chat_sep=None,
                               suffix=[['eos_token_id']],template_cls=RswaTemplate))
register_model(ModelMeta(model_type=MODEL_TYPE,
    model_groups=[ModelGroup([Model('OpenMOSS-Team/MOSS-Transcribe-Diarize','OpenMOSS-Team/MOSS-Transcribe-Diarize')])],
    loader=RswaLoader,template=MODEL_TYPE,model_arch=base.MODEL_TYPE,
    architectures=['MossTranscribeDiarizeForConditionalGeneration'],is_multimodal=True,
    tags=['audio'],requires=['soundfile','soxr']))
