"""Package native weights with original processor assets; no tensor conversion."""
import argparse
import json
import hashlib
import struct
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'evaluation'))
from common import BASE, sha, write


def compatible_weights(source,destination,config):
    with source.open('rb') as f:
        header_size=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(header_size))
    head='lm_head.weight'
    embed='model.language_model.embed_tokens.weight'
    if head not in header:
        destination.symlink_to(source)
        return dict(action='unchanged_symlink',source=str(source))
    assert config.get('tie_word_embeddings') and config['text_config'].get('tie_word_embeddings')
    assert embed in header, 'Expected explicit tied embedding in native checkpoint'
    assert all(header[head][k]==header[embed][k] for k in ['dtype','shape'])
    payload_start=8+header_size
    def digest(name):
        start,end=header[name]['data_offsets'];h=hashlib.sha256()
        with source.open('rb') as f:
            f.seek(payload_start+start)
            remaining=end-start
            while remaining:
                block=f.read(min(8*1024*1024,remaining));assert block
                h.update(block);remaining-=len(block)
        return h.hexdigest()
    shared_hash=digest(head)
    assert shared_hash==digest(embed), 'Output head differs from tied embeddings; cannot remove it'
    entries=sorted(((k,v) for k,v in header.items() if k not in (head,'__metadata__')),
                   key=lambda item:item[1]['data_offsets'][0])
    new_header={'__metadata__':header.get('__metadata__',{'format':'pt'})}
    offset=0
    for name,info in entries:
        start,end=info['data_offsets'];new_header[name]=dict(info,data_offsets=[offset,offset+end-start]);offset+=end-start
    encoded=json.dumps(new_header,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    expected=hashlib.sha256()
    with source.open('rb') as original,destination.open('xb') as output:
        output.write(struct.pack('<Q',len(encoded)));output.write(encoded)
        for name,info in entries:
            start,end=info['data_offsets'];original.seek(payload_start+start);remaining=end-start
            while remaining:
                block=original.read(min(8*1024*1024,remaining));assert block
                expected.update(block);output.write(block);remaining-=len(block)
    actual=hashlib.sha256()
    with destination.open('rb') as f:
        f.seek(8+len(encoded))
        for block in iter(lambda:f.read(8*1024*1024),b''):
            actual.update(block)
    assert actual.hexdigest()==expected.hexdigest(), 'Exported tensor payload changed'
    return dict(action='omit_verified_identical_tied_lm_head_alias',source=str(source),
        shared_tensor_sha256=shared_hash,remaining_payload_sha256=actual.hexdigest(),
        all_remaining_tensor_bytes_identical=True,dtype_conversion=False)


def export(checkpoint, destination):
    checkpoint=Path(checkpoint).resolve()
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    # Native Swift saves processor_class without the remote AutoProcessor mapping.
    # Restore the same original processor files in an inference-only directory.
    for source in BASE.iterdir():
        if source.is_file() and source.suffix in ('.json','.py','.jinja','.txt'):
            if source.name.endswith('.safetensors.index.json'):
                continue
            shutil.copy2(source,destination/source.name)
    shutil.copy2(checkpoint/'config.json',destination/'config.json')
    weights=sorted(checkpoint.glob('*.safetensors'))
    assert weights, 'Missing native safetensors checkpoint'
    config=json.loads((checkpoint/'config.json').read_text())
    weight_receipts=[]
    for source in weights:
        weight_receipts.append(compatible_weights(source,destination/source.name,config))
    for index in checkpoint.glob('*.safetensors.index.json'):
        content=json.loads(index.read_text())
        if any(r['action']!='unchanged_symlink' for r in weight_receipts):
            content['weight_map'].pop('lm_head.weight',None)
            content.pop('metadata',None)
        write(destination/index.name,content)
    # A changed model configuration must be preserved, never silently reset to Base.
    assert sha(checkpoint/'config.json')==sha(destination/'config.json')
    assert json.loads((destination/'processor_config.json').read_text())['auto_map']['AutoProcessor']
    write(destination/'export_receipt.json',dict(checkpoint=str(checkpoint),
        weights=weight_receipts,
        processor_source=str(BASE),files={p.name:sha(p) for p in sorted(destination.iterdir())}))
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    export(a.checkpoint,a.destination)
