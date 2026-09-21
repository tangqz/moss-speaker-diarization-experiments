import hashlib
import json
from pathlib import Path
import zipfile

r=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
config=read(r/'design_protocol.json')
summary=read(r/'evidence_summary.json')
sentinels=read(r/'sentinel_manifest.json')['records']
assert config['training']['persistent_parameters']=='float32'
assert config['training']['attention_policy']=='full_causal'
assert config['training']['scheduler_horizon_steps']==402
assert config['training']['effective_batch_meetings']==4
assert len(sentinels)==6 and len({x['key'] for x in sentinels})==6
assert all('/dev/' in x['key'] for x in sentinels)
assert config['status']=='design_not_submitted'
assert summary['actual_train_records']==16 and summary['optimizer_steps']==4
assert summary['supervised_tokens']==329169
assert not summary['full_generation_validated']
account=(r/'slurm-accounting.txt').read_text(encoding='utf-8')
for job in ['63401','63402']:
    assert f'{job}|COMPLETED|0:0|' in account
for job in ['63399','63400','63401','63402']:
    assert (r/f'evidence-{job}/slurm.out').exists()
    assert (r/f'evidence-{job}/slurm.err').exists()
for name in ['prefix_replay','update_writeback','pilot_validation']:
    assert all((r/f'figures/{name}.{ext}').is_file() for ext in ['png','svg'])

checks=dict(design_only=True,paired_pilots_completed=True,post_reload_completed=True,
    failures_retained=True,unique_dev_sentinels=6,figure_groups=3,
    visual_inspection='All three PNG research figures inspected for data ranges, legends and clipping.',
    remaining_formal_entry_gates=config['entry_gates'])
(r/'delivery_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
files=[p for p in sorted(r.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='file_manifest.json']
manifest=[dict(path=str(p.relative_to(r)).replace('\\','/'),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(r/'file_manifest.json').write_text(json.dumps(dict(files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
files.append(r/'file_manifest.json')
archive=r.parent/'next_sft_design_20260912.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for p in files:z.write(p,arcname='next_sft/'+str(p.relative_to(r)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for item in manifest:
        assert hashlib.sha256(z.read('next_sft/'+item['path'])).hexdigest()==item['sha256']
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),all_file_hashes_verified=True)
(r.parent/'next_sft_design_20260912.verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True))
