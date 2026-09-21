"""Compact audit view; numerical qualification remains explicit and reviewable."""
import argparse,json,math
from pathlib import Path
from common import read,write

def summarize(run):
    reports={}
    for path in sorted(Path(run).glob('*/hf*alignment.json')):
        data=read(path);checks=data['token_checks'];records=data['records']
        value=dict(prefix=data['prefix'],tokens=data['generated_tokens'],window=data['window'],
            original_fixed_tolerance_passed=data['passed'],reference_unchanged=data['all_layer_reference_unchanged'],
            top1_agreement=sum(x['hf_top1']==x['vllm_top1'] for x in checks)/len(checks),
            high_margin_disagreements=[x for x in checks if x['high_margin_disagreement']],
            raw_error_maxima={key:max(x[key] for x in records) for key in ['max_abs','mean_abs','p99']})
        if data['window'] is not None:
            expected=data['prefix']+data['window']
            value['hf_storage_bounded']=all(s['capacity']==s['occupied']==expected and
                s['storage_bytes']==expected*8*128*2*2 and
                s['logical_length']==data['prefix']+data['generated_tokens']-1 for s in data['hf_cache'])
        blocks=path.parent/'kv_blocks.jsonl'
        if blocks.exists():
            rows=[json.loads(x) for x in blocks.read_text().splitlines() if x]
            rows=[r for r in rows if r['prefix_tokens'] is not None]
            if rows:
                value['maximum_live_kv_pages']=max(r['physical_blocks'] for r in rows)
                value['maximum_logical_kv_pages']=max(r['logical_blocks'] for r in rows)
                if data['window'] is not None:
                    value['vllm_pages_bounded']=all(r['physical_blocks']<=math.ceil(r['prefix_tokens']/r['block_size'])+
                        math.ceil(r['window']/r['block_size'])+2 for r in rows)
                    value['vllm_page_bound_note']='ceil(P/page)+ceil(k/page)+2; includes rounded pages and one current-write boundary'
        reports[str(path.relative_to(run))]=value
    report=dict(records=reports,qualification='requires explicit numerical review; original tolerance failures preserved')
    write(Path(run)/'compact_audit.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();summarize(a.run)
