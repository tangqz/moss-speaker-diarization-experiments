"""Resume only unfinished CE/scoring, preserving already saved free generations."""
import argparse,subprocess
from pathlib import Path
from common import ROOT,read,write,sha
from run_eval import workers,HENV,METRIC,HERE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True);a=ap.parse_args()
    items=read(a.pilot/'inputs.json');assert len(items)==2
    before={}
    for label in ['base','sft']:
        for item in items:
            path=a.pilot/'predictions'/label/f"{item['key']}.json"
            record=read(path);assert record['status']=='ok' and not record['engineering_smoke']
            before[str(path)]=sha(path)
    workers('evaluate_ce.py',HENV,['--run',a.pilot,'--checkpoint',a.pilot/'inference_model',
            '--window','128','--verify'],a.pilot,'ce_resumed')
    subprocess.run([str(METRIC),str(HERE/'score_v1.py'),'--run',str(a.pilot)],check=True)
    subprocess.run([str(METRIC),str(HERE/'analysis_contract.py')],check=True)
    subprocess.run([str(METRIC),str(HERE/'protocol_contract.py')],check=True)
    assert all(sha(Path(path))==digest for path,digest in before.items())
    assert read(a.pilot/'metrics_summary.json')['complete']
    write(a.run/'pilot_completion.json',dict(passed=True,pilot=str(a.pilot),meetings=2,
        free_generations_reused_unchanged=before,ce_source_manifest_sha256=sha(HERE/'source_manifest.json'),
        quality_claim=False,meaning='pipeline operation only; original processor API failure retained'))

if __name__=='__main__':main()
