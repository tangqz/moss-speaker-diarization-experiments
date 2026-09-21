"""Verify actual forwards, excluding dataloader prefetch/collation."""
import collections,json,time
from pathlib import Path
from common import ROOT,write

def prepare(training,completed_step):
    path=Path(training)/'sample_order.jsonl'
    if not path.exists():
        assert completed_step==0
    for current in ([path] if path.exists() else [])+sorted(path.parent.glob('sample_order.rank*.jsonl')):
        lines=current.read_text().splitlines();keep=4*completed_step
        assert len(lines)>=keep
        if len(lines)>keep:
            backup=current.with_name(f'abandoned_{current.stem}_{time.time_ns()}.jsonl')
            backup.write_text('\n'.join(lines[keep:])+'\n')
            current.write_text('\n'.join(lines[:keep])+('\n' if keep else ''))

def verify(training,step):
    training=Path(training)
    records=[json.loads(line) for line in (training/'sample_order.jsonl').read_text().splitlines() if line]
    assert len(records)==4*step,('Unexpected actual microbatch count',len(records),step)
    for rank in range(1,4):
        other=[json.loads(line) for line in (training/f'sample_order.rank{rank}.jsonl').read_text().splitlines() if line]
        assert records==other,('Sequence-parallel ranks saw different meetings',rank)
    for index,record in enumerate(records):
        assert record['optimizer_step']==index//4+1 and record['microbatch']==index%4+1,record
        assert 0<record['prefix']<record['length']<=131072,record
    manifest=ROOT/'dkucc/rswa_20260920/data/train.jsonl'
    expected=[json.loads(line)['audios'][0] for line in manifest.read_text().splitlines() if line]
    assert len(expected)==536 and len(set(expected))==536
    assert all(r['audio'] in set(expected) for r in records),'Unexpected training meeting'
    epochs=[]
    for epoch in range(len(records)//536):
        observed=[r['audio'] for r in records[epoch*536:(epoch+1)*536]]
        assert collections.Counter(observed)==collections.Counter(expected),('Missing or duplicate epoch meetings',epoch)
        epochs.append(dict(epoch=epoch,actual_meetings=536,unique_meetings=536))
    partial=[r['audio'] for r in records[(len(records)//536)*536:]]
    assert len(set(partial))==len(partial),'Duplicate meeting in incomplete epoch'
    result=dict(passed=True,optimizer_steps=step,actual_meetings=len(records),all_four_rank_orders_equal=True,complete_epochs=epochs,
        partial_epoch_meetings=len(records)%536,log_semantics='actual outer-model training forwards, not collation')
    write(training/'order_audit.json',result);return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True);ap.add_argument('--step',type=int,required=True)
    args=ap.parse_args();print(json.dumps(verify(args.training,args.step)),flush=True)
