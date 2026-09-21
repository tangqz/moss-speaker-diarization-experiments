"""Exact model/scheduler replay audit after native FSDP checkpoint recovery."""
import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open

def main():
    ap=argparse.ArgumentParser();ap.add_argument('continuous',type=Path);ap.add_argument('resumed',type=Path)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(4)
    stats=[]
    with safe_open(a.continuous/'model.safetensors',framework='pt',device='cpu') as x, \
         safe_open(a.resumed/'model.safetensors',framework='pt',device='cpu') as y:
        assert set(x.keys())==set(y.keys())
        for name in x.keys():
            t,u=x.get_tensor(name),y.get_tensor(name)
            error=float((t-u).abs().max())
            stats.append(dict(name=name,maximum_error=error,identical=torch.equal(t,u)))
    for directory in [a.continuous,a.resumed]:
        state=json.loads((directory/'trainer_state.json').read_text());assert state['global_step']==2
    sx=torch.load(a.continuous/'scheduler.pt',weights_only=True)
    sy=torch.load(a.resumed/'scheduler.pt',weights_only=True)
    assert sx==sy,(sx,sy)
    ox=torch.load(a.continuous/'optimizer.bin',map_location='cpu',weights_only=True)
    oy=torch.load(a.resumed/'optimizer.bin',map_location='cpu',weights_only=True)
    assert ox['param_groups']==oy['param_groups']
    assert set(ox['state'])==set(oy['state'])
    square_diff=0.;square_norm=0.;optimizer_steps=[]
    for key,state in ox['state'].items():
        other=oy['state'][key]
        assert float(state['step'])==float(other['step'])==2
        optimizer_steps.append(float(other['step']))
        for moment in ['exp_avg','exp_avg_sq']:
            t,u=state[moment],other[moment]
            square_diff+=float((t-u).double().square().sum())
            square_norm+=float(t.double().square().sum())
    optimizer_relative_l2=(square_diff/max(square_norm,1e-30))**.5
    # A tiny tolerance permits nondeterministic collective arithmetic. Updates
    # are only 1e-7, so larger deviations cannot be waved away as "close".
    maximum=max(r['maximum_error'] for r in stats)
    report=dict(passed=maximum<=2e-7 and optimizer_relative_l2<=1e-4,maximum_parameter_error=maximum,
                exact_tensors=sum(r['identical'] for r in stats),total_tensors=len(stats),
                scheduler_identical=True,scheduler=sx,optimizer_relative_l2=optimizer_relative_l2,
                optimizer_states_at_step2=len(optimizer_steps),different=[r for r in stats if not r['identical']])
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='different'}),flush=True)
    assert report['passed'],'Native save/resume differs from uninterrupted updates'

if __name__=='__main__':main()
