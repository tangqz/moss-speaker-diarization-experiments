"""Check state immediately after native FSDP load, before any next update."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,StateDictType,
                                    FullStateDictConfig,FullOptimStateDictConfig)

def watch_rng_restore(trainer,checkpoint,output_dir):
    import random,numpy as np
    original=trainer._load_rng_state
    rank=dist.get_rank()
    def checked(*args,**kwargs):
        result=original(*args,**kwargs)
        expected=torch.load(Path(checkpoint)/f'rng_state_{rank}.pth',map_location='cpu',weights_only=False)
        numpy_state=np.random.get_state()
        np_ok=all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b
                  for a,b in zip(numpy_state,expected['numpy']))
        cuda_expected=expected['cuda']
        cuda_actual=torch.cuda.get_rng_state_all() if isinstance(cuda_expected,list) else torch.cuda.get_rng_state()
        cuda_ok=all(torch.equal(a,b) for a,b in zip(cuda_actual,cuda_expected)) if isinstance(cuda_expected,list) else torch.equal(cuda_actual,cuda_expected)
        report=dict(python=random.getstate()==expected['python'],numpy=np_ok,
                    cpu=torch.equal(torch.get_rng_state(),expected['cpu']),cuda=cuda_ok,rank=rank)
        report['passed']=all(report[k] for k in ['python','numpy','cpu','cuda'])
        Path(output_dir,f'rng_restore_rank{rank}.json').write_text(json.dumps(report,indent=2))
        assert report['passed'],report
        return result
    trainer._load_rng_state=checked

def audit(trainer,checkpoint,output):
    checkpoint=Path(checkpoint);wrapped=trainer.model_wrapped
    optim=getattr(trainer.optimizer,'optimizer',trainer.optimizer)
    with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,
          FullStateDictConfig(offload_to_cpu=True,rank0_only=True),
          FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        weights=wrapped.state_dict()
        optimizer=FSDP.optim_state_dict(wrapped,optim)
    result=None
    if dist.get_rank()==0:
        different=[];keys=[]
        with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as saved:
            assert set(weights)==set(saved.keys()),(set(weights)-set(saved.keys()),set(saved.keys())-set(weights))
            for name,value in weights.items():
                expected=saved.get_tensor(name);keys.append(name)
                if not torch.equal(value,expected):
                    different.append(dict(name=name,max_abs=float((value-expected).abs().max())))
        del weights
        expected_opt=torch.load(checkpoint/'optimizer.bin',map_location='cpu',weights_only=True)
        assert optimizer['param_groups']==expected_opt['param_groups'],'Optimizer groups changed at restore'
        assert set(optimizer['state'])==set(expected_opt['state'])
        opt_diff=[];states=0
        for name,state in optimizer['state'].items():
            for key,value in state.items():
                expected=expected_opt['state'][name][key]
                equal=torch.equal(value,expected) if torch.is_tensor(value) else value==expected
                if not equal:opt_diff.append(dict(name=name,state=key))
            states+=1
        expected_scheduler=torch.load(checkpoint/'scheduler.pt',weights_only=True)
        scheduler_equal=trainer.lr_scheduler.state_dict()==expected_scheduler
        result=dict(passed=not different and not opt_diff and scheduler_equal,
                    model_tensor_count=len(keys),model_differences=different,
                    optimizer_parameter_states=states,optimizer_differences=opt_diff,
                    scheduler_exact=scheduler_equal,checkpoint=str(checkpoint),
                    meaning='state immediately after native load, before first resumed forward')
        Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    status=[result['passed'] if result is not None else None]
    dist.broadcast_object_list(status,src=0)
    assert status[0],'Native FSDP checkpoint did not restore exactly'
    dist.barrier()
