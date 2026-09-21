"""Release old optimizer copies only inside this campaign after state commit."""
import re
from pathlib import Path
from common import read,write,MILESTONES

def prune(root,state):
    root=Path(root).resolve();training=(root/'training').resolve()
    assert training.is_relative_to(root) and training.name=='training'
    removed=[]
    for key,g in state['groups'].items():
        assert re.fullmatch(r'(full|128|256)-seed[01]',key)
        directory=(training/key).resolve();assert directory.is_relative_to(training)
        keep=set(MILESTONES)|{5,g['step'],max(0,(g['step']-1)//10*10)}
        for checkpoint in directory.glob('checkpoint-*'):
            if not checkpoint.is_dir() or checkpoint.is_symlink():continue
            match=re.fullmatch(r'checkpoint-(\d+)',checkpoint.name)
            if not match or int(match[1]) in keep:continue
            assert checkpoint.resolve().is_relative_to(directory)
            for name in ['optimizer.bin','scheduler.pt','pytorch_model_fsdp.bin']:
                path=checkpoint/name
                if path.exists():
                    assert not path.is_symlink() and path.resolve().is_relative_to(directory)
                    removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
            if removed:write(checkpoint/'retention.json',dict(full_training_resume_retained=False,
                inference_weights_retained=True,reason='old nonmilestone; current, preceding, step5 and all milestones preserved'))
    if removed:
        file=root/'retention_log.json';old=read(file) if file.exists() else []
        write(file,old+removed)
