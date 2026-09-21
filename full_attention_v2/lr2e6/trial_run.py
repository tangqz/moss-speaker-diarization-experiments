"""Independent 2e-6 full-attention trial: 10 -> probe -> 25 -> 50 -> review."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback
from common import ROOT, BASE, TRAIN, HERE, OLD_EVAL, read, write, sha, now
from run import status, execute, fast, prepare, workers, model_manifest, verify_actual_order, METRIC_PY

PARENT = ROOT/'results/full-attention-v2-63421'
GATE = ROOT/'results/full-attention-v2-audit-63414'
TARGET = 'ami/dev/TS3004c'


def assess_complete(summary, quality, expected):
    assert summary['complete'] and summary['scored_predictions']==2*expected
    assert summary['expected_predictions']==2*expected
    assert not summary['missing'] and not summary['errors']
    groups=[v for k,v in summary['groups'].items() if k.startswith('sft/')]
    assert sum(g['completed'] for g in groups)==expected
    assert all(g['complete'] for g in groups)
    return not (any(g['truncated'] or g['empty_predictions'] for g in groups) or quality['major_parse_loss'])


def verify_lineage(run):
    manifest=read(HERE/'source_manifest.json')
    for name,digest in manifest.items():
        assert sha(HERE/name)==digest,name
    lineage=read(HERE/'lineage.json')
    for name,digest in lineage['original_hashes'].items():
        assert sha(PARENT/'source'/name)==digest,name
    gate=read(GATE/'entry_gates_passed.json')
    assert gate['passed']
    # Reuse measured algorithm/precision qualification only for unchanged code;
    # the new LR is independently asserted at every live optimizer update.
    for name in ['common.py','memory_sft.py','preflight.json','dev_inputs.json','sentinel_manifest.json']:
        assert manifest[name]==gate['source_manifest'][name],name
    old_train=(PARENT/'source/train.py').read_text()
    assert old_train.count('1e-5')==3
    assert (HERE/'train.py').read_text()==old_train.replace('1e-5','2e-6')
    assert sha(HERE/'production_generate.py')==lineage['production_generate_sha256']
    corrected=read(ROOT/'results/full-attention-v2-diagnostic-63441/outcome.json')
    assert corrected['status']=='completed' and corrected['base_recheck']['all_generated_tokens_equal']
    environment=read(GATE/'environment.json')
    for path,digest in environment['hashes'].items():
        assert sha(path)==digest,path
    design=read(HERE/'design_protocol.json')
    assert design['training']['learning_rate']==2e-6
    assert design['training']['scheduler_horizon_steps']==402
    assert design['training']['warmup_steps']==0 and design['training']['max_steps']==402
    assert sha(TRAIN)==design['data']['sha256']
    assert sha(BASE/'model-00000-of-00001.safetensors')==design['initialization']['weights_sha256']
    previous_base=read(ROOT/'checkpoints/full-attention-v2-63421/step-0-base.json')
    assert model_manifest(BASE)==previous_base,'Base config or model source changed'
    write(run/'entry_lineage_verified.json',dict(utc=now(),passed=True,parent_job=63421,
        qualification_job=63414,source_manifest=manifest,
        note='Prior precision/algorithm qualification reused with exact derivative audit; not a claim that the new LR was already trained.'))


def prepare_probe(folder, checkpoint):
    item=dict(next(r for r in read(HERE/'dev_inputs.json') if r['key']==TARGET),rank=0)
    folder.mkdir(parents=True,exist_ok=True)
    write(folder/'inputs.json',[item])
    write(folder/'references.json',{TARGET:read(OLD_EVAL/'references.json')[TARGET]})
    write(folder/'model_manifest.json',model_manifest(checkpoint))
    dest=folder/'predictions/base'/(TARGET+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD_EVAL/'predictions/base'/(TARGET+'.json'),dest)


def generate_and_score(run, checkpoint, step, scope):
    folder=run/'evaluations'/f'{scope}-{step}'
    status(run,'full_generation',step=step,scope=scope,checkpoint=str(checkpoint),learning_rate=2e-6)
    if scope=='probe':
        prepare_probe(folder,checkpoint)
    else:
        assert scope=='sentinel' and step==50
        prepare(folder,checkpoint,'sentinel')
    write(folder/'protocol.json',dict(scope=scope,step=step,learning_rate=2e-6,
        audio_preprocessing='production_cuda_bfloat16_autocast',attention='full_causal_sdpa',
        source_manifest=read(HERE/'source_manifest.json'),selection_performed=False,
        base_comparator='Historical production base reused for quality only; full TS3004c identity verified in 63441.'))
    if not (folder/'scoring_complete.json').exists():
        if scope=='probe':
            execute(run,[sys.executable,HERE/'production_generate.py','--run',folder,
                         '--checkpoint',checkpoint,'--rank','0'],f'probe-{step}-rank-0')
        else:
            workers(run,folder,'production_generate.py',checkpoint,f'sentinel-{step}')
        execute(run,[METRIC_PY,HERE/'selection.py','--run',folder],f'score-{scope}-{step}')
        summary=read(folder/'metrics_summary.json')
        quality=read(folder/'quality_status.json')
        passed=assess_complete(summary,quality,1 if scope=='probe' else 6)
        write(folder/'scoring_complete.json',dict(complete=True,quality_pass=passed,utc=now()))
    passed=assess_complete(read(folder/'metrics_summary.json'),read(folder/'quality_status.json'),
                           1 if scope=='probe' else 6)
    return folder,passed


def verify_training(run, checkpoints, through):
    verify_actual_order(run,checkpoints,through)
    for rank in range(4):
        candidate=[json.loads(line) for line in (checkpoints/f'train-rank-{rank}.jsonl').read_text().splitlines()]
        original=[json.loads(line) for line in (ROOT/'checkpoints/full-attention-v2-63421'/f'train-rank-{rank}.jsonl').read_text().splitlines()]
        for row,base in zip(candidate,original[:through]):
            assert row['step']==base['step']
            assert abs(row['lr']-base['lr']/5)<1e-12
            m,b=row['microbatches'][0],base['microbatches'][0]
            for name in ['indices','sequence_tokens','local_supervised_tokens','global_supervised_tokens']:
                assert m[name]==b[name],(rank,row['step'],name)
    for step in [10,25,50]:
        if step>through:continue
        folder=checkpoints/f'checkpoint-{step}'
        assert read(folder/'checkpoint_complete.json')['complete']
        required=['optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{rank}.pth' for rank in range(4)]
        assert all((folder/name).is_file() for name in required)
    evidence=read(run/'actual_data_order_verified.json')
    if through==50:assert evidence['supervised_tokens']==4076380
    write(run/'matched_exposure_verified.json',dict(through_step=through,utc=now(),
        original_training_job=63421,all_batch_indices_and_token_counts_equal=True,
        learning_rate_ratio=.2,supervised_tokens=evidence['supervised_tokens']))


def retain_early_weights(checkpoints):
    for step in [1,5]:
        folder=(checkpoints/f'checkpoint-{step}').resolve()
        assert folder.is_relative_to(checkpoints.resolve())
        assert read(folder/'checkpoint_complete.json')['complete']
        removed=[]
        for name in ['optimizer.pt','scheduler.pt']+[f'rng_state_{r}.pth' for r in range(4)]:
            path=(folder/name).resolve()
            assert path.parent==folder and path.is_relative_to(checkpoints.resolve())
            if path.is_file():path.unlink();removed.append(name)
        write(folder/'retention.json',dict(weights_preserved=True,full_resume_state=False,
            removed=removed,reason='Short-trial plan retains full states at 10/25/50 and weights at 1/5.'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,required=True)
    a=ap.parse_args();run=a.run;checkpoints=a.checkpoints
    try:
        if (run/'outcome.json').exists():
            print('This short trial already stopped for review; no further updates are authorized by this runner.',flush=True)
            return
        status(run,'validating_execution_bundle',learning_rate=2e-6)
        verify_lineage(run)
        assert shutil.disk_usage(checkpoints.parent).free>=110*2**30
        write(run/'execution_contract.json',dict(initialization=str(BASE),learning_rate=2e-6,
            attention='full_causal_sdpa',all_parameters_trainable=True,schedule_horizon=402,
            phase_boundaries=[10,25,50],stop_after=50,comparison_training_job=63421,
            corrected_comparison_job=63441,full_dev_performed=False,test_performed=False))
        execute(run,[sys.executable,HERE/'check_device.py','--run',run],'allocation-preflight')
        execute(run,[sys.executable,HERE/'trial_checks.py'],'trial-gate-checks')
        if not (run/'data_order.json').exists():
            execute(run,[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                         HERE/'sampler_gate.py','--run',run],'shuffled-sampler-gate')
        assert read(run/'data_order.json')['epochs']==read(PARENT/'data_order.json')['epochs']
        assert 10 in read(run/'data_order.json')['verified_resume_steps']
        base_source=PARENT/'evaluations/fast-0'
        base_summary=read(base_source/'fast_summary.json')
        assert base_summary['complete'] and len(base_summary['records'])==6
        if not (run/'evaluations/fast-0').exists():
            shutil.copytree(base_source,run/'evaluations/fast-0')
        assert sha(run/'evaluations/fast-0/fast_summary.json')==sha(base_source/'fast_summary.json')
        write(run/'inherited_baselines.json',dict(teacher_step0_from=str(base_source),
            summary_sha256=sha(base_source/'fast_summary.json'),
            note='Existing unchanged teacher protocol and base reused; not a new measurement or production generation.'))
        checkpoints.mkdir(parents=True,exist_ok=True)
        write(checkpoints/'step-0-base.json',model_manifest(BASE))
        previous=None
        for step in [10,25,50]:
            status(run,'training',step_target=step,resume=str(previous) if previous else None,
                learning_rate=2e-6,schedule_horizon=402,checkpoints=str(checkpoints))
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                     HERE/'host_runtime.py','--output',checkpoints,'--stop',str(step)]
            if previous:command+=['--resume',previous]
            if not (checkpoints/f'phase-{step}-complete.json').exists():
                expected_last=0 if previous is None else int(previous.name.split('-')[-1])
                for rank in range(4):
                    log=checkpoints/f'train-rank-{rank}.jsonl'
                    if log.exists():
                        observed=[json.loads(line)['step'] for line in log.read_text().splitlines()]
                        assert observed==list(range(1,expected_last+1)), 'Partial phase requires checkpoint recovery and archival of later logs before retry.'
                execute(run,command,f'train-to-{step}')
            checkpoint=checkpoints/f'checkpoint-{step}'
            assert read(checkpoints/f'phase-{step}-complete.json')['step']==step
            verify_training(run,checkpoints,step)
            model_manifest(checkpoint)
            if step==10:
                retain_early_weights(checkpoints)
                folder,passed=generate_and_score(run,checkpoint,step,'probe')
                if not passed:
                    status(run,'paused_for_probe_failure',step=step,evaluation=str(folder),learning_rate=2e-6)
                    write(run/'outcome.json',dict(status='paused_for_probe_failure',step=step,
                        learning_rate=2e-6,quality_pass=False,full_dev_performed=False,test_performed=False,
                        completed_utc=now(),reason='Complete step-10 TS3004c failed the frozen catastrophic-output gate.'))
                    return
            if step in [25,50]:fast(run,checkpoint,step)
            if step==50:
                folder,passed=generate_and_score(run,checkpoint,step,'sentinel')
                stage='completed_short_trial' if passed else 'paused_for_sentinel_failure'
                status(run,stage,step=step,evaluation=str(folder),learning_rate=2e-6)
                write(run/'outcome.json',dict(status=stage,step=step,learning_rate=2e-6,
                    quality_pass=passed,full_dev_performed=False,test_performed=False,selected_step=None,
                    completed_utc=now(),reason='Short trial ends at 50 for review; no formal selection or automatic continuation.'))
                return
            previous=checkpoint
    except Exception as exc:
        write(run/'failure.json',dict(utc=now(),type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        status(run,'failed',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
