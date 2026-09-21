"""One-shot server-side install -> vLLM gate -> native training dispatch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
TASK=Path(__file__).resolve().parent
sys.path.insert(0,str(TASK/'evaluation'))
from common import read,write,now


def status(stage,**details):
    payload=dict(stage=stage,utc=now(),**details)
    write(TASK/'vllm_dispatch.json',payload)
    print(json.dumps(payload),flush=True)


def main():
    status('waiting_for_vllm_install')
    deadline=time.monotonic()+3600
    while 'VLLM_ENV_READY' not in (TASK/'bootstrap_vllm.log').read_text(errors='replace'):
        if time.monotonic()>deadline:
            raise TimeoutError('vLLM environment not ready after one hour')
        time.sleep(30)
    assert read('/work/qt28/moss/results/ms-swift-resume-63512/resume_check.json')['native_resume_and_eval_passed']
    job=subprocess.check_output(['sbatch','--parsable',str(TASK/'vllm_check.slurm')],text=True).strip()
    assert job.isdigit(),job
    gate=Path('/work/qt28/moss/results')/f'vllm-check-{job}'
    status('vllm_gate_submitted',validation_job=int(job),validation_run=str(gate))
    deadline=time.monotonic()+3*3600
    while True:
        state=subprocess.check_output(['sacct','-j',job,'-n','-X','-o','State'],text=True).strip()
        if state=='COMPLETED':
            break
        if any(f in state for f in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']):
            raise RuntimeError(f'vLLM gate {job}: {state}; see {gate}')
        if time.monotonic()>deadline:
            raise TimeoutError(f'vLLM gate {job} has not completed within three hours')
        time.sleep(30)
    assert read(gate/'vllm_check.json')['loading_and_long_audio_passed']
    env=dict(os.environ,MOSS_GATE_RUN='/work/qt28/moss/results/ms-swift-fsdp-check-63506',
             MOSS_RESUME_GATE='/work/qt28/moss/results/ms-swift-resume-63512',MOSS_VLLM_GATE=str(gate))
    training=subprocess.check_output(['sbatch','--parsable',str(TASK/'train.slurm')],env=env,text=True).strip()
    assert training.isdigit(),training
    state=read(TASK/'current_run.json')
    state.update(stage='vllm_enabled_training_submitted',training_job=int(training),
        formal_remote_run=f'/work/qt28/moss/results/ms-swift-{training}',
        remote_run=f'/work/qt28/moss/results/ms-swift-{training}',
        vllm_validation_job=int(job),vllm_validation_passed=True,formal_submission='submitted_after_all_gates_passed')
    write(TASK/'current_run.json',state)
    status('training_submitted',validation_job=int(job),validation_run=str(gate),
           training_job=int(training),remote_run=state['remote_run'])


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('dispatch_failed',error=str(exc),traceback=traceback.format_exc())
        raise
