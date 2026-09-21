#!/usr/bin/env bash
set -euo pipefail
OLD_JOB=${1:?old job id}
RUN=${2:?existing run}
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
while ! python3 - "$RUN/selection.json" <<'PY'
import json,sys
try:
    data=json.load(open(sys.argv[1]))
    ok=[row['step'] for row in data['history']][-1] == 20
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
do
    sleep 5
done
cp "$RUN/job_exit.json" "$RUN/job_exit_before_resume.json" 2>/dev/null || true
scancel "$OLD_JOB"
NEW_JOB=$(sbatch --parsable --export=ALL,MOSS_RESUME_RUN="$RUN",MOSS_RESUME_STEP=20 "$TASK/resume_10step.slurm")
python3 - "$RUN" "$NEW_JOB" "$OLD_JOB" "$TASK/current_run.json" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
run,new,old,current=Path(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),Path(sys.argv[4])
payload={'old_job':old,'resume_job':new,'run':str(run),'start_step':20,
         'dev_interval':10,'utc':datetime.now(timezone.utc).isoformat()}
(run/'resume_dispatch.json').write_text(json.dumps(payload,indent=2))
state=json.loads(current.read_text())
state['training_job']=new
state['formal_remote_run']=str(run)
state['remote_run']=str(run)
state['resumed_from_job']=old
state['resume_start_step']=20
state['dev_interval_after_step20']=10
current.write_text(json.dumps(state,indent=2))
print(json.dumps(payload),flush=True)
PY
