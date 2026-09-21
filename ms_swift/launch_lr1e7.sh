#!/usr/bin/env bash
set -euo pipefail

OLD_TASK=/work/qt28/moss/dkucc/ms_swift_20260914
NEW_TASK=/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915
GATE=/work/qt28/moss/results/ms-swift-fsdp-check-63506
RESUME_GATE=/work/qt28/moss/results/ms-swift-resume-63512
VLLM_GATE=/work/qt28/moss/results/vllm-check-63520

test -d "$OLD_TASK"
test -f "$GATE/outcome.json"
test -f "$RESUME_GATE/resume_check.json"
test -f "$VLLM_GATE/vllm_check.json"
if test -e "$NEW_TASK"; then
  echo "Refusing to overwrite existing task: $NEW_TASK" >&2
  exit 2
fi

cp -a "$OLD_TASK" "$NEW_TASK"
python3 - "$OLD_TASK" "$NEW_TASK" <<'PY'
import json
import sys
from pathlib import Path

old = sys.argv[1]
root = Path(sys.argv[2])

def replace_exact(name, before, after, expected=1):
    path = root / name
    text = path.read_text()
    count = text.count(before)
    if count != expected:
        raise RuntimeError(f'{name}: expected {expected} occurrence(s) of {before!r}, found {count}')
    path.write_text(text.replace(before, after))

replace_exact('native_sft.sh', '--learning_rate 1e-6', '--learning_rate 1e-7')
replace_exact('pipeline.py', 'initial_lr=1e-6', 'initial_lr=1e-7')
for name in ['train.slurm', 'resume_10step.slurm', 'switch_after_step20.sh']:
    replace_exact(name, f'TASK={old}', f'TASK={root}')

# Documentation travels with the submitted source and records the sole
# optimization change. These replacements do not affect execution.
readme = root / 'README.md'
text = readme.read_text()
text = text.replace('| 学习率 | 1e-6；沿用原 linear scheduler、402 步衰减尺度、无 warmup |',
                    '| 学习率 | 1e-7；沿用原 linear scheduler、402 步衰减尺度、无 warmup |')
text = text.replace('学习率等沿用已有配置，目的是先观察框架迁移的影响，不声称 1e-6 已被证明最优。',
                    '除初始学习率从 1e-6 改为 1e-7 外，其余训练与评估配置沿用正式对照运行。')
readme.write_text(text)

(root / 'lr1e7_change.json').write_text(json.dumps({
    'comparison_run': '/work/qt28/moss/results/ms-swift-63521',
    'sole_model_optimization_change': {'learning_rate': {'from': 1e-6, 'to': 1e-7}},
    'unchanged': [
        'base_model', 'train_and_dev_data', 'full_parameter_update',
        'full_causal_attention', 'sequence_parallel_size_4', 'fsdp1_cpu_offload',
        'bf16_autocast_fp32_master_parameters', 'optimizer', 'linear_scheduler_402_steps',
        'zero_warmup', 'global_batch_4', 'seeds', 'checkpoint_interval',
        'native_train_and_dev_ce', 'vllm_greedy_dev_generation',
        'selection_and_catastrophic_stop_rules', 'evaluation_schedule_0_5_10_15_20_30_40_50'
    ]
}, indent=2))
PY

test "$(grep -c -- '--learning_rate 1e-7' "$NEW_TASK/native_sft.sh")" -eq 1
test "$(grep -c -- '--learning_rate 1e-6' "$NEW_TASK/native_sft.sh")" -eq 0
python3 -m py_compile "$NEW_TASK/pipeline.py" "$NEW_TASK/resume_10step.py" "$NEW_TASK/runtime_callback.py"
bash -n "$NEW_TASK/native_sft.sh" "$NEW_TASK/train.slurm" \
  "$NEW_TASK/resume_10step.slurm" "$NEW_TASK/switch_after_step20.sh"

JOB=$(sbatch --parsable \
  --export=ALL,MOSS_GATE_RUN="$GATE",MOSS_RESUME_GATE="$RESUME_GATE",MOSS_VLLM_GATE="$VLLM_GATE" \
  "$NEW_TASK/train.slurm")
RUN=/work/qt28/moss/results/ms-swift-$JOB
mkdir -p "$RUN/logs"
nohup bash "$NEW_TASK/switch_after_step20.sh" "$JOB" "$RUN" \
  > "$RUN/logs/switch-after-step20.log" 2>&1 < /dev/null &
WATCHER=$!
printf '%s\n' "$WATCHER" > "$RUN/switch-after-step20.pid"

python3 - "$OLD_TASK/current_run.json" "$NEW_TASK/current_run.json" "$NEW_TASK" "$RUN" "$JOB" "$WATCHER" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

old_state, new_state, task, run, job, watcher = sys.argv[1:]
state = {
    'active_task': 'native_ms_swift_full_attention_lr1e7_control',
    'stage': 'submitted',
    'training_job': int(job),
    'remote_run': run,
    'formal_remote_run': run,
    'remote_task': task,
    'remote_environment': '/work/qt28/moss/envs/ms-swift-20260914',
    'comparison_run': '/work/qt28/moss/results/ms-swift-63521',
    'initial_learning_rate': 1e-7,
    'sole_model_optimization_change': 'learning_rate_1e-6_to_1e-7',
    'sequence_parallel_size': 4,
    'dev_schedule': [0, 5, 10, 15, 20, 30, 40, 50],
    'full_dataset_training_started': False,
    'follow_remote_state': True,
    'watcher_pid': int(watcher),
    'submitted_utc': datetime.now(timezone.utc).isoformat(),
}
payload = json.dumps(state, ensure_ascii=False, indent=2)
Path(old_state).write_text(payload)
Path(new_state).write_text(payload)
Path(run, 'launch_receipt.json').write_text(payload)
PY

echo "JOB=$JOB"
echo "RUN=$RUN"
echo "TASK=$NEW_TASK"
echo "WATCHER=$WATCHER"
scontrol show job -o "$JOB" | sed -n '1p'
diff -u "$OLD_TASK/native_sft.sh" "$NEW_TASK/native_sft.sh" || true
