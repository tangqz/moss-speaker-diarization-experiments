# MOSS short recovery v2 sparse

This directory implements `DESIGN.md` without modifying the baseline plugin or pinned Swift source. There is no GPU acceptance result in this checkout. G0-G5 remain pending, and existing D0 diagnostics are not reused as gate evidence.

## Implemented contract

- `core.py` implements numeric timestamp body spans, eligible anchors, exact clean/aux native `loss_scale` budgets, sparse model-history decisions, token transforms, and raw-ID loop scans.
- `prepare.py` verifies canonical train/dev bytes and all audio hashes, rejects path/recording/audio overlap, calls the official MOSS collator, fixes every K-label budget before HF prefetch, and writes ordered `clean,aux` rows plus tokenizer/plan hashes.
- `plugin.py` registers the independent `moss_short_recovery_v2_sparse` template, preserves full audio tensors and original token IDs, and changes only IDs, labels, positions, masks, and loss scale.
- `trainer.py` uses Swift's sized `DataLoaderShard` with an ordered sequence-parallel sampler. Its BF16 `no_grad` probe preserves RNG, module modes, existing gradients, and shared audio; native Swift retains loss, backward, clipping, optimizer, checkpoint, and scheduler.
- `train.py` pins FSDP1 full shard with activation CPU offload and `CELOSS_PARALLEL_SIZE=512`, verifies the effective native denominator `raw/world_size=N`, writes component/event logs, and supports formal milestone stops at 5, 15, and 30.
- `calibrate.py` probes at most 64 uniformly spaced all-body and 64 eligible Dev positions per each of 26 meetings in one bounded forward per meeting. It reports raw/eligible legal greedy error, expected temperature error, Wilson descriptions, and downward-only probability bounds.
- `injected_eval.py` builds one arm-independent fixed Dev library. Sparse substitution can be not applicable when greedy/sample equals gold or fallback occurs; repeat2/3 remain fixed pressure cases. Local generation explicitly uses greedy decoding with repetition penalty 1 and no n-gram ban.

## CPU checks

Run from the `dkucc` directory:

```bash
short_recovery/.venv-test/Scripts/python.exe -m pytest short_recovery/tests -q
python -m short_recovery.prepare tokenizer-audit \
  --manifest analysis/20260915-label-overlap/train.jsonl \
  --tokenizer-json analysis/20260915-label-overlap/tokenizer.json \
  --output short_recovery/artifacts/tokenizer_audit.json
python -m short_recovery.preflight --config short_recovery/config.default.json \
  dry-run --output short_recovery/artifacts/dry_run.json
```

The local tokenizer audit verifies real canonical train/tokenizer bytes. It cannot validate remote audio features, model logits, Swift SP/FSDP behavior, or CUDA memory.

## Cluster preparation order

Use the verified Swift checkout:

```bash
cd /work/qt28/moss/dkucc
export SWIFT_ROOT=/work/qt28/moss/dkucc/ms_swift_20260914/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076
export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:$SWIFT_ROOT:/work/qt28/moss/dkucc
```

For `sub` or `mix`, first run calibration with the proposed conservative probabilities. Then copy its recommended bounds into a new calibrated config; probabilities may only decrease. Build the manifest from that final config, then bind the gate. This order avoids a config/gate hash cycle.

```bash
CUDA_VISIBLE_DEVICES=0 python -m short_recovery.calibrate \
  --config PROPOSED_CONFIG.json --output short_recovery/artifacts/dev_calibration.json
python -m short_recovery.prepare manifest \
  --config CALIBRATED_CONFIG.json --output-dir PLAN_DIR
python -m short_recovery.preflight --config CALIBRATED_CONFIG.json init-gate \
  --manifest-summary PLAN_DIR/manifest_summary.json \
  --output short_recovery/artifacts/gate.json
```

Gate binding includes the final config, canonical splits, initial model tree, manifest, tokenizer/special/legal vocab, target corpus, calibration artifact for sparse arms, pinned upstream commits, and a whitelist hash of runtime code/config files. Artifacts, evidence, tests, logs, and `.venv-test` are excluded from the source hash.

## Bounded engineering evidence

`--engineering-stage` breaks the gate bootstrap loop for G1-G3. It still requires a bound pending gate and all data/source/model/tokenizer/calibration identities, forces debug contracts, writes only below `short-recovery-engineering/`, saves each update, permits one or two updates, and never records a stage pass.

```bash
# G1: use an SP1 config and one process
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m short_recovery.train \
  --config SP1_CONFIG.json --manifest-summary SP1_PLAN/manifest_summary.json \
  --run-id g1-evidence --engineering-stage G1 --stop-at 1 --launch

# G2: use an SP4 config and four processes
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m short_recovery.train \
  --config SP4_CONFIG.json --manifest-summary SP4_PLAN/manifest_summary.json \
  --run-id g2-evidence --engineering-stage G2 --stop-at 1 --launch

# G3: use the separately reviewed longest-meeting evidence plan
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m short_recovery.train \
  --config SP4_CONFIG.json --manifest-summary G3_LONGEST_PLAN/manifest_summary.json \
  --run-id g3-evidence --engineering-stage G3 --stop-at 2 --launch
```

These commands produce runtime/checkpoint logs for review. They do not themselves perform G2 numerical comparison or prove the G3 plan is the longest case. `gpu_checks.py` accepts only a fresh stage harness JSON with every mandatory check, exact bindings, command, and artifacts; `preflight record-stage` applies the same validator.

## Formal milestones

Formal launch requires all G0-G5 stages to pass. The Slurm file launches one task containing one four-process `torchrun`, with FSDP1 and four GPUs. Run three segments and perform the frozen external evaluations after each stop:

```bash
export RECOVERY_CONFIG=/work/qt28/moss/dkucc/short_recovery/config.calibrated.json
export RECOVERY_MANIFEST_SUMMARY=/work/qt28/moss/dkucc/short_recovery/plan/manifest_summary.json
export RECOVERY_RUN_ID=v2-sparse-main
export RECOVERY_LAUNCH=1
RECOVERY_STOP_AT=5  sbatch short_recovery/train.slurm
# Run clean Dev CE and full Dev generation/scoring for milestone 5.
RECOVERY_STOP_AT=15 RECOVERY_RESUME=.../checkpoint-5 sbatch short_recovery/train.slurm
# Run the milestone 15 evaluations.
RECOVERY_STOP_AT=30 RECOVERY_RESUME=.../checkpoint-15 sbatch short_recovery/train.slurm
# Run the milestone 30 evaluations.
```

The repository does not automatically orchestrate clean Dev CE or full official vLLM generation at 5/15/30. Each external result must preserve all 26 Dev rows, parser failures, raw token IDs, model/config/source hashes, and checkpoint identity.

## Fixed injected evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m short_recovery.injected_eval build-library \
  --config CALIBRATED_CONFIG.json --output FIXED_DEV.json
CUDA_VISIBLE_DEVICES=0 python -m short_recovery.injected_eval run-library \
  --config ARM_CONFIG.json --library FIXED_DEV.json --checkpoint CHECKPOINT \
  --output injected_predictions.jsonl
python -m short_recovery.evaluate scan \
  --predictions injected_predictions.jsonl --output injected_diagnostics.json
```

The 256-token injected continuation is a local censored horizon. Its body edit S/D/I counts include window-boundary effects and are not full-meeting CER. Full CER/cpCER/DER025, early-EOS completion, and structural time progress require the official complete-Dev evaluation.

## Pending validation

- G0 official remote collator/audio identity and EOS preservation.
- G1 SP1 probe/native-CE and gradient/RNG/mode parity.
- G2 SP4/FSDP1 loss/gradient, shard-boundary, zero-label-rank, and denominator parity. The numerical comparison harness is not implemented here.
- G3 reviewed longest-meeting-plus-two plan, full prefetch window, native checkpoint, and peak memory at most 42 GiB.
- G4 continuous/resume equivalence and HF/vLLM prefix parity; no automatic cross-backend harness is included.
- G5 shared initialization/order/budget, fixed library readiness, clean Dev CE, and complete official Dev generation; no full-Dev vLLM orchestrator is included.
- Dev calibration records every meeting and global all-body/eligible totals, but does not aggregate results by corpus.

Thirty mix updates expose at most 60 repeat slots and expect about three insertions at probability 0.05. Any event type with fewer than 20 realized cases supports only an engineering-pilot conclusion.
