# Recovery checkpoint generation evaluation

## Requested protocol

- Checkpoint: `short-recovery-sr-v2-sub-150-20260916-r3/sub/seed-0/checkpoint-150`.
- First evaluate the complete 1776.99-second meeting `alimeeting/test/R8005_M8009`.
- Greedy decoding: temperature 0, repetition penalty 1.0, presence/frequency penalties 0.
- Preserve the previous inference processor, parser, EOS and 65,536 output-token cap.
- Inspect raw token repetitions, truncation, parser results, coverage beyond the former 984-second failure, and the meeting ending near 1773 seconds.
- Compare CER/cpCER and deletion counts with the previous checkpoint. Do not infer successful recovery solely from EOS or training loss.
- Start the full 56-meeting evaluation only after reviewing this sample. `run.py` requires a passing `review_decision.json` bound to the problem report's SHA-256.

## Jobs

- 63719: initial full evaluation cancelled at the user's request before any new predictions.
- 63720: initial single-meeting attempt failed during engine startup because the allocated device had insufficient free memory; no prediction produced.
- 63721: completed single-meeting retry on idle node `dkucc-core-gpu-dkurc-d-it09-8`; review failed due to 48,237 consecutive 外 tokens and output-cap truncation. Full evaluation was not started. See `problem-63721/RESULTS.md`.

`current_run.json` identifies the latest job and remote paths. `monitor_current.ps1` follows that file. Evaluation events are mirrored into the existing local TensorBoard on port 6006.

## Scripts

- `problem.py` and `problem.slurm`: full-length problem meeting, reused historical Base prediction, standard metrics, raw diagnostics and review report.
- `run.py` and `evaluate.slurm`: gated full evaluation with 56 newly generated predictions and 56 verified historical Base predictions.
- The existing validated vLLM generator and metric implementation remain the source of decoding and scoring behavior.

The cancelled, failed and successful attempts retain separate output directories.
