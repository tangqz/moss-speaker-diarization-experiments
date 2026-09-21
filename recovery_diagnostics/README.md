# D0/D1 recovery diagnostics

This package begins the no-retraining diagnostics for the `R8005_M8009` repetition failure.

## Scope

- D0 compares Transformers BF16, vLLM BF16, and a bounded Transformers FP32 replay.
- D0 repeats each vLLM case twice and records the first raw-token divergence across paths.
- D1 replays identical unfinished assistant histories against Base, Step 60, Step 90, and Step 150.
- Controlled histories append actual token ID `47815` at lengths 0, 8, 32, 64, and 128.
- Each Transformers decode records the repeated-token logit margin, exit length, EOS, parsed progress, and raw continuation.

`R8005_M8009` is a known Test failure and is diagnostic-only. It must not be used for training, checkpoint selection, or a claim of fresh blind-test improvement.

## Evidence boundaries

- Exiting token `47815` does not alone establish recovery. The continuation must also advance valid transcription rather than switch loops, jump timestamps, or end early.
- No zero-update Step 0 checkpoint exists in the run. The report marks that comparison unavailable instead of silently substituting Base.
- FP32 is intentionally bounded to D0 cases on Base and Step 90. If the long-context path cannot run within A40 memory, the failure is retained as an explicit result.
- These jobs do not train, edit checkpoints, change CE, or use decoding penalties.

## Output

Remote results are written under:

`/work/qt28/moss/results/ms-swift-63643/evaluations/recovery-diagnostics-d0-d1`

The final archive is:

`/work/qt28/moss/results/ms-swift-63643/recovery-diagnostics-d0-d1.tar.gz`
