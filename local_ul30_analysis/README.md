# Local UL30 CPU audit

`audit.py` is a small, model-free audit for the known `R8005_M8009` problem
meeting. It reads the saved Base/B/C prediction JSON files and the frozen
reference JSON. It reports:

- longest same-token run with token-ID context and the existing short-period
  loop detector (`period=1` and periods 2--16);
- EOS, truncation, parse, generated-length, and major-parse-loss fields;
- imported per-record CER/cpCER/DeltaCP and error counts when
  `per_record_metrics.json` is supplied. Use `--base-metrics`, `--b-metrics`,
  and `--c-metrics` when each evaluator wrote its own file. `--metrics` is a
  fallback only; rows must match both key and model uniquely, otherwise the
  report marks metrics unavailable;
- reference-speech interval coverage after subtracting prediction-segment
  coverage, with uncovered intervals, seconds, and ratio;
- prompt ID hash, prompt length, audio, and decoding parity across Base/B/C;
  decoding parity includes recorded dtype and the actual `max_new_tokens` cap;
- text-repeat candidates for review. These are explicitly candidates: natural
  spoken repetition is not classified as an error by this tool.

The coverage quantity is a time-coverage diagnostic only. It does not measure
text correctness, speaker correctness, or diarization quality.

Example using the existing saved artifacts:

```powershell
python dkucc/local_ul30_analysis/audit.py `
  --base dkucc/artifacts/ms-swift-63521/truncated-test-examples/R8005_M8009/base_prediction.json `
  --reference dkucc/artifacts/ms-swift-63643/raw/evaluations/test-150/references.json `
  --metrics dkucc/artifacts/ms-swift-63643/raw/evaluations/test-150/per_record_metrics.json `
  --out dkucc/local_ul30_analysis/validation_base.json
```

For separate B/C evaluator outputs:

```powershell
python dkucc/local_ul30_analysis/audit.py `
  --base path/to/base_prediction.json --b path/to/B_prediction.json --c path/to/C_prediction.json `
  --reference path/to/references.json `
  --base-metrics path/to/base/per_record_metrics.json `
  --b-metrics path/to/B/per_record_metrics.json `
  --c-metrics path/to/C/per_record_metrics.json
```

For the formal UL30 run, B and C should be saved as one prediction JSON per
model and passed with `--b` and `--c`:

Recommended producer contract (the execution agent may choose an equivalent
root, but should keep these three names together for handoff):

```text
<ul30-run>/evaluations/problem-R8005_M8009-step-30/predictions/base.json
<ul30-run>/evaluations/problem-R8005_M8009-step-30/predictions/b.json
<ul30-run>/evaluations/problem-R8005_M8009-step-30/predictions/c.json
<ul30-run>/evaluations/problem-R8005_M8009-step-30/references.json
<ul30-run>/evaluations/problem-R8005_M8009-step-30/per_record_metrics.json  # if scored
```

```powershell
python dkucc/local_ul30_analysis/audit.py `
  --base path/to/base_prediction.json `
  --b path/to/B_prediction.json `
  --c path/to/C_prediction.json `
  --reference path/to/references.json `
  --metrics path/to/per_record_metrics.json `
  --out path/to/ul30_audit.json
```

No model or tokenizer is loaded and no GPU is used. The report's
`input_consistency.all_input_checks_pass` is useful for gating prompt and
decoding parity, but it only covers fields recorded in the prediction JSON.
