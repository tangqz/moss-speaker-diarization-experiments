# Decoding intervention pipeline

This pipeline runs after the D0/D1 recovery diagnostic and before any retraining.

1. Run only greedy `repetition_penalty` values 1.00, 1.02, 1.05, and 1.10 on `R8005_M8009`, using Base and Step 150.
2. Evaluate a complete from-scratch generation of the problematic meeting; the fixed-prefix backend replay is omitted to prioritize this experiment.
3. Select the smallest non-default penalty that passes structural coverage and keeps each model's CER, cpCER, and DER@0.25 within 1 percentage point of its A0 result.
4. If a penalty passes, do not run temperature experiments. Regenerate all 56 Test meetings from scratch for Base and Step 150 under the identical selected penalty, then score them.
5. Only if every penalty fails, run temperature 0.2/0.4 with seeds 0, 1, and 2 as a fallback.

Any full Test score is an engineering re-evaluation, not a fresh blind Test, because the known Test failure `R8005_M8009` participates in decoding-configuration selection. Historical greedy outputs remain untouched.
