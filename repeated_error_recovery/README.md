# Seed-1 random-error repetition experiment

This run keeps the preceding 150-step recovery experiment's optimizer, model,
loss, batching, sequence parallelism, scheduling, tokenizer, data and inference
parameters. It changes the experiment seed from 0 to 1 and changes each actual
substitution error into a run of three or four copies of that same wrong token.

The 2% temperature branch and the 98% greedy branch are unchanged. A gold,
illegal or low-mass result remains a clean auxiliary view. For every optimizer
update, two source slots are assigned total repeat length 3 and two are assigned
total repeat length 4; seed 1 determines the slot assignment.

Training runs in 30-step segments. After each checkpoint, the complete known
problem meeting `alimeeting/test/R8005_M8009` is decoded with temperature 0 and
all repetition/presence/frequency penalties disabled (repetition penalty 1.0).
Raw token loops, output-cap truncation, EOS, parse integrity and transcript end
coverage are checked. A failed gate stops before the next training segment.
