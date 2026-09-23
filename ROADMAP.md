# Next experiments

The first release establishes one compact, local model and its full evaluation
path. The project now needs results that change how people use it.

## 1. Make a decision benchmark people can argue with

Publish a fixed suite of state + schema tasks at 2, 4, 8, 16, 32, and 64
candidates. Report accuracy, Brier score, ECE, latency, state-shuffle control,
and candidate-order invariance for every release.

## 2. Show the application primitive

Ship three end-to-end examples where generated text is the wrong interface:
ticket routing, policy triage, and evidence-backed extraction. Each example
should expose the input schema, returned distribution, execution threshold, and
human escalation path.

## 3. Stress the architecture

Measure the packed tree mask against independent per-question requests:
throughput, latency, memory, exactness of isolated branches, and scaling with
the number of questions and candidates.

## 4. Train the next checkpoint

Expand beyond the first 0.85B checkpoint with a published data recipe,
training configuration, and the benchmark record above. The next model earns a
release by moving the decision-quality/calibration frontier, not by adding a
larger parameter count.

## 5. Add vision as a decision input

Train and evaluate a checkpoint that receives images in the packed state,
alongside a visual decision benchmark with calibrated output metrics.
