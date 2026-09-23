# Contributing

MoJev is useful when a change moves one of these questions forward:

1. **Decision quality:** does the model use the state to rank the schema's
   candidate values?
2. **Probability quality:** do reported probabilities track observed outcomes?
3. **Parallelism:** do independently specified questions remain independent
   when packed into one forward pass?
4. **Serving fidelity:** does the SDK request receive exactly the same decision
   as the in-process model?

Start with a minimal issue or pull request containing a state, a schema, an
expected observable, and the command that demonstrates it. For code changes:

```sh
pip install -e '.[dev]'
pytest -q
```

Add the smallest relevant test. The suite has dedicated checks for candidate
permutation, tree-mask reachability, calibration metrics, and the served path.
For a new result, include the split, the metric, and an appropriate control.

Keep checkpoints and generated datasets in the Hugging Face repositories; keep
this repository focused on code, tests, evaluation recipes, and documentation.
