# MoJev in the browser

Text decisions run locally with ONNX Runtime Web. The browser tokenizes the state,
question and candidates, runs all 24 text layers, and applies MoJev's trained
FP32 scoring head. Candidate probabilities come from the model's logits.
Candidates use the server's Unicode-code-point sort order and are mapped back to
the input order after scoring. Questions use the schema's choice-field format.
Question, state and candidate tokenization are untruncated.

## Model format

- Transformer linear weights: asymmetric INT4, block size 32, FP32 computation.
- Token embeddings: per-row INT8 with FP32 scales.
- Normalization and decision head: FP32.
- Execution: WebAssembly CPU, or WebGPU with CPU fallback for unsupported operators.
- Dynamic sequence lengths and candidate counts; no candidate-token truncation.
- Text input through the text encoder and decision head.

The export follows the pinned scorer's execution. Its six full-attention layers
receive the additive tree mask. In Transformers 5.17.0, the 18 Gated DeltaNet
layers ignore a four-dimensional attention mask and run sequential recurrence.
`export.py` preserves that behavior. It replaces chunked delta-rule evaluation
with a mathematically equivalent dynamic ONNX loop, leaving weights unchanged
before quantization.

## Export and verify

Use Python 3.12 and the source checkpoint identified in `export.py`.

```sh
python -m pip install -r browser/requirements.txt
python browser/check_export.py --config /path/to/source/config.json --output /path/to/probes
python browser/export.py --source /path/to/source --output /path/to/export
python browser/validate.py --source /path/to/source --export /path/to/export
```

`manifest.json` records the source checkpoint hash and checksums of all runtime
assets. `validation.json` contains layer-level export comparisons.
`comparison.json` records full-model probability changes on four English/Chinese
smoke examples. All four retain the source model's highest-probability candidate.
Their maximum absolute probability change after quantization is 4.53 percentage
points. Browser WebGPU probabilities match desktop ONNX within 0.000003 on these
cases.

## Build the static page

```sh
cd browser
npm install
npm test
npm run build
```

The build copies the page and pinned runtime files into `docs/try/`.
Serve the repository using an HTTP server, and set the model bundle URL with
`?model=https://your-host/path/to/export/`. Use `browser/verify.html` with the same
model parameter to compare browser outputs with `comparison.json`.

The worker verifies downloaded asset hashes and caches the model locally.
Scoring inputs stay in the worker. No inference server is called. WebAssembly
uses one thread so the page works without cross-origin-isolation headers.
The dense attention mask uses `4 × sequence_length²` bytes; memory requirements
therefore grow with the packed context, question and all candidate tokens.
