---
title: MoJev
emoji: 🧭
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.15.1
python_version: 3.12
app_file: app.py
pinned: false
license: mit
models:
  - MoLeMo-Lab/mojev
---

# MoJev

Text and image decisions with runtime-defined candidates. Inference runs on
Hugging Face ZeroGPU; no model weights are downloaded to your browser.

Five saved examples include precomputed probabilities from the source checkpoint.
Selecting an example displays its inputs and results without inference or GPU
allocation. Editing an input clears the saved result; click **Score candidates**
to compute a new prediction.

[Homepage](https://molemo-lab.github.io/mojev/) ·
[Model](https://huggingface.co/MoLeMo-Lab/mojev) ·
[Source](https://github.com/MoLeMo-Lab/mojev) ·
[Preprint](https://github.com/MoLeMo-Lab/mojev/blob/master/paper/mojev-preprint.pdf)

Images and text are sent to this Space for inference. The app does not log inputs.
Temporary uploaded files are removed by Gradio's cache cleanup.
