"""Export MoJev's text scorer to length-dynamic, layer-wise browser graphs.

Run with the versions in requirements.txt. No training weights are changed.
The scripted delta-rule loop keeps sequence length dynamic in ONNX.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

logging.getLogger('onnxruntime.quantization.matmul_nbits_quantizer').setLevel(logging.WARNING)


@torch.jit.script
def delta_loop(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    q = q.float().transpose(1, 2)
    k = k.float().transpose(1, 2)
    v = v.float().transpose(1, 2)
    g = g.float().transpose(1, 2)
    beta = beta.float().transpose(1, 2)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q / (q.size(-1) ** 0.5)
    state = torch.zeros((v.size(0), v.size(1), k.size(-1), v.size(-1)), dtype=torch.float32)
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    for i in range(q.size(2)):
        kt = k[:, :, i]
        state = state * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        prediction = (state * kt.unsqueeze(-1)).sum(-2)
        update = (v[:, :, i] - prediction) * beta[:, :, i].unsqueeze(-1)
        state = state + kt.unsqueeze(-1) * update.unsqueeze(-2)
        outputs.append((state * q[:, :, i].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=2).transpose(1, 2)


def export_delta(query, key, value, g, beta, **kwargs):
    return delta_loop(query, key, value, g, beta).to(query.dtype), None


class Layer(nn.Module):
    def __init__(self, layer, rotary):
        super().__init__()
        self.layer = layer
        self.rotary = rotary

    def forward(self, hidden, mask):
        pos = torch.arange(hidden.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        rope = self.rotary(hidden, pos)
        return self.layer(hidden, position_embeddings=rope,
                          attention_mask=mask if self.layer.block_type == 'full_attention' else None)


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_norm = model.encoder.language_model.norm
        self.norm = model.norm
        self.context_proj = model.context_proj
        self.option_proj = model.option_proj
        self.rank = model.rank

    def forward(self, hidden, context, fields, options):
        h = self.norm(self.final_norm(hidden).float())
        c = (h * context.unsqueeze(-1)).sum(1) / context.sum(-1, keepdim=True).clamp_min(1)
        f = torch.einsum('bfl,blw->bfw', fields, h) / fields.sum(-1, keepdim=True).clamp_min(1)
        o = torch.einsum('bfnl,blw->bfnw', options, h) / options.sum(-1, keepdim=True).clamp_min(1)
        query = self.context_proj(c)[:, None, :] + self.context_proj(f)
        return (query[:, :, None, :] * self.option_proj(o)).sum(-1) / self.rank ** 0.5


def export(module, args, path, names, dynamic):
    torch.onnx.export(module, args, str(path), input_names=names, output_names=['output'],
                      dynamic_axes=dynamic, opset_version=17, dynamo=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / 'manifest.json').exists():
        prior = json.loads((out / 'manifest.json').read_text())
        if prior['precision']['layers'] != 'asymmetric int4, block 32':
            raise ValueError('Existing export uses a different quantization policy; use a new output directory')
    with (Path(args.source) / 'model.safetensors').open('rb') as handle:
        source_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    if source_sha != 'eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50':
        raise ValueError('Source checkpoint differs from the pinned MoJev release')
    model = AutoModel.from_pretrained(args.source, trust_remote_code=True).eval()
    model.encoder.set_attn_implementation('eager')
    text = model.encoder.language_model
    # Export starts from the saved BF16 values, represented in FP32 for ONNX.
    model.float()
    qwen.torch_chunk_gated_delta_rule = export_delta
    hidden = torch.randn(1, 9, model.width)
    mask = torch.zeros(1, 1, 9, 9)
    report = []
    for index, layer in enumerate(text.layers):
        wrapper = Layer(layer, text.rotary_emb).eval()
        path = out / f'layer-{index:02d}.onnx'
        if not args.resume or not path.exists():
            with torch.no_grad():
                export(wrapper, (hidden, mask), path, ['hidden', 'mask'],
                       {'hidden': {1: 'length'}, 'mask': {2: 'length', 3: 'length'}, 'output': {1: 'length'}})
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        for length in (5, 13):
            x = torch.randn(1, length, model.width)
            m = torch.zeros(1, 1, length, length)
            feeds = {'hidden': x.numpy(), 'mask': m.numpy()}
            feeds = {v.name: feeds[v.name] for v in session.get_inputs()}
            actual = session.run(None, feeds)[0]
            with torch.no_grad():
                expected = wrapper(x, m).numpy()
            error = float(np.max(np.abs(actual - expected)))
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
            report.append({'layer': index, 'length': length, 'max_abs_error': error})
        print(f'Validated layer {index}: {layer.block_type}', flush=True)
        quant_path = out / f'layer-{index:02d}-q4.onnx'
        if not args.resume or not quant_path.exists():
            quantizer = MatMulNBitsQuantizer(str(path), block_size=32, is_symmetric=False,
                                            accuracy_level=1, op_types_to_quantize=('MatMul',))
            quantizer.process()
            quantizer.model.save_model_to_file(str(quant_path), use_external_data_format=False)
        if args.probe:
            break
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.probe:
        return
    with torch.no_grad():
        weights = text.embed_tokens.weight.numpy()
        scale = np.maximum(np.abs(weights).max(axis=1) / 127, 1e-12).astype('<f4')
        quant = np.clip(np.round(weights / scale[:, None]), -127, 127).astype(np.int8)
        quant.tofile(out / 'embeddings-int8.bin')
        scale.tofile(out / 'embeddings-scales.bin')
        head = Head(model).eval()
        export(head, (hidden, torch.ones(1, 9), torch.ones(1, 1, 9), torch.ones(1, 1, 2, 9)),
               out / 'head.onnx', ['hidden', 'context', 'fields', 'options'],
               {'hidden': {1: 'length'}, 'context': {1: 'length'},
                'fields': {1: 'fields', 2: 'length'}, 'options': {1: 'fields', 2: 'options', 3: 'length'},
                'output': {1: 'fields', 2: 'options'}})
    AutoTokenizer.from_pretrained(args.source, trust_remote_code=True).save_pretrained(out)
    files = ['embeddings-int8.bin', 'embeddings-scales.bin', 'head.onnx', 'tokenizer.json', 'tokenizer_config.json'] + [f'layer-{i:02d}-q4.onnx' for i in range(len(text.layers))]
    manifest = {
        'format': 'mojev-browser-v1', 'source': 'MoLeMo-Lab/mojev',
        'source_revision': 'd439315bd9a11409584e16758bb76a9d75b5bea7',
        'source_sha256': source_sha,
        'width': model.width, 'layer_types': list(text.config.layer_types),
        'precision': {'layers': 'asymmetric int4, block 32', 'embeddings': 'int8 per row', 'head': 'float32'},
        'modality': 'text', 'files': {},
        'semantics': 'Matches the pinned Transformers scorer: full-attention layers use the tree mask; linear-attention layers use sequential recurrence.',
    }
    for filename in files:
        data = (out / filename).read_bytes()
        manifest['files'][filename] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
