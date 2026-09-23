"""Check dynamic recurrence and full-attention export before quantizing weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoConfig
import export as exporter

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
torch.set_num_threads(4)
config = AutoConfig.for_model(**json.loads(Path(args.config).read_text())['encoder_config']['text_config'])
config._attn_implementation = 'eager'
reference_delta = exporter.qwen.torch_chunk_gated_delta_rule
reports = []
for length in (5, 13, 65):
    q, k, v = [torch.randn(1, length, 2, 8) for _ in range(3)]
    g = -torch.rand(1, length, 2)
    beta = torch.rand(1, length, 2)
    expected, _ = reference_delta(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    actual = exporter.delta_loop(q, k, v, g, beta)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    reports.append({'check': 'recurrent-vs-chunk', 'length': length, 'max_abs_error': float((actual-expected).abs().max())})
exporter.qwen.torch_chunk_gated_delta_rule = exporter.export_delta
for index in (0, 3):
    wrapper = exporter.Layer(exporter.qwen.Qwen3_5DecoderLayer(config, index),
                             exporter.qwen.Qwen3_5TextRotaryEmbedding(config)).eval()
    path = out / f'probe-{index}.onnx'
    with torch.no_grad():
        exporter.export(wrapper, (torch.randn(1, 9, 1024), torch.zeros(1, 1, 9, 9)), path,
                        ['hidden', 'mask'], {'hidden': {1:'length'}, 'mask':{2:'length',3:'length'},'output':{1:'length'}})
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        for length in (5, 13, 65):
            hidden = torch.randn(1, length, 1024)
            mask = torch.triu(torch.full((1, 1, length, length), -3.4028234663852886e38), diagonal=1)
            feeds = {'hidden': hidden.numpy(), 'mask':mask.numpy()}
            actual = session.run(None, {x.name:feeds[x.name] for x in session.get_inputs()})[0]
            expected = wrapper(hidden, mask).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
            reports.append({'check':config.layer_types[index], 'length':length,'max_abs_error':float(np.abs(actual-expected).max())})
        quant_path = out / f'probe-{index}-q4.onnx'
        quantizer = exporter.MatMulNBitsQuantizer(str(path), block_size=32, is_symmetric=False, accuracy_level=1)
        quantizer.process()
        quantizer.model.save_model_to_file(str(quant_path), use_external_data_format=False)
        quant_session = ort.InferenceSession(str(quant_path), providers=['CPUExecutionProvider'])
        feeds = {'hidden': np.ones((1, 13, 1024), dtype=np.float32), 'mask': np.zeros((1, 1, 13, 13), dtype=np.float32)}
        result = quant_session.run(None, {x.name: feeds[x.name] for x in quant_session.get_inputs()})[0]
        (out / f'probe-{index}.json').write_text(json.dumps(result.flatten().tolist()))
(out / 'checks.json').write_text(json.dumps(reports, indent=2)+'\n')
print(json.dumps(reports, indent=2))
