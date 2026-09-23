"""Compare the complete quantized text scorer with the source checkpoint."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

CASES = [
    {'context':'The customer was charged twice for the same order and wants the duplicate payment returned.',
     'question':'Which team should handle this request?', 'candidates':['Billing','Technical support','Sales']},
    {'context':'The red box is empty. The blue box contains the key.',
     'question':'Which box contains the key?', 'candidates':['The red box','The blue box']},
    {'context':'A parcel must arrive by Friday. Express delivery arrives Thursday; standard delivery arrives Monday.',
     'question':'Which delivery option meets the deadline?', 'candidates':['Express delivery','Standard delivery']},
    {'context':'用户说：订单被重复扣款，希望退回多付的钱。',
     'question':'应该把请求转给哪个团队？', 'candidates':['账单与退款','技术支持','销售咨询']},
]

def batch_for(tokenizer, case):
    menu = sorted(case['candidates'])
    parts = [case['question'].replace('_',' '), 'kind: choice']
    if len(menu) <= 8: parts.append('options: ' + ', '.join(menu))
    pieces = [tokenizer.encode(case['context']), tokenizer.encode(' | '.join(parts))]
    pieces += [tokenizer.encode(c) for c in menu]
    length = sum(map(len, pieces))
    batch = {'packed_ids':torch.tensor([sum(pieces, [])]), 'packed_mask':torch.ones(1,length,dtype=torch.bool),
             'context_span':torch.zeros(1,length), 'field_span':torch.zeros(1,1,length),
             'option_span':torch.zeros(1,1,len(case['candidates']),length),
             'option_mask':torch.ones(1,1,len(case['candidates']),dtype=torch.bool)}
    offset = 0
    for i, piece in enumerate(pieces):
        target = batch['context_span'][0] if i == 0 else batch['field_span'][0,0] if i == 1 else batch['option_span'][0,0,i-2]
        target[offset:offset+len(piece)] = 1
        offset += len(piece)
    return batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--export',required=True)
    args = parser.parse_args()
    root = Path(args.export)
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(args.source, trust_remote_code=True).eval()
    model.encoder.set_attn_implementation('eager')
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    manifest = json.loads((root/'manifest.json').read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    sessions = [ort.InferenceSession(str(root/f'layer-{i:02d}-q4.onnx'),sess_options=options) for i in range(len(manifest['layer_types']))]
    head = ort.InferenceSession(str(root/'head.onnx'),sess_options=options)
    embeddings = np.memmap(root/'embeddings-int8.bin',dtype=np.int8,mode='r').reshape(-1,manifest['width'])
    scales = np.fromfile(root/'embeddings-scales.bin',dtype='<f4')
    report = {'source_revision':manifest['source_revision'], 'source_encoder_dtype':str(next(model.encoder.parameters()).dtype),
              'scope':'Four English/Chinese text cases for numerical comparison.', 'cases':[]}
    for case in CASES:
        batch = batch_for(tokenizer,case)
        with torch.no_grad():
            logits = model(batch).float().numpy()[0,0]
        reference = torch.softmax(torch.from_numpy(logits),-1).tolist()
        ids = batch['packed_ids'].numpy()
        hidden = (embeddings[ids].astype(np.float32) * scales[ids][...,None])
        mask = model.build_mask(batch['context_span'],batch['field_span'],batch['option_span']).numpy()
        start = time.perf_counter()
        for session in sessions:
            feeds = {'hidden':hidden,'mask':mask}
            hidden = session.run(None,{x.name:feeds[x.name] for x in session.get_inputs()})[0]
        quant_logits = head.run(None,{'hidden':hidden,'context':batch['context_span'].numpy(),
                                     'fields':batch['field_span'].numpy(),'options':batch['option_span'].numpy()})[0][0,0]
        probs = torch.softmax(torch.from_numpy(quant_logits),-1).tolist()
        order = sorted(range(len(case['candidates'])),key=lambda i:case['candidates'][i])
        inverse = np.argsort(order)
        reference = [reference[i] for i in inverse]
        probs = [probs[i] for i in inverse]
        quant_logits = quant_logits[inverse]
        record = {**case,'tokens':ids.shape[1],'reference':reference,'quantized':probs,'quantized_logits':quant_logits.tolist(),
                  'top1_agrees':int(np.argmax(reference))==int(np.argmax(probs)),
                  'max_probability_error':float(np.max(np.abs(np.array(reference)-probs))),
                  'onnx_cpu_seconds':time.perf_counter()-start}
        report['cases'].append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
    (root/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')

if __name__ == '__main__': main()
