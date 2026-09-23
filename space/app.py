import os
import time
import json
from pathlib import Path
import spaces
import gradio as gr
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

MODEL = 'MoLeMo-Lab/mojev'
REVISION = '0c8695b6252f4205907433d4e196a94f032e60c3'
SOURCE = os.environ.get('MOJEV_SOURCE', MODEL)
kwargs = {} if SOURCE != MODEL else {'revision': REVISION}
processor = AutoProcessor.from_pretrained(SOURCE, trust_remote_code=True, **kwargs)
model = AutoModel.from_pretrained(SOURCE, trust_remote_code=True, **kwargs).eval()
model.encoder.set_attn_implementation('eager')
device = os.environ.get('MOJEV_DEVICE', 'cuda')
model.to(device)
EXAMPLES = json.loads(Path(__file__).with_name('examples.json').read_text())['examples']

def preview_images(paths):
    paths = paths or []
    return gr.update(value=[(path, f'Image {i + 1}') for i, path in enumerate(paths)],
                     visible=bool(paths), columns=1 if len(paths) == 1 else 2,
                     preview=False, selected_index=None)

def saved_example(index):
    example = EXAMPLES[index]
    return (example['context'], example['question'], '\n'.join(example['candidates']),
            example['files'] or None, example['probabilities'], preview_images(example['files']),
            f"Saved example · {example['tokens']:,} packed tokens · no inference")

def prepare(context, question, candidates, files):
    menu = [line.strip() for line in candidates.splitlines() if line.strip()]
    if len(menu) < 2: raise gr.Error('Enter at least two candidates.')
    if len(set(menu)) != len(menu): raise gr.Error('Candidates must be distinct.')
    if not context.strip() and not files: raise gr.Error('Add text or an image.')
    if not question.strip(): raise gr.Error('Enter a question.')
    for text in [context, question, *menu]:
        if any(tag in text for tag in ['<|image_pad|>','<|vision_start|>','<|vision_end|>','<|video_pad|>']):
            raise gr.Error('Use the image upload field for visual inputs.')
    images = []
    for file in files or []:
        with Image.open(file) as image:
            images.append(ImageOps.exif_transpose(image).convert('RGB'))
    prefix = '<|vision_start|><|image_pad|><|vision_end|>\n' * len(images)
    encoded = processor(text=[prefix + context], images=images or None, return_tensors='pt', truncation=False)
    order = sorted(range(len(menu)), key=lambda i:menu[i])
    sorted_menu = [menu[i] for i in order]
    prompt = question.replace('_',' ') + ' | kind: choice'
    if len(menu) <= 8: prompt += ' | options: ' + ', '.join(sorted_menu)
    pieces = [encoded['input_ids'][0].tolist(), processor.tokenizer.encode(prompt)]
    pieces += [processor.tokenizer.encode(candidate) for candidate in sorted_menu]
    length = sum(map(len,pieces))
    # Reject oversized requests before allocating the dense attention matrix.
    # No candidate is silently truncated.
    if length > 16384: raise gr.Error(f'This demo accepts 16,384 packed tokens per request; received {length:,}.')
    batch = dict(packed_ids=torch.tensor([sum(pieces,[])]),packed_mask=torch.ones(1,length,dtype=torch.bool),
                 context_span=torch.zeros(1,length),field_span=torch.zeros(1,1,length),
                 option_span=torch.zeros(1,1,len(menu),length),option_mask=torch.ones(1,1,len(menu),dtype=torch.bool))
    offset = 0
    for i,piece in enumerate(pieces):
        target = batch['context_span'][0] if i == 0 else batch['field_span'][0,0] if i == 1 else batch['option_span'][0,0,i-2]
        target[offset:offset+len(piece)] = 1
        offset += len(piece)
    for key in ['pixel_values','image_grid_thw','mm_token_type_ids']:
        if key in encoded: batch[key] = encoded[key]
    return batch,menu,order,length

@spaces.GPU(duration=30)
def infer(batch):
    with torch.inference_mode():
        batch = {key:value.to(device) for key,value in batch.items()}
        return model(batch)[0,0].float().softmax(-1).cpu().tolist()

def score(context, question, candidates, files):
    batch,menu,order,length = prepare(context,question,candidates,files)
    start = time.perf_counter()
    probabilities = infer(batch)
    result = {menu[index]:float(probabilities[i]) for i,index in enumerate(order)}
    return result, f'{length:,} packed tokens · {time.perf_counter()-start:.2f} s including GPU allocation'

with gr.Blocks(title='MoJev', delete_cache=(300,300)) as demo:
    gr.Markdown('# MoJev\n### Your context. Your candidates.\nScore custom candidates from text and images in one forward pass.')
    gr.Markdown('[Homepage](https://molemo-lab.github.io/mojev/) · [Model](https://huggingface.co/MoLeMo-Lab/mojev) · [Source](https://github.com/MoLeMo-Lab/mojev)')
    example_picker = gr.Radio([e['name'] for e in EXAMPLES],value=EXAMPLES[0]['name'],type='index',label='Saved examples — instant results, no GPU')
    with gr.Row():
        with gr.Column():
            context = gr.Textbox(label='Context', lines=5, value='The customer was charged twice for the same order and wants the duplicate payment returned.')
            files = gr.File(label='Images (optional)',file_count='multiple',file_types=['image'],type='filepath')
            question = gr.Textbox(label='Question',value='Which team should handle this request?')
            candidates = gr.Textbox(label='Candidates — one per line',lines=4,value='Billing\nTechnical support\nSales')
            button = gr.Button('Score candidates',variant='primary')
        with gr.Column():
            gallery = gr.Gallery(label='Image preview · click to enlarge',visible=False,
                                 elem_id='image-preview',
                                 interactive=False,columns=2,rows=1,height=340,object_fit='contain',
                                 allow_preview=True,buttons=['fullscreen'])
            output = gr.Label(label='Probability distribution',value=EXAMPLES[0]['probabilities'])
            timing = gr.Textbox(label='Execution',interactive=False,value=saved_example(0)[-1])
    gr.Markdown('Runs on Hugging Face ZeroGPU. Text and images are sent to this Space; no model download is needed. GPU access uses Hugging Face’s shared queue and daily quota.')
    button.click(score,[context,question,candidates,files],[output,timing],api_name='score',concurrency_limit=1)
    example_picker.change(saved_example,example_picker,[context,question,candidates,files,output,gallery,timing],queue=False,api_name=False)
    files.change(preview_images,files,gallery,queue=False,api_name=False,show_progress='hidden')
    for component in [context,question,candidates]:
        component.input(lambda: (None, 'Inputs changed · click Score candidates to run inference'),None,[output,timing],queue=False,api_name=False)
    for event in [files.upload,files.clear,files.delete]:
        event(lambda: (None, 'Inputs changed · click Score candidates to run inference'),None,[output,timing],queue=False,api_name=False)
demo.queue(max_size=20)
if __name__ == '__main__':
    demo.launch(max_file_size='20mb',css='''
    #image-preview .thumbnail-item { height: 300px !important; min-height: 0 !important; }
    #image-preview .thumbnail-item img { width: 100% !important; height: 100% !important; object-fit: contain !important; }
    ''')
