import * as ort from './vendor/ort.all.min.mjs';
import {Tokenizer} from './vendor/tokenizers.mjs';
import {pack, embed, softmax, canonicalMenu} from './core.mjs';

ort.env.wasm.numThreads = 1; // Works on GitHub Pages without COOP/COEP headers.
ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
let tokenizer, manifest, weights, scales, sessions = [], head;
const status = message => postMessage({type: 'status', message});

async function asset(base, name) {
  const url = new URL(name, base).href;
  const cache = await caches.open('mojev-browser-v1');
  const entry = manifest.files[name];
  if (!entry) throw new Error(`${name}: missing manifest checksum`);
  const cacheUrl = new URL(url);
  cacheUrl.searchParams.set('mojev_sha256', entry.sha256);
  const cacheKey = cacheUrl.href;
  let response = await cache.match(cacheKey);
  if (!response) {
    response = await fetch(url);
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    await cache.put(cacheKey, response.clone());
  }
  const bytes = await response.arrayBuffer();
  if (entry) {
    if (bytes.byteLength !== entry.bytes) throw new Error(`${name}: incomplete download`);
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
      .map(x => x.toString(16).padStart(2, '0')).join('');
    if (digest !== entry.sha256) {
      await cache.delete(cacheKey);
      throw new Error(`${name}: checksum mismatch. Load again to retry.`);
    }
  }
  return bytes;
}

async function load(base, backend) {
  base = new URL(base, self.location).href.replace(/\/?$/, '/');
  const response = await fetch(new URL('manifest.json', base));
  if (!response.ok) throw new Error(`Model manifest: HTTP ${response.status}`);
  manifest = await response.json();
  if (manifest.format !== 'mojev-browser-v1') throw new Error('Unsupported model format');
  status('Loading tokenizer');
  // Tokenizer files are copied from the pinned model into the export bundle.
  const decodeJson = buffer => JSON.parse(new TextDecoder().decode(buffer));
  tokenizer = new Tokenizer(decodeJson(await asset(base, 'tokenizer.json')),
                            decodeJson(await asset(base, 'tokenizer_config.json')));
  status('Downloading and verifying embeddings');
  weights = new Int8Array(await asset(base, 'embeddings-int8.bin'));
  scales = new Float32Array(await asset(base, 'embeddings-scales.bin'));
  const options = {executionProviders: backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
                   graphOptimizationLevel: 'all'};
  for (let i = 0; i < manifest.layer_types.length; i++) {
    status(`Loading layer ${i + 1} / ${manifest.layer_types.length}`);
    const filename = `layer-${String(i).padStart(2, '0')}-q4.onnx`;
    sessions.push(await ort.InferenceSession.create(await asset(base, filename), options));
  }
  head = await ort.InferenceSession.create(await asset(base, 'head.onnx'), options);
  postMessage({type: 'ready', manifest});
}

async function score(context, question, candidates) {
  if (!head) throw new Error('Load the model first');
  const tokens = text => tokenizer.encode(text, {add_special_tokens: true}).ids;
  const {order, sorted, prompt} = canonicalMenu(question, candidates);
  const packed = pack(tokens(context), [tokens(prompt)], [sorted.map(tokens)]);
  const {length, width} = packed;
  const start = performance.now();
  let hidden = new ort.Tensor('float32', embed(packed.ids, weights, scales, manifest.width), [1, length, manifest.width]);
  const mask = new ort.Tensor('float32', packed.mask, [1, 1, length, length]);
  for (let i = 0; i < sessions.length; i++) {
    status(`Scoring layer ${i + 1} / ${sessions.length} · ${length} tokens`);
    const feeds = {hidden, mask};
    const output = await sessions[i].run(Object.fromEntries(sessions[i].inputNames.map(name => [name, feeds[name]])));
    hidden.dispose();
    hidden = output.output;
  }
  const feeds = {
    hidden, context: new ort.Tensor('float32', packed.state, [1, length]),
    fields: new ort.Tensor('float32', packed.field, [1, 1, length]),
    options: new ort.Tensor('float32', packed.option, [1, 1, width, length]),
  };
  const output = await head.run(feeds);
  const sortedLogits = [...output.output.data];
  const sortedProbabilities = softmax(sortedLogits);
  const logits = Array(candidates.length), probabilities = Array(candidates.length);
  order.forEach((original, i) => { logits[original] = sortedLogits[i]; probabilities[original] = sortedProbabilities[i]; });
  for (const t of Object.values(feeds)) t.dispose();
  mask.dispose();
  output.output.dispose();
  postMessage({type: 'result', candidates, logits, probabilities, tokens: length, elapsed: performance.now() - start});
}

let busy = false;
self.onmessage = async ({data}) => {
  if (busy) return;
  busy = true;
  try {
    if (data.type === 'load') await load(data.base, data.backend);
    else if (data.type === 'score') await score(data.context, data.question, data.candidates);
  } catch (error) {
    postMessage({type: 'error', message: error.message || String(error)});
  } finally { busy = false; }
};
