const $ = id => document.getElementById(id);
if (navigator.gpu) $('backend').value = 'webgpu';
let worker;
function resetWorker() {
  worker?.terminate();
  worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
  worker.onerror = event => fail(event.message);
  worker.onmessage = ({data}) => {
    if (data.type === 'status') $('status').textContent = data.message;
    if (data.type === 'error') fail(data.message);
    if (data.type === 'ready') {
      $('status').textContent = 'Model loaded. All scoring runs on this device.';
      $('score').disabled = false;
      const bytes = Object.values(data.manifest.files).reduce((sum, f) => sum + f.bytes, 0);
      $('size').textContent = `${Math.ceil(bytes / 2 ** 20)} MiB · cached locally`;
    }
    if (data.type === 'result') {
      $('score').disabled = false;
      $('status').textContent = 'Done';
      $('results').replaceChildren();
      data.candidates.forEach((candidate, i) => {
        const li = document.createElement('li'); li.className = 'result';
        const line = document.createElement('span');
        const label = document.createElement('span'); label.textContent = candidate;
        const probability = document.createElement('strong'); probability.textContent = `${(data.probabilities[i] * 100).toFixed(2)}%`;
        line.append(label, probability);
        const bar = document.createElement('div'); bar.className = 'bar'; bar.style.width = `${data.probabilities[i] * 100}%`;
        li.append(line, bar); $('results').append(li);
      });
      $('timing').textContent = `${data.tokens} packed tokens · ${(data.elapsed / 1000).toFixed(2)} s on this device`;
      $('output').hidden = false;
    }
  };
}
function fail(message) {
  worker?.terminate();
  $('status').textContent = message; $('status').classList.add('error');
  $('load').disabled = false; $('score').disabled = true;
  $('backend').disabled = false; $('unload').hidden = true;
}
$('load').onclick = () => {
  resetWorker(); $('load').disabled = true; $('score').disabled = true;
  $('backend').disabled = true; $('unload').hidden = false;
  $('status').classList.remove('error'); $('status').textContent = 'Starting runtime…';
  const base = new URLSearchParams(location.search).get('model') || 'https://huggingface.co/MoLeMo-Lab/mojev/resolve/main/browser/';
  worker.postMessage({type: 'load', base: new URL(base, location.href).href, backend: $('backend').value});
};
$('unload').onclick = () => {
  worker?.terminate();
  $('load').disabled = false; $('score').disabled = true; $('backend').disabled = false;
  $('unload').hidden = true; $('status').classList.remove('error');
  $('status').textContent = 'Model unloaded. Downloaded files remain cached.';
};
$('form').onsubmit = event => {
  event.preventDefault();
  const candidates = $('candidates').value.split('\n').map(x => x.trim()).filter(Boolean);
  if (candidates.length < 2) { $('status').textContent = 'Enter at least two candidates.'; return; }
  $('score').disabled = true; $('output').hidden = true;
  worker.postMessage({type: 'score', context: $('context').value, question: $('question').value, candidates});
};
