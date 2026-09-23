// Pure packing/scoring utilities shared by the worker and Node checks.
export function canonicalMenu(question, candidates) {
  if (new Set(candidates).size !== candidates.length) throw new Error('Candidates must be distinct.');
  const compare = (a, b) => {
    const x = Array.from(a, c => c.codePointAt(0)), y = Array.from(b, c => c.codePointAt(0));
    for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
    return x.length - y.length;
  };
  const order = candidates.map((_, i) => i).sort((a, b) => compare(candidates[a], candidates[b]));
  const sorted = order.map(i => candidates[i]);
  const parts = [question.replaceAll('_', ' '), 'kind: choice'];
  if (sorted.length <= 8) parts.push('options: ' + sorted.join(', '));
  return {order, sorted, prompt: parts.join(' | ')};
}

export function pack(context, questions, menus) {
  const ids = [...context];
  const nodes = context.map(() => ({kind: 'state', field: -1, option: -1}));
  questions.forEach((tokens, f) => {
    ids.push(...tokens);
    nodes.push(...tokens.map(() => ({kind: 'question', field: f, option: -1})));
    menus[f].forEach((tokens, n) => {
      ids.push(...tokens);
      nodes.push(...tokens.map(() => ({kind: 'option', field: f, option: n})));
    });
  });
  if (!ids.length || !context.length) throw new Error('Enter a non-empty context.');
  const length = ids.length, fields = questions.length;
  const width = Math.max(...menus.map(x => x.length));
  const state = new Float32Array(length);
  const field = new Float32Array(fields * length);
  const option = new Float32Array(fields * width * length);
  nodes.forEach((n, i) => {
    if (n.kind === 'state') state[i] = 1;
    if (n.kind === 'question') field[n.field * length + i] = 1;
    if (n.kind === 'option') option[(n.field * width + n.option) * length + i] = 1;
  });
  // Match PackedScorer.build_mask, including bidirectional attention within nodes.
  const mask = new Float32Array(length * length);
  for (let i = 0; i < length; i++) {
    const q = nodes[i];
    for (let j = 0; j < length; j++) {
      const k = nodes[j];
      const allow = k.kind === 'state' ||
        (q.kind !== 'state' && q.field === k.field && k.kind === 'question') ||
        (q.kind === 'option' && k.kind === 'option' && q.field === k.field && q.option === k.option);
      mask[i * length + j] = allow ? 0 : -3.4028234663852886e38;
    }
  }
  return {ids, mask, state, field, option, length, fields, width};
}

export function softmax(logits) {
  const max = logits.reduce((a, b) => Math.max(a, b), -Infinity);
  const values = logits.map(x => Math.exp(x - max));
  const sum = values.reduce((a, b) => a + b, 0);
  return values.map(x => x / sum);
}

export function embed(ids, weights, scales, width) {
  const output = new Float32Array(ids.length * width);
  ids.forEach((id, i) => {
    if (!Number.isInteger(id) || id < 0 || id >= scales.length) throw new Error('Invalid token ID');
    for (let j = 0; j < width; j++) output[i * width + j] = weights[id * width + j] * scales[id];
  });
  return output;
}
