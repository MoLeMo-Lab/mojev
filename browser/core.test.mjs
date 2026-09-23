import assert from 'node:assert/strict';
import {pack, softmax, embed, canonicalMenu} from './core.mjs';
const p = pack([1, 2], [[3], [6]], [[[4], [5]], [[7], [8]]]);
assert.equal(p.length, 8);
const sees = (q, k) => p.mask[q * p.length + k] === 0;
assert.ok(sees(0, 1)); // state is bidirectional
assert.ok(sees(3, 0));
assert.ok(sees(3, 2));
assert.ok(sees(3, 3));
assert.ok(!sees(3, 4)); // sibling candidate
assert.ok(!sees(3, 5)); // other question
assert.ok(!sees(5, 2)); // sibling question
assert.ok(!sees(0, 2)); // state cannot see question
assert.deepEqual(softmax([1000, 1000]), [0.5, 0.5]);
assert.deepEqual([...embed([1], new Int8Array([1, 2, -3, 4]), new Float32Array([1, 2]), 2)], [-6, 8]);
const long = Array(321).fill(9);
assert.equal(pack([1], [[2]], [[long, [3]]]).ids.length, 324);
assert.deepEqual(canonicalMenu('q', ['😀','\uE000','A']).sorted, ['A','\uE000','😀']);
assert.equal(canonicalMenu('q', ['B','A']).prompt, canonicalMenu('q', ['A','B']).prompt);
assert.throws(() => canonicalMenu('q', ['A','A']), /distinct/);
console.log('Packing, sibling mask, stable softmax, embedding and uncapped candidates: passed');
