import {mkdir, copyFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root = path.dirname(fileURLToPath(import.meta.url));
const modules = path.resolve(process.env.MOJEV_NODE_MODULES || path.join(root, 'node_modules'));
const output = path.resolve(process.argv[2] || path.join(root, '../docs/try'));
await mkdir(path.join(output, 'vendor'), {recursive:true});
for (const name of ['index.html','core.mjs','ui.mjs','worker.mjs']) {
  if (path.join(root,name) !== path.join(output,name)) await copyFile(path.join(root,name),path.join(output,name));
}
for (const name of ['ort.all.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
                    'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) {
  await copyFile(path.join(modules,'onnxruntime-web/dist',name),path.join(output,'vendor',name));
}
await copyFile(path.join(modules,'@huggingface/tokenizers/dist/tokenizers.mjs'),path.join(output,'vendor/tokenizers.mjs'));
await copyFile(path.join(modules,'@huggingface/tokenizers/LICENSE'),path.join(output,'vendor/tokenizers-LICENSE'));
await copyFile(path.join(root,'onnxruntime-LICENSE'),path.join(output,'vendor/onnxruntime-LICENSE'));
await copyFile(path.join(root,'onnxruntime-ThirdPartyNotices.txt'),path.join(output,'vendor/onnxruntime-ThirdPartyNotices.txt'));
console.log(`Built ${output}`);
