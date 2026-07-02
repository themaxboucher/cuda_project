import { TraceBuilder } from '../sim/TraceBuilder.js';
import {
  enumerateBlocks,
  enumerateThreads,
  blockKey,
  threadKey,
  chunk,
  WARP_SIZE,
} from '../sim/topology.js';
import { GLOBAL_LATENCY } from '../sim/memory.js';

const source = [
  '__global__ void vector_add(int* c, int* a, int* b, int n) {',
  '  int i = blockIdx.x * blockDim.x + threadIdx.x;',
  '  if (i < n) {',
  '    c[i] = a[i] + b[i];',
  '  }',
  '}',
];

export const vectorAdd = {
  id: 'vectorAdd',
  name: 'Vector Add (1D)',
  dims: 1,
  source,
  blurb:
    'Each thread adds one pair of numbers: c[i] = a[i] + b[i]. The classic 1D kernel — a single row of threads split into blocks.',
  params: [
    { key: 'threadsPerBlock', label: 'threads / block (blockDim.x)', min: 1, max: 16, default: 4 },
    { key: 'blocks', label: 'blocks (gridDim.x)', min: 1, max: 12, default: 4 },
    { key: 'blocksPerWave', label: 'blocks per wave (SMs)', min: 1, max: 8, default: 2 },
  ],

  makeConfig(params) {
    const gridDim = { x: params.blocks, y: 1, z: 1 };
    const blockDim = { x: params.threadsPerBlock, y: 1, z: 1 };
    const n = gridDim.x * blockDim.x;
    return { gridDim, blockDim, n };
  },

  makeDefaultData(params) {
    const { n } = this.makeConfig(params);
    const a = Array.from({ length: n }, (_, i) => i);
    const b = Array.from({ length: n }, (_, i) => i * 2);
    const c = Array.from({ length: n }, () => null);
    return { a, b, c };
  },

  dataPanels(params) {
    const { n } = this.makeConfig(params);
    return [
      { key: 'a', label: 'a (input)', rows: 1, cols: n, kind: 'input' },
      { key: 'b', label: 'b (input)', rows: 1, cols: n, kind: 'input' },
      { key: 'c', label: 'c (output)', rows: 1, cols: n, kind: 'output' },
    ];
  },

  generateTrace(params, data) {
    const { gridDim, blockDim, n } = this.makeConfig(params);
    const tb = new TraceBuilder({
      id: this.id,
      name: this.name,
      dims: this.dims,
      gridDim,
      blockDim,
      source,
      dataPanels: this.dataPanels(params),
      data: { a: [...data.a], b: [...data.b], c: [...data.c] },
    });

    const blocks = enumerateBlocks(gridDim);
    const threads = enumerateThreads(blockDim);
    const waves = chunk(blocks, params.blocksPerWave);

    waves.forEach((wave, wIdx) => {
      const waveThreadKeys = wave.flatMap((b) => threads.map((t) => threadKey(b, t)));
      tb.step({
        line: 0,
        phase: 'schedule',
        caption: `Wave ${wIdx + 1}: the GPU schedules ${wave.length} block(s) onto its SMs — they light up together.`,
        activeBlocks: wave.map(blockKey),
        activeThreads: waveThreadKeys,
      });

      for (const b of wave) {
        const warps = chunk(threads, WARP_SIZE);
        warps.forEach((warp, warpIdx) => {
          const warpKeys = warp.map((t) => threadKey(b, t));
          const indices = warp.map((t) => b.bx * blockDim.x + t.tx);

          tb.step({
            line: 1,
            phase: 'index',
            caption:
              `Block ${b.bx}, warp ${warpIdx}: each thread computes its global index ` +
              `i = blockIdx.x(${b.bx})·blockDim.x(${blockDim.x}) + threadIdx.x → i ∈ {${indices.join(', ')}}`,
            activeBlocks: [blockKey(b)],
            activeThreads: warpKeys,
          });

          const nextC = [...tb.data.c];
          for (const i of indices) {
            if (i < n) nextC[i] = tb.data.a[i] + tb.data.b[i];
          }
          tb.data = { ...tb.data, c: nextC };
          tb.markThreadsDone(warpKeys);

          tb.step({
            line: 3,
            phase: 'compute',
            memory: 'global',
            cost: 3 * GLOBAL_LATENCY,
            caption: `c[i] = a[i] + b[i] — two global reads plus one global write per thread; the warp writes ${indices.length} result(s).`,
            activeBlocks: [blockKey(b)],
            activeThreads: warpKeys,
            reads: { a: indices, b: indices },
            writes: { c: indices },
          });
        });
        tb.markBlockDone(blockKey(b));
      }
    });

    tb.step({
      line: 5,
      phase: 'done',
      caption: 'Kernel complete — every thread wrote its output.',
    });

    return tb.build();
  },
};
