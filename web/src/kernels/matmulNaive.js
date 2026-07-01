import { TraceBuilder } from '../sim/TraceBuilder.js';
import {
  enumerateBlocks,
  enumerateThreads,
  blockKey,
  threadKey,
  chunk,
  WARP_SIZE,
} from '../sim/topology.js';

const source = [
  '__global__ void matmul_naive(float* C, float* A, float* B,',
  '                             int M, int N, int K) {',
  '  int row = blockIdx.y * blockDim.y + threadIdx.y;',
  '  int col = blockIdx.x * blockDim.x + threadIdx.x;',
  '  float sum = 0.0f;',
  '  for (int k = 0; k < K; k++) {',
  '    sum += A[row * K + k] * B[k * N + col];',
  '  }',
  '  C[row * N + col] = sum;',
  '}',
];

export const matmulNaive = {
  id: 'matmulNaive',
  name: 'Matrix Multiply — Naive (2D)',
  dims: 2,
  source,
  blurb:
    'Each thread computes one output cell C[row][col] by reading a whole row of A and a whole column of B from slow global memory.',
  params: [
    { key: 'blockX', label: 'blockDim.x', min: 1, max: 4, default: 2 },
    { key: 'blockY', label: 'blockDim.y', min: 1, max: 4, default: 2 },
    { key: 'gridX', label: 'gridDim.x (block cols)', min: 1, max: 4, default: 2 },
    { key: 'gridY', label: 'gridDim.y (block rows)', min: 1, max: 4, default: 2 },
    { key: 'K', label: 'K (shared dim)', min: 1, max: 8, default: 4 },
    { key: 'blocksPerWave', label: 'blocks per wave (SMs)', min: 1, max: 8, default: 2 },
  ],

  makeConfig(params) {
    const gridDim = { x: params.gridX, y: params.gridY, z: 1 };
    const blockDim = { x: params.blockX, y: params.blockY, z: 1 };
    const M = gridDim.y * blockDim.y;
    const N = gridDim.x * blockDim.x;
    const K = params.K;
    return { gridDim, blockDim, M, N, K };
  },

  makeDefaultData(params) {
    const { M, N, K } = this.makeConfig(params);
    // A[r][k] = r+1, B[k][c] = c+1  ->  C[r][c] = K*(r+1)*(c+1), easy to verify.
    const A = [];
    for (let r = 0; r < M; r++) for (let k = 0; k < K; k++) A.push(r + 1);
    const B = [];
    for (let k = 0; k < K; k++) for (let c = 0; c < N; c++) B.push(c + 1);
    const C = Array.from({ length: M * N }, () => null);
    return { A, B, C };
  },

  dataPanels(params) {
    const { M, N, K } = this.makeConfig(params);
    return [
      { key: 'A', label: 'A (M×K, input)', rows: M, cols: K, kind: 'input' },
      { key: 'B', label: 'B (K×N, input)', rows: K, cols: N, kind: 'input' },
      { key: 'C', label: 'C (M×N, output)', rows: M, cols: N, kind: 'output' },
    ];
  },

  generateTrace(params, data) {
    const { gridDim, blockDim, M, N, K } = this.makeConfig(params);
    const tb = new TraceBuilder({
      id: this.id,
      name: this.name,
      dims: this.dims,
      gridDim,
      blockDim,
      source,
      dataPanels: this.dataPanels(params),
      data: { A: [...data.A], B: [...data.B], C: [...data.C] },
    });

    const blocks = enumerateBlocks(gridDim);
    const threads = enumerateThreads(blockDim);
    const waves = chunk(blocks, params.blocksPerWave);

    waves.forEach((wave, wIdx) => {
      const waveThreadKeys = wave.flatMap((b) => threads.map((t) => threadKey(b, t)));
      tb.step({
        line: 0,
        phase: 'schedule',
        caption: `Wave ${wIdx + 1}: ${wave.length} block(s) scheduled onto SMs.`,
        activeBlocks: wave.map(blockKey),
        activeThreads: waveThreadKeys,
      });

      for (const b of wave) {
        const warps = chunk(threads, WARP_SIZE);
        warps.forEach((warp, warpIdx) => {
          const warpKeys = warp.map((t) => threadKey(b, t));
          const coords = warp.map((t) => ({
            row: b.by * blockDim.y + t.ty,
            col: b.bx * blockDim.x + t.tx,
            key: threadKey(b, t),
          }));

          tb.step({
            line: 3,
            phase: 'index',
            caption: `Block (${b.bx},${b.by}), warp ${warpIdx}: each thread maps to one C cell (row, col).`,
            activeBlocks: [blockKey(b)],
            activeThreads: warpKeys,
          });

          const readsA = [];
          const readsB = [];
          for (const { row, col } of coords) {
            for (let k = 0; k < K; k++) {
              readsA.push(row * K + k);
              readsB.push(k * N + col);
            }
          }
          tb.step({
            line: 6,
            phase: 'compute',
            caption:
              'Each thread walks k = 0..K-1, multiplying A[row][k] · B[k][col] and accumulating into sum. ' +
              'Note: every value comes from slow global memory.',
            activeBlocks: [blockKey(b)],
            activeThreads: warpKeys,
            reads: { A: readsA, B: readsB },
          });

          const nextC = [...tb.data.C];
          const writes = [];
          for (const { row, col } of coords) {
            let sum = 0;
            for (let k = 0; k < K; k++) sum += tb.data.A[row * K + k] * tb.data.B[k * N + col];
            nextC[row * N + col] = sum;
            writes.push(row * N + col);
          }
          tb.data = { ...tb.data, C: nextC };
          tb.markThreadsDone(warpKeys);

          tb.step({
            line: 8,
            phase: 'write',
            caption: 'C[row][col] = sum — the warp writes its output cells.',
            activeBlocks: [blockKey(b)],
            activeThreads: warpKeys,
            writes: { C: writes },
          });
        });
        tb.markBlockDone(blockKey(b));
      }
    });

    tb.step({ line: 9, phase: 'done', caption: 'Kernel complete — C is fully computed.' });
    return tb.build();
  },
};
