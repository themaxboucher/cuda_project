import { TraceBuilder } from '../sim/TraceBuilder.js';
import { enumerateBlocks, enumerateThreads, blockKey, threadKey, chunk } from '../sim/topology.js';

const source = [
  '#define TILE 2',
  '__global__ void matmul_tiled(float* C, float* A, float* B,',
  '                             int M, int N, int K) {',
  '  __shared__ float aTile[TILE][TILE];',
  '  __shared__ float bTile[TILE][TILE];',
  '  int tx = threadIdx.x, ty = threadIdx.y;',
  '  int row = blockIdx.y * TILE + ty;',
  '  int col = blockIdx.x * TILE + tx;',
  '  float sum = 0.0f;',
  '  for (int p = 0; p < K / TILE; p++) {',
  '    aTile[ty][tx] = A[row * K + p * TILE + tx];',
  '    bTile[ty][tx] = B[(p * TILE + ty) * N + col];',
  '    __syncthreads();',
  '    for (int i = 0; i < TILE; i++)',
  '      sum += aTile[ty][i] * bTile[i][tx];',
  '    __syncthreads();',
  '  }',
  '  C[row * N + col] = sum;',
  '}',
];

export const matmulTiled = {
  id: 'matmulTiled',
  name: 'Matrix Multiply — Tiled (shared memory)',
  dims: 2,
  source,
  blurb:
    'Threads in a block cooperate to copy a small TILE×TILE tile of A and B into fast shared memory, sync at a barrier, then reuse it — the real GPU speed trick.',
  params: [
    { key: 'TILE', label: 'TILE (block is TILE×TILE)', min: 2, max: 4, default: 2 },
    { key: 'gridX', label: 'gridDim.x (block cols)', min: 1, max: 4, default: 2 },
    { key: 'gridY', label: 'gridDim.y (block rows)', min: 1, max: 4, default: 2 },
    { key: 'phases', label: 'phases (K = phases × TILE)', min: 1, max: 4, default: 2 },
    { key: 'blocksPerWave', label: 'blocks per wave (SMs)', min: 1, max: 8, default: 2 },
  ],

  makeConfig(params) {
    const TILE = params.TILE;
    const gridDim = { x: params.gridX, y: params.gridY, z: 1 };
    const blockDim = { x: TILE, y: TILE, z: 1 };
    const M = gridDim.y * TILE;
    const N = gridDim.x * TILE;
    const K = params.phases * TILE;
    return { gridDim, blockDim, M, N, K, TILE };
  },

  makeDefaultData(params) {
    const { M, N, K } = this.makeConfig(params);
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
    const { gridDim, blockDim, M, N, K, TILE } = this.makeConfig(params);
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
    const phases = K / TILE;
    const emptyTile = () => Array.from({ length: TILE * TILE }, () => null);

    waves.forEach((wave, wIdx) => {
      const waveThreadKeys = wave.flatMap((b) => threads.map((t) => threadKey(b, t)));
      tb.step({
        line: 1,
        phase: 'schedule',
        caption: `Wave ${wIdx + 1}: ${wave.length} block(s) scheduled onto SMs.`,
        activeBlocks: wave.map(blockKey),
        activeThreads: waveThreadKeys,
      });

      for (const b of wave) {
        const blockThreadKeys = threads.map((t) => threadKey(b, t));
        const total = threads.length;
        const sums = {};
        for (const t of threads) sums[threadKey(b, t)] = 0;

        tb.shared = { aTile: emptyTile(), bTile: emptyTile(), TILE, block: `(${b.bx},${b.by})` };
        tb.step({
          line: 8,
          phase: 'init',
          caption: `Block (${b.bx},${b.by}) begins. sum = 0 for every thread; shared tiles are empty.`,
          activeBlocks: [blockKey(b)],
          activeThreads: blockThreadKeys,
        });

        for (let p = 0; p < phases; p++) {
          // LOAD phase
          const aTile = emptyTile();
          const bTile = emptyTile();
          const readsA = [];
          const readsB = [];
          for (const t of threads) {
            const row = b.by * TILE + t.ty;
            const col = b.bx * TILE + t.tx;
            const aIdx = row * K + p * TILE + t.tx;
            const bIdx = (p * TILE + t.ty) * N + col;
            aTile[t.ty * TILE + t.tx] = tb.data.A[aIdx];
            bTile[t.ty * TILE + t.tx] = tb.data.B[bIdx];
            readsA.push(aIdx);
            readsB.push(bIdx);
          }
          tb.shared = { aTile, bTile, TILE, block: `(${b.bx},${b.by})`, highlight: 'load' };
          tb.step({
            line: 10,
            phase: 'load',
            caption: `Phase ${p + 1}/${phases}: the ${total} threads cooperatively load one TILE×TILE tile of A and B into shared memory.`,
            activeBlocks: [blockKey(b)],
            activeThreads: blockThreadKeys,
            reads: { A: readsA, B: readsB },
          });

          // SYNC
          tb.shared = { ...tb.shared, highlight: null };
          tb.step({
            line: 12,
            phase: 'sync',
            caption: '__syncthreads(): every thread stops at the barrier until the tile is fully loaded.',
            activeBlocks: [blockKey(b)],
            activeThreads: blockThreadKeys,
            barrier: { arrived: total, total },
          });

          // COMPUTE from shared memory
          for (const t of threads) {
            const key = threadKey(b, t);
            let partial = 0;
            for (let i = 0; i < TILE; i++) {
              partial += aTile[t.ty * TILE + i] * bTile[i * TILE + t.tx];
            }
            sums[key] += partial;
          }
          tb.shared = { aTile, bTile, TILE, block: `(${b.bx},${b.by})`, highlight: 'compute' };
          tb.step({
            line: 14,
            phase: 'compute',
            caption:
              'Each thread multiplies its row of aTile by its column of bTile — fast shared-memory reads — and adds to sum.',
            activeBlocks: [blockKey(b)],
            activeThreads: blockThreadKeys,
          });

          // SYNC again before reusing shared memory
          tb.shared = { ...tb.shared, highlight: null };
          tb.step({
            line: 15,
            phase: 'sync',
            caption: '__syncthreads() again, so no thread overwrites the tile while others still read it.',
            activeBlocks: [blockKey(b)],
            activeThreads: blockThreadKeys,
            barrier: { arrived: total, total },
          });
        }

        // WRITE final sums
        const nextC = [...tb.data.C];
        const writes = [];
        for (const t of threads) {
          const row = b.by * TILE + t.ty;
          const col = b.bx * TILE + t.tx;
          nextC[row * N + col] = sums[threadKey(b, t)];
          writes.push(row * N + col);
        }
        tb.data = { ...tb.data, C: nextC };
        tb.markThreadsDone(blockThreadKeys);
        tb.markBlockDone(blockKey(b));

        tb.step({
          line: 17,
          phase: 'write',
          caption: 'After all phases, each thread writes its accumulated sum to C in global memory.',
          activeBlocks: [blockKey(b)],
          activeThreads: blockThreadKeys,
          writes: { C: writes },
        });

        tb.shared = null;
      }
    });

    tb.step({ line: 18, phase: 'done', caption: 'Kernel complete — C computed using shared-memory tiles.' });
    return tb.build();
  },
};
