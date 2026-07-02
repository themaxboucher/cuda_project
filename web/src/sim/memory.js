// Toy latency model used by the traces. The numbers are rough real-GPU orders
// of magnitude (global DRAM ~400 cycles, on-chip shared memory ~30 cycles),
// chosen so the playback speed difference is obvious, not to be exact.
export const GLOBAL_LATENCY = 400; // cycles per global-memory access
export const SHARED_LATENCY = 30; // cycles per shared-memory access
export const SYNC_COST = 20; // __syncthreads() barrier
export const BASE_COST = 4; // index math, scheduling, etc.
