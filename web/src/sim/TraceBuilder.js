// Accumulates a fully replayable list of steps. Every step carries a complete
// snapshot of the visible state (data arrays, shared memory, which threads glow)
// so the player can jump to any step forward OR backward with no re-simulation.
//
// Invariant: data/shared arrays are never mutated in place. On a write we clone
// the array, mutate the clone, and swap it in. That lets each step cheaply hold
// references to the arrays as they were at that moment.

export class TraceBuilder {
  constructor({ id, name, dims, gridDim, blockDim, source, dataPanels, data }) {
    this.meta = { id, name, dims, gridDim, blockDim, source, dataPanels };
    this.data = data; // { a: [...], b: [...], ... } current arrays
    this.shared = null; // current shared-memory snapshot or null
    this.steps = [];
    this.doneBlocks = [];
    this.doneThreads = [];
  }

  markBlockDone(key) {
    if (!this.doneBlocks.includes(key)) this.doneBlocks.push(key);
  }

  markThreadsDone(keys) {
    for (const k of keys) {
      if (!this.doneThreads.includes(k)) this.doneThreads.push(k);
    }
  }

  step(partial) {
    this.steps.push({
      line: -1,
      phase: '',
      caption: '',
      activeBlocks: [],
      activeThreads: [],
      reads: {},
      writes: {},
      barrier: null,
      shared: this.shared ? cloneShared(this.shared) : null,
      doneBlocks: [...this.doneBlocks],
      doneThreads: [...this.doneThreads],
      data: { ...this.data },
      ...partial,
    });
  }

  build() {
    return { ...this.meta, steps: this.steps };
  }
}

function cloneShared(shared) {
  return {
    ...shared,
    aTile: shared.aTile ? [...shared.aTile] : shared.aTile,
    bTile: shared.bTile ? [...shared.bTile] : shared.bTile,
  };
}
