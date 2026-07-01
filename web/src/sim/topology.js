// Helpers for enumerating the CUDA thread hierarchy and grouping it the way the
// hardware does: blocks are scheduled in waves onto SMs, and threads inside a
// block run in warps of 32.

export const WARP_SIZE = 32;

export function enumerateBlocks(gridDim) {
  const blocks = [];
  for (let bz = 0; bz < gridDim.z; bz++) {
    for (let by = 0; by < gridDim.y; by++) {
      for (let bx = 0; bx < gridDim.x; bx++) {
        blocks.push({ bx, by, bz, index: blocks.length });
      }
    }
  }
  return blocks;
}

export function enumerateThreads(blockDim) {
  const threads = [];
  for (let tz = 0; tz < blockDim.z; tz++) {
    for (let ty = 0; ty < blockDim.y; ty++) {
      for (let tx = 0; tx < blockDim.x; tx++) {
        threads.push({ tx, ty, tz, tid: threads.length });
      }
    }
  }
  return threads;
}

export function blockKey(b) {
  return `${b.bx}_${b.by}_${b.bz}`;
}

export function threadKey(b, t) {
  return `${b.bx}_${b.by}_${b.bz}__${t.tx}_${t.ty}_${t.tz}`;
}

export function chunk(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) {
    out.push(arr.slice(i, i + size));
  }
  return out;
}
