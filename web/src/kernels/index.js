import { vectorAdd } from './vectorAdd.js';
import { matmulNaive } from './matmulNaive.js';
import { matmulTiled } from './matmulTiled.js';

export const kernels = [vectorAdd, matmulNaive, matmulTiled];

export function getKernel(id) {
  return kernels.find((k) => k.id === id) ?? kernels[0];
}

export function defaultParams(kernel) {
  const params = {};
  for (const p of kernel.params) params[p.key] = p.default;
  return params;
}
