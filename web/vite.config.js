import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base is relative so a static build works when served from a subpath later.
export default defineConfig({
  plugins: [react()],
  base: './',
});
