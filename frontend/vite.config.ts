import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backend = 'http://localhost:8090';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': backend,
      '/healthz': backend
    }
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 2000
  }
});
