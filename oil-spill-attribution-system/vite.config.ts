import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        // Standard shadcn/ui convention: '@/*' -> 'src/*'. Nothing in this
        // codebase used '@/' before this (grepped, none found), so
        // repointing it from the project root to src/ is a safe, one-time
        // fix -- required for shadcn-style imports like
        // '@/components/ui/holographic-card' to resolve to the real
        // src/components/ui/ folder instead of a nonexistent root one.
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modifyâfile watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
      // Proxies to the real backend (backend/main.py, FastAPI) so the
      // frontend can just fetch('/api/...') with no CORS/base-URL config --
      // see backend/main.py's own module docstring for how to run it.
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
