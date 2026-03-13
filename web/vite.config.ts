import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    open: true,
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  build: {
    sourcemap: true,
  },
  plugins: [{
    name: 'strip-import-query',
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        if (req.url?.endsWith('.mjs?import')) {
          req.url = req.url.replace('?import', '');
        }
        next();
      });
    },
  }],
});
