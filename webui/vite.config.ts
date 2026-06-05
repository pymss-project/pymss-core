import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/ui/",
  plugins: [svelte(), tailwindcss()],
  build: {
    outDir: "../pymss/server/webui_static",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/v1": "http://127.0.0.1:8000",
    },
  },
});
