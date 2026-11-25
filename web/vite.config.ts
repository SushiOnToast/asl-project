import { defineConfig, loadEnv } from "vite";

// base url for gh-pages
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  return { base: env.VITE_BASE || "/" };
});
