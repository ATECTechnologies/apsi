import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";

const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"));

export default defineConfig({
  plugins: [react()],
  define: { __SETUP_VERSION__: JSON.stringify(pkg.setupVersion || pkg.version) },
  base: "./",
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false, target: "chrome120" },
});
