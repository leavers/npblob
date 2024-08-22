import { resolve } from "path";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  plugins: [
    dts({
      insertTypesEntry: true,
      exclude: ["vite.config.ts", "test/**/*"],
      strictOutput: true,
    }),
  ],
  build: {
    target: "esnext",
    sourcemap: true,
    minify: "esbuild",
    lib: {
      entry: resolve(__dirname, "src/npblob.ts"),
      name: "npblob",
      fileName: (format) => `npblob.${format}.js`,
      formats: ["es", "cjs", "umd"],
    },
  },
});
