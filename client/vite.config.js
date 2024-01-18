import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  assetsInclude: ["**/*.xlsx", "**/*.csv"],
  server: {
    host: true,
    strictPort: true,
    port: 5173,
  },
});
